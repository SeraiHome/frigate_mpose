"""Shared pool for pose activity detectors.

A `PoseActivityPool` hosts ONE `PoseActivityDetector` instance in a dedicated
worker process (`PoseActivityDetectProcess`) and exposes a thread-safe
request/response interface to multiple cameras in the main Frigate process.

Design rationale
----------------

Frigate's per-camera `PoseDetectProcess` model is correct for pose detection
(MediaPipe VIDEO mode holds an implicit in-interpreter ROI hint that cannot
cross process boundaries — see `docs/PLUGIN_ARCHITECTURE.md` §2b). Activity
classification is a different kind of temporal state: the sliding window is
held as an explicit Python dict keyed by `(camera, pose_id)` in the detector
instance. That state can safely centralize into one pool worker as long as
FIFO-per-(camera, pose_id) ordering is preserved by the request queue — which
it trivially is because each camera's `TrackedPoseProcessor` publishes calls
in order.

Benefits on CPU (today):
  - One TFLite interpreter instead of N per-camera instances (RAM savings)
  - Fire-and-forget async means camera threads never block on classifier latency
  - Per-track state is isolated by `(camera, pose_id)` — no cross-camera leakage
  - Ready for hardware acceleration: when a quantized/Coral-compatible model
    exists, pointing `device: usb:0` at one pool entry serves all cameras
    from a single accelerator with zero other changes.

Message protocol
----------------

Input queue (camera → pool):

    ("detect", camera: str, pose_id: str, keypoints: np.ndarray,
     frame_width: int, frame_height: int)

    ("forget", camera: str, pose_id: str)

    None  → shutdown sentinel

Output queue (pool → listener):

    (camera: str, pose_id: str, action_value: str, confidence: float)

The output tuple uses the string value of the action enum (e.g. "falling")
so the consumer doesn't need to import `PoseActionTypeEnum` — it rehydrates
the enum on the receiving side.
"""

from __future__ import annotations

import logging
import queue
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict, Optional, Tuple

import numpy as np

from frigate.const import PROCESS_PRIORITY_MED
from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors.base import PoseActivityDetector
from frigate.pose_activity_detectors.detector_config import (
    BasePoseActivityDetectorConfig,
)
from frigate.util.process import FrigateProcess

logger = logging.getLogger(__name__)


# Sentinel values for message op codes in the input queue
_OP_DETECT = "detect"
_OP_FORGET = "forget"


class PoseActivityDetectProcess(FrigateProcess):
    """Dedicated worker process hosting one activity detector instance.

    Drains requests from `input_queue` and publishes results on `output_queue`.
    The detector is (re)instantiated inside `run()` because the detector's
    underlying TFLite interpreter is not safe to move across a process boundary
    via pickle.
    """

    def __init__(
        self,
        name: str,
        input_queue: MpQueue,
        output_queue: MpQueue,
        detector_config: BasePoseActivityDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(
            stop_event,
            PROCESS_PRIORITY_MED,
            name=name,
            daemon=True,
        )
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector_config = detector_config

    def run(self) -> None:
        # Import inside run() to keep the plugin registry load local to the
        # worker process. The main process's DETECTOR_REGISTRY is not shared
        # through fork on all platforms (Windows spawn semantics).
        from frigate.pose_activity_detectors import (
            create_activity_detector,
            load_detector_plugins,
        )

        self.pre_run_setup(None)

        try:
            load_detector_plugins()
            detector = create_activity_detector(self.detector_config)
        except Exception:
            logger.exception("Pool worker failed to instantiate activity detector")
            return

        if detector is None or not getattr(detector, "initialized", False):
            logger.error(
                "Pool worker activity detector failed to initialize "
                "(type=%s). Worker exiting; pool will fall back to CPU-local "
                "detector on each camera.",
                self.detector_config.type,
            )
            return

        logger.info(
            f"PoseActivityDetectProcess '{self.name}' ready "
            f"(type={self.detector_config.type})"
        )

        while not self.stop_event.is_set():
            try:
                msg = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if msg is None:
                # Shutdown sentinel
                break

            try:
                op = msg[0]
            except (IndexError, TypeError):
                logger.debug(f"pool worker received malformed message: {msg!r}")
                continue

            if op == _OP_DETECT:
                try:
                    (_op, camera, pose_id, keypoints, frame_width, frame_height) = msg
                except ValueError:
                    logger.debug(f"malformed detect message: {msg!r}")
                    continue
                try:
                    action, confidence = detector.detect(
                        keypoints,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        pose_id=pose_id,
                        camera=camera,
                    )
                    self.output_queue.put_nowait(
                        (
                            camera,
                            pose_id,
                            action.value if hasattr(action, "value") else str(action),
                            float(confidence),
                        )
                    )
                except queue.Full:
                    logger.warning(
                        "PoseActivity pool output queue full — dropping result "
                        f"({camera}/{pose_id})"
                    )
                except Exception:
                    logger.exception(
                        f"pool detect failed for ({camera}/{pose_id})"
                    )

            elif op == _OP_FORGET:
                try:
                    (_op, camera, pose_id) = msg
                except ValueError:
                    logger.debug(f"malformed forget message: {msg!r}")
                    continue
                try:
                    detector.forget(pose_id, camera=camera)
                except TypeError:
                    # Legacy detectors without the camera kwarg
                    try:
                        detector.forget(pose_id)
                    except Exception:
                        logger.debug(
                            f"pool legacy forget failed for ({camera}/{pose_id})",
                            exc_info=True,
                        )
                except Exception:
                    logger.debug(
                        f"pool forget failed for ({camera}/{pose_id})",
                        exc_info=True,
                    )

            else:
                logger.debug(f"pool worker received unknown op {op!r}")

        logger.info(f"PoseActivityDetectProcess '{self.name}' exiting")


class PoseActivityPool:
    """Main-process wrapper for a pool worker.

    Owns the input/output queues, spawns the `PoseActivityDetectProcess`, and
    exposes a thread-safe `submit` / `forget` / `get_result` API for the
    `TrackedPoseProcessor`'s listener thread. Multiple cameras share one
    pool instance.
    """

    def __init__(
        self,
        name: str,
        detector_config: BasePoseActivityDetectorConfig,
        stop_event: MpEvent,
        max_queue_size: int = 1024,
    ) -> None:
        self.name = name
        self.detector_config = detector_config
        self.stop_event = stop_event
        self.input_queue: MpQueue = MpQueue(maxsize=max_queue_size)
        self.output_queue: MpQueue = MpQueue(maxsize=max_queue_size)
        self.process = PoseActivityDetectProcess(
            name=f"pose-activity-pool-{name}",
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            detector_config=detector_config,
            stop_event=stop_event,
        )
        self.process.start()

    def submit(
        self,
        camera: str,
        pose_id: str,
        keypoints: np.ndarray,
        frame_width: Optional[int],
        frame_height: Optional[int],
    ) -> bool:
        """Enqueue a classification request. Non-blocking.

        Returns True if the request was accepted, False if the queue is full
        (in which case the caller keeps its last-known action until the next
        successful submit).
        """
        try:
            self.input_queue.put_nowait(
                (_OP_DETECT, camera, pose_id, keypoints, frame_width, frame_height)
            )
            return True
        except queue.Full:
            logger.debug(
                f"pool '{self.name}' input queue full — dropping detect "
                f"({camera}/{pose_id})"
            )
            return False

    def forget(self, camera: str, pose_id: str) -> None:
        """Enqueue a per-track state cleanup. Non-blocking."""
        try:
            self.input_queue.put_nowait((_OP_FORGET, camera, pose_id))
        except queue.Full:
            logger.debug(
                f"pool '{self.name}' input queue full — dropping forget "
                f"({camera}/{pose_id})"
            )

    def get_result(
        self, timeout: float = 0.0
    ) -> Optional[Tuple[str, str, str, float]]:
        """Retrieve one inference result. Returns None if none available."""
        try:
            if timeout > 0:
                return self.output_queue.get(timeout=timeout)
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the worker to exit and join."""
        try:
            self.input_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self.process.join(timeout=5.0)
        except Exception:
            pass


class PoolActivityDetectorProxy(PoseActivityDetector):
    """Drop-in replacement for a per-camera `PoseActivityDetector` instance
    that forwards `detect()` calls into a shared `PoseActivityPool`.

    `detect()` is fire-and-forget: it submits the request and immediately
    returns the track's last-known `(action, confidence)` pair. A listener
    thread running in `TrackedPoseProcessor` drains the pool's output queue
    and updates the last-known cache keyed by `(camera, pose_id)`, so the
    next `detect()` call returns fresh classification — at worst one frame
    behind (~200 ms at 5 fps).

    This proxy is assigned to `TrackedPose.active_detector` when the camera
    is configured to use a pool (via `pose.activity_detector_pool`). The
    existing `_analyze_pose_action()` path in `tracked_pose.py` continues
    to work unchanged — it just happens to be talking to a proxy instead
    of a concrete detector.
    """

    def __init__(self, pool: PoseActivityPool, last_known_cache: Dict[Any, Any]):
        # PoseActivityDetector.__init__ in base.py is abstract; we bypass
        # it deliberately since the proxy holds no real detector state.
        self.initialized = True
        self.pool = pool
        self._last_known = last_known_cache

    def reset(self) -> None:
        # Nothing to reset on the proxy itself; the pool worker holds state.
        return

    def detect(
        self,
        keypoints: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        pose_id: Optional[str] = None,
        camera: Optional[str] = None,
    ) -> Tuple[PoseActionTypeEnum, float]:
        if pose_id is None:
            # Can't route without a stable track id; return a no-op.
            return PoseActionTypeEnum.standing, 0.0

        self.pool.submit(
            camera=camera or "",
            pose_id=pose_id,
            keypoints=keypoints,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        cached = self._last_known.get((camera, pose_id))
        if cached is not None:
            return cached
        return PoseActionTypeEnum.standing, 0.0

    def forget(self, pose_id: str, camera: Optional[str] = None) -> None:
        self.pool.forget(camera=camera or "", pose_id=pose_id)
        self._last_known.pop((camera, pose_id), None)
