import base64
import json
import logging
import queue
import threading
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from frigate.pose_activity_detectors.pool import PoseActivityPool

import cv2

from frigate.camera.state import CameraState
from frigate.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.comms.dispatcher import Dispatcher
from frigate.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.config import FrigateConfig
from frigate.config.camera import CameraConfig
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import FAST_QUEUE_TIMEOUT
from frigate.events.pose_types import (
    PoseActionTypeEnum,
    PoseEventStateEnum,
    PoseEventTypeEnum,
)
from frigate.pose_activity_detectors import create_activity_detector
from frigate.pose_activity_detectors.base import PoseActivityDetector
from frigate.pose_activity_detectors.detector_config import create_detector_config
from frigate.track.tracked_pose import TrackedPose
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


class PoseProcessingState(str, Enum):
    complete = "complete"
    start = "start"
    end = "end"


class TrackedPoseProcessor(threading.Thread):
    """
    Processes tracked poses from pose detectors and assigns appropriate activity detectors.

    Each camera can have its own activity detector configuration, and poses from different
    cameras will be analyzed with their specific detectors.
    """

    def __init__(
        self,
        config: FrigateConfig,
        dispatcher: Dispatcher,
        tracked_poses_queue: MpQueue,
        stop_event: MpEvent,
        ptz_autotracker_thread=None,
        activity_pools: Optional[Dict[str, "PoseActivityPool"]] = None,
    ) -> None:
        super().__init__(name="pose_processor")
        self.config = config
        self.dispatcher = dispatcher
        self.tracked_poses_queue = tracked_poses_queue
        self.stop_event: MpEvent = stop_event
        self.ptz_autotracker_thread = ptz_autotracker_thread
        self.camera_states: dict[str, CameraState] = {}
        self.frame_manager = SharedMemoryFrameManager()

        # Activity detectors for each camera. Each entry is either a
        # locally-instantiated `PoseActivityDetector` (today's default) or
        # a `PoolActivityDetectorProxy` that forwards to a shared pool
        # worker when the camera's config references
        # `pose.activity_detector_pool`.
        self.activity_detectors: Dict[str, PoseActivityDetector] = {}

        # Pool wrappers keyed by name, spawned upstream in frigate/app.py.
        # Empty dict means "no pools configured" → all cameras use local
        # detectors via the existing initialize_activity_detector() path.
        self.activity_pools: Dict[str, "PoseActivityPool"] = activity_pools or {}

        # Shared last-known-classification cache for pool-routed cameras.
        # Keyed by (camera, pose_id) → (PoseActionTypeEnum, confidence).
        # The pool result listener thread (below) mutates this; the pool
        # proxies read from it when a camera's tracked_pose asks for the
        # current classification.
        self._pool_last_known: Dict[tuple, tuple] = {}

        # Listener thread started lazily on first camera that needs a pool.
        # None until initialize_activity_detector() binds a camera to a
        # pool and kicks off the drain loop.
        self._pool_listener_thread: Optional[threading.Thread] = None

        self.camera_config_subscriber = CameraConfigUpdateSubscriber(
            self.config,
            self.config.cameras,
            [
                CameraConfigUpdateEnum.add,
                CameraConfigUpdateEnum.enabled,
                CameraConfigUpdateEnum.remove,
                CameraConfigUpdateEnum.zones,
            ],
        )

        self.detection_publisher = DetectionPublisher(DetectionTypeEnum.all.value)
        # Side-channel for shipping per-frame webp thumbnail bytes from
        # integration.py through to pose_consumer without changing the
        # tuple shape on the main "video" sub-topic (which other consumers
        # expect to be a fixed 6-tuple).
        self.pose_thumb_publisher = DetectionPublisher(DetectionTypeEnum.pose.value)
        self.event_sender = EventUpdatePublisher()
        self.event_end_subscriber = EventEndSubscriber()

        self.camera_activity: dict[str, dict[str, Any]] = {}

        # Zone data for pose tracking
        self.pose_zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.active_pose_zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Initialize camera states and activity detectors for enabled cameras with pose detection
        for camera in self.config.cameras.keys():
            camera_config = self.config.cameras[camera]
            if (
                camera_config.enabled
                and hasattr(camera_config, "pose")
                and camera_config.pose.enabled
            ):
                logger.info(
                    f"Initializing pose tracking and activity detection for camera {camera}"
                )
                self.create_camera_state(camera)
                self.initialize_activity_detector(camera)
            else:
                reasons = []
                if not camera_config.enabled:
                    reasons.append("camera disabled")
                if not hasattr(camera_config, "pose") or not camera_config.pose.enabled:
                    reasons.append("pose detection disabled")
                if reasons:
                    logger.debug(
                        f"Skipping pose activity detection for camera {camera}: {', '.join(reasons)}"
                    )

    def initialize_activity_detector(self, camera: str) -> None:
        """
        Initialize the activity detector for a specific camera based on its configuration.
        Falls back to heuristic detector if specified detector fails to initialize.

        Args:
            camera: Camera name to initialize the detector for
        """
        if camera not in self.config.cameras:
            return

        camera_config: CameraConfig = self.config.cameras[camera]

        # Skip if pose detection is not enabled for this camera
        if not hasattr(camera_config, "pose") or not camera_config.pose.enabled:
            logger.debug(
                f"Pose detection not enabled for camera {camera}, skipping activity detector initialization"
            )
            return

        # Pool routing takes precedence over inline activity_detector config.
        # When set, assign a proxy detector to this camera that forwards
        # classification calls into the named shared pool worker and returns
        # the track's last-known action without blocking.
        pool_name = getattr(
            camera_config.pose, "activity_detector_pool", None
        )
        if pool_name:
            pool = self.activity_pools.get(pool_name)
            if pool is None:
                logger.error(
                    f"Camera {camera} references activity_detector_pool "
                    f"'{pool_name}' but no such pool is configured — "
                    f"falling back to inline activity_detector."
                )
            else:
                from frigate.pose_activity_detectors.pool import (
                    PoolActivityDetectorProxy,
                )

                proxy = PoolActivityDetectorProxy(
                    pool=pool,
                    last_known_cache=self._pool_last_known,
                )
                self.activity_detectors[camera] = proxy
                self._ensure_pool_listener_started()
                logger.info(
                    f"Camera {camera} bound to pose activity detector pool "
                    f"'{pool_name}' (fire-and-forget async)"
                )
                return

        # Create detector from camera config
        if (
            not hasattr(camera_config.pose, "activity_detector")
            or not camera_config.pose.activity_detector
        ):
            # No specific activity detector configured, use default heuristic detector
            logger.info(
                f"No activity detector configured for {camera}, using default heuristic detector"
            )
            config_dict = {"type": "heuristic"}
        else:
            # Use camera-specific activity detector configuration
            detector_config = camera_config.pose.activity_detector
            config_dict = detector_config.model_dump()
            logger.info(
                f"Initializing {detector_config.type} activity detector for camera {camera}"
            )

        # Try to initialize the specified detector
        success = self._try_initialize_detector(camera, config_dict)

        # Fall back to heuristic detector if the specified detector failed
        if not success and config_dict.get("type") != "heuristic":
            logger.warning(
                f"Specified detector failed, falling back to heuristic detector for camera {camera}"
            )
            success = self._try_initialize_detector(camera, {"type": "heuristic"})

            if not success:
                logger.error(
                    f"Both primary and fallback detectors failed for camera {camera}. "
                    f"Pose detection will continue but activity detection will be disabled."
                )

    def _ensure_pool_listener_started(self) -> None:
        """Lazy-start the single background thread that drains every pool's
        output queue and updates the shared `_pool_last_known` cache keyed
        by (camera, pose_id).

        One listener is sufficient regardless of how many pools are
        configured — the listener round-robins non-blocking polls across
        every registered pool per tick.
        """
        if self._pool_listener_thread is not None:
            return

        def _run() -> None:
            logger.info(
                f"Pose activity pool listener started "
                f"({len(self.activity_pools)} pool(s))"
            )
            while not self.stop_event.is_set():
                progressed = False
                for pool in self.activity_pools.values():
                    try:
                        result = pool.get_result(timeout=0.0)
                    except Exception:
                        result = None
                    if result is None:
                        continue
                    progressed = True
                    try:
                        camera, pose_id, action_value, confidence = result
                        try:
                            action_enum = PoseActionTypeEnum(action_value)
                        except ValueError:
                            logger.debug(
                                f"pool result with unknown action {action_value!r}"
                            )
                            continue
                        self._pool_last_known[(camera, pose_id)] = (
                            action_enum,
                            float(confidence),
                        )
                    except Exception:
                        logger.exception("pool listener failed to apply result")
                if not progressed:
                    # No pool had a ready result — small sleep to avoid
                    # busy-waiting. Tuned to give worst-case ~50 ms lag
                    # from worker publish to cache update, well under
                    # the 200 ms frame interval at 5 fps.
                    try:
                        self.stop_event.wait(timeout=0.05)
                    except Exception:
                        pass
            logger.info("Pose activity pool listener exiting")

        self._pool_listener_thread = threading.Thread(
            name="pose-activity-pool-listener",
            target=_run,
            daemon=True,
        )
        self._pool_listener_thread.start()

    def _try_initialize_detector(self, camera: str, config_dict: dict) -> bool:
        """
        Attempt to initialize an activity detector with the given configuration.

        Args:
            camera: Camera name
            config_dict: Detector configuration dictionary

        Returns:
            True if initialization was successful, False otherwise
        """
        # Create detector configuration
        detector_config = create_detector_config(config_dict)
        if detector_config is None:
            logger.error(f"Failed to create detector configuration for camera {camera}")
            return False

        # Create activity detector instance
        try:
            activity_detector = create_activity_detector(detector_config)
            if activity_detector and activity_detector.initialized:
                logger.info(
                    f"Successfully initialized activity detector for camera {camera}: {detector_config.type}"
                )
                self.activity_detectors[camera] = activity_detector
                return True
            else:
                logger.error(
                    f"Failed to initialize activity detector for camera {camera}"
                )
                return False
        except Exception as e:
            logger.error(f"Error creating activity detector for camera {camera}: {e}")
            return False

    def create_camera_state(self, camera: str) -> None:
        """Creates a new camera state for pose tracking."""

        from frigate.track.tracked_object import TrackedObject

        class _PoseAdapter:
            """Adapter to present a TrackedObject or dict as a TrackedPose-like object
            for the pose processing callbacks.
            """

            def __init__(self, source):
                # source may be TrackedPose, TrackedObject, or dict
                self._source = source

                if isinstance(source, TrackedPose):
                    return

                if isinstance(source, TrackedObject):
                    data = source.to_dict()
                elif isinstance(source, dict):
                    data = source
                else:
                    data = {}

                self.obj_data = data
                self.previous = getattr(source, "previous", data.copy())
                self.has_snapshot = data.get("has_snapshot", False)
                self.has_clip = data.get("has_clip", False)
                self.false_positive = data.get("false_positive", True)
                self.entered_zones = set(data.get("entered_zones", []))
                self.current_zones = set(data.get("current_zones", []))

                # Action may be stored as explicit 'action' or inside sub_label
                action = data.get("action")
                if action is None:
                    sub = data.get("sub_label")
                    if sub and isinstance(sub, (list, tuple)) and len(sub) > 0:
                        action = sub[0]
                self.action = action

                # Confidence may be in action_confidence or score
                self.confidence = data.get("action_confidence", data.get("score", 0.0))

            def to_dict(self):
                if isinstance(self._source, TrackedPose):
                    return self._source.to_dict()
                return self.obj_data

        def start(camera: str, pose: TrackedPose, frame_name: str) -> None:
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.start,
                    camera,
                    frame_name,
                    (
                        pose.to_dict()
                        if hasattr(pose, "to_dict")
                        else _PoseAdapter(pose).to_dict()
                    ),
                )
            )

        def update(camera: str, pose: TrackedPose, frame_name: str) -> None:
            adapter = pose if isinstance(pose, TrackedPose) else _PoseAdapter(pose)
            # set snapshot/clip flags based on adapter
            try:
                adapter.has_snapshot = self.should_save_pose_snapshot(camera, adapter)
            except Exception:
                adapter.has_snapshot = False

            try:
                adapter.has_clip = self.should_retain_pose_recording(camera, adapter)
            except Exception:
                adapter.has_clip = False

            after = adapter.to_dict() if hasattr(adapter, "to_dict") else {}
            message = {
                "before": getattr(adapter, "previous", {}),
                "after": after,
                "type": "new"
                if getattr(adapter, "previous", {}).get("false_positive", True)
                else "update",
            }
            self.dispatcher.publish("pose_events", json.dumps(message), retain=False)
            if hasattr(adapter, "previous"):
                adapter.previous = after
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.update,
                    camera,
                    frame_name,
                    after,
                )
            )

        def end(camera: str, pose: TrackedPose, frame_name: str) -> None:
            adapter = pose if isinstance(pose, TrackedPose) else _PoseAdapter(pose)
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.end,
                    camera,
                    frame_name,
                    adapter.to_dict() if hasattr(adapter, "to_dict") else {},
                )
            )

        def camera_activity(camera: str, activity: dict[str, Any]) -> None:
            last_activity = self.camera_activity.get(camera)
            if not last_activity or activity != last_activity:
                self.camera_activity[camera] = activity

        camera_state = CameraState(
            name=camera,
            config=self.config,
            frame_manager=self.frame_manager,
            ptz_autotracker_thread=self.ptz_autotracker_thread,
        )

        camera_state.on("start", start)
        camera_state.on("update", update)
        camera_state.on("end", end)
        camera_state.on("camera_activity", camera_activity)

        self.camera_states[camera] = camera_state

    def should_save_pose_snapshot(self, camera: str, pose: TrackedPose) -> bool:
        """Determine if pose snapshot should be saved."""
        camera_config = self.config.cameras[camera]

        # Check if pose detection snapshots are enabled
        if not getattr(camera_config.snapshots, "enabled", False):
            return False

        # Check if this pose action should trigger a snapshot
        pose_config = getattr(camera_config, "pose", None)
        if pose_config and hasattr(pose_config, "snapshot_actions"):
            if pose.action not in pose_config.snapshot_actions:
                return False

        # Check confidence threshold
        if pose.confidence < getattr(pose_config, "confidence_threshold", 0.4):
            return False

        return True

    def should_retain_pose_recording(self, camera: str, pose: TrackedPose) -> bool:
        """Determine if pose recording should be retained."""
        camera_config = self.config.cameras[camera]

        # Check if pose detection recording is enabled
        if not getattr(camera_config.record, "enabled", False):
            return False

        # Check if this pose action should trigger recording
        pose_config = getattr(camera_config, "pose", None)
        if pose_config and hasattr(pose_config, "record_actions"):
            if pose.action not in pose_config.record_actions:
                return False

        # Check confidence threshold
        if pose.confidence < getattr(pose_config, "confidence_threshold", 0.4):
            return False

        return True

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Check for camera config updates
                updated_topics = self.camera_config_subscriber.check_for_updates()

                if "enabled" in updated_topics:
                    for camera in updated_topics["enabled"]:
                        camera_config = self.config.cameras[camera]
                        current_enabled = camera_config.enabled

                        # If camera is now enabled and has pose detection enabled, initialize activity detector
                        if (
                            current_enabled
                            and hasattr(camera_config, "pose")
                            and camera_config.pose.enabled
                        ):
                            # Create camera state if it doesn't exist
                            if camera not in self.camera_states:
                                logger.info(
                                    f"Camera {camera} now enabled with pose detection, initializing"
                                )
                                self.create_camera_state(camera)
                                self.initialize_activity_detector(camera)
                            elif (
                                camera in self.camera_states
                                and self.camera_states[camera].prev_enabled is None
                            ):
                                # Just update prev_enabled if state already exists
                                self.camera_states[
                                    camera
                                ].prev_enabled = current_enabled

                        # If camera is now disabled and had an activity detector, clean it up
                        elif not current_enabled and camera in self.activity_detectors:
                            logger.info(
                                f"Camera {camera} now disabled, cleaning up activity detector"
                            )
                            self.activity_detectors.pop(camera)

                        # Update prev_enabled for the camera state if it exists
                        if camera in self.camera_states:
                            self.camera_states[camera].prev_enabled = current_enabled
                elif "add" in updated_topics:
                    for camera in updated_topics["add"]:
                        # Update the camera config
                        self.config.cameras[camera] = (
                            self.camera_config_subscriber.camera_configs[camera]
                        )

                        # Get the updated camera config
                        camera_config = self.config.cameras[camera]

                        # Only initialize if camera is enabled AND has pose detection enabled
                        if (
                            camera_config.enabled
                            and hasattr(camera_config, "pose")
                            and camera_config.pose.enabled
                        ):
                            logger.info(
                                f"New camera {camera} added with pose detection enabled, initializing"
                            )
                            self.create_camera_state(camera)
                            self.initialize_activity_detector(camera)
                        else:
                            reasons = []
                            if not camera_config.enabled:
                                reasons.append("camera disabled")
                            if (
                                not hasattr(camera_config, "pose")
                                or not camera_config.pose.enabled
                            ):
                                reasons.append("pose detection disabled")
                            if reasons:
                                logger.debug(
                                    f"Skipping pose activity detection for new camera {camera}: {', '.join(reasons)}"
                                )
                elif "remove" in updated_topics:
                    for camera in updated_topics["remove"]:
                        if camera in self.camera_states:
                            camera_state = self.camera_states[camera]
                            camera_state.shutdown()
                            self.camera_states.pop(camera)
                        # Clean up activity detector
                        if camera in self.activity_detectors:
                            self.activity_detectors.pop(camera)

                # Manage camera disabled state
                for camera, config in self.config.cameras.items():
                    if (
                        not config.enabled_in_config
                        or not hasattr(config, "pose")
                        or not config.pose.enabled
                    ):
                        continue

                    if camera not in self.camera_states:
                        continue

                    current_enabled = config.enabled
                    camera_state = self.camera_states[camera]

                    if (
                        hasattr(camera_state, "prev_enabled")
                        and camera_state.prev_enabled
                        and not current_enabled
                    ):
                        logger.debug(
                            f"Not processing poses for disabled camera {camera}"
                        )

                    camera_state.prev_enabled = current_enabled

                    if not current_enabled:
                        continue

                # Get the next data from the queue
                try:
                    queue_data = self.tracked_poses_queue.get(True, 1)

                    # Check the format of the data we received.
                    #
                    # Three known shapes (oldest first):
                    #   5-tuple: (camera, frame_time, tracked_poses, motion_boxes, regions)
                    #   6-tuple A: (camera, frame_name, frame_time, tracked_poses, motion_boxes, regions)
                    #   6-tuple B: (camera, frame_time, tracked_poses, motion_boxes, regions, thumbnail_bytes)
                    #
                    # Disambiguate 6-tuple A vs B by the type of the 2nd item:
                    # A has a string `frame_name`, B has a float `frame_time`.
                    thumbnail_bytes = None
                    if len(queue_data) == 6 and isinstance(queue_data[1], str):
                        # 6-tuple A: includes frame_name
                        (
                            camera,
                            frame_name,
                            frame_time,
                            tracked_poses,
                            motion_boxes,
                            regions,
                        ) = queue_data
                    elif len(queue_data) == 6:
                        # 6-tuple B: includes thumbnail_bytes (the new format
                        # produced by integration.py)
                        (
                            camera,
                            frame_time,
                            tracked_poses,
                            motion_boxes,
                            regions,
                            thumbnail_bytes,
                        ) = queue_data
                        frame_name = f"{camera}_{frame_time}"
                    else:
                        # 5-tuple: legacy
                        camera, frame_time, tracked_poses, motion_boxes, regions = (
                            queue_data
                        )
                        frame_name = f"{camera}_{frame_time}"

                except queue.Empty:
                    continue

                # Skip processing for disabled cameras or those without pose detection
                if (
                    camera not in self.config.cameras
                    or not self.config.cameras[camera].enabled
                    or not hasattr(self.config.cameras[camera], "pose")
                    or not self.config.cameras[camera].pose.enabled
                ):
                    logger.debug(
                        f"Camera {camera} disabled or pose detection disabled, skipping update"
                    )
                    continue

                # Skip if we don't have a state for this camera
                if camera not in self.camera_states:
                    logger.debug(
                        f"No camera state for {camera}, skipping pose processing"
                    )
                    continue

                camera_state = self.camera_states[camera]

                # Process tracked poses
                try:
                    # Get camera's activity detector
                    activity_detector = self.activity_detectors.get(camera)
                    if (
                        activity_detector is None
                        and camera not in self.activity_detectors
                    ):
                        # Try to initialize the detector if it doesn't exist
                        self.initialize_activity_detector(camera)
                        activity_detector = self.activity_detectors.get(camera)

                    # Update pose zones and trigger action analysis before sending to camera state
                    for pose in tracked_poses:
                        if isinstance(pose, TrackedPose):
                            self._update_pose_zones(camera, pose)
                            # Set the camera name for the pose if not already set
                            if not pose.camera_name:
                                pose.camera_name = camera

                            # Assign the appropriate activity detector to this pose
                            if activity_detector and activity_detector.initialized:
                                pose.active_detector = activity_detector

                            # Only run the activity classifier on fresh
                            # keypoints. `tracked_poses` is returned from
                            # `PoseDetectionIntegration.detect_poses()` and
                            # intentionally includes stale TrackedPose
                            # instances (time_since_update > 0) kept alive
                            # under `track_max_disappeared_seconds` so
                            # downstream CameraState lifecycles don't
                            # flicker on brief pose-detector drops.
                            # Classifying frozen keypoints would feed any
                            # sliding-window classifier a constant-valued
                            # buffer and produce spurious events; this gate
                            # is a data-quality filter and the single site
                            # that triggers classification. `TrackedPose.update()`
                            # (called from integration.py's match/create path)
                            # only advances tracking state; classification
                            # is lifted here to avoid firing twice per frame.
                            if getattr(pose, "time_since_update", 0) == 0:
                                pose.classify()

                    # Convert tracked poses list to a dictionary keyed by pose ID
                    # CameraState.update expects a dictionary with keys, not a list
                    # AND it expects specific fields like score_history for TrackedObject creation
                    tracked_poses_dict = {}
                    for pose in tracked_poses:
                        pose_dict = {}

                        if isinstance(pose, TrackedPose):
                            # If it's a TrackedPose object, convert to dict with its ID as key
                            pose_dict = pose.to_dict()
                            pose_id = pose.pose_id

                            # Add required fields for TrackedObject
                            pose_dict["score_history"] = [pose.confidence]
                            pose_dict["start_time"] = frame_time
                            pose_dict["frame_time"] = frame_time
                            pose_dict["label"] = (
                                "person"  # Poses are always associated with persons
                            )
                            pose_dict["top_score"] = pose.confidence
                            pose_dict["score"] = pose.confidence
                            pose_dict["position_changes"] = pose.hit_streak
                            pose_dict["motionless_count"] = pose.time_since_update
                            pose_dict["attributes"] = []

                            # Ensure bbox is in the expected format [x1, y1, x2, y2]
                            if pose.bbox and len(pose.bbox) == 4:
                                x, y, w, h = pose.bbox
                                pose_dict["box"] = [x, y, x + w, y + h]
                                # compute centroid as center of box
                                try:
                                    cx = int(
                                        (pose_dict["box"][0] + pose_dict["box"][2])
                                        / 2.0
                                    )
                                    cy = int(
                                        (pose_dict["box"][1] + pose_dict["box"][3])
                                        / 2.0
                                    )
                                except Exception:
                                    cx, cy = 0, 0
                                pose_dict["centroid"] = (cx, cy)
                                pose_dict["area"] = w * h
                                pose_dict["ratio"] = w / h if h > 0 else 1.0
                                pose_dict["region"] = [0, 0, 0, 0]  # Default region
                            else:
                                # Default values if no bbox
                                pose_dict["box"] = [0, 0, 10, 10]
                                pose_dict["centroid"] = (5, 5)
                                pose_dict["area"] = 100
                                pose_dict["ratio"] = 1.0
                                pose_dict["region"] = [0, 0, 0, 0]

                            tracked_poses_dict[pose_id] = pose_dict

                        elif isinstance(pose, dict):
                            # If it's already a dict, ensure it has required fields
                            pose_dict = pose.copy()

                            # Determine ID field
                            if "id" in pose:
                                pose_id = pose["id"]
                            elif "pose_id" in pose:
                                pose_id = pose["pose_id"]
                            else:
                                # Generate a unique ID if none exists
                                pose_id = f"pose_{len(tracked_poses_dict)}"

                            # Ensure the dict has an explicit id field matching the key
                            pose_dict["id"] = pose_id

                            # Add required fields if they don't exist
                            pose_dict.setdefault(
                                "score_history", [pose.get("confidence", 0.5)]
                            )
                            pose_dict.setdefault("start_time", frame_time)
                            pose_dict.setdefault("frame_time", frame_time)
                            pose_dict.setdefault("label", "person")
                            pose_dict.setdefault(
                                "top_score", pose.get("confidence", 0.5)
                            )
                            pose_dict.setdefault("score", pose.get("confidence", 0.5))
                            pose_dict.setdefault("position_changes", 1)
                            pose_dict.setdefault("motionless_count", 0)
                            pose_dict.setdefault("attributes", [])

                            # Ensure bbox is in the expected format
                            if "bbox" in pose and "box" not in pose:
                                x, y, w, h = pose["bbox"]
                                pose_dict["box"] = [x, y, x + w, y + h]
                                try:
                                    cx = int(
                                        (pose_dict["box"][0] + pose_dict["box"][2])
                                        / 2.0
                                    )
                                    cy = int(
                                        (pose_dict["box"][1] + pose_dict["box"][3])
                                        / 2.0
                                    )
                                except Exception:
                                    cx, cy = 0, 0
                                pose_dict["centroid"] = (cx, cy)
                                pose_dict["area"] = w * h
                                pose_dict["ratio"] = w / h if h > 0 else 1.0
                            elif "box" not in pose:
                                pose_dict["box"] = [0, 0, 10, 10]
                                pose_dict["centroid"] = (5, 5)
                                pose_dict["area"] = 100
                                pose_dict["ratio"] = 1.0

                            if "region" not in pose:
                                pose_dict["region"] = [0, 0, 0, 0]

                            tracked_poses_dict[pose_id] = pose_dict

                    # Now update the camera state with the dictionary of tracked poses
                    camera_state.update(
                        frame_name,
                        frame_time,
                        tracked_poses_dict,
                        motion_boxes,
                        regions,
                    )

                    # Publish detection info for this frame on the standard
                    # "video" sub-topic.  Tuple shape MUST stay at 6 elements
                    # because the recording maintainer, review maintainer,
                    # embeddings maintainer, and output process all subscribe
                    # to this topic and unpack a fixed-length 6-tuple.
                    self.detection_publisher.publish(
                        (
                            camera,
                            frame_name,
                            frame_time,
                            [
                                p.to_dict() if hasattr(p, "to_dict") else p
                                for p in tracked_poses
                            ],
                            motion_boxes,
                            regions,
                        ),
                        DetectionTypeEnum.video.value,
                    )

                    # Side-channel: publish the per-frame webp thumbnail bytes
                    # on a separate "pose" sub-topic that only pose_consumer
                    # subscribes to.  Bytes are base64-encoded so json.dumps
                    # in the ZMQ Publisher can serialize them.  pose_consumer
                    # decodes and persists them to disk when a pose-driven
                    # Event finalizes.
                    if thumbnail_bytes:
                        try:
                            thumbnail_b64 = base64.b64encode(thumbnail_bytes).decode(
                                "ascii"
                            )
                            self.pose_thumb_publisher.publish(
                                (camera, frame_time, thumbnail_b64),
                                DetectionTypeEnum.pose.value,
                            )
                        except Exception:
                            logger.debug(
                                f"Failed to publish pose thumbnail for {camera}"
                            )

                    # Update camera activity based on the poses
                    self._update_camera_activity(camera, tracked_poses)

                    # Check for any events that need to be ended
                    while not self.stop_event.is_set():
                        update = self.event_end_subscriber.check_for_update(
                            timeout=FAST_QUEUE_TIMEOUT
                        )

                        if not update:
                            break

                        event_id, event_camera, _ = update
                        if event_camera == camera and camera in self.camera_states:
                            self.camera_states[camera].finished(event_id)

                except Exception as e:
                    logger.error(f"Error processing poses for camera {camera}: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"Error in pose processor main loop: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Cleanup when stopping
        for state in self.camera_states.values():
            state.shutdown()

        self.detection_publisher.stop()
        self.pose_thumb_publisher.stop()
        self.event_sender.stop()
        self.event_end_subscriber.stop()
        self.camera_config_subscriber.stop()

        logger.info("Exiting pose processor...")

    def _update_pose_zones(self, camera: str, pose: TrackedPose) -> None:
        """Update pose zone tracking with inertia, loitering time, and zone filters.

        This applies the same zone filtering logic as object detection, including:
        - Object type filtering (zones can specify which objects/poses trigger them)
        - Inertia (consecutive frames required before entering zone)
        - Loitering time (seconds required to be considered in zone)
        - Zone filters (min/max area, ratio, threshold)
        """
        camera_config = self.config.cameras[camera]

        if not hasattr(camera_config, "zones") or not pose.bbox:
            return

        # Compute bottom center and area/ratio for filtering
        x, y, w, h = pose.bbox
        bottom_center = (
            x + w / 2,
            y + h,
        )  # Bottom center is more reliable for zone detection

        pose_area = w * h
        pose_ratio = w / h if h > 0 else 1.0

        current_zones = []

        for zone_name, zone_config in camera_config.zones.items():
            # Skip zones that don't include poses/persons
            if hasattr(zone_config, "objects") and len(zone_config.objects) > 0:
                if (
                    "person" not in zone_config.objects
                    and "pose" not in zone_config.objects
                ):
                    continue

            if not hasattr(zone_config, "contour"):
                continue

            contour = zone_config.contour
            zone_score = pose.zone_presence.get(zone_name, 0) + 1

            # Check if the pose is inside the zone polygon
            if cv2.pointPolygonTest(contour, bottom_center, False) >= 0:
                # Apply zone filters if configured
                if not self._pose_zone_filtered(
                    pose, zone_config, pose_area, pose_ratio
                ):
                    # Update zone presence count
                    pose.zone_presence[zone_name] = zone_score

                    # Check inertia requirement
                    inertia = getattr(zone_config, "inertia", 3)
                    if zone_score >= inertia:
                        # Update loitering count
                        loitering_score = pose.zone_loitering.get(zone_name, 0) + 1
                        pose.zone_loitering[zone_name] = loitering_score

                        # Check loitering time requirement
                        loitering_time = getattr(zone_config, "loitering_time", 0)
                        detect_fps = camera_config.detect.fps

                        # loitering_time is in seconds, convert to frames
                        loitering_frames_required = loitering_time * detect_fps

                        if loitering_score >= loitering_frames_required:
                            current_zones.append(zone_name)

                            if zone_name not in pose.entered_zones:
                                pose.entered_zones.add(zone_name)
                                logger.debug(
                                    f"Pose {pose.pose_id} entered zone {zone_name}"
                                )
            else:
                # Reset zone presence and loitering when pose leaves zone polygon
                if zone_name in pose.zone_presence:
                    del pose.zone_presence[zone_name]
                if zone_name in pose.zone_loitering:
                    del pose.zone_loitering[zone_name]

        # Update current zones
        pose.current_zones = set(current_zones)

    def _pose_zone_filtered(
        self, pose: TrackedPose, zone_config, pose_area: float, pose_ratio: float
    ) -> bool:
        """Check if pose should be filtered out based on zone filters.

        Args:
            pose: The TrackedPose to check
            zone_config: Zone configuration with filters
            pose_area: Computed area of the pose bounding box
            pose_ratio: Computed width/height ratio of the pose bbox

        Returns:
            True if the pose should be filtered OUT (not included), False otherwise
        """
        # Check if there are person-specific filters in the zone
        filters = getattr(zone_config, "filters", {})
        if not filters:
            return False

        # Look for "person" or "pose" filters
        obj_settings = filters.get("person") or filters.get("pose")
        if not obj_settings:
            return False

        # Check min_area filter
        min_area = getattr(obj_settings, "min_area", 0)
        if min_area > 0 and pose_area < min_area:
            logger.debug(
                f"Pose {pose.pose_id} filtered: area {pose_area} < min_area {min_area}"
            )
            return True

        # Check max_area filter
        max_area = getattr(obj_settings, "max_area", float("inf"))
        if max_area < float("inf") and pose_area > max_area:
            logger.debug(
                f"Pose {pose.pose_id} filtered: area {pose_area} > max_area {max_area}"
            )
            return True

        # Check threshold (confidence) filter
        threshold = getattr(obj_settings, "threshold", 0)
        if threshold > 0 and pose.confidence < threshold:
            logger.debug(
                f"Pose {pose.pose_id} filtered: confidence {pose.confidence} < threshold {threshold}"
            )
            return True

        # Check min_ratio filter
        min_ratio = getattr(obj_settings, "min_ratio", 0)
        if min_ratio > 0 and pose_ratio < min_ratio:
            logger.debug(
                f"Pose {pose.pose_id} filtered: ratio {pose_ratio} < min_ratio {min_ratio}"
            )
            return True

        # Check max_ratio filter
        max_ratio = getattr(obj_settings, "max_ratio", float("inf"))
        if max_ratio < float("inf") and pose_ratio > max_ratio:
            logger.debug(
                f"Pose {pose.pose_id} filtered: ratio {pose_ratio} > max_ratio {max_ratio}"
            )
            return True

        return False

    def _update_camera_activity(self, camera: str, poses: List[TrackedPose]) -> None:
        """Update camera activity based on pose detections."""
        if not poses:
            return

        # Filter active (non-false-positive) poses
        active_poses = []
        for p in poses:
            if hasattr(p, "false_positive"):
                if not p.false_positive:
                    active_poses.append(p)
            else:
                # If false_positive isn't defined, consider it as active
                active_poses.append(p)

        activity = {
            "enabled": self.config.cameras[camera].enabled,
            "pose_count": len(active_poses),
            "actions": defaultdict(int),
            "zones": defaultdict(int),
        }

        # Count actions and zones
        for pose in active_poses:
            # Get action, handling both enum and string types
            if hasattr(pose, "action"):
                action = pose.action
                if hasattr(action, "value"):
                    action = action.value
                activity["actions"][action] += 1

            # Count poses by zone
            if hasattr(pose, "current_zones"):
                for zone in pose.current_zones:
                    activity["zones"][zone] += 1

        # Call camera activity callback for this camera
        camera_state = self.camera_states.get(camera)
        if camera_state and "camera_activity" in camera_state.callbacks:
            for callback in camera_state.callbacks["camera_activity"]:
                callback(camera, activity)
