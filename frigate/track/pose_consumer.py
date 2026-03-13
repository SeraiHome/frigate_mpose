import json
import logging
import queue
import threading
import time
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, List, Optional

import numpy as np

from frigate.pose_detection.privacy_renderer import keypoints_to_mqtt_payload

from frigate.camera.state import CameraState
from frigate.comms.detections_updater import (
    DetectionPublisher,
    DetectionSubscriber,
    DetectionTypeEnum,
)
from frigate.comms.dispatcher import Dispatcher
from frigate.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import (
    FrigateConfig,
    RecordConfig,
    SnapshotsConfig,
)
from frigate.events.types import EventStateEnum, EventTypeEnum
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


class PoseAsTrackedObject:
    """Lightweight adapter that presents a TrackedPose as a TrackedObject-like
    interface for the event pipeline. Only implements what's needed by
    the consumers (to_dict, previous, false_positive, has_snapshot, has_clip,
    entered_zones, thumbnail_data, write_snapshot_to_disk/write_thumbnail_to_disk).
    """

    def __init__(self, obj_data: dict[str, Any]):
        self.obj_data = obj_data
        self.previous = obj_data.copy()
        self.false_positive = obj_data.get("false_positive", False)
        self.has_snapshot = obj_data.get("has_snapshot", False)
        self.has_clip = obj_data.get("has_clip", False)
        self.entered_zones = obj_data.get("entered_zones", [])
        self.thumbnail_data = obj_data.get("thumbnail_data")

    def to_dict(self) -> dict[str, Any]:
        return self.obj_data

    def write_thumbnail_to_disk(self) -> None:
        # No-op by default. Implement if needed later.
        return

    def write_snapshot_to_disk(self) -> None:
        # No-op by default. Implement if needed later.
        return

    def is_stationary(self) -> bool:
        # Poses are not considered stationary in the object-sense
        return False


class PoseConsumer(threading.Thread):
    """Consumes `tracked_poses_queue` and feeds pose detections into the
    same event/detection flow used by object detections. This makes pose
    detections produce the same EventTypeEnum.tracked_object events and
    get routed to the same DB/clip/review logic without going through
    the main `detected_objects_queue` (optional).
    """

    def __init__(
        self,
        config: FrigateConfig,
        dispatcher: Dispatcher,
        stop_event: MpEvent,
        camera_metrics: dict = None,
        ptz_autotracker_thread=None,
        detected_frames_queue: Optional[MpQueue] = None,
    ) -> None:
        super().__init__(name="pose_consumer")
        self.config = config
        self.dispatcher = dispatcher
        self.stop_event = stop_event
        self.camera_metrics = camera_metrics or {}
        self.ptz_autotracker_thread = ptz_autotracker_thread

        self.frame_manager = SharedMemoryFrameManager()
        self.requestor = InterProcessRequestor()
        self.detection_publisher = DetectionPublisher(DetectionTypeEnum.all.value)
        self.event_sender = EventUpdatePublisher()
        self.event_end_subscriber = EventEndSubscriber()

        # Optional queue to publish synthesized detected objects into the
        # existing object processing pipeline.
        self.detected_frames_queue = detected_frames_queue

        # Track per-camera privacy override expiry to debounce triggers
        self._privacy_override_until: dict[str, float] = {}

        # Subscribe to processed pose detections published by TrackedPoseProcessor
        self.detection_subscriber = DetectionSubscriber(DetectionTypeEnum.video.value)

        self.camera_states: dict[str, CameraState] = {}

        # Create camera states for all configured cameras
        for camera in self.config.cameras.keys():
            self._create_camera_state(camera)

        # Subscribe to camera config updates
        from frigate.config.camera.updater import (
            CameraConfigUpdateEnum,
            CameraConfigUpdateSubscriber,
        )

        self.camera_config_subscriber = CameraConfigUpdateSubscriber(
            self.config,
            self.config.cameras,
            [CameraConfigUpdateEnum.enabled, CameraConfigUpdateEnum.add],
        )

    def _create_camera_state(self, camera: str) -> None:
        # Create a CameraState instance and register handlers that mirror
        # TrackedObjectProcessor behavior but operate on PoseAsTrackedObject
        camera_state = CameraState(
            camera, self.config, self.frame_manager, self.ptz_autotracker_thread
        )

        def start(camera_name: str, obj: PoseAsTrackedObject, frame_name: str) -> None:
            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.start,
                    camera_name,
                    frame_name,
                    obj.to_dict(),
                )
            )

        def should_save_snapshot(camera_name: str, obj: PoseAsTrackedObject) -> bool:
            if obj.false_positive:
                return False

            snapshot_config: SnapshotsConfig = self.config.cameras[
                camera_name
            ].snapshots

            if not snapshot_config.enabled:
                return False

            # object never changed position
            if obj.obj_data.get("position_changes", 0) == 0:
                return False

            # if there are required zones and there is no overlap
            required_zones = snapshot_config.required_zones
            if len(required_zones) > 0 and not set(obj.entered_zones) & set(
                required_zones
            ):
                logger.debug(
                    f"Not creating snapshot for {obj.obj_data.get('id')} because it did not enter required zones"
                )
                return False

            return True

        def should_retain_recording(camera_name: str, obj: PoseAsTrackedObject) -> bool:
            if obj.false_positive:
                return False

            record_config: RecordConfig = self.config.cameras[camera_name].record

            # Recording is disabled
            if not record_config.enabled:
                return False

            # object never changed position
            if obj.obj_data.get("position_changes", 0) == 0:
                return False

            # If the object is not considered an alert or detection
            if getattr(obj, "max_severity", None) is None:
                return False

            return True

        def update(camera_name: str, obj: PoseAsTrackedObject, frame_name: str) -> None:
            # mark snapshot/clip flags if configuration requests it
            obj.has_snapshot = should_save_snapshot(camera_name, obj)
            obj.has_clip = should_retain_recording(camera_name, obj)

            after = obj.to_dict()
            message = {
                "before": obj.previous,
                "after": after,
                "type": "new" if obj.previous.get("false_positive", True) else "update",
            }
            self.dispatcher.publish("events", json.dumps(message), retain=False)
            obj.previous = after
            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.update,
                    camera_name,
                    frame_name,
                    obj.to_dict(),
                )
            )

        def autotrack(
            camera_name: str, obj: PoseAsTrackedObject, frame_name: str
        ) -> None:
            if self.ptz_autotracker_thread:
                self.ptz_autotracker_thread.ptz_autotracker.autotrack_object(
                    camera_name, obj
                )

        def end(camera_name: str, obj: PoseAsTrackedObject, frame_name: str) -> None:
            # populate has_snapshot
            obj.has_snapshot = should_save_snapshot(camera_name, obj)
            obj.has_clip = should_retain_recording(camera_name, obj)

            # write thumbnail to disk if it will be saved as an event
            if obj.has_snapshot or obj.has_clip:
                try:
                    obj.write_thumbnail_to_disk()
                except Exception:
                    logger.exception(
                        f"Error writing thumbnail for pose {obj.obj_data.get('id')}"
                    )

        self.camera_states[camera] = camera_state

    def _convert_tracked_poses(
        self, tracked_poses: List[Any], frame_time: float, frame_name: str
    ) -> dict[str, dict[str, Any]]:
        """Convert TrackedPose objects or dicts into a dictionary keyed by id
        matching the shape expected by CameraState.update() and the object
        pipeline.
        """
        results: dict[str, dict[str, Any]] = {}

        for pose in tracked_poses:
            if hasattr(pose, "to_dict"):
                pd = pose.to_dict()
                pose_id = (
                    pd.get("id") or f"pose_{pd.get('person_id', 0)}_{int(frame_time)}"
                )
            elif isinstance(pose, dict):
                pd = pose.copy()
                pose_id = (
                    pd.get("id")
                    or pd.get("pose_id")
                    or f"pose_{int(frame_time)}_{len(results)}"
                )
            else:
                continue

            # Ensure required fields
            pd.setdefault("score_history", [pd.get("confidence", 0.0)])
            pd.setdefault("start_time", frame_time)
            pd.setdefault("frame_time", frame_time)
            pd.setdefault("label", "person")
            pd.setdefault("top_score", pd.get("confidence", 0.0))
            pd.setdefault("score", pd.get("confidence", 0.0))
            pd.setdefault("position_changes", 1)
            pd.setdefault("motionless_count", 0)
            pd.setdefault("attributes", [])

            # Normalize/ensure bbox/box/centroid/area/ratio are present and well-formed.
            # Accept either model bbox [x,y,w,h] or existing box [x1,y1,x2,y2].
            try:
                if "bbox" in pd and ("box" not in pd or not pd.get("box")):
                    try:
                        x, y, w, h = pd["bbox"]
                        x1 = int(x)
                        y1 = int(y)
                        x2 = int(x + w)
                        y2 = int(y + h)
                        pd["box"] = [x1, y1, x2, y2]
                    except Exception:
                        pd["box"] = [0, 0, 10, 10]

                # Ensure box is a 4-int list
                box = pd.get("box", [0, 0, 10, 10])
                if not isinstance(box, list) or len(box) < 4:
                    box = [int(box[0]) if box else 0, 0, 10, 10]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                pd["box"] = [x1, y1, x2, y2]

                # Compute centroid as bottom-center (x center, bottom y)
                pd["centroid"] = (int((x1 + x2) / 2), int(y2))

                # Area and aspect ratio
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                pd["area"] = int(pd.get("area", w * h))
                pd["ratio"] = float(pd.get("ratio", (w / h) if h else 1.0))
            except Exception:
                pd["box"] = [0, 0, 10, 10]
                pd["centroid"] = (5, 5)
                pd["area"] = 100
                pd["ratio"] = 1.0

            if "region" not in pd:
                pd.setdefault("region", [0, 0, 0, 0])

            results[pose_id] = pd

        return results

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                update = self.camera_config_subscriber.check_for_updates()
                if update:
                    # recreate camera states if cameras were added/removed
                    # keep it simple for now
                    pass

                # Read processed pose detections published by TrackedPoseProcessor
                topic_payload = self.detection_subscriber.check_for_update(timeout=0.5)
                if not topic_payload:
                    continue
                topic, payload = topic_payload
                if payload is None:
                    continue

                try:
                    (
                        camera,
                        frame_name,
                        frame_time,
                        tracked_poses,
                        motion_boxes,
                        regions,
                    ) = payload
                except Exception:
                    # Unexpected payload format
                    logger.debug(
                        "Received unexpected detection payload from DetectionSubscriber"
                    )
                    continue
                frame_name = f"{camera}_{frame_time}"

                if camera not in self.camera_states:
                    # create camera state on demand
                    self._create_camera_state(camera)

                camera_state = self.camera_states[camera]

                # Convert poses to dict keyed by id
                tracked_dict = self._convert_tracked_poses(
                    tracked_poses, frame_time, frame_name
                )

                # Update camera state which will trigger start/update/end callbacks
                camera_state.update(
                    frame_name, frame_time, tracked_dict, motion_boxes, regions
                )

                # Publish detection summary for UI/consumers
                try:
                    self.detection_publisher.publish(
                        (
                            camera,
                            frame_name,
                            frame_time,
                            list(tracked_dict.values()),
                            motion_boxes,
                            regions,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to publish pose detection via DetectionPublisher"
                    )

                # Publish keypoints to MQTT for the privacy proxy sidecar
                cam_config = self.config.cameras.get(camera)
                if (
                    cam_config
                    and hasattr(cam_config, "pose")
                    and cam_config.pose
                    and cam_config.pose.publish_keypoints
                    and tracked_poses
                ):
                    try:
                        frame_w = cam_config.detect.width
                        frame_h = cam_config.detect.height
                        payload = keypoints_to_mqtt_payload(
                            tracked_poses, frame_time, frame_w, frame_h
                        )
                        self.dispatcher.publish(
                            f"{camera}/pose_keypoints",
                            json.dumps(payload),
                            retain=False,
                        )
                    except Exception:
                        logger.debug(
                            f"Failed to publish keypoints for {camera}"
                        )

                # Ensure the frame is available in the CameraState frame cache so
                # snapshots/clips can be created even if the object pipeline ran earlier.
                try:
                    raw_frame = self.frame_manager.get(
                        frame_name, camera_state.camera_config.frame_shape_yuv
                    )
                    if (
                        raw_frame is not None
                        and frame_time not in camera_state.frame_cache
                    ):
                        arr = None
                        try:
                            if isinstance(raw_frame, np.ndarray):
                                # unwrap 0-d object arrays that contain a buffer or ndarray
                                if raw_frame.dtype == object and raw_frame.shape == ():
                                    inner = raw_frame.item()
                                else:
                                    inner = raw_frame

                                if isinstance(inner, np.ndarray):
                                    arr = inner.astype(np.uint8, copy=False)
                                elif isinstance(inner, (bytes, bytearray, memoryview)):
                                    try:
                                        arr = np.frombuffer(inner, dtype=np.uint8)
                                    except Exception:
                                        arr = np.asarray(inner, dtype=np.uint8)
                                else:
                                    arr = np.asarray(inner, dtype=np.uint8)
                            else:
                                try:
                                    mv = memoryview(raw_frame)
                                    arr = np.frombuffer(mv, dtype=np.uint8)
                                except Exception:
                                    arr = np.asarray(raw_frame, dtype=np.uint8)
                        except Exception:
                            logger.debug(
                                "PoseConsumer: frame returned from SHM is invalid for conversion"
                            )
                            arr = None

                        if arr is not None:
                            try:
                                # If arr is an object-typed 0-d ndarray, try to unwrap
                                if isinstance(arr, np.ndarray) and (
                                    arr.dtype == object or arr.shape == ()
                                ):
                                    try:
                                        inner = arr.item()
                                        if isinstance(inner, np.ndarray):
                                            arr = inner.astype(np.uint8, copy=False)
                                        elif isinstance(
                                            inner, (bytes, bytearray, memoryview)
                                        ):
                                            arr = np.frombuffer(inner, dtype=np.uint8)
                                        else:
                                            arr = None
                                    except Exception:
                                        arr = None

                                if arr is None:
                                    raise ValueError(
                                        "Unable to coerce frame buffer to ndarray"
                                    )

                                expected_shape = (
                                    camera_state.camera_config.frame_shape_yuv
                                )
                                if arr.ndim == 1:
                                    total = int(np.prod(expected_shape))
                                    if arr.size == total:
                                        arr = arr.reshape(expected_shape)

                                # final validation: must be a uint8 ndarray with at least 2 dims
                                if (
                                    not isinstance(arr, np.ndarray)
                                    or arr.dtype == object
                                ):
                                    raise ValueError(
                                        "Invalid frame ndarray after coercion"
                                    )

                                if arr.ndim < 2:
                                    raise ValueError("Invalid frame dimensions")

                                if arr.dtype != np.uint8:
                                    arr = arr.astype(np.uint8, copy=False)

                                camera_state.frame_cache[frame_time] = {
                                    "frame": arr.copy(),
                                    "object_id": None,
                                }
                            except Exception:
                                logger.debug(
                                    "PoseConsumer: failed to store coerced frame into cache"
                                )
                except Exception:
                    logger.debug("Unable to fetch frame for pose lookback cache")

                # For each tracked pose that indicates an actionable event (e.g., falling),
                # attempt to link to an existing tracked object or synthesize a detected-object
                # entry into the main detected frames queue so the object pipeline will
                # create events, snapshots and review segments.
                try:
                    for pd in tracked_dict.values():
                        action = pd.get("action")
                        # ignore neutral/standing actions
                        if not action or action == "standing":
                            continue

                        # Try to match pose to an existing tracked person by IoU
                        matched_id = None
                        try:
                            pose_box = pd.get("box")
                            if pose_box and len(pose_box) == 4:
                                px1, py1, px2, py2 = pose_box
                                best_iou = 0.3  # minimum IoU threshold
                                for obj in camera_state.tracked_objects.values():
                                    if obj.obj_data.get("label") != "person":
                                        continue
                                    obj_box = obj.obj_data.get("box", [])
                                    if len(obj_box) != 4:
                                        continue
                                    ox1, oy1, ox2, oy2 = obj_box
                                    ix1 = max(px1, ox1)
                                    iy1 = max(py1, oy1)
                                    ix2 = min(px2, ox2)
                                    iy2 = min(py2, oy2)
                                    if ix2 > ix1 and iy2 > iy1:
                                        intersection = (ix2 - ix1) * (iy2 - iy1)
                                        pose_area = max(1, (px2 - px1) * (py2 - py1))
                                        obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                                        iou = intersection / (pose_area + obj_area - intersection)
                                        if iou > best_iou:
                                            best_iou = iou
                                            matched_id = obj.obj_data.get("id")
                        except Exception:
                            matched_id = None

                        if matched_id and matched_id in camera_state.tracked_objects:
                            # Attach action as a sub_label to the matched object and
                            # trigger an update callback so downstream systems see it.
                            try:
                                matched_obj = camera_state.tracked_objects[matched_id]
                                # attach action and confidence
                                matched_obj.obj_data["sub_label"] = (
                                    action,
                                    pd.get("action_confidence", 0.0),
                                )
                                matched_obj.obj_data["action"] = action
                                matched_obj.obj_data["action_confidence"] = pd.get(
                                    "action_confidence", 0.0
                                )

                                # mark as true positive so snapshot/clip logic runs
                                try:
                                    matched_obj.false_positive = False
                                except Exception:
                                    pass

                                # If camera requests snapshots/recording for this action,
                                # ensure the matched object flags are set so the
                                # object pipeline will write thumbnails/snapshots.
                                try:
                                    cam_pose_cfg = self.config.cameras[camera].pose
                                except Exception:
                                    cam_pose_cfg = None

                                try:
                                    if (
                                        cam_pose_cfg
                                        and getattr(
                                            cam_pose_cfg, "snapshot_actions", None
                                        )
                                        and action in cam_pose_cfg.snapshot_actions
                                    ):
                                        matched_obj.has_snapshot = True
                                        matched_obj.obj_data["has_snapshot"] = True

                                    if (
                                        cam_pose_cfg
                                        and getattr(
                                            cam_pose_cfg, "record_actions", None
                                        )
                                        and action in cam_pose_cfg.record_actions
                                    ):
                                        matched_obj.has_clip = True
                                        matched_obj.obj_data["has_clip"] = True

                                    # Trigger privacy override for sidecar
                                    # (switches recording from skeleton to real frames)
                                    # Debounce: only fire if not already overriding
                                    now = time.time()
                                    already_overriding = (
                                        now
                                        < self._privacy_override_until.get(camera, 0)
                                    )
                                    if (
                                        not already_overriding
                                        and cam_pose_cfg
                                        and getattr(cam_pose_cfg, "privacy_mode", False)
                                        and getattr(
                                            cam_pose_cfg, "record_actions", None
                                        )
                                        and action in cam_pose_cfg.record_actions
                                    ):
                                        post_event_s = getattr(
                                            cam_pose_cfg, "privacy_override_seconds", 60
                                        )
                                        override_until = now + post_event_s
                                        self._privacy_override_until[camera] = (
                                            override_until
                                        )
                                        # SHM override (web UI / birdseye) via shared mp.Value
                                        cam_metrics = self.camera_metrics.get(camera)
                                        if cam_metrics:
                                            cam_metrics.privacy_override_until.value = override_until
                                        # MQTT override (sidecar proxy)
                                        self.dispatcher.publish(
                                            f"{camera}/privacy_override",
                                            json.dumps(
                                                {"duration_seconds": post_event_s}
                                            ),
                                            retain=False,
                                        )
                                        logger.info(
                                            f"Privacy override triggered for {camera} "
                                            f"({post_event_s}s) due to '{action}'"
                                        )

                                    # Ensure an alert severity exists so retention logic
                                    # that checks `max_severity` will consider this object
                                    # significant.
                                    try:
                                        if (
                                            getattr(matched_obj, "max_severity", None)
                                            is None
                                        ):
                                            matched_obj.max_severity = 1
                                    except Exception:
                                        pass
                                except Exception:
                                    # best-effort only; don't fail the whole loop
                                    logger.exception(
                                        "Error setting snapshot/record flags for matched pose action"
                                    )

                                # update scoring so object is considered significant
                                try:
                                    ac = float(pd.get("action_confidence", 0.0) or 0.0)
                                    # push recent action confidence into score history
                                    if hasattr(matched_obj, "score_history"):
                                        matched_obj.score_history.append(ac)
                                        matched_obj.score_history = (
                                            matched_obj.score_history[-10:]
                                        )
                                    matched_obj.obj_data["score"] = ac
                                    if hasattr(matched_obj, "top_score"):
                                        matched_obj.top_score = max(
                                            getattr(matched_obj, "top_score", 0.0), ac
                                        )
                                except Exception:
                                    pass

                                logger.info(
                                    f"Pose action '{action}' matched tracked object {matched_id} on {camera} (attached action_confidence={matched_obj.obj_data.get('action_confidence')})"
                                )

                                # invoke update callbacks to trigger snapshot/event logic
                                for c in camera_state.callbacks["update"]:
                                    try:
                                        c(camera, matched_obj, frame_name)
                                    except Exception:
                                        logger.exception(
                                            "Error invoking update callback for matched pose action"
                                        )
                            except Exception:
                                logger.exception(
                                    "Failed to attach pose action to matched tracked object"
                                )
                        else:
                            # Synthesize a detection dict for the pose and push to the
                            # detected frames queue if available so the object pipeline
                            # will process it exactly like an object detection.
                            if self.detected_frames_queue is not None:
                                det = {
                                    pd.get("id", f"pose_{int(frame_time)}"): {
                                        "id": pd.get("id", f"pose_{int(frame_time)}"),
                                        "label": "person",
                                        "sub_label": (
                                            action,
                                            pd.get("action_confidence", 0.0),
                                        ),
                                        "box": pd.get("box", [0, 0, 10, 10]),
                                        "centroid": pd.get("centroid", (5, 5)),
                                        "estimate": tuple(
                                            pd.get("box", [0, 0, 10, 10])
                                        ),
                                        "estimate_velocity": (0, 0),
                                        "start_time": frame_time,
                                        "frame_time": frame_time,
                                        "motionless_count": 0,
                                        "position_changes": pd.get(
                                            "position_changes", 1
                                        ),
                                        "attributes": [],
                                        "score_history": [
                                            pd.get("action_confidence", 0.0)
                                        ],
                                        "score": pd.get(
                                            "action_confidence", pd.get("score", 0.0)
                                        ),
                                        "top_score": pd.get(
                                            "action_confidence",
                                            pd.get("top_score", 0.0),
                                        ),
                                        "area": pd.get("area", 100),
                                        "ratio": pd.get("ratio", 1.0),
                                        "region": pd.get("region", [0, 0, 0, 0]),
                                    }
                                }
                                try:
                                    logger.info(
                                        f"Synthesizing detection for pose action '{action}' on {camera} at {frame_time}"
                                    )
                                    # avoid blocking indefinitely if queue is full
                                    self.detected_frames_queue.put(
                                        (
                                            camera,
                                            frame_name,
                                            frame_time,
                                            det,
                                            motion_boxes,
                                            regions,
                                        ),
                                        block=True,
                                        timeout=0.5,
                                    )
                                except queue.Full:
                                    logger.warning(
                                        "Detected frames queue is full; dropped synthesized pose detection"
                                    )
                                except Exception:
                                    logger.exception(
                                        "Failed to put synthesized pose detection into detected_frames_queue"
                                    )
                except Exception:
                    logger.exception("Error while synthesizing pose-driven detections")

            except Exception:
                logger.exception("Error in PoseConsumer main loop")

        # cleanup
        for state in self.camera_states.values():
            state.shutdown()

        self.detection_publisher = None
        self.event_sender = None
