import base64
import json
import logging
import os
import threading
import time
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, List, Optional

import numpy as np

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
from frigate.const import THUMB_DIR
from frigate.events.types import EventStateEnum, EventTypeEnum
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


def _keypoints_to_mqtt_payload(
    tracked_poses,
    frame_time: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Convert tracked poses to a JSON-serializable dict for MQTT publishing.

    Consumers subscribed to frigate/{camera}/pose_keypoints can use this
    payload to drive downstream visualizations or second-stage classifiers.
    """
    import numpy as np

    poses_data = []
    for pose in tracked_poses:
        kps = (
            pose.keypoints if hasattr(pose, "keypoints") else pose.get("keypoints", [])
        )
        if isinstance(kps, np.ndarray):
            kps_list = kps.tolist()
        else:
            kps_list = [[float(k[0]), float(k[1]), float(k[2])] for k in kps]

        pose_id = getattr(pose, "pose_id", None) or (
            pose.get("id") if isinstance(pose, dict) else None
        )
        poses_data.append(
            {
                "id": str(pose_id) if pose_id else "unknown",
                "keypoints": kps_list,
            }
        )

    return {
        "timestamp": frame_time,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "poses": poses_data,
    }


# Seconds without a fresh pose-action match before we publish an `end`
# event for a pose-driven Event row. The pose pipeline only pushes
# detections while a pose action is actively present, so we synthesize
# the lifecycle ourselves: start on first match, update on subsequent
# matches, end after this timeout of silence.
POSE_EVENT_END_TIMEOUT = 5.0


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
        tracked_object_processor=None,
    ) -> None:
        super().__init__(name="pose_consumer")
        self.config = config
        self.dispatcher = dispatcher
        self.stop_event = stop_event
        self.camera_metrics = camera_metrics or {}
        self.ptz_autotracker_thread = ptz_autotracker_thread
        # Reference to the standard frigate tracker (TrackedObjectProcessor).
        # When the activity classifier attaches an action label to a pose,
        # pose_consumer mutates the real TrackedObject.obj_data via this
        # reference so the standard MQTT update flow carries the new
        # sub_label. Both threads run in the same main frigate process and
        # share memory; without this reference pose_consumer was mutating its
        # own local CameraState instance, which never reached MQTT consumers.
        self.tracked_object_processor = tracked_object_processor

        self.frame_manager = SharedMemoryFrameManager()
        self.requestor = InterProcessRequestor()
        self.detection_publisher = DetectionPublisher(DetectionTypeEnum.all.value)
        self.event_sender = EventUpdatePublisher()
        self.event_end_subscriber = EventEndSubscriber()

        # Optional queue to publish synthesized detected objects into the
        # existing object processing pipeline.
        self.detected_frames_queue = detected_frames_queue

        # Lifecycle state for pose-driven Event rows.
        # _pose_event_id_by_track maps "{camera}:{pose_track_id}:{action}" -> active pose_event_id.
        # _active_pose_events maps pose_event_id -> {
        #     'camera', 'frame_name', 'last_seen', 'top_score', 'event_data'
        # }.
        # See `_publish_pose_action_event` and `_check_pose_event_timeouts`.
        self._pose_event_id_by_track: dict[str, str] = {}
        self._active_pose_events: dict[str, dict[str, Any]] = {}
        # Cooldown dedup for same-track + same-action re-fires (issue follow-up
        # to #60). Maps track_key -> {"expires_at", "event_id", "active"}, where
        # `active` is the full _active_pose_events entry from the previous
        # lifecycle, preserved so we can reopen it without re-publishing start.
        # See `_publish_pose_action_event` reopen branch and
        # `_check_pose_event_timeouts` end branch.
        self._pose_event_cooldown: dict[str, dict[str, Any]] = {}

        # Most recent webp thumbnail bytes received per camera, encoded by
        # `pose_detection.integration.detect_poses` in the detect process
        # while the SHM frame was still valid.  Used by the lifecycle to
        # persist a thumbnail file when a pose-driven Event finalizes.
        self._latest_thumbnail_bytes: dict[str, bytes] = {}

        # Subscribe to processed pose detections published by TrackedPoseProcessor
        self.detection_subscriber = DetectionSubscriber(DetectionTypeEnum.video.value)

        # Side-channel subscriber for per-frame webp thumbnail bytes that
        # `pose_processing` publishes on a separate `pose` sub-topic.
        self.pose_thumb_subscriber = DetectionSubscriber(DetectionTypeEnum.pose.value)

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

    def _build_pose_event_data(
        self,
        camera: str,
        pd: dict[str, Any],
        frame_time: float,
        event_id: str,
        start_time: float,
        end_time: Optional[float],
        top_score: float,
    ) -> dict[str, Any]:
        """Build an event_data dict shaped for `events.maintainer.handle_pose_detection`.

        Includes every key that `should_update_db`, `should_update_state`,
        and `handle_pose_detection` read, so the events maintainer can
        compare against this dict without raising KeyError.
        """
        action = pd.get("action")
        ac = float(pd.get("action_confidence", 0.0) or 0.0)

        cam_pose_cfg = getattr(self.config.cameras[camera], "pose", None)
        snapshot_actions = (
            getattr(cam_pose_cfg, "snapshot_actions", None) if cam_pose_cfg else None
        )
        record_actions = (
            getattr(cam_pose_cfg, "record_actions", None) if cam_pose_cfg else None
        )
        record_enabled = bool(self.config.cameras[camera].record.enabled)

        has_snapshot = bool(snapshot_actions and action and action in snapshot_actions)
        has_clip = bool(
            record_enabled and record_actions and action and action in record_actions
        )

        box = pd.get("box") or [0, 0, 0, 0]
        region = pd.get("region") or [0, 0, 0, 0]
        keypoints = pd.get("keypoints")

        return {
            "id": event_id,
            "camera": camera,
            "label": action or "pose",
            "sub_label": (action, ac) if action else None,
            "start_time": start_time,
            "end_time": end_time,
            "false_positive": False,
            "top_score": top_score,
            "score": ac,
            "entered_zones": [],
            "current_zones": [],
            "thumbnail": None,
            "has_clip": has_clip,
            "has_snapshot": has_snapshot,
            "snapshot": {
                "frame_time": frame_time,
                "score": ac,
                "confidence": ac,
                "region": region,
                "box": box,
                "attributes": [],
                "keypoints": keypoints,
            },
            "confidence": ac,
            "action": action,
            "action_confidence": ac,
            "stationary": False,
            "attributes": [],
            "average_estimated_speed": 0,
            "velocity_angle": 0,
            "recognized_license_plate": None,
            "path_data": None,
        }

    def _latest_pose_thumbnail_bytes(self, camera: str) -> Optional[bytes]:
        """Return the most recent webp thumbnail bytes received for `camera`.

        The bytes are produced by `pose_detection.integration.detect_poses`
        in the detect process, where the SHM-backed YUV frame is still
        valid, base64-encoded by `pose_processing` for ZMQ transport, and
        decoded back to bytes by `_drain_pose_thumb_subscriber` here.  We
        just cache the latest one per camera and return it on demand.
        """
        return self._latest_thumbnail_bytes.get(camera)

    def _drain_pose_thumb_subscriber(self) -> None:
        """Drain all pending messages from the pose-thumbnail side-channel
        and stash the latest bytes per camera in `_latest_thumbnail_bytes`.

        Called once per main run-loop iteration.  Non-blocking — uses a
        zero-timeout poll so we don't slow the main loop down.  Multiple
        thumbnails per camera in the queue are folded — only the most
        recent one is kept (which is fine since we only care about the
        thumbnail at the moment a pose action fires).
        """
        if self.pose_thumb_subscriber is None:
            return
        # Drain at most a small bounded number of pending messages per
        # call to avoid an infinite loop if the subscriber's
        # check_for_update keeps returning the (None, None) "no message"
        # sentinel (which is truthy but contains no real payload).
        for _ in range(16):
            try:
                msg = self.pose_thumb_subscriber.check_for_update(timeout=0)
            except Exception:
                break
            if not msg:
                break
            topic, payload = msg
            if payload is None:
                # No message currently waiting on the socket — done.
                break
            try:
                camera, _frame_time, thumb_b64 = payload
            except Exception:
                logger.debug("Unexpected pose-thumb payload format on the side-channel")
                continue
            if not thumb_b64:
                continue
            try:
                self._latest_thumbnail_bytes[camera] = base64.b64decode(thumb_b64)
            except Exception:
                logger.debug(f"Failed to base64-decode pose thumbnail for {camera}")

    def _write_pose_thumbnail_to_disk(
        self, camera: str, event_id: str, thumbnail_bytes: bytes
    ) -> None:
        """Write thumbnail bytes to {THUMB_DIR}/{camera}/{event_id}.webp.

        Matches the layout that `frigate.util.path.get_event_thumbnail_bytes`
        reads from, so the standard `/api/events/{id}/thumbnail.{ext}` route
        will serve it without further changes.
        """
        directory = os.path.join(THUMB_DIR, camera)
        try:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f"{event_id}.webp"), "wb") as f:
                f.write(thumbnail_bytes)
        except Exception:
            logger.exception(f"Failed to write pose thumbnail to disk for {event_id}")

    def _should_create_pose_event(self, camera: str) -> bool:
        """Decide whether pose_consumer should synthesize a tracked_pose Event.

        The standard tracker (and the pose-as-person-detector path in
        video.py:1170) already creates a regular `person` Event with a
        thumbnail in every camera configuration EXCEPT one:

            detect.enabled: false  AND  pose.detect_persons: false

        In that single case nothing else creates an Event row for the
        fall, so the pose lifecycle in pose_consumer is the only source.
        For every other config a posefall-* Event would be a duplicate
        of an existing person Event, the CCC would prefer it because of
        the explicit `falling` label, and it would render with no
        thumbnail (broken card in the app).
        """
        cam_config = self.config.cameras.get(camera)
        if cam_config is None:
            return False

        detect_enabled = bool(getattr(cam_config.detect, "enabled", True))
        pose_cfg = getattr(cam_config, "pose", None)
        detect_persons = bool(
            getattr(pose_cfg, "detect_persons", True) if pose_cfg else True
        )

        return (not detect_enabled) and (not detect_persons)

    def _publish_pose_action_event(
        self,
        camera: str,
        pose_track_id: Any,
        pd: dict[str, Any],
        frame_time: float,
        frame_name: str,
    ) -> None:
        """Publish a tracked_pose start/update event for the given pose action.

        First match for a (camera, track_id, action) triple → start.
        Subsequent matches → update with bumped top_score and refreshed last_seen.

        No-op when another code path is already responsible for creating
        the Event row (see _should_create_pose_event).
        """
        action = pd.get("action")
        if not action or self.event_sender is None:
            return

        if not self._should_create_pose_event(camera):
            return

        ac = float(pd.get("action_confidence", 0.0) or 0.0)
        track_key = f"{camera}:{pose_track_id}:{action}"
        existing_event_id = self._pose_event_id_by_track.get(track_key)

        # Cooldown reopen path: if this (camera, track_id, action) had a
        # lifecycle that recently ended within event_cooldown_seconds, treat
        # the new match as a silent continuation of the previous Event
        # instead of starting a fresh one. This prevents fragmenting a
        # single fall incident into multiple posefall-* rows when the pose
        # detector briefly drops and re-fires while the subject is still
        # on the ground. See plan bright-forging-knuth.md.
        if existing_event_id is None:
            cooldown_entry = self._pose_event_cooldown.get(track_key)
            now = time.time()
            if cooldown_entry is not None:
                if now < cooldown_entry["expires_at"]:
                    # Reopen: restore the previous lifecycle state and fall
                    # through to the "subsequent match" update branch below.
                    # Do NOT publish a new tracked_pose start; the DB row
                    # already exists and re-publishing start would fire a
                    # spurious frigate/events type=new on MQTT.
                    reopened_id = cooldown_entry["event_id"]
                    restored = cooldown_entry["active"]
                    restored["last_seen"] = now
                    restored["frame_name"] = frame_name
                    restored["last_pd"] = pd
                    self._active_pose_events[reopened_id] = restored
                    self._pose_event_id_by_track[track_key] = reopened_id
                    self._pose_event_cooldown.pop(track_key, None)
                    existing_event_id = reopened_id
                    logger.debug(
                        f"Pose event lifecycle reopened via cooldown: "
                        f"{reopened_id} ({action} on {camera})"
                    )
                else:
                    # Expired entry; genuine new incident after recovery.
                    self._pose_event_cooldown.pop(track_key, None)

        if existing_event_id is None:
            # New pose-action lifecycle. Use a distinct id namespace so it
            # never collides with synthesized-detection ids in events_in_process.
            event_id = f"posefall-{camera}-{pose_track_id}-{int(frame_time * 1000)}"
            event_data = self._build_pose_event_data(
                camera,
                pd,
                frame_time,
                event_id=event_id,
                start_time=frame_time,
                end_time=None,
                top_score=0.0,  # bumped on first update so should_update_db fires
            )
            try:
                self.event_sender.publish(
                    (
                        EventTypeEnum.tracked_pose,
                        EventStateEnum.start,
                        camera,
                        frame_name,
                        event_data,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish tracked_pose start event for %s", event_id
                )
                return

            # Grab the latest thumbnail bytes shipped from the detect
            # process for this camera, so the standard
            # /api/events/{id}/thumbnail.{ext} route can serve it once
            # the lifecycle ends.
            thumb_bytes = self._latest_pose_thumbnail_bytes(camera)

            self._pose_event_id_by_track[track_key] = event_id
            self._active_pose_events[event_id] = {
                "camera": camera,
                "frame_name": frame_name,
                "last_seen": time.time(),
                "top_score": 0.0,
                "start_time": frame_time,
                "track_key": track_key,
                "last_pd": pd,
                "thumbnail_bytes": thumb_bytes,
            }
            logger.debug(
                f"Pose event lifecycle started: {event_id} ({action} on {camera})"
            )
            return

        # Subsequent match: publish an update with bumped top_score so the
        # events maintainer's should_update_db comparison fires.
        active = self._active_pose_events.get(existing_event_id)
        if active is None:
            # Stale mapping; clean up and re-enter as a new lifecycle.
            self._pose_event_id_by_track.pop(track_key, None)
            self._publish_pose_action_event(
                camera, pose_track_id, pd, frame_time, frame_name
            )
            return

        new_top_score = max(active["top_score"], ac)
        event_data = self._build_pose_event_data(
            camera,
            pd,
            frame_time,
            event_id=existing_event_id,
            start_time=active["start_time"],
            end_time=None,
            top_score=new_top_score,
        )
        try:
            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_pose,
                    EventStateEnum.update,
                    camera,
                    frame_name,
                    event_data,
                )
            )
        except Exception:
            logger.exception(
                "Failed to publish tracked_pose update event for %s", existing_event_id
            )
            return

        active["last_seen"] = time.time()
        active["top_score"] = new_top_score
        active["frame_name"] = frame_name
        active["last_pd"] = pd

        # Refresh thumbnail with the latest bytes shipped from the
        # detect process so the saved image reflects the most recent
        # frame seen during the lifecycle.
        new_thumb = self._latest_pose_thumbnail_bytes(camera)
        if new_thumb is not None:
            active["thumbnail_bytes"] = new_thumb

    def _check_pose_event_timeouts(self) -> None:
        """End any pose-driven Event whose action hasn't fired in POSE_EVENT_END_TIMEOUT.

        Called once per main loop iteration. Publishes a tracked_pose `end`
        message and removes the entry from the lifecycle dicts.
        """
        if not self._active_pose_events or self.event_sender is None:
            return

        now = time.time()
        ended_event_ids: list[str] = []

        for event_id, active in self._active_pose_events.items():
            if now - active["last_seen"] < POSE_EVENT_END_TIMEOUT:
                continue

            camera = active["camera"]
            frame_name = active["frame_name"]
            pd = active.get("last_pd") or {}

            # Persist the captured thumbnail bytes to disk so the
            # standard /api/events/{id}/thumbnail.{ext} route can serve
            # them.  No-op if we never captured a frame for this event.
            thumb_bytes = active.get("thumbnail_bytes")
            if thumb_bytes:
                self._write_pose_thumbnail_to_disk(camera, event_id, thumb_bytes)

            event_data = self._build_pose_event_data(
                camera,
                pd,
                frame_time=active["start_time"],
                event_id=event_id,
                start_time=active["start_time"],
                end_time=active["last_seen"],
                top_score=active["top_score"],
            )

            try:
                self.event_sender.publish(
                    (
                        EventTypeEnum.tracked_pose,
                        EventStateEnum.end,
                        camera,
                        frame_name,
                        event_data,
                    )
                )
                logger.info(
                    f"Pose event lifecycle ended: {event_id} "
                    f"(top_score={active['top_score']:.3f}, "
                    f"duration={active['last_seen'] - active['start_time']:.1f}s)"
                )
            except Exception:
                logger.exception(
                    "Failed to publish tracked_pose end event for %s", event_id
                )

            # Also publish a frontend-shaped end message to the dispatcher
            # `events` topic so the live grid removes the synthesized
            # tracked object from its `objects` array. The frontend keys
            # by the standard tracker's object id, which is the
            # `pose_track_id` segment of `track_key` (and matches the id
            # that `pose_consumer` synthesized into detected_frames_queue
            # at line ~1175). Without this the red border + sub_label
            # tooltip stay stuck on the live tile until page reload,
            # because the standard tracker's CameraState only sees what's
            # pushed to detected_frames_queue and never gets a removal
            # signal once pose actions go silent.
            track_key = active.get("track_key", "")
            parts = track_key.split(":") if track_key else []
            synth_id = parts[1] if len(parts) >= 2 else None
            if synth_id:
                try:
                    # The frontend's use-camera-activity hook early-returns
                    # on `after.camera !== camera.name`, so `camera` must
                    # be present in `after` for the removal to fire.
                    synth_payload = {"id": synth_id, "camera": camera}
                    end_message = {
                        "before": synth_payload,
                        "after": synth_payload,
                        "type": "end",
                    }
                    self.dispatcher.publish(
                        "events", json.dumps(end_message), retain=False
                    )
                except Exception:
                    logger.exception(
                        "Failed to publish dispatcher 'events' end for "
                        "synthesized pose object %s",
                        synth_id,
                    )

            ended_event_ids.append(event_id)

        for event_id in ended_event_ids:
            active = self._active_pose_events.pop(event_id, None)
            if active is not None:
                track_key = active.get("track_key", "")
                self._pose_event_id_by_track.pop(track_key, None)
                # Record a cooldown window so a same-(camera, track, action)
                # re-fire within event_cooldown_seconds reopens THIS event
                # instead of creating a new posefall-* row. The reopen path
                # in _publish_pose_action_event reads `active` back out.
                if track_key:
                    camera = active.get("camera")
                    cam_pose_cfg = (
                        self.config.cameras[camera].pose
                        if camera and camera in self.config.cameras
                        else None
                    )
                    cooldown_seconds = int(
                        getattr(cam_pose_cfg, "event_cooldown_seconds", 60)
                        if cam_pose_cfg is not None
                        else 60
                    )
                    if cooldown_seconds > 0:
                        self._pose_event_cooldown[track_key] = {
                            "expires_at": time.time() + cooldown_seconds,
                            "event_id": event_id,
                            "active": active,
                        }

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
                # Close out pose-driven Event lifecycles whose action has
                # gone silent.  Runs every loop iteration so end events
                # fire within ~POSE_EVENT_END_TIMEOUT + 0.5s of last match.
                self._check_pose_event_timeouts()

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

                # Drain the pose-thumbnail side-channel into the per-camera
                # latest_thumbnail_bytes cache.  pose_processing publishes
                # webp bytes (base64-encoded for JSON transport) on the
                # `pose` sub-topic at every pose detection cycle.  We
                # accumulate the latest one per camera so the lifecycle
                # helpers below can grab it.
                self._drain_pose_thumb_subscriber()

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

                # Publish keypoints to MQTT for downstream consumers.
                # Filter to FRESH poses only (time_since_update == 0). Stale
                # tracks are kept alive by track_max_disappeared_seconds for
                # CameraState lifecycle continuity, but publishing their
                # frozen last-known keypoints would leak ghost skeletons to
                # anything rendering the stream. Same principle as the
                # pose_processing freshness gate: stale poses are not new
                # information.
                fresh_poses = [
                    p for p in tracked_poses if getattr(p, "time_since_update", 0) == 0
                ]
                cam_config = self.config.cameras.get(camera)
                if (
                    cam_config
                    and hasattr(cam_config, "pose")
                    and cam_config.pose
                    and cam_config.pose.publish_keypoints
                    and fresh_poses
                ):
                    try:
                        frame_w = cam_config.detect.width
                        frame_h = cam_config.detect.height
                        payload = _keypoints_to_mqtt_payload(
                            fresh_poses, frame_time, frame_w, frame_h
                        )
                        self.dispatcher.publish(
                            f"{camera}/pose_keypoints",
                            json.dumps(payload),
                            retain=False,
                        )
                    except Exception:
                        logger.debug(f"Failed to publish keypoints for {camera}")

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

                # Pose-action matching lifecycle + posefall-* Event row
                # creation. Runs ONLY for poses with a fresh action
                # classification this frame (falling, standing, etc.).
                #
                # We do NOT push to detected_frames_queue from here anymore.
                # video.py injects tracked poses directly into the detections
                # dict on every camera frame (see the detect_persons=false
                # branch), so object_processing's CameraState sees continuous
                # presence from a single push per frame. That eliminates the
                # create/update/end cycling on frigate/events MQTT that the
                # old pose_consumer push (inside-the-match-loop or hoisted)
                # caused when interleaved with video.py's empty-detections
                # push for the same camera.
                try:
                    # Resolve the camera's configured pose.actions list once
                    # for both the clear-stale and set-new passes below.
                    try:
                        cam_actions = self.config.cameras[camera].pose.actions
                    except Exception:
                        cam_actions = None

                    # Resolve the shared standard-tracker CameraState once.
                    shared_tracked_objects = {}
                    if (
                        self.tracked_object_processor is not None
                        and camera in self.tracked_object_processor.camera_states
                    ):
                        shared_tracked_objects = (
                            self.tracked_object_processor.camera_states[
                                camera
                            ].tracked_objects
                        )

                    # Helper: IoU-match a pose bbox to an existing tracked
                    # object id in shared_tracked_objects. Returns the
                    # matched id or None.
                    def _iou_match_pose_to_tracked_object(
                        pose_box_xyxy,
                    ) -> Optional[str]:
                        if not pose_box_xyxy or len(pose_box_xyxy) != 4:
                            return None
                        px1, py1, px2, py2 = pose_box_xyxy
                        best_iou = 0.3
                        matched = None
                        for obj in list(shared_tracked_objects.values()):
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
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                p_area = max(1, (px2 - px1) * (py2 - py1))
                                o_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                                iou = inter / (p_area + o_area - inter)
                                if iou > best_iou:
                                    best_iou = iou
                                    matched = obj.obj_data.get("id")
                        return matched

                    # Pass 1 — clear stale sub_label / action on tracked
                    # objects whose matched pose is no longer classified as
                    # a configured action. Without this, a sub_label set
                    # during a previous fall would STICK on the tracked
                    # object forever (video.py's per-frame detection dict
                    # intentionally omits sub_label so pose_consumer is
                    # the sole writer — but if pose_consumer only ever
                    # SETS and never CLEARS, once a fall is classified
                    # the tracked_object stays "falling" until the track
                    # itself ends).
                    for pd in tracked_dict.values():
                        pose_box = pd.get("box")
                        current_action = pd.get("action")
                        is_configured_action = bool(
                            current_action
                            and cam_actions
                            and current_action in cam_actions
                        )
                        if is_configured_action:
                            # The set-new pass below handles these.
                            continue
                        cleared_id = _iou_match_pose_to_tracked_object(pose_box)
                        if cleared_id and cleared_id in shared_tracked_objects:
                            try:
                                cleared_obj = shared_tracked_objects[cleared_id]
                                # Only write if there's something to clear,
                                # to avoid firing spurious update callbacks.
                                if (
                                    cleared_obj.obj_data.get("sub_label") is not None
                                    or cleared_obj.obj_data.get("action") is not None
                                ):
                                    cleared_obj.obj_data["sub_label"] = None
                                    cleared_obj.obj_data["action"] = None
                                    cleared_obj.obj_data["action_confidence"] = 0.0
                            except Exception:
                                pass

                    # Pass 2 — the existing set-new loop: for poses with
                    # actions in the configured list, run the IoU match,
                    # attach the sub_label, set has_snapshot / has_clip.
                    for pd in tracked_dict.values():
                        action = pd.get("action")
                        if not action:
                            continue

                        if cam_actions and action not in cam_actions:
                            continue

                        # IoU-match the pose to an existing tracked_object
                        # via the helper resolved at the top of this try
                        # block. (shared_tracked_objects is the standard
                        # tracker's CameraState, not pose_consumer's local
                        # one — see the note above.)
                        matched_id = _iou_match_pose_to_tracked_object(pd.get("box"))

                        if matched_id and matched_id in shared_tracked_objects:
                            # Attach action as a sub_label to the matched object and
                            # trigger an update callback so downstream systems see it.
                            try:
                                matched_obj = shared_tracked_objects[matched_id]
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
                            except Exception:
                                logger.exception(
                                    "Failed to attach pose action to matched tracked object"
                                )

                        # Publish a tracked_pose lifecycle event so
                        # frigate.events.maintainer.handle_pose_detection
                        # writes an Event row.  Independent of the
                        # synthesized-detection push below (which only drives
                        # ReviewSegment creation), and required because
                        # object_processing's camera_state never fires `end`
                        # for pose-only cameras: nothing else pushes to
                        # detected_frames_queue, so the synthesized object id
                        # is never removed and `end` callbacks never run.
                        try:
                            self._publish_pose_action_event(
                                camera,
                                pd.get("id", f"pose_{int(frame_time)}"),
                                pd,
                                frame_time,
                                frame_name,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to publish pose-action lifecycle event"
                            )

                except Exception:
                    logger.exception("Error while processing pose-action matches")

            except Exception:
                logger.exception("Error in PoseConsumer main loop")

        # cleanup
        for state in self.camera_states.values():
            state.shutdown()

        self.detection_publisher = None
        self.event_sender = None
