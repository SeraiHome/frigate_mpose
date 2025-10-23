import json
import logging
import queue
import threading
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, List

import cv2

from frigate.camera.state import CameraState
from frigate.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.comms.dispatcher import Dispatcher
from frigate.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.config import FrigateConfig
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import FAST_QUEUE_TIMEOUT
from frigate.events.pose_types import PoseEventStateEnum, PoseEventTypeEnum
from frigate.track.tracked_pose import TrackedPose
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


class PoseProcessingState(str, Enum):
    complete = "complete"
    start = "start"
    end = "end"


class TrackedPoseProcessor(threading.Thread):
    def __init__(
        self,
        config: FrigateConfig,
        dispatcher: Dispatcher,
        tracked_poses_queue: MpQueue,
        stop_event: MpEvent,
        ptz_autotracker_thread=None,
    ) -> None:
        super().__init__(name="pose_processor")
        self.config = config
        self.dispatcher = dispatcher
        self.tracked_poses_queue = tracked_poses_queue
        self.stop_event: MpEvent = stop_event
        self.ptz_autotracker_thread = ptz_autotracker_thread
        self.camera_states: dict[str, CameraState] = {}
        self.frame_manager = SharedMemoryFrameManager()

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

        for camera in self.config.cameras.keys():
            if (
                camera in self.config.cameras
                and self.config.cameras[camera].pose.enabled
            ):
                self.create_camera_state(camera)

    def create_camera_state(self, camera: str) -> None:
        """Creates a new camera state for pose tracking."""

        def start(camera: str, pose: TrackedPose, frame_name: str) -> None:
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.start,
                    camera,
                    frame_name,
                    pose.to_dict(),
                )
            )

        def update(camera: str, pose: TrackedPose, frame_name: str) -> None:
            pose.has_snapshot = self.should_save_pose_snapshot(camera, pose)
            pose.has_clip = self.should_retain_pose_recording(camera, pose)
            after = pose.to_dict()
            message = {
                "before": pose.previous,
                "after": after,
                "type": "new"
                if pose.previous.get("false_positive", True)
                else "update",
            }
            self.dispatcher.publish("pose_events", json.dumps(message), retain=False)
            pose.previous = after
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.update,
                    camera,
                    frame_name,
                    pose.to_dict(),
                )
            )

        def end(camera: str, pose: TrackedPose, frame_name: str) -> None:
            self.event_sender.publish(
                (
                    PoseEventTypeEnum.pose_detected,
                    PoseEventStateEnum.end,
                    camera,
                    frame_name,
                    pose.to_dict(),
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
                        if (
                            camera in self.camera_states
                            and self.camera_states[camera].prev_enabled is None
                        ):
                            self.camera_states[
                                camera
                            ].prev_enabled = self.config.cameras[camera].enabled
                elif "add" in updated_topics:
                    for camera in updated_topics["add"]:
                        self.config.cameras[camera] = (
                            self.camera_config_subscriber.camera_configs[camera]
                        )
                        if self.config.cameras[camera].pose.enabled:
                            self.create_camera_state(camera)
                elif "remove" in updated_topics:
                    for camera in updated_topics["remove"]:
                        if camera in self.camera_states:
                            camera_state = self.camera_states[camera]
                            camera_state.shutdown()
                            self.camera_states.pop(camera)

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

                    # Check the format of the data we received
                    if len(queue_data) == 6:
                        # Data format: (camera, frame_name, frame_time, tracked_poses, motion_boxes, regions)
                        (
                            camera,
                            frame_name,
                            frame_time,
                            tracked_poses,
                            motion_boxes,
                            regions,
                        ) = queue_data
                    else:
                        # Older format: (camera, frame_time, tracked_poses, motion_boxes, regions)
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
                    # Update pose zones before sending to camera state
                    for pose in tracked_poses:
                        if isinstance(pose, TrackedPose):
                            self._update_pose_zones(camera, pose)

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
                                pose_dict["area"] = w * h
                                pose_dict["ratio"] = w / h if h > 0 else 1.0
                                pose_dict["region"] = [0, 0, 0, 0]  # Default region
                            else:
                                # Default values if no bbox
                                pose_dict["box"] = [0, 0, 10, 10]
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
                                pose_dict["area"] = w * h
                                pose_dict["ratio"] = w / h if h > 0 else 1.0
                            elif "box" not in pose:
                                pose_dict["box"] = [0, 0, 10, 10]
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

                    # Publish detection info for this frame
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
                        DetectionTypeEnum.video.value,  # Use "video" type since pose is not defined in enum
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
        self.event_sender.stop()
        self.event_end_subscriber.stop()
        self.camera_config_subscriber.stop()

        logger.info("Exiting pose processor...")

    def _update_pose_zones(self, camera: str, pose: TrackedPose) -> None:
        """Update pose zone tracking."""
        camera_config = self.config.cameras[camera]

        # Check which zones the pose is currently in
        current_zones = set()

        if hasattr(camera_config, "zones") and pose.bbox:
            # Use bottom center of bounding box for zone detection
            x, y, w, h = pose.bbox
            bottom_center = (
                x + w / 2,
                y + h,
            )  # Bottom center is more reliable for zone detection

            for zone_name, zone_config in camera_config.zones.items():
                # Skip zones that don't include poses/persons
                if hasattr(zone_config, "objects") and len(zone_config.objects) > 0:
                    if (
                        "person" not in zone_config.objects
                        and "pose" not in zone_config.objects
                    ):
                        continue

                if hasattr(zone_config, "contour"):
                    # Use the contour for zone detection
                    if (
                        cv2.pointPolygonTest(zone_config.contour, bottom_center, False)
                        >= 0
                    ):
                        current_zones.add(zone_name)

        # Update pose zones
        new_zones = current_zones - pose.current_zones
        pose.entered_zones.update(new_zones)
        pose.current_zones = current_zones

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
