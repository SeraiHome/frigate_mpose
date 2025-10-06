import json
import logging
import queue
import threading
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict

import numpy as np

from frigate.camera.state import CameraState
from frigate.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.comms.dispatcher import Dispatcher
from frigate.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.config import FrigateConfig
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
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
    ) -> None:
        super().__init__(name="pose_processor")
        self.config = config
        self.dispatcher = dispatcher
        self.tracked_poses_queue = tracked_poses_queue
        self.stop_event: MpEvent = stop_event
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
                "type": "new" if pose.previous.get("false_positive", True) else "update",
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

        camera_state = CameraState(
            name=camera,
            config=self.config.cameras[camera],
            frame_manager=self.frame_manager,
        )

        camera_state.on("start", start)
        camera_state.on("update", update)
        camera_state.on("end", end)

        self.camera_states[camera] = camera_state

    def should_save_pose_snapshot(self, camera: str, pose: TrackedPose) -> bool:
        """Determine if pose snapshot should be saved."""
        camera_config = self.config.cameras[camera]
        
        # Check if pose detection snapshots are enabled
        if not getattr(camera_config.snapshots, 'enabled', False):
            return False
        
        # Check if this pose action should trigger a snapshot
        pose_config = getattr(camera_config, 'pose', None)
        if pose_config and hasattr(pose_config, 'snapshot_actions'):
            if pose.action not in pose_config.snapshot_actions:
                return False
        
        # Check confidence threshold
        if pose.confidence < getattr(pose_config, 'confidence_threshold', 0.4):
            return False
        
        return True

    def should_retain_pose_recording(self, camera: str, pose: TrackedPose) -> bool:
        """Determine if pose recording should be retained."""
        camera_config = self.config.cameras[camera]
        
        # Check if pose detection recording is enabled
        if not getattr(camera_config.record, 'enabled', False):
            return False
        
        # Check if this pose action should trigger recording
        pose_config = getattr(camera_config, 'pose', None)
        if pose_config and hasattr(pose_config, 'record_actions'):
            if pose.action not in pose_config.record_actions:
                return False
        
        # Check confidence threshold
        if pose.confidence < getattr(pose_config, 'confidence_threshold', 0.4):
            return False
        
        return True

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Check for camera config updates
                self.camera_config_subscriber.check_for_updates()

                # Process pose detections
                try:
                    (
                        camera,
                        frame_time,
                        tracked_poses,
                        motion_boxes,
                        regions,
                    ) = self.tracked_poses_queue.get(True, 1)
                except queue.Empty:
                    continue

                camera_state = self.camera_states.get(camera)
                if camera_state is None:
                    continue

                # Process each tracked pose
                for pose in tracked_poses:
                    # Update pose zones
                    self._update_pose_zones(camera, pose)
                    
                    # Trigger camera state updates
                    if pose.time_since_update == 0:  # New or updated pose
                        if pose.age == 1:  # New pose
                            camera_state.on("start", camera, pose, f"{camera}_{frame_time}")
                        else:  # Updated pose
                            camera_state.on("update", camera, pose, f"{camera}_{frame_time}")
                    elif pose.time_since_update > 10:  # Lost pose
                        camera_state.on("end", camera, pose, f"{camera}_{frame_time}")

                # Update camera activity
                self._update_camera_activity(camera, tracked_poses)

            except Exception as e:
                logger.error(f"Error in pose processor: {e}")

        logger.info("Exiting pose processor...")

    def _update_pose_zones(self, camera: str, pose: TrackedPose) -> None:
        """Update pose zone tracking."""
        camera_config = self.config.cameras[camera]
        
        # Check which zones the pose is currently in
        current_zones = set()
        
        if hasattr(camera_config, 'zones') and pose.bbox:
            x, y, w, h = pose.bbox
            pose_center = (x + w/2, y + h/2)
            
            for zone_name, zone_config in camera_config.zones.items():
                if self._point_in_zone(pose_center, zone_config.coordinates):
                    current_zones.add(zone_name)
        
        # Update pose zones
        new_zones = current_zones - pose.current_zones
        pose.entered_zones.update(new_zones)
        pose.current_zones = current_zones

    def _point_in_zone(self, point, zone_coords) -> bool:
        """Check if a point is inside a zone."""
        try:
            import cv2
            # Convert zone coordinates to the right format for cv2.pointPolygonTest
            zone_array = np.array(zone_coords, dtype=np.int32)
            result = cv2.pointPolygonTest(zone_array, point, False)
            return result >= 0
        except Exception:
            return False

    def _update_camera_activity(self, camera: str, poses: list[TrackedPose]) -> None:
        """Update camera activity based on pose detections."""
        active_poses = [p for p in poses if not p.false_positive]
        
        activity = {
            "pose_count": len(active_poses),
            "actions": {},
            "zones": defaultdict(int),
        }
        
        # Count actions and zones
        for pose in active_poses:
            action = pose.action.value
            activity["actions"][action] = activity["actions"].get(action, 0) + 1
            
            for zone in pose.current_zones:
                activity["zones"][zone] += 1
        
        self.camera_activity[camera] = activity