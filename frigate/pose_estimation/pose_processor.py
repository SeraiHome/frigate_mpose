import logging
import threading
from collections import defaultdict
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frigate.comms.events_updater import EventUpdatePublisher
from frigate.config import CameraConfig, FrigateConfig
from frigate.events.types import EventStateEnum, EventTypeEnum
from frigate.models import Event
from frigate.track.tracked_object import TrackedObject
from frigate.util.builtin import to_relative_box

from .base import Pose, PoseKeypoint, RemotePoseDetector

logger = logging.getLogger(__name__)


class PoseEvent:
    """Represents a pose detection event."""
    def __init__(
        self,
        pose_id: str,
        camera: str,
        pose: Pose,
        frame_time: float,
        zones: List[str] = None,
        actions: List[str] = None,
    ):
        self.pose_id = pose_id
        self.camera = camera
        self.pose = pose
        self.frame_time = frame_time
        self.start_time = frame_time
        self.end_time = None
        self.zones = zones or []
        self.actions = actions or []
        self.has_clip = False
        self.has_snapshot = False
        self.top_score = pose.confidence
        self.current_zones = zones or []
        self.entered_zones = zones or []
        self.stationary_count = 0
        self.active = True
        self.thumbnail_data = None
        self.attributes = {}

    def update(self, pose: Pose, frame_time: float, zones: List[str] = None, actions: List[str] = None):
        """Update the pose event with new data."""
        self.pose = pose
        self.frame_time = frame_time
        self.zones = zones or []
        self.actions = actions or []
        self.current_zones = zones or []
        self.entered_zones = list(set(self.entered_zones + self.current_zones))
        self.top_score = max(self.top_score, pose.confidence)
        self.active = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose event to dictionary for database storage."""
        return {
            "id": self.pose_id,
            "camera": self.camera,
            "label": "pose",
            "sub_label": ",".join(self.actions) if self.actions else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "score": self.pose.confidence,
            "top_score": self.top_score,
            "zones": self.entered_zones,
            "has_clip": self.has_clip,
            "has_snapshot": self.has_snapshot,
            "pose_data": self.pose.to_dict(),
            "attributes": self.attributes,
        }


class PoseTracker:
    """Tracks poses across frames."""
    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.tracked_poses: Dict[str, PoseEvent] = {}
        self.pose_id_counter = 0
        self.max_disappeared = camera_config.pose.max_disappeared
        self.disappeared_counts: Dict[str, int] = defaultdict(int)

    def update(self, poses: List[Pose], frame_time: float, zones: List[str] = None) -> List[PoseEvent]:
        """Update tracked poses with new detections."""
        # For now, simple tracking - match poses based on proximity
        # In a real implementation, you'd use more sophisticated tracking
        
        updated_events = []
        matched_ids = set()
        
        for pose in poses:
            # Find closest existing pose
            best_match_id = None
            best_distance = float('inf')
            
            if pose.bbox:
                pose_center = (
                    (pose.bbox[0] + pose.bbox[2]) / 2,
                    (pose.bbox[1] + pose.bbox[3]) / 2
                )
                
                for pose_id, tracked_event in self.tracked_poses.items():
                    if pose_id in matched_ids:
                        continue
                    
                    if tracked_event.pose.bbox:
                        tracked_center = (
                            (tracked_event.pose.bbox[0] + tracked_event.pose.bbox[2]) / 2,
                            (tracked_event.pose.bbox[1] + tracked_event.pose.bbox[3]) / 2
                        )
                        
                        distance = np.sqrt(
                            (pose_center[0] - tracked_center[0]) ** 2 +
                            (pose_center[1] - tracked_center[1]) ** 2
                        )
                        
                        if distance < best_distance and distance < 100:  # threshold
                            best_distance = distance
                            best_match_id = pose_id
            
            if best_match_id:
                # Update existing pose
                matched_ids.add(best_match_id)
                self.tracked_poses[best_match_id].update(pose, frame_time, zones)
                self.disappeared_counts[best_match_id] = 0
                updated_events.append(self.tracked_poses[best_match_id])
            else:
                # Create new pose event
                pose_id = f"{self.camera_config.name}_pose_{self.pose_id_counter}"
                self.pose_id_counter += 1
                pose_event = PoseEvent(pose_id, self.camera_config.name, pose, frame_time, zones)
                self.tracked_poses[pose_id] = pose_event
                self.disappeared_counts[pose_id] = 0
                updated_events.append(pose_event)
        
        # Handle disappeared poses
        disappeared_ids = []
        for pose_id in list(self.tracked_poses.keys()):
            if pose_id not in matched_ids:
                self.disappeared_counts[pose_id] += 1
                if self.disappeared_counts[pose_id] > self.max_disappeared:
                    # Mark pose as ended
                    self.tracked_poses[pose_id].active = False
                    self.tracked_poses[pose_id].end_time = frame_time
                    updated_events.append(self.tracked_poses[pose_id])
                    disappeared_ids.append(pose_id)
        
        # Remove disappeared poses
        for pose_id in disappeared_ids:
            del self.tracked_poses[pose_id]
            del self.disappeared_counts[pose_id]
        
        return updated_events


class PoseActionDetector:
    """Detects actions from pose keypoints."""
    def __init__(self, config: CameraConfig):
        self.config = config
        self.action_config = config.pose.actions
        self.action_threshold = self.action_config.action_threshold

    def detect_actions(self, pose: Pose) -> List[str]:
        """Detect actions from pose keypoints."""
        if not self.action_config.enabled:
            return []
        
        actions = []
        
        # Simple action detection based on keypoint positions
        # This is a placeholder - real implementation would use ML models
        
        if "falling" in self.action_config.actions:
            if self._is_falling(pose):
                actions.append("falling")
        
        if "fighting" in self.action_config.actions:
            if self._is_fighting(pose):
                actions.append("fighting")
        
        if "running" in self.action_config.actions:
            if self._is_running(pose):
                actions.append("running")
        
        return actions

    def _is_falling(self, pose: Pose) -> bool:
        """Detect if pose indicates falling."""
        # Placeholder logic - check if torso is horizontal
        if len(pose.keypoints) < 17:
            return False
        
        # Get shoulder and hip keypoints (indices 5,6,11,12 in COCO format)
        left_shoulder = pose.keypoints[5]
        right_shoulder = pose.keypoints[6]
        left_hip = pose.keypoints[11]
        right_hip = pose.keypoints[12]
        
        # Check if all keypoints are valid
        if any(kp.confidence < 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return False
        
        # Calculate angle of torso
        shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, 
                          (left_shoulder.y + right_shoulder.y) / 2)
        hip_center = ((left_hip.x + right_hip.x) / 2,
                     (left_hip.y + right_hip.y) / 2)
        
        # If shoulder is not significantly above hip, might be falling
        vertical_diff = shoulder_center[1] - hip_center[1]
        horizontal_diff = abs(shoulder_center[0] - hip_center[0])
        
        return horizontal_diff > abs(vertical_diff) * 2

    def _is_fighting(self, pose: Pose) -> bool:
        """Detect if pose indicates fighting."""
        # Placeholder - check for raised arms
        if len(pose.keypoints) < 17:
            return False
        
        # Check if hands are raised above shoulders
        left_wrist = pose.keypoints[9]
        right_wrist = pose.keypoints[10]
        left_shoulder = pose.keypoints[5]
        right_shoulder = pose.keypoints[6]
        
        if any(kp.confidence < 0.5 for kp in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False
        
        hands_raised = (left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y)
        return hands_raised

    def _is_running(self, pose: Pose) -> bool:
        """Detect if pose indicates running."""
        # Placeholder - check leg positions
        if len(pose.keypoints) < 17:
            return False
        
        # Check if legs are in running position
        left_knee = pose.keypoints[13]
        right_knee = pose.keypoints[14]
        left_ankle = pose.keypoints[15]
        right_ankle = pose.keypoints[16]
        
        if any(kp.confidence < 0.5 for kp in [left_knee, right_knee, left_ankle, right_ankle]):
            return False
        
        # Check if knees are at different heights (one leg raised)
        knee_diff = abs(left_knee.y - right_knee.y)
        ankle_spread = abs(left_ankle.x - right_ankle.x)
        
        return knee_diff > 30 and ankle_spread > 50


class PoseEventProcessor(threading.Thread):
    """Processes pose detection events."""
    def __init__(
        self,
        config: FrigateConfig,
        camera_config: CameraConfig,
        pose_queue: Queue,
        timeline_queue: Queue,
        stop_event: MpEvent,
    ):
        super().__init__(name=f"{camera_config.name}_pose_processor")
        self.config = config
        self.camera_config = camera_config
        self.pose_queue = pose_queue
        self.timeline_queue = timeline_queue
        self.stop_event = stop_event
        
        self.pose_tracker = PoseTracker(camera_config)
        self.action_detector = PoseActionDetector(camera_config)
        self.event_publisher = EventUpdatePublisher()
        
        # Initialize pose detector
        model_config = None
        detector_config = None
        
        if config.pose_detectors:
            detector_config = list(config.pose_detectors.values())[0]
            if detector_config and detector_config.model:
                model_config = detector_config.model
        
        if model_config:
            self.pose_detector = RemotePoseDetector(
                camera_config.name,
                {},  # No labels needed for pose
                pose_queue,
                model_config,
                stop_event,
            )
        else:
            self.pose_detector = None
            logger.warning(f"No pose detector configured for camera {camera_config.name}")

    def run(self) -> None:
        """Main processing loop."""
        while not self.stop_event.is_set():
            try:
                # Get frame data from queue
                frame_data = self.pose_queue.get(timeout=1)
                if frame_data is None:
                    continue
                
                frame, frame_time, zones = frame_data
                
                if self.pose_detector:
                    # Detect poses in frame
                    poses = self.pose_detector.detect_poses(frame, threshold=self.camera_config.pose.keypoint_threshold)
                    
                    # Detect actions for each pose
                    for pose in poses:
                        actions = self.action_detector.detect_actions(pose)
                        pose.actions = actions
                    
                    # Update tracked poses
                    pose_events = self.pose_tracker.update(poses, frame_time, zones)
                    
                    # Process events
                    for event in pose_events:
                        self._process_pose_event(event)
                        
            except Exception as e:
                logger.error(f"Error in pose processor: {e}")
        
        self.event_publisher.stop()
        if self.pose_detector:
            self.pose_detector.cleanup()
        logger.info(f"Exiting pose processor for {self.camera_config.name}")

    def _process_pose_event(self, event: PoseEvent):
        """Process a single pose event."""
        # Determine event state
        if not event.active and event.end_time:
            event_state = EventStateEnum.end
        elif event.start_time == event.frame_time:
            event_state = EventStateEnum.start
        else:
            event_state = EventStateEnum.update
        
        # Send to timeline queue
        self.timeline_queue.put(
            (
                event.camera,
                EventTypeEnum.pose,
                event_state,
                None,
                event.to_dict(),
            )
        )
        
        # Publish event update
        self.event_publisher.publish(
            (
                EventTypeEnum.pose,
                event_state,
                event.camera,
                event.pose_id,
                event.to_dict(),
            )
        )