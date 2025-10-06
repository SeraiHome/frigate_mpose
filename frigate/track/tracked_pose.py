import logging
from typing import Any, Dict, Optional

import numpy as np

from frigate.events.pose_types import PoseActionTypeEnum

logger = logging.getLogger(__name__)


class TrackedPose:
    def __init__(
        self,
        pose_id: str,
        person_id: int,
        keypoints: np.ndarray,
        confidence: float,
        bbox: Optional[list] = None,
        frame_time: float = 0.0,
    ):
        self.pose_id = pose_id
        self.person_id = person_id
        self.keypoints = keypoints  # Shape: (17, 3) for COCO format
        self.confidence = confidence
        self.bbox = bbox or [0, 0, 0, 0]
        self.frame_time = frame_time
        
        # Tracking state
        self.age = 0
        self.hit_streak = 0
        self.time_since_update = 0
        
        # Pose analysis
        self.action = PoseActionTypeEnum.standing
        self.action_confidence = 0.0
        
        # Event tracking
        self.has_snapshot = False
        self.has_clip = False
        self.false_positive = True
        self.zone_history = []
        self.entered_zones = set()
        self.current_zones = set()
        
        # History for smoothing and analysis
        self.keypoint_history = []
        self.action_history = []
        
        # Store previous state for event comparison
        self.previous = self.to_dict()

    def update(self, keypoints: np.ndarray, confidence: float, bbox: Optional[list] = None):
        """Update pose with new detection."""
        self.keypoints = keypoints
        self.confidence = confidence
        if bbox:
            self.bbox = bbox
        
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Add to history
        self.keypoint_history.append(keypoints.copy())
        if len(self.keypoint_history) > 10:  # Keep last 10 frames
            self.keypoint_history.pop(0)
        
        # Analyze pose action
        self._analyze_pose_action()

    def predict(self):
        """Predict next pose state (for tracking)."""
        self.age += 1
        self.time_since_update += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0

    def _analyze_pose_action(self):
        """Analyze keypoints to determine pose action."""
        if len(self.keypoints) < 17 or self.keypoints.shape[1] < 3:
            return
        
        try:
            # Extract key points for pose analysis
            nose = self.keypoints[0]
            left_shoulder = self.keypoints[5]
            right_shoulder = self.keypoints[6]
            left_hip = self.keypoints[11]
            right_hip = self.keypoints[12]
            left_knee = self.keypoints[13]
            right_knee = self.keypoints[14]
            left_ankle = self.keypoints[15]
            right_ankle = self.keypoints[16]
            
            # Check if key points are visible
            key_points_visible = (
                nose[2] > 0.3 and
                left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3 and
                left_hip[2] > 0.3 and right_hip[2] > 0.3
            )
            
            if not key_points_visible:
                self.action = PoseActionTypeEnum.standing
                self.action_confidence = 0.3
                return
            
            # Calculate body orientation and pose
            shoulder_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2,
                               (left_shoulder[1] + right_shoulder[1]) / 2]
            hip_midpoint = [(left_hip[0] + right_hip[0]) / 2,
                          (left_hip[1] + right_hip[1]) / 2]
            
            # Body height (shoulder to hip distance)
            body_height = abs(shoulder_midpoint[1] - hip_midpoint[1])
            
            # Analyze pose based on body posture
            if body_height < 50:  # Very low body height
                self.action = PoseActionTypeEnum.lying
                self.action_confidence = 0.8
            elif nose[1] > hip_midpoint[1]:  # Head below hips
                self.action = PoseActionTypeEnum.sitting
                self.action_confidence = 0.7
            else:
                # Check leg positions for standing/walking/running
                if (left_knee[2] > 0.3 and right_knee[2] > 0.3 and
                    left_ankle[2] > 0.3 and right_ankle[2] > 0.3):
                    
                    # Calculate leg spread
                    leg_spread = abs(left_ankle[0] - right_ankle[0])
                    
                    if leg_spread > 100:  # Wide stance
                        self.action = PoseActionTypeEnum.walking
                        self.action_confidence = 0.6
                    else:
                        self.action = PoseActionTypeEnum.standing
                        self.action_confidence = 0.8
                else:
                    self.action = PoseActionTypeEnum.standing
                    self.action_confidence = 0.5
            
            # Add to action history for smoothing
            self.action_history.append(self.action)
            if len(self.action_history) > 5:
                self.action_history.pop(0)
            
            # Smooth action based on history
            if len(self.action_history) >= 3:
                most_common_action = max(set(self.action_history), 
                                       key=self.action_history.count)
                if self.action_history.count(most_common_action) >= 3:
                    self.action = most_common_action
                    self.action_confidence = min(self.action_confidence + 0.2, 1.0)
        
        except Exception as e:
            logger.warning(f"Error analyzing pose action: {e}")
            self.action = PoseActionTypeEnum.standing
            self.action_confidence = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose to dictionary for event processing."""
        return {
            "id": self.pose_id,
            "person_id": self.person_id,
            "keypoints": self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "frame_time": self.frame_time,
            "action": self.action,
            "action_confidence": self.action_confidence,
            "age": self.age,
            "hit_streak": self.hit_streak,
            "time_since_update": self.time_since_update,
            "has_snapshot": self.has_snapshot,
            "has_clip": self.has_clip,
            "false_positive": self.false_positive,
            "entered_zones": list(self.entered_zones),
            "current_zones": list(self.current_zones),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPose":
        """Create TrackedPose from dictionary."""
        keypoints = np.array(data["keypoints"]) if data.get("keypoints") else np.zeros((17, 3))
        
        pose = cls(
            pose_id=data["id"],
            person_id=data.get("person_id", 0),
            keypoints=keypoints,
            confidence=data.get("confidence", 0.0),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            frame_time=data.get("frame_time", 0.0),
        )
        
        # Restore state
        pose.action = PoseActionTypeEnum(data.get("action", PoseActionTypeEnum.standing))
        pose.action_confidence = data.get("action_confidence", 0.0)
        pose.age = data.get("age", 0)
        pose.hit_streak = data.get("hit_streak", 0)
        pose.time_since_update = data.get("time_since_update", 0)
        pose.has_snapshot = data.get("has_snapshot", False)
        pose.has_clip = data.get("has_clip", False)
        pose.false_positive = data.get("false_positive", True)
        pose.entered_zones = set(data.get("entered_zones", []))
        pose.current_zones = set(data.get("current_zones", []))
        
        return pose