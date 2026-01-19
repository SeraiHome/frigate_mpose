import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors.base import PoseActivityDetector

logger = logging.getLogger(__name__)


class TrackedPose:
    """
    Tracked pose object for storing and analyzing detected poses.

    Each pose is associated with a camera and can be analyzed using
    camera-specific activity detectors.
    """

    def __init__(
        self,
        pose_id: str,
        person_id: int,
        keypoints: np.ndarray,
        confidence: float,
        bbox: Optional[list] = None,
        frame_time: float = 0.0,
        camera_name: str = "",
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ):
        self.pose_id = pose_id
        self.person_id = person_id
        self.keypoints = keypoints  # Shape: (17, 3) for COCO format
        self.confidence = confidence
        self.bbox = bbox or [0, 0, 0, 0]
        self.frame_time = frame_time
        self.camera_name = camera_name
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Activity detector specific to this pose's camera
        self._active_detector: Optional[PoseActivityDetector] = None
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

        # Zone presence tracking (for inertia and loitering time)
        self.zone_presence: dict[str, int] = {}  # Frames present in zone
        self.zone_loitering: dict[str, int] = {}  # Frames loitering in zone

        # History for smoothing and analysis
        self.keypoint_history = deque(maxlen=10)
        self.action_history = deque(maxlen=5)

    def __getstate__(self):
        """Exclude unpicklable objects (like TFLite interpreters) when pickling."""
        state = self.__dict__.copy()
        # Remove the activity detector - it contains TFLite interpreter which can't be pickled
        state["_active_detector"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # _active_detector will be None after unpickling
        # It needs to be re-assigned by the receiving process if needed

    @property
    def active_detector(self) -> Optional[PoseActivityDetector]:
        """Get the activity detector for this pose."""
        return self._active_detector

    @active_detector.setter
    def active_detector(self, detector: Optional[PoseActivityDetector]):
        """Set the activity detector for this pose."""
        self._active_detector = detector
        # Store previous state for event comparison
        self.previous = self.to_dict()

    def update(
        self, keypoints: np.ndarray, confidence: float, bbox: Optional[list] = None
    ):
        """Update pose with new detection."""
        self.keypoints = keypoints
        self.confidence = confidence
        if bbox:
            self.bbox = bbox

        self.hit_streak += 1
        self.time_since_update = 0

        # Add to history
        self.keypoint_history.append(
            keypoints.copy()
        )  # Will automatically maintain max length

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
        # Validate keypoints data
        if len(self.keypoints) < 17 or self.keypoints.shape[1] < 3:
            logger.debug(f"Insufficient keypoints data: shape={self.keypoints.shape}")
            return

        # Check for valid keypoint values
        non_zero_coords = np.count_nonzero(self.keypoints[:, :2])
        if non_zero_coords == 0:
            logger.debug("All keypoint coordinates are zero - possible detection issue")
            return

        try:
            # Log keypoints data for debugging
            logger.debug(
                f"Processing keypoints for pose {self.pose_id}: "
                f"shape={self.keypoints.shape}, "
                f"non-zero coords={non_zero_coords}, "
                f"max_x={np.max(self.keypoints[:, 0]):.1f}, "
                f"max_y={np.max(self.keypoints[:, 1]):.1f}, "
                f"avg_conf={np.mean(self.keypoints[:, 2]):.2f}"
            )

            # Log frame dimensions if available
            if self.frame_width and self.frame_height:
                logger.debug(
                    f"Frame dimensions for pose {self.pose_id}: "
                    f"{self.frame_width}x{self.frame_height}"
                )

            # Try using the active detector if available
            if self.active_detector and self.active_detector.initialized:
                try:
                    # Log the detector we're using
                    logger.debug(
                        f"Using {self.active_detector.__class__.__name__} for pose {self.pose_id}"
                    )

                    # Verify keypoints before passing to detector
                    if np.isnan(self.keypoints).any():
                        logger.warning(
                            "NaN values found in keypoints - replacing with zeros"
                        )
                        self.keypoints = np.nan_to_num(self.keypoints)

                    # Pass frame dimensions if available
                    action, confidence = self.active_detector.detect(
                        self.keypoints,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                    )
                    logger.debug(
                        f"Detector result: action={action}, confidence={confidence:.2f}"
                    )

                    # Only update if confidence is high enough
                    if confidence > 0.4:
                        self.action = action
                        self.action_confidence = confidence

                        # Add to action history for smoothing
                        self.action_history.append(self.action)
                        if len(self.action_history) > 5:
                            self.action_history.pop(0)

                        # Smooth action based on history
                        if len(self.action_history) >= 3:
                            most_common_action = max(
                                set(self.action_history), key=self.action_history.count
                            )
                            if self.action_history.count(most_common_action) >= 1:
                                self.action = most_common_action
                                self.action_confidence = min(
                                    self.action_confidence + 0.1, 1.0
                                )
                                logger.debug(
                                    f"Smoothed action: {self.action}, confidence: {self.action_confidence:.2f}"
                                )

                        return
                except Exception as e:
                    logger.warning(
                        f"Error using activity detector: {e}, falling back to default standing pose"
                    )

            # If no detector is available or it failed, fall back to default values
            self.action = PoseActionTypeEnum.standing
            self.action_confidence = 0.3
            logger.debug("Using default standing pose")

        except Exception as e:
            logger.warning(f"Error analyzing pose action: {e}")
            self.action = PoseActionTypeEnum.standing
            self.action_confidence = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose to dictionary for event processing."""
        result = {
            "id": self.pose_id,
            "person_id": self.person_id,
            "keypoints": self.keypoints.tolist()
            if isinstance(self.keypoints, np.ndarray)
            else self.keypoints,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "frame_time": self.frame_time,
            # Ensure action is serialized as a plain string (e.g., 'standing')
            "action": self.action.value
            if hasattr(self.action, "value")
            else str(self.action),
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

        # Add frame dimensions if available
        if self.frame_width is not None:
            result["frame_width"] = self.frame_width
        if self.frame_height is not None:
            result["frame_height"] = self.frame_height

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPose":
        """Create TrackedPose from dictionary."""
        keypoints = (
            np.array(data["keypoints"]) if data.get("keypoints") else np.zeros((17, 3))
        )

        pose = cls(
            pose_id=data["id"],
            person_id=data.get("person_id", 0),
            keypoints=keypoints,
            confidence=data.get("confidence", 0.0),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            frame_time=data.get("frame_time", 0.0),
            frame_width=data.get("frame_width"),
            frame_height=data.get("frame_height"),
        )

        # Restore state
        pose.action = PoseActionTypeEnum(
            data.get("action", PoseActionTypeEnum.standing)
        )
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
