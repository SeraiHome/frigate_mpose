import logging
from collections import deque
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors import create_activity_detector
from frigate.pose_activity_detectors.base import PoseActivityDetector
from frigate.pose_activity_detectors.detector_config import create_detector_config

logger = logging.getLogger(__name__)


class TrackedPose:
    # Active detector instance
    active_detector: Optional[PoseActivityDetector] = None

    @classmethod
    def init_activity_detector(
        cls, detector_type: str, model_path: Optional[str] = None, **kwargs
    ):
        """
        Initialize the activity detector with the specified type and model.

        Args:
            detector_type: Type of detector to use (must be in activity_detectors registry)
            model_path: Path to the model file (if required by the detector)
            **kwargs: Additional keyword arguments for the detector

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # Create configuration for the detector
        config_dict = {"type": detector_type, "model_path": model_path, **kwargs}
        detector_config = create_detector_config(config_dict)
        
        if detector_config is None:
            logger.error(f"Failed to create configuration for detector type: {detector_type}")
            return False

        # Create detector instance
        try:
            cls.active_detector = create_activity_detector(detector_config)
            if cls.active_detector:
                logger.info(f"Activity detector initialized: {detector_type}")
                return cls.active_detector.initialized
            else:
                logger.error(f"Failed to create activity detector: {detector_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize activity detector: {e}")
            cls.active_detector = None
            return False

    @classmethod
    def init_from_config(cls, config):
        """
        Initialize the activity detector from configuration.

        Args:
            config: PoseConfig instance containing activity detector configuration

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not config or not config.activity_detector:
            # No activity detector configured, use default heuristic detector
            logger.info(
                "No activity detector configured, using default heuristic detector"
            )
            return cls.init_activity_detector("heuristic")

        detector_config = config.activity_detector
        detector_type = detector_config.type

        # Extract parameters from config
        kwargs = {
            "model_path": detector_config.model_path,
            "confidence_threshold": detector_config.confidence_threshold,
            "num_threads": detector_config.num_threads,
        }

        logger.info(f"Initializing activity detector from config: {detector_type}")
        return cls.init_activity_detector(detector_type, **kwargs)

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
            # Try using the active detector if available
            if self.active_detector and self.active_detector.initialized:
                try:
                    action, confidence = self.active_detector.detect(self.keypoints)

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
                            if self.action_history.count(most_common_action) >= 3:
                                self.action = most_common_action
                                self.action_confidence = min(
                                    self.action_confidence + 0.1, 1.0
                                )

                        return
                except Exception as e:
                    logger.warning(
                        f"Error using activity detector: {e}, falling back to default standing pose"
                    )

            # If no detector is available or it failed, fall back to default values
            self.action = PoseActionTypeEnum.standing
            self.action_confidence = 0.3

        except Exception as e:
            logger.warning(f"Error analyzing pose action: {e}")
            self.action = PoseActionTypeEnum.standing
            self.action_confidence = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose to dictionary for event processing."""
        return {
            "id": self.pose_id,
            "person_id": self.person_id,
            "keypoints": self.keypoints.tolist()
            if isinstance(self.keypoints, np.ndarray)
            else self.keypoints,
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