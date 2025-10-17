"""
Heuristic Pose Detector Plugin

This module provides a rule-based detector that uses heuristics
to analyze pose keypoints and determine various activities.
"""

import logging
from collections import deque
from typing import Tuple

import numpy as np

from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors import register_detector
from frigate.pose_activity_detectors.base import PoseActivityDetector

logger = logging.getLogger(__name__)


class HeuristicPoseDetector(PoseActivityDetector):
    """
    Heuristic-based pose activity detector.

    This detector uses simple heuristics based on keypoint positions
    to determine the pose activity.
    """

    def __init__(
        self,
        body_height_threshold: int = 50,
        leg_spread_threshold: int = 100,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize the heuristic pose detector.

        Args:
            body_height_threshold: Threshold for body height to detect lying pose (in pixels)
            leg_spread_threshold: Threshold for leg spread to detect walking (in pixels)
            confidence_threshold: Minimum confidence for keypoints to be considered valid
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.body_height_threshold = body_height_threshold
        self.leg_spread_threshold = leg_spread_threshold
        self.confidence_threshold = confidence_threshold
        self.action_history = deque(maxlen=5)
        self.initialized = True

    def reset(self):
        """Reset the detector's internal state."""
        self.action_history.clear()

    def detect(self, keypoints: np.ndarray) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect the pose activity using heuristics.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]

        Returns:
            Tuple of (action_type, confidence)
        """
        if len(keypoints) < 17 or keypoints.shape[1] < 3:
            return PoseActionTypeEnum.standing, 0.3

        try:
            # Extract key points for pose analysis
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            # Check if key points are visible
            key_points_visible = (
                nose[2] > self.confidence_threshold
                and left_shoulder[2] > self.confidence_threshold
                and right_shoulder[2] > self.confidence_threshold
                and left_hip[2] > self.confidence_threshold
                and right_hip[2] > self.confidence_threshold
            )

            if not key_points_visible:
                return PoseActionTypeEnum.standing, 0.3

            # Calculate body orientation and pose
            shoulder_midpoint = [
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            ]
            hip_midpoint = [
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
            ]

            # Body height (shoulder to hip distance)
            body_height = abs(shoulder_midpoint[1] - hip_midpoint[1])

            # Analyze pose based on body posture
            if body_height < self.body_height_threshold:  # Very low body height
                action = PoseActionTypeEnum.lying
                confidence = 0.8
            elif nose[1] > hip_midpoint[1]:  # Head below hips
                action = PoseActionTypeEnum.sitting
                confidence = 0.7
            else:
                # Check leg positions for standing/walking/running
                if (
                    left_knee[2] > self.confidence_threshold
                    and right_knee[2] > self.confidence_threshold
                    and left_ankle[2] > self.confidence_threshold
                    and right_ankle[2] > self.confidence_threshold
                ):
                    # Calculate leg spread
                    leg_spread = abs(left_ankle[0] - right_ankle[0])

                    if leg_spread > self.leg_spread_threshold:  # Wide stance
                        action = PoseActionTypeEnum.walking
                        confidence = 0.6
                    else:
                        action = PoseActionTypeEnum.standing
                        confidence = 0.8
                else:
                    action = PoseActionTypeEnum.standing
                    confidence = 0.5

            # Add to action history for smoothing
            self.action_history.append(action)

            # Smooth action based on history
            if len(self.action_history) >= 3:
                most_common_action = max(
                    set(self.action_history), key=self.action_history.count
                )
                if self.action_history.count(most_common_action) >= 3:
                    action = most_common_action
                    confidence = min(confidence + 0.2, 1.0)

            return action, confidence

        except Exception as e:
            logger.warning(f"Error analyzing pose action: {e}")
            return PoseActionTypeEnum.standing, 0.3


# Register this detector with the registry
register_detector("heuristic", HeuristicPoseDetector)