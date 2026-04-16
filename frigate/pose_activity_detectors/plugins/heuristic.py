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
        fall_window_size: int = 10,
        fall_vertical_drop_ratio: float = 0.25,
        fall_hold_frames: int = 6,
        **kwargs,
    ):
        """
        Initialize the heuristic pose detector.

        Args:
            body_height_threshold: Threshold for body height to detect lying pose (in pixels)
            leg_spread_threshold: Threshold for leg spread to detect walking (in pixels)
            confidence_threshold: Minimum confidence for keypoints to be considered valid
            fall_window_size: Number of recent frames kept for fall detection
            fall_vertical_drop_ratio: Minimum fraction of frame height the hip
                must drop across the window to register a fall
            fall_hold_frames: Number of subsequent frames to hold the 'falling'
                label after a fall is first detected
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.body_height_threshold = body_height_threshold
        self.leg_spread_threshold = leg_spread_threshold
        # Allow overriding via detector config kwargs (e.g. from config.yml)
        self.confidence_threshold = float(
            kwargs.get("confidence_threshold", confidence_threshold)
        )
        self.action_history = deque(maxlen=5)
        # Per-track hip-y history, keyed by (camera, pose_id)
        self._hip_y_history: dict = {}
        # Per-track remaining hold frames after a fall fires
        self._fall_hold: dict = {}
        self.fall_window_size = int(fall_window_size)
        self.fall_vertical_drop_ratio = float(fall_vertical_drop_ratio)
        self.fall_hold_frames = int(fall_hold_frames)
        self.initialized = True

    def reset(self):
        """Reset the detector's internal state."""
        self.action_history.clear()
        self._hip_y_history.clear()
        self._fall_hold.clear()

    def forget(self, pose_id: str, camera=None) -> None:
        """Drop per-track state when a track expires."""
        key = (camera, pose_id)
        self._hip_y_history.pop(key, None)
        self._fall_hold.pop(key, None)

    def _check_fall(
        self,
        key: tuple,
        hip_y: float,
        frame_height: int,
    ) -> bool:
        """Simple vertical-drop fall heuristic.

        Maintains a per-track sliding window of hip midpoint Y coordinates and
        fires when the hip drops by more than `fall_vertical_drop_ratio` of
        frame height across the window. This is a deliberately modest reference
        implementation — a joint-angle + velocity rule is enough to show the
        plugin interface working on synthetic and clean lab footage, but real
        deployments will want a proper classifier plugged in via the same
        `PoseActivityDetector` contract.
        """
        if frame_height is None or frame_height <= 0:
            return False

        history = self._hip_y_history.setdefault(
            key, deque(maxlen=self.fall_window_size)
        )
        history.append(hip_y)

        if len(history) < self.fall_window_size:
            return False

        drop = history[-1] - history[0]
        return drop >= (self.fall_vertical_drop_ratio * frame_height)

    def detect(
        self,
        keypoints: np.ndarray,
        frame_width=None,
        frame_height=None,
        pose_id=None,
        camera=None,
        **kwargs,
    ) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect the pose activity using heuristics.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]
            frame_width/frame_height: Frame dimensions used by the fall heuristic
            pose_id: Stable track id used for per-track state keying
            camera: Camera name used for per-track state keying

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

            track_key = (camera, pose_id)

            # Fall detection: vertical hip drop over a short window.
            # Fires once and then holds the 'falling' label for a few frames
            # so downstream consumers have time to observe the event.
            fall_fired = self._check_fall(
                track_key, float(hip_midpoint[1]), frame_height
            )
            hold_remaining = self._fall_hold.get(track_key, 0)
            if fall_fired:
                hold_remaining = max(hold_remaining, self.fall_hold_frames)
                self._fall_hold[track_key] = hold_remaining

            if hold_remaining > 0:
                self._fall_hold[track_key] = hold_remaining - 1
                return PoseActionTypeEnum.falling, 0.8

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
