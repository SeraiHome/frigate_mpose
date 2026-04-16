"""
Base Pose Activity Detector

This module defines the abstract base class for pose activity detectors.
All specific detector implementations should inherit from this base class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from frigate.events.pose_types import PoseActionTypeEnum

logger = logging.getLogger(__name__)


class PoseActivityDetector(ABC):
    """
    Base class for pose activity detectors.

    This abstract class defines the interface that all pose activity detectors must implement.
    Specific implementations can use different models and techniques to detect activities.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the pose activity detector.

        Args:
            **kwargs: Additional keyword arguments for the detector
        """
        self.initialized = False

    @abstractmethod
    def reset(self):
        """Reset the detector's internal state."""
        pass

    @abstractmethod
    def detect(
        self,
        keypoints: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        pose_id: Optional[str] = None,
        camera: Optional[str] = None,
    ) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect the pose activity from keypoints.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is
                [x, y, confidence]
            frame_width: Width of the frame in pixels (optional).
            frame_height: Height of the frame in pixels (optional).
            pose_id: Stable id of the track feeding keypoints. Detectors that
                maintain per-track state (sliding windows, smoothing buffers)
                must key on this so keypoints from distinct subjects do not
                interleave. Stateless detectors may ignore it.
            camera: Camera name that owns the track. Used together with
                pose_id to key per-track state when a single detector
                instance is shared across multiple cameras (e.g. via a
                shared pool worker). Optional for legacy per-camera
                detector instances — stateless detectors may ignore it.

        Returns:
            Tuple of (action_type, confidence)
        """
        pass

    def forget(self, pose_id: str, camera: Optional[str] = None) -> None:
        """Drop any per-track state held for `(camera, pose_id)`.

        Called by the integration layer when a TrackedPose is deleted.
        `camera` is optional for backwards compatibility with legacy
        callers. Default no-op covers stateless detectors (e.g. the
        heuristic).
        """
        return
