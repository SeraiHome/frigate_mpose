"""
Base Pose Activity Detector

This module defines the abstract base class for pose activity detectors.
All specific detector implementations should inherit from this base class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

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
    def detect(self, keypoints: np.ndarray) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect the pose activity from keypoints.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]

        Returns:
            Tuple of (action_type, confidence)
        """
        pass
