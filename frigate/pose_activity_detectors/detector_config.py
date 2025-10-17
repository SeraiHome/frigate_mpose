"""
Pose Activity Detector Configuration

This module defines the configuration classes for pose activity detectors.
"""

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PoseActivityDetectorTypeEnum(str, Enum):
    """Enum for the available pose activity detector types."""
    stgcn_fall = "stgcn_fall"
    heuristic = "heuristic"


class BasePoseActivityDetectorConfig(BaseModel):
    """
    Base configuration class for pose activity detectors.
    
    All activity detector configurations should extend this class.
    """
    type: str = Field(default="heuristic", title="Activity Detector Type")
    confidence_threshold: float = Field(
        default=0.5,
        title="Confidence threshold",
        description="Minimum confidence threshold for activity detection.",
    )

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, protected_namespaces=()
    )


class STGCNFallDetectorConfig(BasePoseActivityDetectorConfig):
    """
    Configuration for STGCN-based fall detectors.
    """
    type: str = Field(
        default="stgcn_fall",
        title="Activity Detector Type",
        description="Must be 'stgcn_fall' for STGCN-based fall detectors",
    )
    model_path: str = Field(
        default="",
        title="Model path",
        description="Path to the TFLite model for fall detection",
    )
    num_threads: int = Field(
        default=3,
        title="Number of threads",
        description="Number of threads to use for inference",
    )


class HeuristicDetectorConfig(BasePoseActivityDetectorConfig):
    """
    Configuration for heuristic-based pose activity detectors.
    """
    type: str = Field(
        default="heuristic",
        title="Activity Detector Type",
        description="Must be 'heuristic' for heuristic-based pose activity detectors",
    )
    # Heuristic detectors don't need a model path, but may have other parameters
    # Add any heuristic-specific parameters here
    body_height_threshold: int = Field(
        default=50,
        title="Body height threshold",
        description="Threshold for body height to detect lying pose (in pixels)",
    )
    leg_spread_threshold: int = Field(
        default=100,
        title="Leg spread threshold",
        description="Threshold for leg spread to detect walking (in pixels)",
    )


def create_detector_config(config_dict: dict) -> Optional[BasePoseActivityDetectorConfig]:
    """
    Create a detector config instance from a dictionary.

    Args:
        config_dict: Dictionary containing detector configuration

    Returns:
        A detector config instance if creation was successful, None otherwise
    """
    if not config_dict or not isinstance(config_dict, dict):
        return None

    detector_type = config_dict.get("type", "heuristic")

    try:
        if detector_type == PoseActivityDetectorTypeEnum.stgcn_fall:
            return STGCNFallDetectorConfig(**config_dict)
        elif detector_type == PoseActivityDetectorTypeEnum.heuristic:
            return HeuristicDetectorConfig(**config_dict)
        else:
            logger.warning(f"Unknown activity detector type: {detector_type}")
            return None
    except Exception as e:
        logger.error(f"Error creating detector config: {e}")
        return None