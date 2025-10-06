import logging
from typing import Optional

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from .pose_detector_types import pose_api_types

logger = logging.getLogger(__name__)


def create_pose_detector(detector_config: BaseDetectorConfig) -> Optional[DetectionApi]:
    """Create a pose detector instance based on the configuration."""
    if detector_config is None:
        raise ValueError("Detector config must be provided")

    detector_type = detector_config.type
    
    if detector_type not in pose_api_types:
        raise ValueError(f"Unknown pose detector type: {detector_type}")

    api_class = pose_api_types[detector_type]
    
    logger.info(f"Creating pose detector of type: {detector_type}")
    
    try:
        return api_class(detector_config)
    except Exception as e:
        logger.error(f"Failed to create pose detector: {e}")
        raise