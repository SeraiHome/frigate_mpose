import logging

from .detector_config import InputTensorEnum, PoseModelConfig, PixelFormatEnum  # noqa: F401
from .detector_types import PoseDetectorConfig, PoseDetectorTypeEnum, pose_api_types  # noqa: F401

logger = logging.getLogger(__name__)


def create_pose_detector(detector_config):
    if detector_config.type == PoseDetectorTypeEnum.cpu:
        logger.warning(
            "CPU pose detectors are not recommended and should only be used for testing or for trial purposes."
        )

    api = pose_api_types.get(detector_config.type)
    if not api:
        raise ValueError(detector_config.type)
    return api(detector_config)