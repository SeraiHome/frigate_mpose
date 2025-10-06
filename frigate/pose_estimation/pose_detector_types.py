import importlib
import logging
import pkgutil
from enum import Enum
from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from . import plugins
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)


# Import all pose detector plugins
_included_modules = pkgutil.iter_modules(plugins.__path__, plugins.__name__ + ".")

plugin_modules = []

for _, name, _ in _included_modules:
    try:
        plugin_modules.append(importlib.import_module(name))
    except ImportError as e:
        logger.error(f"Error importing pose detector runtime: {e}")


# Get all pose detector API types
pose_api_types = {det.type_key: det for det in DetectionApi.__subclasses__() 
                  if hasattr(det, 'type_key') and ('pose' in det.type_key or 'mediapipe' in det.type_key)}


class StrEnum(str, Enum):
    pass


PoseDetectorTypeEnum = StrEnum("PoseDetectorTypeEnum", {k: k for k in pose_api_types})


class PoseDetectorConfig(BaseDetectorConfig):
    """Configuration for pose detectors."""
    keypoint_thresh: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for keypoints"
    )
    pose_thresh: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for poses"
    )
    max_poses: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of poses to detect"
    )


PoseDetectorConfigUnion = Annotated[
    Union[tuple(PoseDetectorConfig.__subclasses__() if PoseDetectorConfig.__subclasses__() else [PoseDetectorConfig])],
    Field(discriminator="type"),
]