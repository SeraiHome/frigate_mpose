import importlib
import logging
import pkgutil
from enum import Enum
from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from . import plugins
from .detection_api import PoseDetectionApi
from .detector_config import BasePoseDetectorConfig

logger = logging.getLogger(__name__)


_included_modules = pkgutil.iter_modules(plugins.__path__, plugins.__name__ + ".")

plugin_modules = []

for _, name, _ in _included_modules:
    try:
        plugin_modules.append(importlib.import_module(name))
    except ImportError as e:
        logger.error(f"Error importing pose detector runtime: {e}")


pose_api_types = {det.type_key: det for det in PoseDetectionApi.__subclasses__()}


class StrEnum(str, Enum):
    pass


PoseDetectorTypeEnum = StrEnum("PoseDetectorTypeEnum", {k: k for k in pose_api_types})

PoseDetectorConfig = Annotated[
    Union[tuple(BasePoseDetectorConfig.__subclasses__())],
    Field(discriminator="type"),
]
