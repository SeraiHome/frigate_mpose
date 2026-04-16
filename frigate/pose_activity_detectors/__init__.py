"""
Pose Activity Detectors

This package contains the detectors used to analyze pose data and
determine activities such as falling, standing, etc.
"""

import importlib
import logging
import os
from typing import Dict, Optional, Type

from frigate.pose_activity_detectors.base import PoseActivityDetector
from frigate.pose_activity_detectors.detector_config import (
    BasePoseActivityDetectorConfig,
)

logger = logging.getLogger(__name__)

# Registry to keep track of available detector types
DETECTOR_REGISTRY: Dict[str, Type[PoseActivityDetector]] = {}


def register_detector(name: str, detector_cls: Type[PoseActivityDetector]) -> None:
    """
    Register a detector type with the system.

    Args:
        name: The name to register the detector under (e.g., "heuristic")
        detector_cls: The detector class to register
    """
    DETECTOR_REGISTRY[name] = detector_cls
    logger.debug(f"Registered pose activity detector: {name}")


def get_detector_class(detector_type: str) -> Optional[Type[PoseActivityDetector]]:
    """
    Get a detector class by type name.

    Args:
        detector_type: The name of the detector type to retrieve

    Returns:
        The detector class if found, None otherwise
    """
    return DETECTOR_REGISTRY.get(detector_type)


def load_detector_plugins() -> None:
    """
    Load all detector plugins found in the plugins directory.

    This function dynamically imports all Python modules in the plugins directory
    to register their detector types.
    """
    plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")
    if not os.path.exists(plugins_dir):
        logger.warning(f"Plugins directory not found: {plugins_dir}")
        return

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove the .py extension
            try:
                importlib.import_module(
                    f"frigate.pose_activity_detectors.plugins.{module_name}"
                )
                logger.debug(f"Loaded pose activity detector plugin: {module_name}")
            except ImportError as e:
                logger.error(
                    f"Failed to load pose activity detector plugin {module_name}: {e}"
                )


def create_activity_detector(
    config: BasePoseActivityDetectorConfig,
) -> Optional[PoseActivityDetector]:
    """
    Create an activity detector instance based on the configuration.

    Args:
        config: The detector configuration

    Returns:
        A detector instance if creation was successful, None otherwise
    """
    if not config:
        logger.warning("No activity detector configuration provided")
        return None

    detector_type = config.type
    detector_class = get_detector_class(detector_type)

    if not detector_class:
        logger.error(f"Unknown activity detector type: {detector_type}")
        return None

    try:
        logger.info(f"Creating pose activity detector: {detector_type}")
        return detector_class(**config.model_dump(exclude={"type"}))
    except Exception as e:
        logger.error(f"Failed to create activity detector {detector_type}: {e}")
        return None


# Load all detector plugins
load_detector_plugins()
