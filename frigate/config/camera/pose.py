from typing import Dict, List, Optional, Set

from pydantic import Field

from frigate.config.base import FrigateBaseModel
from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors.detector_config import (
    BasePoseActivityDetectorConfig,
)


class ActivityDetectorConfig(BasePoseActivityDetectorConfig):
    """Configuration for pose activity detectors."""

    pass


class PoseFilterConfig(FrigateBaseModel):
    min_area: int = Field(default=0, title="Minimum pose area in pixels.")
    max_area: int = Field(default=24000000, title="Maximum pose area in pixels.")
    min_ratio: float = Field(
        default=0, title="Minimum width/height ratio for pose bounding box."
    )
    max_ratio: float = Field(
        default=24000000, title="Maximum width/height ratio for pose bounding box."
    )
    threshold: float = Field(
        default=0.4, title="Minimum confidence threshold for pose detection."
    )
    min_score: float = Field(
        default=0.4, title="Minimum score for pose to be considered valid."
    )


class PoseConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enable pose detection for camera.")
    confidence_threshold: float = Field(
        default=0.4, title="Minimum confidence threshold for pose detection."
    )
    keypoint_threshold: float = Field(
        default=0.3, title="Minimum confidence threshold for individual keypoints."
    )
    activity_detector: Optional[ActivityDetectorConfig] = Field(
        default=None,
        title="Activity detector configuration",
        description="Configuration for the pose activity detector.",
    )
    actions: Set[PoseActionTypeEnum] = Field(
        default_factory=lambda: {
            PoseActionTypeEnum.standing,
            PoseActionTypeEnum.walking,
            PoseActionTypeEnum.sitting,
            PoseActionTypeEnum.lying,
        },
        title="Pose actions to track.",
    )
    snapshot_actions: Set[PoseActionTypeEnum] = Field(
        default_factory=lambda: {
            PoseActionTypeEnum.waving,
            PoseActionTypeEnum.pointing,
        },
        title="Pose actions that trigger snapshots.",
    )
    record_actions: Set[PoseActionTypeEnum] = Field(
        default_factory=lambda: {
            PoseActionTypeEnum.walking,
            PoseActionTypeEnum.running,
            PoseActionTypeEnum.jumping,
        },
        title="Pose actions that trigger recording retention.",
    )
    filters: Dict[str, PoseFilterConfig] = Field(
        default_factory=dict, title="Filters for specific pose actions."
    )
    mask: str = Field(default="", title="Pose detection mask.")
    required_zones: List[str] = Field(
        default_factory=list,
        title="List of required zones for pose detection to trigger events.",
    )
    fps: int = Field(
        default=5, title="FPS for pose detection (should be <= camera detect fps)."
    )
    publish_to_detected_objects: bool = Field(
        default=True,
        title="Publish pose detections to the main detected_objects queue.",
        description=(
            "When true, pose detections will be added to the standard "
            "detected_objects_queue in addition to being sent to the pose queues. "
            "Set to false to avoid timing interference and use the pose-specific "
            "processing pipeline instead."
        ),
    )

    def __init__(self, **config):
        super().__init__(**config)

        # Ensure pose detection fps doesn't exceed reasonable limits
        if self.fps > 30:
            self.fps = 30
