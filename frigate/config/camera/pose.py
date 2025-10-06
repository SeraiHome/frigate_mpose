from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from frigate.config.base import FrigateBaseModel


class PoseFilterConfig(BaseModel):
    """Filter configuration for specific pose types."""
    min_keypoints: int = Field(
        default=5,
        ge=1,
        le=17,
        description="Minimum number of visible keypoints to consider a valid pose"
    )
    min_confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for pose detection"
    )
    min_area: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum area of bounding box for pose"
    )
    max_area: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum area of bounding box for pose"
    )
    required_zones: List[str] = Field(
        default_factory=list,
        description="List of required zones for pose to be considered"
    )


class PoseActionConfig(BaseModel):
    """Configuration for pose action detection."""
    enabled: bool = Field(default=False, description="Enable pose action detection")
    actions: List[str] = Field(
        default_factory=lambda: ["falling", "fighting", "running"],
        description="List of actions to detect"
    )
    action_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for action detection"
    )


class PoseConfig(FrigateBaseModel):
    """Pose detection configuration for a camera."""
    enabled: bool = Field(default=False, description="Enable pose detection")
    
    width: Optional[int] = Field(
        default=None,
        ge=32,
        le=3840,
        description="Width of pose detection region"
    )
    height: Optional[int] = Field(
        default=None,
        ge=32,
        le=2160,
        description="Height of pose detection region"
    )
    
    fps: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Frames per second for pose detection"
    )
    
    max_disappeared: int = Field(
        default=25,
        ge=1,
        description="Maximum frames a pose can disappear before being removed"
    )
    
    filters: Dict[str, PoseFilterConfig] = Field(
        default_factory=lambda: {"person": PoseFilterConfig()},
        description="Filters for different pose types"
    )
    
    track: List[str] = Field(
        default_factory=lambda: ["person"],
        description="List of pose types to track"
    )
    
    actions: PoseActionConfig = Field(
        default_factory=PoseActionConfig,
        description="Pose action detection configuration"
    )
    
    required_zones: List[str] = Field(
        default_factory=list,
        description="List of zones where pose detection is required"
    )
    
    objects_as_poses: bool = Field(
        default=False,
        description="Convert detected person objects to poses for tracking"
    )
    
    keypoint_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for individual keypoints"
    )
    
    @field_validator("track")
    def validate_track(cls, value: List[str]) -> List[str]:
        """Validate tracked pose types."""
        valid_types = ["person", "animal"]  # Can be extended
        for pose_type in value:
            if pose_type not in valid_types:
                raise ValueError(f"Invalid pose type: {pose_type}. Valid types: {valid_types}")
        return value