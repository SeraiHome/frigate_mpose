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
    activity_detector_pool: Optional[str] = Field(
        default=None,
        title="Activity detector pool name",
        description=(
            "Name of a shared pose activity detector pool defined at the "
            "top-level `pose_activity_detectors:` config section. When set, "
            "this camera routes its classifier calls into the shared pool "
            "worker process instead of instantiating its own local detector "
            "— enabling one interpreter (and one accelerator device) to "
            "serve multiple cameras. Falls back to the inline "
            "`activity_detector` config if left unset."
        ),
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
    skip_frames: int = Field(
        default=0,
        title="Number of frames to skip between pose detections.",
        description=(
            "Set to 0 to process every frame (respecting fps limit). "
            "Set to N to skip N frames between detections for better performance. "
            "Example: skip_frames=2 processes every 3rd frame."
        ),
    )
    use_motion_roi: bool = Field(
        default=False,
        title="Use motion region of interest for pose detection.",
        description=(
            "When enabled, pose detection will only run on the cropped region "
            "containing detected motion, significantly reducing processing time. "
            "Disable if your model doesn't support variable input sizes or if "
            "poses are being missed near frame edges."
        ),
    )
    detect_persons: bool = Field(
        default=True,
        title="Use pose detector as a person presence detector.",
        description=(
            "When true, all detected poses are injected into the object tracker "
            "as 'person' detections, providing presence awareness even without "
            "object detection enabled.  Only poses whose action matches the "
            "configured 'actions' list will create review alerts; other poses "
            "are tracked silently (visible in timeline/debug but no alerts)."
        ),
    )
    event_cooldown_seconds: int = Field(
        default=60,
        title="Cooldown before re-triggering the same (track, action) pose event.",
        description=(
            "After a pose-driven Event ends, suppress creation of a new Event "
            "for the same tracked object + same action within this window. "
            "Matches within the cooldown extend the existing Event's "
            "end_time instead of fragmenting it. Different actions on the same "
            "track (e.g. standing after falling) are NOT affected."
        ),
    )
    track_max_disappeared_seconds: float = Field(
        default=3.0,
        title="Wall-clock seconds to retain a pose track without a detection update.",
        description=(
            "After this many seconds without a matching detection, the tracked "
            "pose is deleted and the next detection for that subject receives "
            "a new pose_track_id. Larger values hold track ids across pose "
            "detector gaps (occlusion, awkward angles during a fall) so "
            "event_cooldown_seconds can dedup them. Smaller values release "
            "ids faster when subjects leave the frame. Default 3.0s is tuned "
            "for fall-detection workloads at 2-10 fps. Prior behavior was a "
            "hardcoded 10-frame threshold, which became 2s at 5fps and "
            "0.33s at 30fps -- too aggressive for fall scenarios."
        ),
    )
    track_match_iou_threshold: float = Field(
        default=0.15,
        title="Minimum bbox IoU for a detection to reuse an existing pose track id.",
        description=(
            "When both the incoming detection and an existing tracked pose "
            "have bbox data, matching uses Intersection-over-Union of the "
            "boxes. IoU >= this threshold is accepted as the same subject. "
            "Higher values are stricter (fewer matches, more new ids); "
            "lower values are more permissive. Default 0.15 is tuned for "
            "low-fps (2-10 fps) fall scenarios where the pose detector "
            "outputs significantly different bbox shapes between "
            "consecutive frames (e.g. a 52x156 standing box becoming "
            "71x334 one frame later as the pose detector re-estimates "
            "the subject's extent). Even below this threshold, the "
            "centroid fallback at track_match_centroid_threshold acts "
            "as a second-chance rescue. IoU is rotation-invariant and "
            "naturally discriminates multiple subjects whose boxes "
            "don't overlap."
        ),
    )
    track_match_centroid_threshold: float = Field(
        default=0.15,
        title="Fallback centroid-distance threshold (normalized by frame diagonal) when IoU is unavailable.",
        description=(
            "When IoU matching cannot run because either the detection or "
            "the tracked pose lacks bbox data, the tracker falls back to "
            "centroid distance normalized by the frame diagonal. A match "
            "is accepted when dist/diagonal <= this value. Default 0.15 "
            "(15% of diagonal); the prior hardcoded value of 0.10 was "
            "too tight for fall transitions where the keypoint-mean "
            "centroid jumps as the subject pivots horizontally."
        ),
    )
    publish_keypoints: bool = Field(
        default=False,
        title="Publish pose keypoints to MQTT.",
        description=(
            "When enabled, publishes COCO 17-keypoint coordinates to "
            "frigate/{camera}/pose_keypoints via MQTT after each detection. "
            "Consumers can render skeletons, feed a second-stage classifier, "
            "or drive downstream automations from the keypoint stream."
        ),
    )
