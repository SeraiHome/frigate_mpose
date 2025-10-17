from enum import Enum


class PoseEventTypeEnum(str, Enum):
    pose_detected = "pose_detected"
    pose_lost = "pose_lost"
    pose_updated = "pose_updated"


class PoseEventStateEnum(str, Enum):
    start = "start"
    update = "update"
    end = "end"


class PoseActionTypeEnum(str, Enum):
    standing = "standing"
    sitting = "sitting"
    lying = "lying"
    falling = "falling"  # New action type for fall detection
    walking = "walking"
    running = "running"
    jumping = "jumping"
    waving = "waving"
    pointing = "pointing"
    custom = "custom"  # For user-defined pose actions
