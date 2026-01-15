import numpy as np

from frigate.pose_detectors.detector_config import InputTensorEnum


def tensor_transform(input_tensor: InputTensorEnum):
    """Convert tensor format enum to numpy transpose axes."""
    if input_tensor == InputTensorEnum.nhwc:
        return None
    elif input_tensor == InputTensorEnum.nchw:
        return (0, 3, 1, 2)
    elif input_tensor == InputTensorEnum.hwnc:
        return (1, 2, 0, 3)
    elif input_tensor == InputTensorEnum.hwcn:
        return (1, 2, 3, 0)
    else:
        return None


from .tensor_utils import (
    POSE_BBOX_END,
    POSE_BBOX_START,
    POSE_KEYPOINTS_END,
    extract_keypoints_from_pose_output,
)


def format_pose_output(raw_poses, threshold=0.4):
    """Format raw pose detection output into standardized format.

    Uses optimized tensor utilities for keypoint extraction.
    """
    formatted_poses = []

    for pose in raw_poses:
        if len(pose) < 2 or pose[1] < threshold:
            continue

        formatted_pose = {
            "person_id": int(pose[0]) if len(pose) > 0 else 0,
            "confidence": float(pose[1]) if len(pose) > 1 else 0.0,
            "keypoints": [],
            "bbox": None,
        }
        # Extract keypoints using optimized helper (returns view)
        if len(pose) >= POSE_KEYPOINTS_END:
            keypoints = extract_keypoints_from_pose_output(pose)
            formatted_pose["keypoints"] = keypoints.tolist()

        # Extract bounding box if available
        if len(pose) >= POSE_BBOX_END:
            formatted_pose["bbox"] = pose[POSE_BBOX_START:POSE_BBOX_END].tolist()

        formatted_poses.append(formatted_pose)

    return formatted_poses


def calculate_pose_bbox(keypoints, confidence_threshold=0.3):
    """Calculate bounding box from keypoints."""
    if keypoints is None or (isinstance(keypoints, np.ndarray) and keypoints.size == 0):
        return None

    if not isinstance(keypoints, np.ndarray):
        keypoints = np.asarray(keypoints)

    # Ensure 2D shape
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(-1, 3)

    if keypoints.shape[0] < 2:
        return None

    # Filter keypoints with sufficient confidence using vectorized operations
    conf_mask = keypoints[:, 2] > confidence_threshold
    valid_count = np.sum(conf_mask)

    if valid_count < 2:
        return None

    valid_keypoints = keypoints[conf_mask, :2]

    # Calculate bounding box using vectorized min/max
    x_min, y_min = np.min(valid_keypoints, axis=0)
    x_max, y_max = np.max(valid_keypoints, axis=0)

    return [x_min, y_min, x_max, y_max]


def pose_similarity(pose1, pose2, threshold=0.5):
    """Calculate similarity between two poses based on keypoint positions.

    Optimized with efficient array handling.
    """
    kp1_raw = pose1.get("keypoints")
    kp2_raw = pose2.get("keypoints")

    if kp1_raw is None or kp2_raw is None:
        return 0.0

    kp1 = np.asarray(kp1_raw, dtype=np.float32)
    kp2 = np.asarray(kp2_raw, dtype=np.float32)

    # Ensure both are 2D
    if kp1.ndim == 1:
        kp1 = kp1.reshape(-1, 3)
    if kp2.ndim == 1:
        kp2 = kp2.reshape(-1, 3)

    if kp1.shape != kp2.shape or kp1.ndim != 2 or kp1.shape[1] < 3:
        return 0.0

    # Only compare keypoints with sufficient confidence (vectorized)
    valid_mask = (kp1[:, 2] > threshold) & (kp2[:, 2] > threshold)

    if not np.any(valid_mask):
        return 0.0

    # Calculate normalized distance between valid keypoints
    valid_kp1 = kp1[valid_mask, :2]
    valid_kp2 = kp2[valid_mask, :2]

    distances = np.linalg.norm(valid_kp1 - valid_kp2, axis=1)
    avg_distance = np.mean(distances)

    # Convert distance to similarity score (higher is more similar)
    similarity = 1.0 / (1.0 + avg_distance)

    return similarity


def calculate_pose_buffer_size(model_config):
    """Calculate consistent buffer size for pose detection shared memory.

    Args:
        model_config: PoseModelConfig with height and width attributes

    Returns:
        int: Buffer size in bytes with safety margin
    """
    if model_config is None:
        return 10485760  # 10MB default

    # Calculate needed size: height * width * channels * dtype_size
    base_size = model_config.height * model_config.width * 3

    # Add 2x safety margin for alignment and overhead
    safe_size = base_size * 2

    # Ensure minimum 10MB
    return max(safe_size, 10485760)


# COCO pose keypoint names for reference
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Connections between keypoints for skeleton drawing
COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]
