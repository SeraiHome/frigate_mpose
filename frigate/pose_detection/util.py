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


def format_pose_output(raw_poses, threshold=0.4):
    """Format raw pose detection output into standardized format."""
    formatted_poses = []
    
    for pose in raw_poses:
        if len(pose) < 2 or pose[1] < threshold:
            continue
            
        formatted_pose = {
            'person_id': int(pose[0]) if len(pose) > 0 else 0,
            'confidence': float(pose[1]) if len(pose) > 1 else 0.0,
            'keypoints': [],
            'bbox': None
        }
        
        # Extract keypoints (assuming COCO format: 17 keypoints * 3 values each)
        if len(pose) >= 53:  # 2 + 17*3 = 53 minimum
            keypoints = pose[2:53].reshape(-1, 3)
            formatted_pose['keypoints'] = keypoints.tolist()
        
        # Extract bounding box if available
        if len(pose) >= 57:  # 53 + 4 bbox values
            formatted_pose['bbox'] = pose[53:57].tolist()
        
        formatted_poses.append(formatted_pose)
    
    return formatted_poses


def calculate_pose_bbox(keypoints, confidence_threshold=0.3):
    """Calculate bounding box from keypoints."""
    if not keypoints or len(keypoints) == 0:
        return None
    
    # Filter keypoints with sufficient confidence
    valid_keypoints = []
    for kp in keypoints:
        if len(kp) >= 3 and kp[2] > confidence_threshold:
            valid_keypoints.append([kp[0], kp[1]])
    
    if len(valid_keypoints) < 2:
        return None
    
    valid_keypoints = np.array(valid_keypoints)
    
    # Calculate bounding box
    x_min = np.min(valid_keypoints[:, 0])
    y_min = np.min(valid_keypoints[:, 1])
    x_max = np.max(valid_keypoints[:, 0])
    y_max = np.max(valid_keypoints[:, 1])
    
    return [x_min, y_min, x_max, y_max]


def pose_similarity(pose1, pose2, threshold=0.5):
    """Calculate similarity between two poses based on keypoint positions."""
    if not pose1.get('keypoints') or not pose2.get('keypoints'):
        return 0.0
    
    kp1 = np.array(pose1['keypoints'])
    kp2 = np.array(pose2['keypoints'])
    
    if kp1.shape != kp2.shape or len(kp1.shape) != 2 or kp1.shape[1] < 2:
        return 0.0
    
    # Only compare keypoints with sufficient confidence
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


# COCO pose keypoint names for reference
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Connections between keypoints for skeleton drawing
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]