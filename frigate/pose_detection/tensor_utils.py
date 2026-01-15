"""
Tensor Utilities for Pose Detection

This module provides canonical tensor shape definitions and helper functions
to standardize tensor handling across the pose detection pipeline.

CANONICAL FORMATS:
- Keypoints: Always (num_keypoints, 3) where 3 = (x, y, confidence)
- COCO format: 17 keypoints, shape (17, 3)
- Kinetics format: 18 keypoints, shape (18, 3)
- Pose output: (20, 57) where 57 = person_id(1) + confidence(1) + keypoints(51) + bbox(4)
- Image input: (H, W, 3) for RGB, or (1, H, W, 3) with batch dimension
"""

import numpy as np

# Keypoint format constants
COCO_NUM_KEYPOINTS = 17
KINETICS_NUM_KEYPOINTS = 18
KEYPOINT_DIMS = 3  # x, y, confidence

# Pose output format constants
MAX_POSES = 20
POSE_OUTPUT_SIZE = 57  # person_id(1) + confidence(1) + keypoints(51) + bbox(4)
POSE_PERSON_ID_IDX = 0
POSE_CONFIDENCE_IDX = 1
POSE_KEYPOINTS_START = 2
POSE_KEYPOINTS_END = 53  # 2 + 17*3
POSE_BBOX_START = 53
POSE_BBOX_END = 57

# Pre-allocated array shapes
POSE_OUTPUT_SHAPE = (MAX_POSES, POSE_OUTPUT_SIZE)
COCO_KEYPOINTS_SHAPE = (COCO_NUM_KEYPOINTS, KEYPOINT_DIMS)
KINETICS_KEYPOINTS_SHAPE = (KINETICS_NUM_KEYPOINTS, KEYPOINT_DIMS)


def ensure_keypoints_2d(
    keypoints: np.ndarray, num_keypoints: int = COCO_NUM_KEYPOINTS
) -> np.ndarray:
    """
    Ensure keypoints are in canonical 2D shape (num_keypoints, 3).

    This function handles various input formats and returns a consistent shape
    without unnecessary copies when possible.

    Args:
        keypoints: Input keypoints array (can be 1D flattened or 2D)
        num_keypoints: Expected number of keypoints (default: 17 for COCO)

    Returns:
        Keypoints array with shape (num_keypoints, 3)
    """
    if keypoints is None:
        return np.zeros((num_keypoints, KEYPOINT_DIMS), dtype=np.float32)

    # Convert list to array if needed
    if isinstance(keypoints, list):
        keypoints = np.asarray(keypoints, dtype=np.float32)

    # Already in correct shape - return view (no copy)
    if keypoints.ndim == 2 and keypoints.shape == (num_keypoints, KEYPOINT_DIMS):
        return keypoints

    # 1D flattened array - reshape (view, no copy)
    if keypoints.ndim == 1:
        expected_size = num_keypoints * KEYPOINT_DIMS
        if keypoints.size == expected_size:
            return keypoints.reshape((num_keypoints, KEYPOINT_DIMS))
        elif keypoints.size >= expected_size:
            # Take only what we need
            return keypoints[:expected_size].reshape((num_keypoints, KEYPOINT_DIMS))
        else:
            # Pad with zeros
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[: keypoints.size] = keypoints
            return padded.reshape((num_keypoints, KEYPOINT_DIMS))

    # 2D but wrong shape - need to handle
    if keypoints.ndim == 2:
        if keypoints.shape[1] == KEYPOINT_DIMS:
            # Correct number of dims per keypoint, just need to adjust keypoint count
            if keypoints.shape[0] >= num_keypoints:
                return keypoints[:num_keypoints]
            else:
                # Pad with zeros
                result = np.zeros((num_keypoints, KEYPOINT_DIMS), dtype=np.float32)
                result[: keypoints.shape[0]] = keypoints
                return result
        elif keypoints.shape[0] == KEYPOINT_DIMS:
            # Transposed - transpose and adjust
            transposed = keypoints.T
            return ensure_keypoints_2d(transposed, num_keypoints)

    # Fallback: return zero array
    return np.zeros((num_keypoints, KEYPOINT_DIMS), dtype=np.float32)


def extract_keypoints_from_pose_output(pose_output: np.ndarray) -> np.ndarray:
    """
    Extract keypoints from pose output array.

    Args:
        pose_output: Raw pose output array of shape (57,) or slice of (20, 57)

    Returns:
        Keypoints array with shape (17, 3)
    """
    keypoints_flat = pose_output[POSE_KEYPOINTS_START:POSE_KEYPOINTS_END]
    # reshape returns a view when possible
    return keypoints_flat.reshape(COCO_KEYPOINTS_SHAPE)


def create_pose_output(
    person_id: int, confidence: float, keypoints: np.ndarray, bbox: np.ndarray = None
) -> np.ndarray:
    """
    Create a standardized pose output array.

    Args:
        person_id: Person identifier
        confidence: Detection confidence
        keypoints: Keypoints array (17, 3) or (51,)
        bbox: Bounding box [x, y, w, h] or None

    Returns:
        Pose output array of shape (57,)
    """
    output = np.zeros(POSE_OUTPUT_SIZE, dtype=np.float32)
    output[POSE_PERSON_ID_IDX] = person_id
    output[POSE_CONFIDENCE_IDX] = confidence

    # Handle keypoints
    if keypoints is not None:
        kp_flat = keypoints.ravel() if keypoints.ndim > 1 else keypoints
        size = min(len(kp_flat), POSE_KEYPOINTS_END - POSE_KEYPOINTS_START)
        output[POSE_KEYPOINTS_START : POSE_KEYPOINTS_START + size] = kp_flat[:size]

    # Handle bbox
    if bbox is not None:
        bbox_arr = np.asarray(bbox, dtype=np.float32).ravel()
        size = min(len(bbox_arr), POSE_BBOX_END - POSE_BBOX_START)
        output[POSE_BBOX_START : POSE_BBOX_START + size] = bbox_arr[:size]

    return output


def create_pose_output_batch() -> np.ndarray:
    """
    Create an empty pose output batch array.

    Returns:
        Zero-initialized array of shape (20, 57)
    """
    return np.zeros(POSE_OUTPUT_SHAPE, dtype=np.float32)


def is_yuv_frame(image: np.ndarray) -> bool:
    """
    Check if image is likely in YUV format based on shape.

    YUV I420 format has height ~1.5x the luma height due to chroma planes.

    Args:
        image: Input image array

    Returns:
        True if image appears to be YUV format
    """
    if image.ndim == 2:
        return True  # 2D is typically I420 YUV
    if image.ndim == 3 and image.shape[0] > image.shape[1] * 1.2:
        return True  # Height >> width suggests YUV with chroma planes
    return False


def ensure_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in RGB HWC format (H, W, 3).

    Handles batch dimension removal and grayscale expansion.

    Args:
        image: Input image (can be NHWC, HWC, or HW)

    Returns:
        Image in HWC format (H, W, 3)
    """
    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]

    # Handle grayscale
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)

    # Already HWC
    if image.ndim == 3:
        if image.shape[-1] == 3:
            return image
        elif image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)

    return image


def ensure_batch_nhwc(image: np.ndarray) -> np.ndarray:
    """
    Ensure image has batch dimension in NHWC format (1, H, W, 3).

    Args:
        image: Input image (HWC or NHWC)

    Returns:
        Image with shape (1, H, W, 3)
    """
    if image.ndim == 3:
        return np.expand_dims(image, axis=0)
    elif image.ndim == 4:
        return image
    else:
        raise ValueError(f"Expected 3D or 4D image, got shape {image.shape}")
