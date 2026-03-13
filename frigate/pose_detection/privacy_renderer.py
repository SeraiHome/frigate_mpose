"""Privacy-preserving skeleton renderer.

Renders COCO 17-keypoint pose skeletons on a black background.
Returns YUV I420 frames compatible with Frigate's SharedMemoryFrameManager.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# COCO 17-keypoint skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # head: nose-eyes-ears
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# Color palette for skeleton parts (BGR)
COLORS = {
    "head": (255, 200, 50),      # light blue
    "torso": (50, 255, 50),      # green
    "arms": (50, 200, 255),      # orange
    "legs": (255, 50, 200),      # purple
    "keypoint": (255, 255, 255), # white
}

# Map edge index to body part for coloring
EDGE_COLORS = (
    [COLORS["head"]] * 4 +
    [COLORS["torso"]] * 1 +
    [COLORS["arms"]] * 4 +
    [COLORS["torso"]] * 2 +
    [COLORS["torso"]] * 1 +
    [COLORS["legs"]] * 4
)

# Confidence threshold for drawing a keypoint/edge
MIN_CONFIDENCE = 0.3


def render_skeleton_bgr(
    tracked_poses: Dict,
    width: int,
    height: int,
    keypoint_radius: int = 5,
    line_thickness: int = 3,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render pose skeletons on a background.

    Args:
        tracked_poses: Dict of tracked pose objects, each with .keypoints
                       as ndarray of shape (17, 3) in COCO format,
                       or a list of dicts with "keypoints" key.
        width: Frame width in pixels.
        height: Frame height in pixels.
        keypoint_radius: Radius of keypoint circles.
        line_thickness: Thickness of skeleton lines.
        bg_color: Background color in BGR (used when bg_image is None).
        bg_image: Optional background image. If provided, used as the canvas
                  instead of a solid color. Will be resized and converted to
                  BGR if needed (e.g. grayscale motion detector avg_frame).

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    if bg_image is not None:
        # Ensure correct size
        if bg_image.shape[0] != height or bg_image.shape[1] != width:
            bg_image = cv2.resize(bg_image, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert grayscale to BGR
        if bg_image.ndim == 2:
            canvas = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = bg_image.copy()
        # Ensure uint8
        if canvas.dtype != np.uint8:
            canvas = cv2.convertScaleAbs(canvas)
    else:
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for pose in (tracked_poses.values() if isinstance(tracked_poses, dict) else tracked_poses):
        # Support both TrackedPose objects (.keypoints) and dicts
        if hasattr(pose, "keypoints"):
            kps = pose.keypoints
        elif isinstance(pose, dict) and "keypoints" in pose:
            kps = pose["keypoints"]
        else:
            continue

        if isinstance(kps, np.ndarray):
            if kps.ndim == 1:
                kps = kps.reshape(-1, 3)
        elif isinstance(kps, list):
            kps = np.array(kps, dtype=np.float32).reshape(-1, 3)

        if len(kps) < 17:
            continue

        # Draw skeleton edges
        for idx, (i, j) in enumerate(SKELETON_EDGES):
            if kps[i][2] > MIN_CONFIDENCE and kps[j][2] > MIN_CONFIDENCE:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                color = EDGE_COLORS[idx]
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        # Draw keypoint circles
        for kp in kps:
            if kp[2] > MIN_CONFIDENCE:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(canvas, pt, keypoint_radius, COLORS["keypoint"], -1, cv2.LINE_AA)

    return canvas


def render_skeleton_yuv(
    tracked_poses: Dict,
    frame_shape: Tuple[int, int],
    keypoint_radius: int = 5,
    line_thickness: int = 3,
    bg_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render pose skeletons and return as YUV I420 frame.

    This format is compatible with Frigate's SharedMemoryFrameManager.

    Args:
        tracked_poses: Dict of tracked pose objects or list of pose dicts.
        frame_shape: (height, width) of the frame.
        bg_image: Optional background image (from motion detector avg_frame).

    Returns:
        YUV I420 numpy array of shape (height * 3 // 2, width).
    """
    height, width = frame_shape[0], frame_shape[1]
    bgr = render_skeleton_bgr(
        tracked_poses, width, height, keypoint_radius, line_thickness,
        bg_image=bg_image,
    )
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv


def render_no_detection_yuv(
    frame_shape: Tuple[int, int],
    bg_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a background-only YUV frame (no poses detected).

    When bg_image is provided, shows the static room background.
    Otherwise shows pure black.
    """
    height, width = frame_shape[0], frame_shape[1]
    if bg_image is not None:
        if bg_image.shape[0] != height or bg_image.shape[1] != width:
            bg_image = cv2.resize(bg_image, (width, height), interpolation=cv2.INTER_LINEAR)
        if bg_image.ndim == 2:
            bgr = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = bg_image
        if bgr.dtype != np.uint8:
            bgr = cv2.convertScaleAbs(bgr)
    else:
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)


def keypoints_to_mqtt_payload(
    tracked_poses: List,
    frame_time: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Convert tracked poses to a JSON-serializable dict for MQTT publishing.

    Args:
        tracked_poses: List of TrackedPose objects.
        frame_time: Timestamp of the frame.
        frame_width: Width of the frame in pixels.
        frame_height: Height of the frame in pixels.

    Returns:
        Dict suitable for json.dumps().
    """
    poses_data = []
    for pose in tracked_poses:
        kps = pose.keypoints if hasattr(pose, "keypoints") else pose.get("keypoints", [])
        if isinstance(kps, np.ndarray):
            kps_list = kps.tolist()
        else:
            kps_list = [[float(k[0]), float(k[1]), float(k[2])] for k in kps]

        pose_id = getattr(pose, "pose_id", None) or (pose.get("id") if isinstance(pose, dict) else None)
        poses_data.append({
            "id": str(pose_id) if pose_id else "unknown",
            "keypoints": kps_list,
        })

    return {
        "timestamp": frame_time,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "poses": poses_data,
    }
