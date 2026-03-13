"""Shared skeleton rendering code for the privacy proxy sidecar.

Renders COCO 17-keypoint pose skeletons on a black background.
This is a standalone copy of the renderer for use outside Frigate.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

# COCO 17-keypoint skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

COLORS = {
    "head": (255, 200, 50),
    "torso": (50, 255, 50),
    "arms": (50, 200, 255),
    "legs": (255, 50, 200),
    "keypoint": (255, 255, 255),
}

EDGE_COLORS = (
    [COLORS["head"]] * 4 +
    [COLORS["torso"]] * 1 +
    [COLORS["arms"]] * 4 +
    [COLORS["torso"]] * 2 +
    [COLORS["torso"]] * 1 +
    [COLORS["legs"]] * 4
)

MIN_CONFIDENCE = 0.3


def render_skeleton_bgr(
    poses: List[dict],
    width: int,
    height: int,
    keypoint_radius: int = 5,
    line_thickness: int = 3,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render pose skeletons on a solid background.

    Args:
        poses: List of pose dicts, each with "keypoints" as list of [x, y, conf].
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for pose in poses:
        kps = pose.get("keypoints", [])
        if len(kps) < 17:
            continue

        for idx, (i, j) in enumerate(SKELETON_EDGES):
            if kps[i][2] > MIN_CONFIDENCE and kps[j][2] > MIN_CONFIDENCE:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(canvas, pt1, pt2, EDGE_COLORS[idx], line_thickness, cv2.LINE_AA)

        for kp in kps:
            if kp[2] > MIN_CONFIDENCE:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(canvas, pt, keypoint_radius, COLORS["keypoint"], -1, cv2.LINE_AA)

    return canvas
