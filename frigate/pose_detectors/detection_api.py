import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from frigate.pose_detection.tensor_utils import (
    COCO_KEYPOINTS_SHAPE,
    POSE_BBOX_END,
    POSE_BBOX_START,
    POSE_KEYPOINTS_END,
    extract_keypoints_from_pose_output,
)
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    PoseModelTypeEnum,
)

logger = logging.getLogger(__name__)


class PoseDetectionApi(ABC):
    type_key: str
    supported_models: List[PoseModelTypeEnum]

    @abstractmethod
    def __init__(self, detector_config: BasePoseDetectorConfig, camera_name=None):
        self.detector_config = detector_config
        self.thresh = 0.4
        # Default dimensions if model is None
        self.height = 640
        self.width = 640
        if detector_config.model is not None:
            self.height = detector_config.model.height
            self.width = detector_config.model.width
        # Store camera name for debugging and identification
        self.camera_name = camera_name

    @abstractmethod
    def detect_raw(self, tensor_input):
        pass

    def postprocess_poses(self, raw_output, threshold=0.4):
        """Post-process raw model output into standardized pose format.

        Uses optimized tensor utilities for consistent keypoint extraction.
        """
        poses = []

        # This is a generic implementation - specific detectors should override
        # Expected format: each pose as [person_id, confidence, keypoints(51), bbox(4)]
        for i, pose_data in enumerate(raw_output):
            if len(pose_data) < 2:
                continue

            confidence = pose_data[1] if len(pose_data) > 1 else 0.0
            if confidence < threshold:
                continue

            # Use optimized keypoint extraction (returns view, no copy)
            if len(pose_data) >= POSE_KEYPOINTS_END:
                keypoints = extract_keypoints_from_pose_output(pose_data)
            else:
                keypoints = np.zeros(COCO_KEYPOINTS_SHAPE, dtype=np.float32)

            pose = {
                "person_id": i,
                "confidence": confidence,
                "keypoints": keypoints,
                "bbox": pose_data[POSE_BBOX_START:POSE_BBOX_END]
                if len(pose_data) >= POSE_BBOX_END
                else None,
            }
            poses.append(pose)

        return poses
