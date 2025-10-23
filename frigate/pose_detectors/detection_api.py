import logging
from abc import ABC, abstractmethod
from typing import List

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
        """Post-process raw model output into standardized pose format."""
        poses = []

        # This is a generic implementation - specific detectors should override
        # Expected format: each pose as [person_id, confidence, keypoints..., bbox...]
        for i, pose_data in enumerate(raw_output):
            if len(pose_data) < 2:
                continue

            confidence = pose_data[1] if len(pose_data) > 1 else 0.0
            if confidence < threshold:
                continue

            pose = {
                "person_id": i,
                "confidence": confidence,
                "keypoints": pose_data[2:53].reshape(-1, 3)
                if len(pose_data) >= 53
                else [],
                "bbox": pose_data[53:57] if len(pose_data) >= 57 else None,
            }
            poses.append(pose)

        return poses
