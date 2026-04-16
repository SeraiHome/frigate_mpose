import logging

import numpy as np

from frigate.pose_detectors.detection_api import PoseDetectionApi
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    PoseModelTypeEnum,
)

logger = logging.getLogger(__name__)


class CPUPoseDetectorConfig(BasePoseDetectorConfig):
    type: str = "cpu"


class CpuPoseApi(PoseDetectionApi):
    type_key = "cpu"
    supported_models = [
        PoseModelTypeEnum.yolo_pose,
        PoseModelTypeEnum.mediapipe,
        PoseModelTypeEnum.openpose,
        PoseModelTypeEnum.movenet,
        PoseModelTypeEnum.posenet,
    ]

    def __init__(self, detector_config: CPUPoseDetectorConfig):
        super().__init__(detector_config)
        logger.warning(
            "CPU pose detection is not optimized and should only be used for testing."
        )

    def detect_raw(self, tensor_input):
        """
        Placeholder CPU implementation - in practice this would load and run
        a model using OpenCV DNN, ONNX Runtime, or TensorFlow Lite
        """
        # This is a dummy implementation that returns empty results
        # Real implementation would:
        # 1. Load the model (ONNX, TensorFlow, etc.)
        # 2. Run inference on tensor_input
        # 3. Post-process results to standard format

        logger.debug("Running CPU pose detection (placeholder)")

        # Return empty pose array in expected format
        # Shape: (max_poses, 57) where 57 = person_id(1) + confidence(1) + keypoints(17*3) + bbox(4)
        return np.zeros((20, 57), dtype=np.float32)
