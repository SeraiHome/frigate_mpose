import logging

import numpy as np

from frigate.pose_detection.tensor_utils import (
    POSE_BBOX_END,
    POSE_BBOX_START,
    POSE_CONFIDENCE_IDX,
    POSE_KEYPOINTS_END,
    POSE_KEYPOINTS_START,
    POSE_PERSON_ID_IDX,
    create_pose_output_batch,
)
from frigate.pose_detectors.detection_api import PoseDetectionApi
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    PoseModelTypeEnum,
)

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    logger.info("ONNX Runtime is available for YOLO pose detection")
except ImportError:
    logger.warning("ONNX Runtime not available. YOLO pose detection will not work.")
    ort = None


class YOLOPoseDetectorConfig(BasePoseDetectorConfig):
    type: str = "yolo_pose"
    device: str = "cpu"  # or "gpu"


class YoloPoseApi(PoseDetectionApi):
    type_key = "yolo_pose"
    supported_models = [PoseModelTypeEnum.yolo_pose]

    def __init__(self, detector_config: YOLOPoseDetectorConfig):
        super().__init__(detector_config)

        if ort is None:
            raise ImportError("ONNX Runtime is required for YOLO pose detection")

        self.device = getattr(detector_config, "device", "cpu")
        self.model_path = detector_config.model_path or detector_config.model.path

        if not self.model_path:
            raise ValueError("Model path is required for YOLO pose detection")

        # Initialize ONNX Runtime session
        providers = ["CPUExecutionProvider"]
        if self.device == "gpu":
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"YOLO pose model loaded: {self.model_path}")

    def detect_raw(self, tensor_input):
        """Run YOLO pose detection on input tensor."""
        try:
            # Add batch dimension if needed
            if tensor_input.ndim == 3:
                tensor_input = np.expand_dims(tensor_input, axis=0)

            # Run inference
            outputs = self.session.run(
                self.output_names, {self.input_name: tensor_input}
            )

            # Post-process YOLO pose output
            return self._postprocess_yolo_pose(outputs[0])

        except Exception as e:
            logger.error(f"YOLO pose detection failed: {e}")
            return create_pose_output_batch()

    def _postprocess_yolo_pose(self, raw_output):
        """Post-process YOLO pose model output.

        Uses pre-allocated output array for efficiency.
        """
        # YOLO pose output format: [batch, num_detections, 56]
        # where 56 = 4 (bbox) + 1 (confidence) + 51 (17 keypoints * 3)

        # Start with pre-allocated output array
        result = create_pose_output_batch()

        detections = raw_output[0] if raw_output.ndim == 3 else raw_output

        pose_idx = 0
        for i, detection in enumerate(detections):
            if pose_idx >= 20 or len(detection) < 5:
                continue

            # Extract bbox and confidence
            confidence = detection[4]

            if confidence < self.thresh:
                continue

            # Extract keypoints (17 keypoints * 3 = 51 values)
            keypoints = (
                detection[5:56]
                if len(detection) >= 56
                else np.zeros(51, dtype=np.float32)
            )

            # Format output directly into result array
            result[pose_idx, POSE_PERSON_ID_IDX] = i
            result[pose_idx, POSE_CONFIDENCE_IDX] = confidence
            result[pose_idx, POSE_KEYPOINTS_START:POSE_KEYPOINTS_END] = keypoints
            result[pose_idx, POSE_BBOX_START:POSE_BBOX_END] = detection[:4]  # bbox

            pose_idx += 1

        return result
