import logging
from typing import List, Optional

import numpy as np

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum

logger = logging.getLogger(__name__)


class YOLOPoseDetectorAPI(DetectionApi):
    """YOLO Pose detector implementation for pose estimation."""
    
    type_key = "yolo_pose"
    supported_models = [ModelTypeEnum.yologeneric]
    
    # COCO pose keypoint mapping
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Body
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    def __init__(self, detector_config: BaseDetectorConfig):
        super().__init__(detector_config)
        self.num_keypoints = 17  # COCO format has 17 keypoints
        self.keypoint_thresh = detector_config.model.model_config.get("keypoint_thresh", 0.5)
        
        # Load model here - this is a placeholder
        # In actual implementation, you would load the YOLO pose model
        logger.info(f"Initializing YOLO Pose detector with model: {detector_config.model.path}")

    def detect_raw(self, tensor_input: np.ndarray) -> List[dict]:
        """
        Detect poses in the input tensor.
        
        Returns a list of pose detections, each containing:
        - confidence: overall pose confidence
        - bbox: bounding box [x1, y1, x2, y2]
        - keypoints: list of [x, y, confidence] for each keypoint
        """
        # This is a placeholder implementation
        # In actual implementation, you would:
        # 1. Run the YOLO pose model on tensor_input
        # 2. Parse the model output to extract poses
        # 3. Format the output as expected
        
        poses = []
        
        # Placeholder: simulate detection of a single pose
        # In real implementation, this would come from the model
        mock_pose = {
            "confidence": 0.85,
            "bbox": [100, 100, 300, 400],  # x1, y1, x2, y2
            "keypoints": []
        }
        
        # Generate mock keypoints
        for i in range(self.num_keypoints):
            # In real implementation, these would be actual detected keypoints
            x = 150 + (i % 3) * 50
            y = 150 + (i // 3) * 50
            conf = 0.8 if i < 5 else 0.6  # Higher confidence for face keypoints
            mock_pose["keypoints"].append([x, y, conf])
        
        poses.append(mock_pose)
        
        return poses

    def _nms_poses(self, poses: List[dict], iou_threshold: float = 0.5) -> List[dict]:
        """Apply Non-Maximum Suppression to remove duplicate poses."""
        if len(poses) <= 1:
            return poses
        
        # Sort poses by confidence
        poses = sorted(poses, key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        for i, pose in enumerate(poses):
            discard = False
            for kept_pose in keep:
                # Calculate IoU between bounding boxes
                iou = self._calculate_iou(pose["bbox"], kept_pose["bbox"])
                if iou > iou_threshold:
                    discard = True
                    break
            
            if not discard:
                keep.append(pose)
        
        return keep

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _filter_keypoints(self, keypoints: List[List[float]]) -> List[List[float]]:
        """Filter keypoints based on confidence threshold."""
        filtered = []
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] >= self.keypoint_thresh:
                filtered.append(kp)
            else:
                filtered.append([0, 0, 0])  # Invalid keypoint
        return filtered