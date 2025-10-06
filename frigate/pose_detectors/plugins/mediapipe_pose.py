import logging

import cv2
import numpy as np

from frigate.pose_detectors.detection_api import PoseDetectionApi
from frigate.pose_detectors.detector_config import BasePoseDetectorConfig, PoseModelTypeEnum

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    logger.info("MediaPipe is available for pose detection")
except ImportError:
    logger.warning("MediaPipe not available. MediaPipe pose detection will not work.")
    mp = None


class MediaPipePoseDetectorConfig(BasePoseDetectorConfig):
    type: str = "mediapipe"
    static_image_mode: bool = False
    model_complexity: int = 1  # 0, 1, or 2
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipePoseApi(PoseDetectionApi):
    type_key = "mediapipe"
    supported_models = [PoseModelTypeEnum.mediapipe]

    def __init__(self, detector_config: MediaPipePoseDetectorConfig):
        super().__init__(detector_config)
        
        if mp is None:
            raise ImportError("MediaPipe is required for MediaPipe pose detection")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=getattr(detector_config, 'static_image_mode', False),
            model_complexity=getattr(detector_config, 'model_complexity', 1),
            smooth_landmarks=getattr(detector_config, 'smooth_landmarks', True),
            enable_segmentation=getattr(detector_config, 'enable_segmentation', False),
            smooth_segmentation=getattr(detector_config, 'smooth_segmentation', True),
            min_detection_confidence=getattr(detector_config, 'min_detection_confidence', 0.5),
            min_tracking_confidence=getattr(detector_config, 'min_tracking_confidence', 0.5),
        )
        
        logger.info("MediaPipe pose detector initialized")

    def detect_raw(self, tensor_input):
        """Run MediaPipe pose detection on input tensor."""
        try:
            # Convert tensor to image format expected by MediaPipe
            if len(tensor_input.shape) == 4:
                image = tensor_input[0]  # Remove batch dimension
            else:
                image = tensor_input
            
            # Convert to RGB if needed (MediaPipe expects RGB)
            if image.shape[-1] == 3:
                image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.astype(np.uint8)
            
            # Process the image
            results = self.pose.process(image_rgb)
            
            # Post-process MediaPipe output
            return self._postprocess_mediapipe_pose(results, image.shape)
            
        except Exception as e:
            logger.error(f"MediaPipe pose detection failed: {e}")
            return np.zeros((20, 57), dtype=np.float32)

    def _postprocess_mediapipe_pose(self, results, image_shape):
        """Post-process MediaPipe pose results."""
        poses = []
        
        if results.pose_landmarks:
            height, width = image_shape[:2]
            
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # MediaPipe returns 33 landmarks, but we'll use the COCO 17 keypoints
            # Mapping from MediaPipe to COCO format
            mp_to_coco_map = {
                0: 0,   # nose
                2: 1,   # left_eye
                5: 2,   # right_eye
                7: 3,   # left_ear
                8: 4,   # right_ear
                11: 5,  # left_shoulder
                12: 6,  # right_shoulder
                13: 7,  # left_elbow
                14: 8,  # right_elbow
                15: 9,  # left_wrist
                16: 10, # right_wrist
                23: 11, # left_hip
                24: 12, # right_hip
                25: 13, # left_knee
                26: 14, # right_knee
                27: 15, # left_ankle
                28: 16, # right_ankle
            }
            
            keypoints = np.zeros(51, dtype=np.float32)  # 17 keypoints * 3
            
            for mp_idx, coco_idx in mp_to_coco_map.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    confidence = landmark.visibility  # MediaPipe uses visibility as confidence
                    
                    keypoints[coco_idx * 3] = x
                    keypoints[coco_idx * 3 + 1] = y
                    keypoints[coco_idx * 3 + 2] = confidence
            
            # Calculate bounding box from keypoints
            valid_points = []
            for i in range(17):
                if keypoints[i * 3 + 2] > 0.3:  # confidence threshold
                    valid_points.append([keypoints[i * 3], keypoints[i * 3 + 1]])
            
            bbox = [0, 0, 0, 0]
            if valid_points:
                valid_points = np.array(valid_points)
                x_min, y_min = np.min(valid_points, axis=0)
                x_max, y_max = np.max(valid_points, axis=0)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # x, y, w, h
            
            # Overall pose confidence (average of visible keypoints)
            visible_keypoints = [kp for i, kp in enumerate(keypoints[2::3]) if kp > 0.3]
            pose_confidence = np.mean(visible_keypoints) if visible_keypoints else 0.0
            
            # Format output: [person_id, confidence, keypoints(51), bbox(4)]
            pose_output = np.zeros(57, dtype=np.float32)
            pose_output[0] = 0  # person_id (MediaPipe only detects one person)
            pose_output[1] = pose_confidence
            pose_output[2:53] = keypoints
            pose_output[53:57] = bbox
            
            poses.append(pose_output)
        
        # Pad to fixed size (20 poses max)
        result = np.zeros((20, 57), dtype=np.float32)
        for i, pose in enumerate(poses[:20]):
            result[i] = pose
            
        return result