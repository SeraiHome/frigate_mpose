import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict, List, Optional

import numpy as np

from frigate.camera import CameraMetrics
from frigate.config import CameraConfig, FrigateConfig, ModelConfig
from frigate.util.image import SharedMemoryFrameManager

from .base import RemotePoseDetector
from .pose_processor import PoseEventProcessor

logger = logging.getLogger(__name__)


def detect_poses(
    camera_config: CameraConfig,
    pose_detector: RemotePoseDetector,
    frame: np.ndarray,
    frame_shape: tuple,
    region: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """Detect poses in a frame region."""
    if not pose_detector or not camera_config.pose.enabled:
        return []
    
    # If region is specified, crop the frame
    if region:
        x_min, y_min, x_max, y_max = region
        cropped_frame = frame[y_min:y_max, x_min:x_max]
    else:
        cropped_frame = frame
        region = (0, 0, frame_shape[1], frame_shape[0])
    
    # Detect poses
    poses = pose_detector.detect_poses(cropped_frame, threshold=camera_config.pose.keypoint_threshold)
    
    # Convert poses to detection format
    detections = []
    for pose in poses:
        if pose.bbox:
            # Adjust bbox coordinates to full frame
            bbox = [
                pose.bbox[0] + region[0],
                pose.bbox[1] + region[1],
                pose.bbox[2] + region[0],
                pose.bbox[3] + region[1],
            ]
            
            # Convert keypoints to full frame coordinates
            adjusted_keypoints = []
            for kp in pose.keypoints:
                adjusted_keypoints.append({
                    "x": kp.x + region[0],
                    "y": kp.y + region[1],
                    "confidence": kp.confidence,
                })
            
            detection = {
                "label": "pose",
                "confidence": pose.confidence,
                "bbox": bbox,
                "keypoints": adjusted_keypoints,
                "pose_id": pose.pose_id,
            }
            
            detections.append(detection)
    
    return detections


def create_pose_detector(
    camera_config: CameraConfig,
    config: FrigateConfig,
    detection_queue: Queue,
    stop_event: MpEvent,
) -> Optional[RemotePoseDetector]:
    """Create a pose detector for a camera."""
    if not camera_config.pose.enabled or not config.pose_detectors:
        return None
    
    # Get first pose detector config
    detector_config = list(config.pose_detectors.values())[0]
    if not detector_config or not detector_config.model:
        logger.warning(f"No pose model configured for camera {camera_config.name}")
        return None
    
    model_config = detector_config.model
    
    return RemotePoseDetector(
        camera_config.name,
        {},  # No labels needed for pose
        detection_queue,
        model_config,
        stop_event,
    )


def should_run_pose_detection(
    camera_config: CameraConfig,
    motion_boxes: List[tuple],
    regions: List[tuple],
) -> bool:
    """Determine if pose detection should run on current frame."""
    if not camera_config.pose.enabled:
        return False
    
    # Check if there's motion in required zones
    if camera_config.pose.required_zones:
        # TODO: Check if motion boxes overlap with required zones
        pass
    
    # For now, run pose detection if there's any motion
    return len(motion_boxes) > 0


def filter_pose_detections(
    detections: List[Dict[str, Any]],
    camera_config: CameraConfig,
) -> List[Dict[str, Any]]:
    """Filter pose detections based on configuration."""
    filtered = []
    
    for detection in detections:
        # Check minimum keypoints
        valid_keypoints = [kp for kp in detection["keypoints"] 
                          if kp["confidence"] >= camera_config.pose.keypoint_threshold]
        
        if len(valid_keypoints) < camera_config.pose.filters["person"].min_keypoints:
            continue
        
        # Check confidence
        if detection["confidence"] < camera_config.pose.filters["person"].min_confidence:
            continue
        
        # Check area if bbox exists
        if detection.get("bbox"):
            bbox = detection["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            min_area = camera_config.pose.filters["person"].min_area
            max_area = camera_config.pose.filters["person"].max_area
            
            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue
        
        filtered.append(detection)
    
    return filtered


def convert_object_to_pose(
    tracked_object: Dict[str, Any],
    camera_config: CameraConfig,
) -> Optional[Dict[str, Any]]:
    """Convert a tracked person object to a pose detection."""
    if not camera_config.pose.objects_as_poses:
        return None
    
    if tracked_object.get("label") != "person":
        return None
    
    # Create a pose detection from the person object
    bbox = tracked_object.get("box", tracked_object.get("region"))
    if not bbox:
        return None
    
    # Generate estimated keypoints based on bbox
    # This is a simple estimation - in real implementation, 
    # you might use a lightweight pose estimator
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Estimate keypoint positions (COCO format)
    # This is very basic - just places keypoints in typical positions
    keypoints = [
        {"x": cx, "y": y1 + height * 0.1, "confidence": 0.5},  # nose
        {"x": cx - width * 0.1, "y": y1 + height * 0.15, "confidence": 0.5},  # left_eye
        {"x": cx + width * 0.1, "y": y1 + height * 0.15, "confidence": 0.5},  # right_eye
        {"x": cx - width * 0.15, "y": y1 + height * 0.15, "confidence": 0.5},  # left_ear
        {"x": cx + width * 0.15, "y": y1 + height * 0.15, "confidence": 0.5},  # right_ear
        {"x": cx - width * 0.25, "y": y1 + height * 0.3, "confidence": 0.5},  # left_shoulder
        {"x": cx + width * 0.25, "y": y1 + height * 0.3, "confidence": 0.5},  # right_shoulder
        {"x": cx - width * 0.25, "y": y1 + height * 0.5, "confidence": 0.5},  # left_elbow
        {"x": cx + width * 0.25, "y": y1 + height * 0.5, "confidence": 0.5},  # right_elbow
        {"x": cx - width * 0.2, "y": y1 + height * 0.6, "confidence": 0.5},  # left_wrist
        {"x": cx + width * 0.2, "y": y1 + height * 0.6, "confidence": 0.5},  # right_wrist
        {"x": cx - width * 0.15, "y": y1 + height * 0.6, "confidence": 0.5},  # left_hip
        {"x": cx + width * 0.15, "y": y1 + height * 0.6, "confidence": 0.5},  # right_hip
        {"x": cx - width * 0.15, "y": y1 + height * 0.8, "confidence": 0.5},  # left_knee
        {"x": cx + width * 0.15, "y": y1 + height * 0.8, "confidence": 0.5},  # right_knee
        {"x": cx - width * 0.1, "y": y1 + height * 0.95, "confidence": 0.5},  # left_ankle
        {"x": cx + width * 0.1, "y": y1 + height * 0.95, "confidence": 0.5},  # right_ankle
    ]
    
    return {
        "label": "pose",
        "confidence": tracked_object.get("score", 0.5),
        "bbox": [x1, y1, x2, y2],
        "keypoints": keypoints,
        "pose_id": f"{tracked_object.get('id', 'unknown')}_pose",
        "from_object": True,
    }


class PoseDetectionIntegration:
    """Integrates pose detection into the video processing pipeline."""
    
    def __init__(
        self,
        camera_config: CameraConfig,
        config: FrigateConfig,
        detection_queue: Queue,
        pose_event_queue: Queue,
        timeline_queue: Queue,
        camera_metrics: CameraMetrics,
        stop_event: MpEvent,
    ):
        self.camera_config = camera_config
        self.config = config
        self.detection_queue = detection_queue
        self.pose_event_queue = pose_event_queue
        self.timeline_queue = timeline_queue
        self.camera_metrics = camera_metrics
        self.stop_event = stop_event
        
        # Create pose detector
        self.pose_detector = create_pose_detector(
            camera_config, config, detection_queue, stop_event
        )
        
        # Create pose event processor
        if self.pose_detector and camera_config.pose.enabled:
            self.pose_processor = PoseEventProcessor(
                config,
                camera_config,
                pose_event_queue,
                timeline_queue,
                stop_event,
            )
            self.pose_processor.start()
        else:
            self.pose_processor = None
        
        self.frame_manager = SharedMemoryFrameManager()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_time: float,
        motion_boxes: List[tuple],
        regions: List[tuple],
        tracked_objects: List[Dict[str, Any]],
        current_zones: List[str],
    ) -> List[Dict[str, Any]]:
        """Process a frame for pose detection."""
        if not self.pose_detector or not self.camera_config.pose.enabled:
            return []
        
        pose_detections = []
        
        # Check if we should run pose detection
        if should_run_pose_detection(self.camera_config, motion_boxes, regions):
            # Detect poses in regions with motion
            for region in regions:
                detections = detect_poses(
                    self.camera_config,
                    self.pose_detector,
                    frame,
                    frame.shape[:2],
                    region,
                )
                pose_detections.extend(detections)
        
        # Optionally convert tracked persons to poses
        if self.camera_config.pose.objects_as_poses:
            for obj in tracked_objects:
                pose = convert_object_to_pose(obj, self.camera_config)
                if pose:
                    pose_detections.append(pose)
        
        # Filter detections
        pose_detections = filter_pose_detections(pose_detections, self.camera_config)
        
        # Send to pose processor
        if pose_detections and self.pose_processor:
            self.pose_event_queue.put((frame, frame_time, current_zones))
        
        # Update metrics
        if hasattr(self.camera_metrics, "pose_fps"):
            self.camera_metrics.pose_fps.value = self.pose_detector.fps.eps()
        
        return pose_detections

    def cleanup(self):
        """Clean up resources."""
        if self.pose_processor:
            self.pose_processor.join(timeout=5)
        
        if self.pose_detector:
            self.pose_detector.cleanup()