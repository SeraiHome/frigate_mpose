import logging
import queue
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent

import numpy as np

from frigate.config import CameraConfig
from frigate.pose_detection.base import RemotePoseDetector
from frigate.pose_detectors.detector_config import PoseModelConfig
from frigate.track.tracked_pose import TrackedPose

logger = logging.getLogger(__name__)


class PoseDetectionIntegration:
    """
    Integrates pose detection with the video pipeline.
    Each camera gets its dedicated pose detector to ensure continuous processing.
    """

    def __init__(
        self,
        camera_name: str,
        config: CameraConfig,
        model_config: PoseModelConfig,
        detection_queue: Queue,
        tracked_poses_queue: Queue,
        stop_event: MpEvent,
    ):
        self.camera_name = camera_name
        self.config = config
        self.pose_config = config.pose
        self.model_config = model_config
        self.detection_queue = detection_queue
        self.tracked_poses_queue = tracked_poses_queue
        self.stop_event = stop_event
        self.pose_detector = None
        self.tracked_poses = {}
        self.next_pose_id = 0

        # Only initialize if pose detection is enabled for this camera
        if self.pose_config.enabled:
            self.initialize_detector()

    def initialize_detector(self):
        """Initialize the pose detector for this camera."""
        try:
            # Create a keypoint names dictionary for the pose detector
            keypoint_dict = {}
            for i, name in enumerate(self.model_config.keypoint_names):
                keypoint_dict[i] = name

            # Initialize the detector using correct shared memory name format
            # The shared memory name for a pose detector should be prefixed with "pose-"
            detector_name = f"pose-{self.camera_name}"

            # Log the dimensions for debugging
            logger.info(
                f"Using pose model dimensions: {self.model_config.height}x{self.model_config.width}"
            )

            # Use the original model dimensions since the buffer is now properly sized in app.py
            self.pose_detector = RemotePoseDetector(
                detector_name,
                keypoint_dict,
                self.detection_queue,
                self.model_config,
                self.stop_event,
            )
            logger.info(
                f"Initialized pose detector for camera: {self.camera_name} detector_name: {detector_name}"
            )
        except Exception as e:
            logger.error(
                f"Error initializing pose detector for {self.camera_name}: {e}"
            )
            self.pose_detector = None

    def detect_poses(self, frame, frame_time, motion_boxes, regions):
        """Detect poses in the frame."""
        if not self.pose_config.enabled or self.pose_detector is None:
            return []

        try:
            # Reshape frame if necessary to match the expected format
            # The frame from the video pipeline is in YUV format (height*1.5 x width)
            # We need to ensure it's in the correct format for the pose detector

            # Detect poses
            detected_poses = self.pose_detector.detect(
                frame, self.pose_config.confidence_threshold
            )

            # Track and process poses
            tracked_poses = self.track_poses(detected_poses, frame_time)

            # Send tracked poses to the queue for further processing
            if tracked_poses:
                try:
                    self.tracked_poses_queue.put(
                        (
                            self.camera_name,
                            frame_time,
                            tracked_poses,
                            motion_boxes,
                            regions,
                        ),
                        False,
                    )
                except queue.Full:
                    logger.debug(f"Pose tracking queue full for {self.camera_name}")

            return tracked_poses
        except Exception as e:
            logger.error(f"Error detecting poses for {self.camera_name}: {e}")
            return []

    def track_poses(self, detected_poses, frame_time):
        """Track poses across frames."""
        # First, predict the state of existing tracked poses
        for pose_id, tracked_pose in list(self.tracked_poses.items()):
            tracked_pose.predict()

            # Remove old poses that haven't been updated in a while
            if tracked_pose.time_since_update > 10:
                del self.tracked_poses[pose_id]

        # Match detected poses with existing tracked poses or create new ones
        tracked = []

        for pose in detected_poses:
            if "keypoints" not in pose or not pose["keypoints"]:
                continue

            # Convert pose format to a NumPy array if needed
            if isinstance(pose["keypoints"], list):
                keypoints = np.array(pose["keypoints"])
            else:
                keypoints = pose["keypoints"]

            # For simplicity, create a new tracked pose for each detection
            # A more sophisticated approach would match existing tracked poses
            pose_id = f"{self.camera_name}_{self.next_pose_id}"
            self.next_pose_id += 1

            # Create a tracked pose object
            tracked_pose = TrackedPose(
                pose_id=pose_id,
                person_id=pose.get("person_id", 0),
                keypoints=keypoints,
                confidence=pose.get("confidence", 0.0),
                bbox=pose.get("bbox"),
                frame_time=frame_time,
            )

            # Mark as not a false positive since it just got detected
            tracked_pose.false_positive = False
            tracked_pose.update(
                keypoints, pose.get("confidence", 0.0), pose.get("bbox")
            )

            # Store in tracked poses dictionary for future tracking
            self.tracked_poses[pose_id] = tracked_pose
            tracked.append(tracked_pose)

        # Return all currently tracked poses that were updated this frame
        return [
            pose for pose in self.tracked_poses.values() if pose.time_since_update == 0
        ]

    def cleanup(self):
        """Clean up resources used by the pose detector."""
        if self.pose_detector:
            try:
                self.pose_detector.cleanup()
                logger.info(f"Cleaned up pose detector for camera: {self.camera_name}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up pose detector for {self.camera_name}: {e}"
                )
