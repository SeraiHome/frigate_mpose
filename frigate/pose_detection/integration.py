import logging
import queue
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent

import numpy as np

from frigate.config import CameraConfig
from frigate.pose_detection.base import RemotePoseDetector
from frigate.pose_detectors.detector_config import PoseModelConfig
from frigate.track.tracked_pose import TrackedPose
from frigate.util.image import yuv_region_2_rgb

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
        self.frame_count = 0

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

        detected_poses = []
        self.frame_count += 1

        try:
            # Convert frame to RGB format required by pose detector
            rgb_frame = self._convert_frame_to_rgb(frame)

            # Verify RGB frame has valid data before detection
            if rgb_frame is not None and np.count_nonzero(rgb_frame) > 0:
                # Create a fresh copy of the frame to ensure memory consistency
                rgb_frame_copy = rgb_frame.copy()

                # Detect poses using the verified RGB frame
                detected_poses = self.pose_detector.detect(
                    rgb_frame_copy, self.pose_config.confidence_threshold
                )
            else:
                logger.error(
                    f"RGB frame is empty or all zeros - cannot detect poses for camera {self.camera_name}"
                )
                detected_poses = []

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
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _convert_frame_to_rgb(self, frame):
        """Convert input frame to RGB format needed by pose detector."""
        try:
            # Log frame shape for debugging
            logger.debug(f"frame.shape: {frame.shape}")

            # Already in RGB format (3D with appropriate dimensions)
            if len(frame.shape) == 3 and not (frame.shape[0] > frame.shape[1] * 1.2):
                return frame

            # Handle YUV conversion - create square regions to avoid dimension mismatch

            # For 2D YUV frames
            if len(frame.shape) == 2:
                # For I420 format, Y plane height is 2/3 of the total height
                y_height = int(frame.shape[0] * 2 / 3)
                frame_width = frame.shape[1]

                # Calculate dimensions for a square region
                # Use the smaller dimension as the size of our square
                square_size = min(frame_width, y_height)

                # Center the square region
                x_center = frame_width // 2
                y_center = y_height // 2

                # Create square region centered in the frame
                region = (
                    max(0, x_center - square_size // 2),  # x_min
                    max(0, y_center - square_size // 2),  # y_min
                    min(frame_width, x_center + square_size // 2),  # x_max
                    min(y_height, y_center + square_size // 2),  # y_max
                )
                logger.debug(f"region: {region}")

                # Ensure the region dimensions are equal (square)
                region_width = region[2] - region[0]
                region_height = region[3] - region[1]

                if region_width != region_height:
                    # Adjust to ensure square dimensions
                    new_size = min(region_width, region_height)
                    region = (
                        region[0],
                        region[1],
                        region[0] + new_size,
                        region[1] + new_size,
                    )

                # Convert YUV to RGB with square region
                return yuv_region_2_rgb(frame, region)

            # For 3D YUV format
            elif len(frame.shape) == 3:
                frame_height = frame.shape[0]
                frame_width = frame.shape[1]

                # Calculate dimensions for a square region
                square_size = min(frame_width, frame_height)

                # Center the square region
                x_center = frame_width // 2
                y_center = frame_height // 2

                # Create square region centered in the frame
                region = (
                    max(0, x_center - square_size // 2),  # x_min
                    max(0, y_center - square_size // 2),  # y_min
                    min(frame_width, x_center + square_size // 2),  # x_max
                    min(frame_height, y_center + square_size // 2),  # y_max
                )
                logger.debugg(f"region: {region}")

                # Convert YUV to RGB with square region
                return yuv_region_2_rgb(frame, region)

            # If we can't determine the format, return as is
            return frame

        except Exception as e:
            logger.error(
                f"Error converting frame to RGB for camera {self.camera_name}: {e}"
            )
            # Create a fallback grayscale RGB image from Y plane if possible
            try:
                if len(frame.shape) == 2:
                    # For 2D YUV, extract just the Y plane as grayscale
                    y_height = int(frame.shape[0] * 2 / 3)
                    y_plane = frame[:y_height]
                    return np.stack([y_plane, y_plane, y_plane], axis=2)
                elif len(frame.shape) >= 3:
                    # For 3D, take first channel or first slice as grayscale
                    if frame.shape[2] >= 1:
                        y_plane = frame[:, :, 0]
                    else:
                        y_plane = frame[:, :]
                    return np.stack([y_plane, y_plane, y_plane], axis=2)
            except Exception as fallback_error:
                logger.error(f"Fallback conversion also failed: {fallback_error}")

            return None

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
            # Skip poses without keypoints or with empty keypoints
            # Properly handle NumPy arrays by checking size rather than using direct boolean evaluation
            if "keypoints" not in pose:
                continue

            # Handle the case where keypoints might be a NumPy array
            keypoints = pose["keypoints"]
            if keypoints is None or (
                isinstance(keypoints, np.ndarray) and keypoints.size == 0
            ):
                continue

            # Convert pose format to a NumPy array if needed
            if isinstance(pose["keypoints"], list):
                keypoints = np.array(pose["keypoints"])
            else:
                keypoints = pose["keypoints"]

            # Create a camera-specific pose ID to ensure uniqueness
            pose_id = f"{self.camera_name}_{self.next_pose_id}"
            self.next_pose_id += 1

            # Get and prepare bbox for TrackedPose (avoid direct NumPy array in boolean context)
            bbox = pose.get("bbox")
            if bbox is None:
                bbox_param = None
            elif isinstance(bbox, np.ndarray) and bbox.size == 0:
                bbox_param = None
            elif isinstance(bbox, np.ndarray):
                bbox_param = bbox.tolist()  # Convert NumPy array to list
            else:
                bbox_param = bbox

            # Create a tracked pose object
            tracked_pose = TrackedPose(
                pose_id=pose_id,
                person_id=pose.get("person_id", 0),
                keypoints=keypoints,
                confidence=pose.get("confidence", 0.0),
                bbox=bbox_param,
                frame_time=frame_time,
            )

            # Mark as not a false positive since it just got detected
            tracked_pose.false_positive = False

            # Handle bbox the same way for the update call as we did for the constructor
            bbox_update = pose.get("bbox")
            if bbox_update is None:
                bbox_update_param = None
            elif isinstance(bbox_update, np.ndarray) and bbox_update.size == 0:
                bbox_update_param = None
            elif isinstance(bbox_update, np.ndarray):
                bbox_update_param = bbox_update.tolist()  # Convert NumPy array to list
            else:
                bbox_update_param = bbox_update

            tracked_pose.update(
                keypoints, pose.get("confidence", 0.0), bbox_update_param
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
