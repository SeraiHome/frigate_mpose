import logging
import queue
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent

import numpy as np

from frigate.config import CameraConfig
from frigate.pose_detection.base import RemotePoseDetector
from frigate.pose_detection.tensor_utils import (
    COCO_NUM_KEYPOINTS,
    ensure_keypoints_2d,
    is_yuv_frame,
)
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
        """Detect poses in the frame.

        Applies motion-based filtering to avoid unnecessary pose detection
        when the scene is static. This is critical for performance.

        Motion filtering only skips detection when NO motion is detected.
        When object detection is disabled, regions will be empty but that
        should not prevent pose detection if there is motion.
        """
        if not self.pose_config.enabled or self.pose_detector is None:
            return []

        # Performance optimization: skip pose detection if no motion detected
        # This prevents wasting CPU/GPU cycles on static scenes.
        # Note: We only require motion_boxes to have content. Regions may be
        # empty when object detection is disabled, which is fine.
        if not motion_boxes:
            logger.debug(
                f"Skipping pose detection for {self.camera_name}: no motion detected"
            )
            return []

        detected_poses = []
        self.frame_count += 1

        try:
            # Convert frame to RGB format required by pose detector and get
            # the region in the original frame that was used for the RGB crop.
            rgb_frame, region = self._convert_frame_to_rgb(frame)

            # Verify RGB frame has valid data before detection
            if rgb_frame is not None and np.count_nonzero(rgb_frame) > 0:
                # Create a fresh copy of the frame to ensure memory consistency
                rgb_frame_copy = rgb_frame.copy()

                # Extract frame dimensions for normalization from the actual RGB
                # frame rather than relying on model config values which may
                # not reflect the resized input used for detection.
                frame_height, frame_width = rgb_frame_copy.shape[:2]
                logger.debug(
                    f"Frame dimensions for camera {self.camera_name}: {frame_width}x{frame_height}"
                )

                # Detect poses using the verified RGB frame
                detected_poses = self.pose_detector.detect(
                    rgb_frame_copy, self.pose_config.confidence_threshold
                )

                # Map returned bboxes from detector/model input coordinates
                # back to the original frame coordinates. The detector typically
                # resizes rgb_frame_copy to the model input size internally; we
                # therefore scale using the rgb_frame dimensions and add the
                # region offset to place the bbox in the original frame.
                try:
                    model_w = getattr(
                        self.model_config, "width", rgb_frame_copy.shape[1]
                    )
                    model_h = getattr(
                        self.model_config, "height", rgb_frame_copy.shape[0]
                    )
                except Exception:
                    model_h, model_w = rgb_frame_copy.shape[:2]

                # region is in original frame coords (x_min,y_min,x_max,y_max)
                if region is None:
                    # If no region was used, treat the whole rgb_frame as region
                    region = (0, 0, rgb_frame_copy.shape[1], rgb_frame_copy.shape[0])

                region_x0, region_y0, region_x1, region_y1 = region
                region_w = max(1, region_x1 - region_x0)
                region_h = max(1, region_y1 - region_y0)

                # detector returns bbox as [x, y, w, h] in model pixels
                for p in detected_poses:
                    bbox = p.get("bbox")
                    if bbox is None or len(bbox) != 4:
                        continue
                    try:
                        bx, by, bw, bh = bbox
                        # scale from model pixels to region pixels
                        x_reg = float(bx) * (region_w / model_w)
                        y_reg = float(by) * (region_h / model_h)
                        w_reg = float(bw) * (region_w / model_w)
                        h_reg = float(bh) * (region_h / model_h)

                        # map to original frame coordinates by adding region offset
                        x_full = int(region_x0 + x_reg)
                        y_full = int(region_y0 + y_reg)
                        w_full = int(max(1, w_reg))
                        h_full = int(max(1, h_reg))

                        # replace bbox in-place with original-frame [x,y,w,h]
                        p["bbox"] = [x_full, y_full, w_full, h_full]
                    except Exception:
                        # leave bbox as-is on failure
                        continue
            else:
                logger.error(
                    f"RGB frame is empty or all zeros - cannot detect poses for camera {self.camera_name}"
                )
                detected_poses = []
                frame_width = None
                frame_height = None

            # Track and process poses
            tracked_poses = self.track_poses(
                detected_poses, frame_time, frame_width, frame_height
            )

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
        """Convert input frame to RGB format needed by pose detector.

        Returns a tuple: (rgb_frame, region). `region` is the (x_min,y_min,x_max,y_max)
        rectangle in the original frame that was used to produce `rgb_frame`.
        If no explicit region was used, `region` will cover the full frame.

        Optimized to minimize redundant shape checks.
        """
        try:
            # Log frame shape for debugging
            logger.debug(f"frame.shape: {frame.shape}")

            frame_ndim = frame.ndim

            # Already in RGB format (3D with appropriate dimensions)
            if (
                frame_ndim == 3
                and frame.shape[2] == 3
                and frame.shape[0] <= frame.shape[1] * 1.2
            ):
                h, w = frame.shape[:2]
                return frame, (0, 0, w, h)

            # Handle YUV conversion - create square regions to avoid dimension mismatch
            if is_yuv_frame(frame):
                if frame_ndim == 2:
                    # For I420 format, Y plane height is 2/3 of the total height
                    y_height = int(frame.shape[0] * 2 / 3)
                    frame_width = frame.shape[1]
                else:
                    # 3D YUV
                    y_height = frame.shape[0]
                    frame_width = frame.shape[1]

                # Calculate dimensions for a square region
                square_size = min(frame_width, y_height)

                # Center the square region
                x_center = frame_width // 2
                y_center = y_height // 2

                # Create square region centered in the frame
                half_size = square_size // 2
                region = (
                    max(0, x_center - half_size),
                    max(0, y_center - half_size),
                    min(frame_width, x_center + half_size),
                    min(y_height, y_center + half_size),
                )

                # Ensure square dimensions
                region_width = region[2] - region[0]
                region_height = region[3] - region[1]
                if region_width != region_height:
                    new_size = min(region_width, region_height)
                    region = (
                        region[0],
                        region[1],
                        region[0] + new_size,
                        region[1] + new_size,
                    )

                logger.debug(f"region: {region}")
                return yuv_region_2_rgb(frame, region), region

            # If we can't determine the format, return as is
            h, w = frame.shape[:2]
            return frame, (0, 0, w, h)

        except Exception as e:
            logger.error(
                f"Error converting frame to RGB for camera {self.camera_name}: {e}"
            )
            # Create a fallback grayscale RGB image from Y plane if possible
            try:
                if frame.ndim == 2:
                    # For 2D YUV, extract just the Y plane as grayscale
                    y_height = int(frame.shape[0] * 2 / 3)
                    y_plane = frame[:y_height]
                    return np.stack([y_plane, y_plane, y_plane], axis=2), (
                        0,
                        0,
                        frame.shape[1],
                        y_height,
                    )
                elif frame.ndim >= 3:
                    # For 3D, take first channel or first slice as grayscale
                    y_plane = frame[:, :, 0] if frame.shape[2] >= 1 else frame[:, :]
                    return np.stack([y_plane, y_plane, y_plane], axis=2), (
                        0,
                        0,
                        frame.shape[1],
                        frame.shape[0],
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback conversion also failed: {fallback_error}")

            return None

    def track_poses(
        self, detected_poses, frame_time, frame_width=None, frame_height=None
    ):
        """
        Track poses across frames.

        Args:
            detected_poses: List of detected poses
            frame_time: Timestamp of the frame
            frame_width: Width of the frame in pixels, used for keypoint normalization
            frame_height: Height of the frame in pixels, used for keypoint normalization
        """
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

            # Ensure keypoints in canonical (17, 3) shape using optimized helper
            keypoints = ensure_keypoints_2d(keypoints, COCO_NUM_KEYPOINTS)

            # Compute detection centroid. Prefer bbox when available (more
            # stable across detectors). Bbox is expected as [x,y,w,h]; if
            # present compute center as (x + w/2, y + h/2). Fall back to
            # keypoints centroid when bbox not available.
            det_centroid = None
            try:
                bbox_val = pose.get("bbox")
                if bbox_val is not None and len(bbox_val) >= 4:
                    # assume format [x, y, w, h]
                    bx, by, bw, bh = bbox_val[0], bbox_val[1], bbox_val[2], bbox_val[3]
                    det_centroid = np.array([bx + bw / 2.0, by + bh / 2.0])
                else:
                    vis = keypoints[:, 2] > 0.1
                    if np.any(vis):
                        det_centroid = np.mean(keypoints[vis, :2], axis=0)
                    else:
                        det_centroid = np.mean(keypoints[:, :2], axis=0)
            except Exception:
                det_centroid = None

            # Prepare detection bbox in x,y,w,h format if available
            det_bbox = None
            bbox_val = pose.get("bbox")
            if bbox_val is not None and not (
                isinstance(bbox_val, np.ndarray) and bbox_val.size == 0
            ):
                if isinstance(bbox_val, np.ndarray):
                    det_bbox = bbox_val.tolist()
                else:
                    det_bbox = bbox_val

            # Attempt to match detection to an existing tracked pose
            matched_id = None
            matched_pose = None
            # Build candidates list of currently tracked poses
            candidates = []
            for tid, tpose in self.tracked_poses.items():
                # Skip poses that were just deleted or have excessive time since update
                if tpose.time_since_update > 10:
                    continue
                # Compute tracked centroid
                try:
                    # Prefer tracked bbox centroid when available
                    t_centroid = None
                    t_bbox_val = tpose.bbox if hasattr(tpose, "bbox") else None
                    if t_bbox_val is not None and len(t_bbox_val) >= 4:
                        # assume [x,y,w,h]
                        tbx, tby, tbw, tbh = (
                            t_bbox_val[0],
                            t_bbox_val[1],
                            t_bbox_val[2],
                            t_bbox_val[3],
                        )
                        t_centroid = np.array([tbx + tbw / 2.0, tby + tbh / 2.0])
                    else:
                        # Use helper to ensure tracked keypoints are (17, 3)
                        t_kp = ensure_keypoints_2d(tpose.keypoints, COCO_NUM_KEYPOINTS)
                        t_vis = t_kp[:, 2] > 0.1
                        if np.any(t_vis):
                            t_centroid = np.mean(t_kp[t_vis, :2], axis=0)
                        else:
                            t_centroid = np.mean(t_kp[:, :2], axis=0)
                except Exception:
                    t_centroid = None

                candidates.append((tid, tpose, t_centroid, tpose.bbox))

            best_score = float("inf")
            best_candidate = None

            # Compute normalization diagonal
            if frame_width and frame_height:
                diag = (frame_width**2 + frame_height**2) ** 0.5
            else:
                diag = None

            for tid, tpose, t_centroid, t_bbox in candidates:
                score = float("inf")
                # distance-based score
                if det_centroid is not None and t_centroid is not None:
                    try:
                        dist = np.linalg.norm(det_centroid - t_centroid)
                        if diag and diag > 0:
                            dist_norm = dist / diag
                        else:
                            dist_norm = dist
                        score = dist_norm
                    except Exception:
                        score = float("inf")

                if score < best_score:
                    best_score = score
                    best_candidate = (tid, tpose, t_centroid, t_bbox)

            # Match only if centroid distance is below threshold. Use strict matching:
            # 0.05 (5%) of diagonal is a reasonable threshold for same person in consecutive frames.
            match = False
            if (
                best_candidate is not None
                and det_centroid is not None
                and diag
                and diag > 0
            ):
                tid, tpose, t_centroid, t_bbox = best_candidate
                try:
                    dist = np.linalg.norm(det_centroid - t_centroid)
                    if (dist / diag) < 0.1:
                        match = True
                        matched_id = tid
                        matched_pose = tpose
                except Exception:
                    match = False

            if matched_pose:
                pose_id = matched_id
                logger.debug(
                    f"Matched detection to existing pose ID: {pose_id} for camera: {self.camera_name}"
                )
                # Update existing tracked pose
                tracked_pose = matched_pose
                tracked_pose.frame_time = frame_time
                tracked_pose.false_positive = False
                # Use detection bbox for update if available
                bbox_for_update = det_bbox
                tracked_pose.update(
                    keypoints, pose.get("confidence", 0.0), bbox_for_update
                )
                # store and continue to next detection (no new TrackedPose created)
                self.tracked_poses[pose_id] = tracked_pose
                tracked.append(tracked_pose)
                continue
            else:
                # Create a camera-specific pose ID to ensure uniqueness
                pose_id = f"{self.camera_name}_{self.next_pose_id}"
                self.next_pose_id += 1
                logger.info(
                    f"Assigning pose ID: {pose_id} for camera: {self.camera_name}"
                )
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
