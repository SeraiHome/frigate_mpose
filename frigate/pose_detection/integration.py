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
        self._skip_counter = 0  # Frame skip counter

        # Pre-compute skip interval for performance (avoid repeated config access)
        self._skip_frames = getattr(self.pose_config, "skip_frames", 0)
        self._use_motion_roi = getattr(self.pose_config, "use_motion_roi", False)

        # Reusable buffer for RGB conversion (shared memory optimization)
        self._rgb_buffer = None
        self._rgb_buffer_shape = None

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
            return []

        # Frame skip optimization: skip N frames between detections
        if self._skip_frames > 0:
            self._skip_counter += 1
            if self._skip_counter <= self._skip_frames:
                return []
            self._skip_counter = 0

        detected_poses = []
        self.frame_count += 1

        try:
            # Get FULL frame dimensions BEFORE cropping to ROI
            # This is critical for consistent keypoint normalization in activity detection
            if frame.ndim == 2:
                # YUV I420 format - Y plane is 2/3 of total height
                full_frame_height = int(frame.shape[0] * 2 / 3)
                full_frame_width = frame.shape[1]
            else:
                full_frame_height, full_frame_width = frame.shape[:2]

            # Compute motion ROI if enabled (union of all motion boxes with padding)
            motion_roi = None
            if self._use_motion_roi and motion_boxes:
                motion_roi = self._compute_motion_roi(frame, motion_boxes)

            # Convert frame to RGB format required by pose detector and get
            # the region in the original frame that was used for the RGB crop.
            rgb_frame, region = self._convert_frame_to_rgb(frame, motion_roi)

            # Verify RGB frame has valid data before detection
            if rgb_frame is not None and rgb_frame.size > 0 and np.any(rgb_frame):
                # Detect poses using the RGB frame (no copy needed - detector handles internally)
                detected_poses = self.pose_detector.detect(
                    rgb_frame, self.pose_config.confidence_threshold
                )

                # Map returned bboxes AND keypoints from detector output coordinates
                # back to the FULL FRAME coordinates.
                #
                # With dynamic SHM, the detector receives the rgb_frame at its ACTUAL size
                # (not resized to model config dimensions). MediaPipe returns coordinates
                # scaled to the input image dimensions, so we use rgb_frame.shape for mapping.
                #
                # The mapping is: detector coords (rgb_frame space) → full frame coords
                rgb_h, rgb_w = rgb_frame.shape[:2]

                # region is in original frame coords (x_min,y_min,x_max,y_max)
                if region is None:
                    # If no region was used, treat the whole rgb_frame as region
                    region = (0, 0, rgb_w, rgb_h)

                region_x0, region_y0, region_x1, region_y1 = region
                region_w = max(1, region_x1 - region_x0)
                region_h = max(1, region_y1 - region_y0)

                # Scale factors from rgb_frame coords to region coords in full frame
                # The detector outputs coords in rgb_frame space (0 to rgb_w/rgb_h)
                # We need to map these to the region in the full frame
                scale_x = region_w / rgb_w
                scale_y = region_h / rgb_h

                # Map bbox AND keypoints for each detected pose to full-frame coordinates
                for p in detected_poses:
                    # Map bbox from model pixels → region pixels → full-frame
                    bbox = p.get("bbox")
                    if bbox is not None and len(bbox) == 4:
                        try:
                            bx, by, bw, bh = bbox
                            # scale from model pixels to region pixels, then to full frame
                            x_full = int(region_x0 + float(bx) * scale_x)
                            y_full = int(region_y0 + float(by) * scale_y)
                            w_full = int(max(1, float(bw) * scale_x))
                            h_full = int(max(1, float(bh) * scale_y))
                            p["bbox"] = [x_full, y_full, w_full, h_full]
                        except Exception:
                            pass  # leave bbox as-is on failure

                    # Map keypoints from model pixels → region pixels → full-frame
                    # This ensures STGCN normalization is consistent regardless of ROI size
                    keypoints = p.get("keypoints")
                    if keypoints is not None:
                        try:
                            kp = np.array(keypoints, dtype=np.float32)
                            if kp.ndim == 1:
                                # Flat array: [x0, y0, conf0, x1, y1, conf1, ...]
                                for i in range(0, len(kp), 3):
                                    if i + 1 < len(kp):
                                        # Transform x, y to full-frame coords
                                        kp[i] = region_x0 + kp[i] * scale_x
                                        kp[i + 1] = region_y0 + kp[i + 1] * scale_y
                            elif kp.ndim == 2:
                                # 2D array: (num_keypoints, 3) with [x, y, conf]
                                kp[:, 0] = region_x0 + kp[:, 0] * scale_x
                                kp[:, 1] = region_y0 + kp[:, 1] * scale_y
                            p["keypoints"] = kp
                        except Exception:
                            pass  # leave keypoints as-is on failure
            else:
                detected_poses = []

            # Track and process poses using FULL FRAME dimensions for consistent
            # STGCN normalization regardless of ROI size changes between frames
            tracked_poses = self.track_poses(
                detected_poses, frame_time, full_frame_width, full_frame_height
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

    def _compute_motion_roi(self, frame, motion_boxes):
        """Compute bounding region containing all motion boxes with padding.

        Returns (x_min, y_min, x_max, y_max) in original frame coordinates,
        or None if motion_boxes is empty.
        """
        if not motion_boxes:
            return None

        # Get frame dimensions
        if frame.ndim == 2:
            # YUV I420 format - Y plane is 2/3 of total height
            frame_height = int(frame.shape[0] * 2 / 3)
            frame_width = frame.shape[1]
        else:
            frame_height, frame_width = frame.shape[:2]

        # Compute union of all motion boxes
        # motion_boxes format: list of (x, y, w, h)
        x_min = frame_width
        y_min = frame_height
        x_max = 0
        y_max = 0

        for box in motion_boxes:
            if len(box) >= 4:
                bx, by, bw, bh = box[:4]
                x_min = min(x_min, bx)
                y_min = min(y_min, by)
                x_max = max(x_max, bx + bw)
                y_max = max(y_max, by + bh)

        if x_max <= x_min or y_max <= y_min:
            return None

        # Add 20% padding to capture full pose even if only part is in motion
        pad_w = int((x_max - x_min) * 0.2)
        pad_h = int((y_max - y_min) * 0.2)

        # Ensure minimum padding of 50 pixels
        pad_w = max(pad_w, 50)
        pad_h = max(pad_h, 50)

        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(frame_width, x_max + pad_w)
        y_max = min(frame_height, y_max + pad_h)

        return (x_min, y_min, x_max, y_max)

    def _convert_frame_to_rgb(self, frame, motion_roi=None):
        """Convert input frame to RGB format needed by pose detector.

        Returns a tuple: (rgb_frame, region). `region` is the (x_min,y_min,x_max,y_max)
        rectangle in the original frame that was used to produce `rgb_frame`.
        If no explicit region was used, `region` will cover the full frame.

        Args:
            frame: Input frame (YUV or RGB)
            motion_roi: Optional (x_min, y_min, x_max, y_max) to crop to motion region

        Optimized to minimize redundant shape checks and memory allocations.
        """
        try:
            frame_ndim = frame.ndim

            # Already in RGB format (3D with appropriate dimensions)
            if (
                frame_ndim == 3
                and frame.shape[2] == 3
                and frame.shape[0] <= frame.shape[1] * 1.2
            ):
                h, w = frame.shape[:2]
                # Apply motion ROI if provided for RGB frames
                if motion_roi is not None:
                    x0, y0, x1, y1 = motion_roi
                    x0 = max(0, min(x0, w - 1))
                    y0 = max(0, min(y0, h - 1))
                    x1 = max(x0 + 1, min(x1, w))
                    y1 = max(y0 + 1, min(y1, h))
                    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)
                return frame, (0, 0, w, h)

            # Handle YUV conversion
            # Note: yuv_region_2_rgb uses yuv_crop_and_resize which forces square output
            # based on region height. We must ensure the region is square to avoid errors.
            if is_yuv_frame(frame):
                if frame_ndim == 2:
                    # For I420 format, Y plane height is 2/3 of the total height
                    y_height = int(frame.shape[0] * 2 / 3)
                    frame_width = frame.shape[1]
                else:
                    # 3D YUV
                    y_height = frame.shape[0]
                    frame_width = frame.shape[1]

                # Use motion ROI if provided, otherwise use largest centered square
                if motion_roi is not None:
                    # Clamp motion_roi to valid frame bounds
                    x0, y0, x1, y1 = motion_roi
                    x0 = max(0, min(x0, frame_width - 1))
                    y0 = max(0, min(y0, y_height - 1))
                    x1 = max(x0 + 1, min(x1, frame_width))
                    y1 = max(y0 + 1, min(y1, y_height))

                    # Make the region square (yuv_crop_and_resize requires it)
                    roi_w = x1 - x0
                    roi_h = y1 - y0
                    if roi_w != roi_h:
                        # Use the smaller dimension to ensure we stay in bounds
                        size = min(roi_w, roi_h)
                        # Round down to multiple of 4 for YUV alignment
                        size = (size // 4) * 4
                        if size < 4:
                            size = 4
                        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                        half = size // 2
                        x0 = max(0, cx - half)
                        y0 = max(0, cy - half)
                        x1 = x0 + size
                        y1 = y0 + size
                        # If we exceed bounds, shift the region back
                        if x1 > frame_width:
                            x1 = frame_width
                            x0 = x1 - size
                        if y1 > y_height:
                            y1 = y_height
                            y0 = y1 - size
                    region = (x0, y0, x1, y1)
                else:
                    # Create largest centered square region
                    square_size = min(frame_width, y_height)
                    # Round down to multiple of 4 for YUV alignment
                    square_size = (square_size // 4) * 4
                    x_center = frame_width // 2
                    y_center = y_height // 2
                    half_size = square_size // 2
                    region = (
                        max(0, x_center - half_size),
                        max(0, y_center - half_size),
                        max(0, x_center - half_size) + square_size,
                        max(0, y_center - half_size) + square_size,
                    )

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
                camera_name=self.camera_name,
                frame_width=frame_width,
                frame_height=frame_height,
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
