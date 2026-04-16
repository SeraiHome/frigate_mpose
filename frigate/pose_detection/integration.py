import logging
import queue
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Optional

import cv2
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
from frigate.util.image import intersection_over_union, yuv_region_2_rgb

logger = logging.getLogger(__name__)


def _bbox_xywh_to_xyxy(bbox):
    """Convert an [x, y, w, h] bbox to the [x1, y1, x2, y2] format expected by
    `frigate.util.image.intersection_over_union`. Returns None if the input
    is malformed so callers can gracefully fall back to centroid matching.
    """
    if bbox is None:
        return None
    try:
        if len(bbox) < 4:
            return None
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if w <= 0 or h <= 0:
            return None
        return [int(x), int(y), int(x + w), int(y + h)]
    except (TypeError, ValueError):
        return None


# Target height (px) for the encoded thumbnail.  Matches the size that
# `frigate.track.tracked_object.TrackedObject.get_thumbnail` uses for
# standard person events, so the two thumbnail sources look identical
# in the dashboard.
_POSE_THUMBNAIL_TARGET_HEIGHT = 175


def _encode_pose_thumbnail(frame: np.ndarray) -> Optional[bytes]:
    """Encode a small webp thumbnail from a YUV I420 frame.

    Used by `PoseDetectionIntegration.detect_poses` to ship a thumbnail
    through the tracked-poses queue so `pose_consumer` can persist it
    when a pose-driven Event is finalized.  Without this, pose-driven
    events have no `THUMB_DIR/{cam}/{event_id}.webp` file and the
    standard `/api/events/{id}/thumbnail.{ext}` route 404s.

    The frame is the live YUV I420 buffer the pose detector just ran
    against.  This function MUST run in the same process and call stack
    as `detect_poses`, while the SHM-backed numpy view is still valid.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return None

    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    except Exception:
        return None

    h, w = bgr.shape[:2]
    if h > _POSE_THUMBNAIL_TARGET_HEIGHT:
        scale = _POSE_THUMBNAIL_TARGET_HEIGHT / float(h)
        bgr = cv2.resize(
            bgr,
            (int(w * scale), _POSE_THUMBNAIL_TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
    if not ok:
        return None
    return buf.tobytes()


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
                    # This keeps downstream classifier normalization consistent
                    # regardless of ROI size
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

            # Track and process poses using FULL FRAME dimensions so downstream
            # classifier normalization is consistent regardless of ROI size
            # changes between frames.
            tracked_poses = self.track_poses(
                detected_poses, frame_time, full_frame_width, full_frame_height
            )

            # Send tracked poses to the queue for further processing.
            #
            # We also encode a small webp thumbnail from the YUV frame in
            # this same process, while the SHM-backed `frame` is still
            # valid.  pose_consumer can't reliably read frames from SHM
            # asynchronously (the camera's circular buffer rotates faster
            # than the pose-action lifecycle), so we ship the thumbnail
            # bytes through the queue.
            if tracked_poses:
                thumbnail_bytes: Optional[bytes] = None
                try:
                    thumbnail_bytes = _encode_pose_thumbnail(frame)
                except Exception:
                    logger.debug(
                        f"Failed to encode pose thumbnail for {self.camera_name}",
                        exc_info=True,
                    )

                try:
                    self.tracked_poses_queue.put(
                        (
                            self.camera_name,
                            frame_time,
                            tracked_poses,
                            motion_boxes,
                            regions,
                            thumbnail_bytes,
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
        # Wall-clock seconds a track may go without a matching detection
        # before we release its id. Seconds-based (not frames-based) so
        # behavior is stable across cameras with different detect.fps.
        # See pose.track_max_disappeared_seconds docs and #60 follow-up.
        max_gap_seconds = float(
            getattr(self.pose_config, "track_max_disappeared_seconds", 3.0)
        )

        # First, predict the state of existing tracked poses
        for pose_id, tracked_pose in list(self.tracked_poses.items()):
            tracked_pose.predict()

            # Remove old poses that haven't been updated in a while.
            # Compare against frame_time (wall-clock) rather than the
            # frames-based time_since_update counter.
            last_seen = getattr(tracked_pose, "frame_time", None) or 0.0
            if frame_time - last_seen > max_gap_seconds:
                # Give the activity detector a chance to drop its per-track
                # state (sliding window + hysteresis deque) so the deleted
                # track's history doesn't leak into any future track that
                # happens to reuse the same pose_id string. Pass the owning
                # camera name so detectors keyed by (camera, pose_id) (e.g.
                # shared pool workers) drop the correct bucket.
                detector = getattr(tracked_pose, "active_detector", None)
                if detector is not None:
                    forget = getattr(detector, "forget", None)
                    if callable(forget):
                        try:
                            forget(pose_id, camera=self.camera_name)
                        except TypeError:
                            # Legacy detectors without the camera kwarg.
                            try:
                                forget(pose_id)
                            except Exception as exc:
                                logger.debug(
                                    f"activity_detector.forget({pose_id}) failed: {exc}"
                                )
                        except Exception as exc:
                            logger.debug(
                                f"activity_detector.forget({pose_id}) failed: {exc}"
                            )
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
                # Skip poses with excessive wall-clock gap since last update.
                # Use the same seconds-based threshold as the deletion pass
                # above so matching and deletion share one source of truth.
                t_last_seen = getattr(tpose, "frame_time", None) or 0.0
                if frame_time - t_last_seen > max_gap_seconds:
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

            # Compute normalization diagonal (for the centroid fallback path)
            if frame_width and frame_height:
                diag = (frame_width**2 + frame_height**2) ** 0.5
            else:
                diag = None

            # Thresholds are PoseConfig fields, validated by Pydantic. Read
            # directly -- no literal fallbacks, so config.yml overrides the
            # default cleanly and there is a single source of truth.
            iou_threshold = self.pose_config.track_match_iou_threshold
            centroid_threshold = self.pose_config.track_match_centroid_threshold

            # Score each candidate. Prefer IoU of bboxes (rotation-invariant,
            # survives a standing->lying fall transition). Also compute the
            # centroid distance as a second-chance rescue for cases where
            # the pose detector outputs wildly different bbox shapes between
            # consecutive frames of the same subject (e.g. a 52x156 standing
            # box becoming 71x334 as the detector re-estimates extent -- the
            # IoU drops below threshold but the centroids are still close).
            #
            # `best_match` holds (tid, tpose, score, score_type) where
            # score_type is "iou" (higher is better) or "centroid" (lower
            # is better). An IoU match always wins over a centroid match,
            # but centroid matches still rescue the IoU-below-threshold case.
            best_match = None
            det_xyxy = _bbox_xywh_to_xyxy(det_bbox) if det_bbox is not None else None

            for tid, tpose, t_centroid, t_bbox in candidates:
                # IoU path (preferred when both sides have bbox)
                iou_cleared = False
                if det_xyxy is not None and t_bbox is not None:
                    t_xyxy = _bbox_xywh_to_xyxy(t_bbox)
                    if t_xyxy is not None:
                        try:
                            iou = float(intersection_over_union(det_xyxy, t_xyxy))
                        except Exception:
                            iou = 0.0
                        if iou >= iou_threshold:
                            iou_cleared = True
                            if (
                                best_match is None
                                or best_match[3] != "iou"
                                or iou > best_match[2]
                            ):
                                best_match = (tid, tpose, iou, "iou")

                # Centroid path: runs unconditionally as a rescue whenever
                # IoU did not clear the threshold for this candidate. When
                # IoU already cleared, skip centroid (IoU is authoritative).
                if iou_cleared:
                    continue
                if (
                    det_centroid is not None
                    and t_centroid is not None
                    and diag
                    and diag > 0
                ):
                    try:
                        dist_norm = (
                            float(np.linalg.norm(det_centroid - t_centroid)) / diag
                        )
                    except Exception:
                        continue
                    if dist_norm <= centroid_threshold:
                        # Only accept a centroid candidate if no IoU match
                        # exists yet -- IoU always beats centroid. Among
                        # centroid candidates, lowest normalized distance wins.
                        if best_match is None or (
                            best_match[3] == "centroid" and dist_norm < best_match[2]
                        ):
                            best_match = (tid, tpose, dist_norm, "centroid")

            if best_match is not None:
                matched_id = best_match[0]
                matched_pose = best_match[1]

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

        # Return all currently ALIVE tracked poses, not just the ones updated
        # this frame. The deletion pass at the top of this method has already
        # pruned poses whose wall-clock gap exceeded track_max_disappeared_seconds,
        # so whatever remains in self.tracked_poses is still considered present.
        # Returning stale poses (time_since_update > 0) during brief pose-detector
        # gaps keeps downstream CameraState lifecycles (pose_consumer + the
        # standard tracker fed by the detected_frames_queue push) from cycling
        # create/destroy on every missed frame. Activity classification runs
        # via the freshness-gated `TrackedPose.classify()` call in
        # pose_processing.py, not on this return value, so there is no risk of
        # running classifier inference on stale keypoints.
        return list(self.tracked_poses.values())

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
