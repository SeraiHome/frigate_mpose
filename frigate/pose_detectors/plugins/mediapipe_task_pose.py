import logging
import os
import tempfile
import urllib.request
from time import time

import cv2
import numpy as np

import frigate.const as const
from frigate.pose_detection.tensor_utils import (
    create_pose_output,
    create_pose_output_batch,
    is_yuv_frame,
)
from frigate.pose_detectors.detection_api import PoseDetectionApi
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    PoseModelTypeEnum,
)
from frigate.util.image import yuv_region_2_rgb

logger = logging.getLogger(__name__)

# MediaPipe task model URLs and paths - updated with correct URLs
MP_TASK_MODEL_VARIANTS = {
    0: "pose_landmarker_lite",  # Lite model
    1: "pose_landmarker_full",  # Full model
    2: "pose_landmarker_heavy",  # Heavy model
}
MP_MODEL_URL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"

# Create a writable directory for MediaPipe models
try:
    MEDIAPIPE_MODEL_DIR = os.path.join(
        tempfile.gettempdir(), "frigate_mediapipe_models"
    )
    os.makedirs(MEDIAPIPE_MODEL_DIR, exist_ok=True)
    logger.info(f"Set MediaPipe model directory to {MEDIAPIPE_MODEL_DIR}")
except Exception as e:
    logger.warning(f"Failed to create MediaPipe model directory: {e}")
    MEDIAPIPE_MODEL_DIR = tempfile.gettempdir()


# Function to pre-download a model to our custom directory
def download_model(model_complexity=1):
    """Download MediaPipe pose task model to our writable directory."""
    model_variant = MP_TASK_MODEL_VARIANTS.get(
        model_complexity, MP_TASK_MODEL_VARIANTS[1]
    )
    model_url = (
        f"{MP_MODEL_URL_BASE}{model_variant}/float16/latest/{model_variant}.task"
    )
    model_path = os.path.join(MEDIAPIPE_MODEL_DIR, f"{model_variant}.task")

    # Only download if it doesn't exist
    if not os.path.exists(model_path):
        logger.info(f"Downloading MediaPipe model from {model_url} to {model_path}")
        try:
            with urllib.request.urlopen(model_url) as response:
                with open(model_path, "wb") as out_file:
                    out_file.write(response.read())
            logger.info(f"Downloaded MediaPipe model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download MediaPipe model: {e}")
            return None
    else:
        logger.info(f"MediaPipe model already exists at {model_path}")

    return model_path


class MediaPipeTaskPoseDetectorConfig(BasePoseDetectorConfig):
    type: str = "mediapipe_task"
    # Parameters for the Task API
    model_complexity: int = 1  # 0: Lite, 1: Full, 2: Heavy
    num_poses: int = 1  # Maximum number of poses to detect
    min_detection_confidence: float = 0.5  # Minimum confidence for detection
    min_pose_presence_confidence: float = 0.5  # Minimum confidence for pose presence
    min_tracking_confidence: float = 0.5  # Minimum confidence for tracking
    output_segmentation_masks: bool = False  # Whether to output segmentation masks
    running_mode: str = "image"  # "image", "video", or "live_stream"
    model_path: str = None  # Optional path to a pre-downloaded model


class MediaPipeTaskPoseApi(PoseDetectionApi):
    type_key = "mediapipe_task"
    supported_models = [PoseModelTypeEnum.mediapipe]

    def __init__(
        self, detector_config: MediaPipeTaskPoseDetectorConfig, camera_name=None
    ):
        super().__init__(detector_config, camera_name)
        self.frame_count = 0
        self.t0 = time()
        # Get configuration parameters
        model_complexity = getattr(detector_config, "model_complexity", 1)
        self.min_detection_confidence = getattr(
            detector_config, "min_detection_confidence", 0.5
        )
        self.running_mode = getattr(detector_config, "running_mode", "video")

        # Check for custom model path
        custom_model_path = getattr(detector_config, "model_path", None)
        if custom_model_path and os.path.exists(custom_model_path):
            logger.info(f"Using custom MediaPipe model at: {custom_model_path}")
            model_path = custom_model_path
        else:
            # If no custom model, download the default model based on complexity
            model_path = download_model(model_complexity)

        if model_path is None or not os.path.exists(model_path):
            raise RuntimeError(f"MediaPipe model not available at {model_path}")

        try:
            # Import the required MediaPipe modules
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Store drawing utilities for visualization (optional, not all builds have it)
            try:
                self.mp_drawing = mp.solutions.drawing_utils
            except AttributeError:
                self.mp_drawing = None
                logger.info("mp.solutions.drawing_utils not available, debug visualization disabled")

            # Import necessary classes for landmark conversion (optional, debug only)
            try:
                from mediapipe.framework.formats import landmark_pb2
                self.landmark_pb2 = landmark_pb2
            except ImportError:
                self.landmark_pb2 = None
                logger.info("mediapipe.framework not available, landmark proto conversion disabled")

            # Attempt to preload an EdgeTPU delegate if configured for this detector.
            # accel = getattr(detector_config, "accelerator", None)
            # device = getattr(detector_config, "device", None)
            # if accel and str(accel).lower() in ("edgetpu", "coral"):
            #     try:
            #         # Try to load the EdgeTPU delegate early so MediaPipe's
            #         # TFLite runtime can pick it up when creating interpreters.
            #         from tflite_runtime.interpreter import load_delegate

            #         if device:
            #             load_delegate("libedgetpu.so.1.0", {"device": device})
            #         else:
            #             load_delegate("libedgetpu.so.1.0")
            #         logger.info("Preloaded EdgeTPU delegate via tflite_runtime")
            #     except Exception as e:
            #         logger.warning(f"Could not preload EdgeTPU delegate: {e}")

            # Create base options for the task
            base_options = python.BaseOptions(model_asset_path=model_path)
            # Create pose landmarker options
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=getattr(vision.RunningMode, self.running_mode.upper()),
                num_poses=getattr(detector_config, "num_poses", 1),
                min_pose_detection_confidence=getattr(
                    detector_config, "min_detection_confidence", 0.5
                ),
                min_pose_presence_confidence=getattr(
                    detector_config, "min_pose_presence_confidence", 0.5
                ),
                min_tracking_confidence=getattr(
                    detector_config, "min_tracking_confidence", 0.5
                ),
                output_segmentation_masks=getattr(
                    detector_config, "output_segmentation_masks", False
                ),
            )

            # Create the pose landmarker
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                pose_options
            )
            logger.info("MediaPipe task pose landmarker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe task pose landmarker: {e}")
            raise RuntimeError(f"Failed to initialize MediaPipe task API: {e}")

        logger.info("MediaPipe task pose detector initialized")

    def detect_raw(self, tensor_input, camera_name=None):
        """Run MediaPipe pose detection on input tensor using the Task API.

        Optimized with reduced shape checks and unified YUV detection.
        """
        try:
            self.frame_count += 1
            self.frame_ts = int((time() - self.t0) * 1000)

            # Remove batch dimension if present
            image = tensor_input[0] if tensor_input.ndim == 4 else tensor_input

            # Process input according to its format
            try:
                # Fast path: already in RGB format (3D with 3 channels)
                if image.ndim == 3 and image.shape[2] == 3 and not is_yuv_frame(image):
                    image_rgb = image.astype(np.uint8)

                # Handle YUV conversion - create square regions to avoid dimension mismatch
                elif is_yuv_frame(image):
                    if image.ndim == 2:
                        # For I420 format, Y plane height is 2/3 of total height
                        y_height = int(image.shape[0] * 2 / 3)
                        frame_width = image.shape[1]
                    else:
                        y_height = image.shape[0]
                        frame_width = image.shape[1]

                    # Calculate dimensions for a square region
                    square_size = min(frame_width, y_height)

                    # Center the square region
                    x_center = frame_width // 2
                    y_center = y_height // 2
                    half_size = square_size // 2

                    # Create square region centered in the frame
                    region = (
                        max(0, x_center - square_size // 2),  # x_min
                        max(0, y_center - square_size // 2),  # y_min
                        min(frame_width, x_center + square_size // 2),  # x_max
                        min(y_height, y_center + square_size // 2),  # y_max
                    )

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

                    # Log dimensions for debugging
                    logger.debug(
                        f"2D YUV frame detected: shape={image.shape}, Y height={y_height}, width={frame_width}, square region={region}"
                    )

                    # Convert YUV to RGB with square region
                    image_rgb = yuv_region_2_rgb(image, region)

                elif len(image.shape) == 3 and image.shape[0] > image.shape[1] * 1.2:
                    # 3D YUV format (possibly NV12 or I420 in 3D array)
                    frame_height = image.shape[0]
                    frame_width = image.shape[1]

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

                    # Log dimensions for debugging
                    logger.debug(
                        f"3D YUV frame detected: shape={image.shape}, height={frame_height}, width={frame_width}, square region={region}"
                    )

                    # Convert YUV to RGB with square region
                    image_rgb = yuv_region_2_rgb(image, region)

                else:
                    # Unknown format, use as-is with warning
                    logger.warning(
                        f"Unknown image format with shape {image.shape}, using as-is"
                    )
                    image_rgb = image.astype(np.uint8)

                # Create a MediaPipe Image object with the correct format from top-level mediapipe
                # Import mediapipe as mp is already done at initialization
                import mediapipe as mp

                # Create the image with RGB format specified
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Process the image with the pose landmarker

                if self.running_mode.upper() == "VIDEO":
                    results = self.pose_landmarker.detect_for_video(
                        mp_image, self.frame_ts
                    )
                else:
                    results = self.pose_landmarker.detect(mp_image)

            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                return np.zeros((20, 57), dtype=np.float32)

            # Save visualization if landmarks are detected and debug dir exists
            debug_dir = os.path.join(const.BASE_DIR, "debug")
            if self.landmark_pb2 and self.mp_drawing:  # results.pose_landmarks and os.path.exists(debug_dir):
                try:
                    # Create a copy of the RGB image for visualization
                    vis_image = image_rgb.copy()

                    # Convert the task API landmarks to the format used by drawing_utils
                    for pose_landmarks in results.pose_landmarks:
                        # Convert to proto landmarks for visualization
                        landmark_list = self.landmark_pb2.NormalizedLandmarkList()
                        for landmark in pose_landmarks:
                            landmark_proto = self.landmark_pb2.NormalizedLandmark()
                            landmark_proto.x = landmark.x
                            landmark_proto.y = landmark.y
                            landmark_proto.z = landmark.z
                            landmark_proto.visibility = landmark.visibility
                            landmark_list.landmark.append(landmark_proto)

                        # Draw landmarks on the image
                        connections = [
                            (0, 1),
                            (0, 4),
                            (1, 2),
                            (2, 3),
                            (3, 7),
                            (4, 5),
                            (5, 6),
                            (6, 8),
                            (9, 10),
                            (11, 12),
                            (11, 13),
                            (13, 15),
                            (12, 14),
                            (14, 16),
                            (11, 23),
                            (12, 24),
                            (23, 24),
                            (23, 25),
                            (24, 26),
                            (25, 27),
                            (26, 28),
                            (27, 29),
                            (28, 30),
                            (29, 31),
                            (30, 32),
                        ]
                        if self.mp_drawing:
                            self.mp_drawing.draw_landmarks(
                                vis_image, landmark_list, connections
                            )

                    # Convert RGB to BGR for OpenCV's imwrite
                    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

                    # Use the camera_name parameter first (if provided), then fall back to self.camera_name
                    # This ensures consistency whether camera_name is passed via initialization or detect_raw
                    current_camera = (
                        camera_name if camera_name is not None else self.camera_name
                    )

                    # Save the visualization with camera name if available
                    if current_camera:
                        vis_path = f"{debug_dir}/pose_landmarks_{current_camera}_{self.frame_count}.jpg"
                    else:
                        vis_path = f"{debug_dir}/pose_landmarks_{self.frame_count}.jpg"
                    cv2.imwrite(vis_path, vis_image_bgr)
                    logger.debug(f"Saved pose visualization to {vis_path}")
                except Exception as e:
                    logger.debug(f"Failed to save pose landmarks visualization: {e}")

            # Post-process MediaPipe output
            return self._postprocess_mediapipe_pose(results, image_rgb.shape)

        except PermissionError as e:
            # Handle permission errors specifically with helpful message
            logger.error(f"MediaPipe permission error: {e}")
            logger.error("MediaPipe cannot download models due to permission issues.")
            logger.error(
                "Try setting a custom model_path or run with elevated permissions."
            )
            return create_pose_output_batch()
        except Exception as e:
            logger.error(f"MediaPipe pose detection failed: {e}")
            return create_pose_output_batch()

    # Class-level mapping from MediaPipe to COCO format (computed once)
    _MP_TO_COCO_MAP = {
        0: 0,  # nose
        2: 1,  # left_eye
        5: 2,  # right_eye
        7: 3,  # left_ear
        8: 4,  # right_ear
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10,  # right_wrist
        23: 11,  # left_hip
        24: 12,  # right_hip
        25: 13,  # left_knee
        26: 14,  # right_knee
        27: 15,  # left_ankle
        28: 16,  # right_ankle
    }

    def _postprocess_mediapipe_pose(self, results, image_shape):
        """Post-process MediaPipe task API pose results using optimized tensor utilities."""
        # Start with pre-allocated output array
        result = create_pose_output_batch()

        if not results.pose_landmarks:
            return result

        height, width = image_shape[:2]

        # Process each detected pose (up to max 20)
        for pose_idx, landmarks in enumerate(results.pose_landmarks[:20]):
            keypoints = np.zeros(51, dtype=np.float32)  # 17 keypoints * 3
            valid_points = []

            for mp_idx, coco_idx in self._MP_TO_COCO_MAP.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    confidence = landmark.visibility

                    base_idx = coco_idx * 3
                    keypoints[base_idx] = x
                    keypoints[base_idx + 1] = y
                    keypoints[base_idx + 2] = confidence

                    if confidence > self.min_detection_confidence:
                        valid_points.append([x, y])

            # Calculate bounding box from valid keypoints
            bbox = np.array([0, 0, 0, 0], dtype=np.float32)
            if valid_points:
                valid_arr = np.array(valid_points, dtype=np.float32)
                x_min, y_min = np.min(valid_arr, axis=0)
                x_max, y_max = np.max(valid_arr, axis=0)
                bbox = np.array(
                    [x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32
                )

            # Overall pose confidence
            if hasattr(results, "pose_score") and results.pose_score:
                pose_confidence = float(results.pose_score[pose_idx])
            else:
                conf_vals = keypoints[2::3]
                visible_mask = conf_vals > 0.3
                pose_confidence = (
                    float(np.mean(conf_vals[visible_mask]))
                    if np.any(visible_mask)
                    else 0.0
                )

            # Use helper to create standardized output
            result[pose_idx] = create_pose_output(
                pose_idx, pose_confidence, keypoints, bbox
            )

        return result
