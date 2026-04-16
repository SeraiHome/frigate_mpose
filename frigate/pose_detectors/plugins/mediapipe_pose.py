import logging
import os
import shutil
import subprocess
import urllib.request

import cv2
import numpy as np

import frigate.const as const
from frigate.const import MODEL_CACHE_DIR
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

# Set to None initially to avoid issues during module loading
mp = None

# MediaPipe model URLs and paths
MP_MODEL_URL_BASE = "https://storage.googleapis.com/mediapipe-assets/"
MP_MODEL_NAMES = {
    0: "pose_landmark_lite.tflite",
    1: "pose_landmark_full.tflite",
    2: "pose_landmark_heavy.tflite",
}
MP_DEFAULT_PATH = (
    "/usr/local/lib/python3.11/dist-packages/mediapipe/modules/pose_landmark/"
)

# Use persistent model cache directory (consistent with other models in the codebase)
MEDIAPIPE_MODEL_DIR = os.path.join(MODEL_CACHE_DIR, "mediapipe")
try:
    os.makedirs(MEDIAPIPE_MODEL_DIR, exist_ok=True)

    # Set environment variable for MediaPipe models
    os.environ["MEDIAPIPE_MODEL_PATH"] = MEDIAPIPE_MODEL_DIR
    logger.info(f"Set MediaPipe model directory to {MEDIAPIPE_MODEL_DIR}")
except Exception as e:
    logger.warning(f"Failed to create MediaPipe model directory: {e}")
    MEDIAPIPE_MODEL_DIR = MODEL_CACHE_DIR


# Function to pre-download a model to our custom directory
def download_model(model_complexity=1):
    """Download MediaPipe pose model to our writable directory."""
    model_name = MP_MODEL_NAMES.get(model_complexity, MP_MODEL_NAMES[1])
    model_url = f"{MP_MODEL_URL_BASE}{model_name}"
    model_path = os.path.join(MEDIAPIPE_MODEL_DIR, model_name)

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


# Function to create a symlink from MediaPipe's expected location to our model
def ensure_model_accessible(model_path, model_complexity=1):
    """Make sure model is accessible at MediaPipe's expected location."""
    if not model_path or not os.path.exists(model_path):
        return False

    model_name = MP_MODEL_NAMES.get(model_complexity, MP_MODEL_NAMES[1])
    mp_expected_path = os.path.join(MP_DEFAULT_PATH, model_name)

    # Check if MediaPipe expected directory exists
    mp_dir = os.path.dirname(mp_expected_path)
    if not os.path.exists(mp_dir):
        logger.warning(f"MediaPipe model directory not found: {mp_dir}")
        return False

    # If model already exists at expected path, we're good
    if os.path.exists(mp_expected_path):
        logger.info(f"MediaPipe model already exists at {mp_expected_path}")
        return True

    # Try to make the model accessible at MediaPipe's expected location
    try:
        # First try to copy the file
        try:
            shutil.copy(model_path, mp_expected_path)
            logger.info(
                f"Copied model to MediaPipe expected location: {mp_expected_path}"
            )
            return True
        except PermissionError:
            logger.warning(
                "Permission denied to copy model file, trying alternatives..."
            )

        # If copy fails, try symbolic link with elevated permissions
        try:
            # Create symlink using sudo
            cmd = ["sudo", "ln", "-sf", model_path, mp_expected_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Created symlink to model at {mp_expected_path}")
                return True
            else:
                logger.warning(f"Failed to create symlink with sudo: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error creating symlink: {e}")

        # As a last resort, see if we can make the directory writable
        try:
            os.chmod(mp_dir, 0o777)  # Try to make writable by all
            shutil.copy(model_path, mp_expected_path)
            logger.info(
                f"Made directory writable and copied model to {mp_expected_path}"
            )
            return True
        except Exception as e:
            logger.warning(f"Could not make directory writable: {e}")

        logger.error("All attempts to make model accessible failed")
        return False
    except Exception as e:
        logger.error(f"Error ensuring model accessibility: {e}")
        return False


# Defer MediaPipe import to help with debugging
def initialize_mediapipe():
    global mp
    if mp is not None:
        return mp

    try:
        import mediapipe as mp_local

        logger.info("MediaPipe is available for pose detection")
        mp = mp_local
        return mp
    except ImportError as e:
        logger.warning(
            f"MediaPipe not available: {e}. MediaPipe pose detection will not work."
        )
        return None


class MediaPipePoseDetectorConfig(BasePoseDetectorConfig):
    type: str = "mediapipe"
    static_image_mode: bool = False
    model_complexity: int = 1  # 0, 1, or 2
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str = None  # Optional path to a pre-downloaded model


class MediaPipePoseApi(PoseDetectionApi):
    type_key = "mediapipe"
    supported_models = [PoseModelTypeEnum.mediapipe]

    def __init__(self, detector_config: MediaPipePoseDetectorConfig, camera_name=None):
        super().__init__(detector_config, camera_name)
        self.frame_count = 0
        # Initialize MediaPipe using the deferred import function
        mp_instance = initialize_mediapipe()
        if mp_instance is None:
            raise ImportError("MediaPipe is required for MediaPipe pose detection")

        self.mp_pose = mp_instance.solutions.pose
        self.mp_drawing = mp_instance.solutions.drawing_utils

        model_complexity = getattr(detector_config, "model_complexity", 1)

        # First, check for a custom model path
        custom_model_path = getattr(detector_config, "model_path", None)
        if custom_model_path and os.path.exists(custom_model_path):
            logger.info(f"Using custom MediaPipe model at: {custom_model_path}")
            # Copy the model to our writable directory if needed
            target_path = os.path.join(
                MEDIAPIPE_MODEL_DIR, os.path.basename(custom_model_path)
            )
            if not os.path.exists(target_path):
                try:
                    shutil.copy(custom_model_path, MEDIAPIPE_MODEL_DIR)
                    logger.info(f"Copied custom model to: {target_path}")
                except Exception as e:
                    logger.warning(f"Failed to copy custom model: {e}")
            model_path = custom_model_path
        else:
            # If no custom model, pre-download the default model
            model_path = download_model(model_complexity)

        # Ensure model is accessible at MediaPipe's expected location
        if model_path:
            ensure_model_accessible(model_path, model_complexity)
        self.min_detection_confidence = getattr(
            detector_config, "min_detection_confidence", 0.5
        )
        try:
            # Initialize MediaPipe Pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=getattr(detector_config, "static_image_mode", False),
                model_complexity=model_complexity,
                smooth_landmarks=getattr(detector_config, "smooth_landmarks", True),
                enable_segmentation=getattr(
                    detector_config, "enable_segmentation", False
                ),
                smooth_segmentation=getattr(
                    detector_config, "smooth_segmentation", True
                ),
                min_detection_confidence=getattr(
                    detector_config, "min_detection_confidence", 0.5
                ),
                min_tracking_confidence=getattr(
                    detector_config, "min_tracking_confidence", 0.5
                ),
            )
            logger.info("MediaPipe pose detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe pose detector: {e}")
            raise RuntimeError(f"Failed to initialize MediaPipe: {e}")

        logger.info("MediaPipe pose detector initialized")

    def detect_raw(self, tensor_input, camera_name=None):
        """Run MediaPipe pose detection on input tensor.

        Optimized with reduced shape checks and unified YUV detection.
        """
        try:
            self.frame_count += 1

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

                elif len(image.shape) == 3 and image.shape[2] == 3:
                    # Input is already RGB (per config.yml input_pixel_format: rgb)
                    image_rgb = image.astype(np.uint8)
                else:
                    # Unknown format, use as-is with warning
                    logger.warning(
                        f"Unknown image format with shape {image.shape}, using as-is"
                    )
                    image_rgb = image.astype(np.uint8)

                # Process the image with MediaPipe (expects RGB input)
                results = self.pose.process(image_rgb)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                return np.zeros((20, 57), dtype=np.float32)

            # Save visualization if landmarks are detected and debug dir exists
            debug_dir = os.path.join(const.BASE_DIR, "debug")
            if results.pose_landmarks and os.path.exists(debug_dir):
                try:
                    # Create a copy of the RGB image for visualization
                    vis_image = image_rgb.copy()

                    # Draw the pose landmarks on the image
                    self.mp_drawing.draw_landmarks(
                        vis_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
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
            return self._postprocess_mediapipe_pose(results, image.shape)

        except PermissionError as e:
            # Handle permission errors specifically with helpful message
            logger.error(f"MediaPipe permission error: {e}")
            logger.error("MediaPipe cannot download models due to permission issues.")
            logger.error(
                "Try setting a custom model_path or run with elevated permissions."
            )
            return np.zeros((20, 57), dtype=np.float32)
        except Exception as e:
            logger.error(f"MediaPipe pose detection failed: {e}")
            return np.zeros((20, 57), dtype=np.float32)

    def _postprocess_mediapipe_pose(self, results, image_shape):
        """Post-process MediaPipe pose results using optimized tensor utilities."""
        # Start with pre-allocated output array
        result = create_pose_output_batch()

        if not results.pose_landmarks:
            return result

        height, width = image_shape[:2]

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # MediaPipe returns 33 landmarks, but we'll use the COCO 17 keypoints
        # Mapping from MediaPipe to COCO format (static, could be class-level)
        mp_to_coco_map = {
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

        keypoints = np.zeros(51, dtype=np.float32)  # 17 keypoints * 3
        valid_points = []

        for mp_idx, coco_idx in mp_to_coco_map.items():
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

        # Overall pose confidence (average of visible keypoints)
        conf_vals = keypoints[2::3]
        visible_mask = conf_vals > 0.3
        pose_confidence = (
            float(np.mean(conf_vals[visible_mask])) if np.any(visible_mask) else 0.0
        )

        # Use helper to create standardized output
        result[0] = create_pose_output(0, pose_confidence, keypoints, bbox)

        return result
