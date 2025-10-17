import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request

import cv2
import numpy as np

import frigate.const as const
from frigate.pose_detectors.detection_api import PoseDetectionApi
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    PoseModelTypeEnum,
)

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

# Create a writable directory for MediaPipe models
try:
    MEDIAPIPE_MODEL_DIR = os.path.join(
        tempfile.gettempdir(), "frigate_mediapipe_models"
    )
    os.makedirs(MEDIAPIPE_MODEL_DIR, exist_ok=True)

    # Set environment variable for MediaPipe models
    os.environ["MEDIAPIPE_MODEL_PATH"] = MEDIAPIPE_MODEL_DIR
    logger.info(f"Set MediaPipe model directory to {MEDIAPIPE_MODEL_DIR}")
except Exception as e:
    logger.warning(f"Failed to create MediaPipe model directory: {e}")
    MEDIAPIPE_MODEL_DIR = tempfile.gettempdir()


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

    def __init__(self, detector_config: MediaPipePoseDetectorConfig):
        super().__init__(detector_config)
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

    def detect_raw(self, tensor_input):
        """Run MediaPipe pose detection on input tensor."""
        try:
            self.frame_count += 1
            logger.info(f"detect_raw mediapipe called for frame {self.frame_count}")

            # Ensure debug directory exists
            debug_dir = os.path.join(const.BASE_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # Save raw tensor input
            logger.info(
                f"Raw tensor_input shape: {tensor_input.shape}, dtype: {tensor_input.dtype}"
            )
            if len(tensor_input.shape) == 4:
                # For 4D tensor, save a slice
                try:
                    # Normalize and convert to uint8 for saving
                    tensor_norm = tensor_input[0].astype(np.float32)
                    if np.max(tensor_norm) > 1.0:
                        tensor_norm = tensor_norm / 255.0
                    tensor_viz = (tensor_norm * 255).astype(np.uint8)

                    # If it's a 3-channel image, save directly
                    if tensor_viz.shape[2] == 3:
                        tensor_path = (
                            f"{debug_dir}/tensor_raw_4d_{self.frame_count}.jpg"
                        )
                        cv2.imwrite(tensor_path, tensor_viz)
                        logger.info(f"Saved raw 4D tensor to {tensor_path}")
                except Exception as e:
                    logger.error(f"Failed to save raw 4D tensor: {e}")

            # Convert tensor to image format expected by MediaPipe
            if len(tensor_input.shape) == 4:
                image = tensor_input[0]  # Remove batch dimension
                logger.info(f"Removed batch dimension, new shape: {image.shape}")

                # Save after batch removal
                try:
                    # Normalize for visualization
                    img_norm = image.astype(np.float32)
                    if np.max(img_norm) > 1.0:
                        img_norm = img_norm / 255.0
                    img_viz = (img_norm * 255).astype(np.uint8)

                    if len(img_viz.shape) == 3 and img_viz.shape[2] == 3:
                        img_path = f"{debug_dir}/tensor_after_batch_removal_{self.frame_count}.jpg"
                        cv2.imwrite(img_path, img_viz)
                        logger.info(f"Saved tensor after batch removal to {img_path}")
                except Exception as e:
                    logger.error(f"Failed to save tensor after batch removal: {e}")
            else:
                image = tensor_input
                logger.info(f"Using tensor as-is, shape: {image.shape}")

            # Detect if this is a YUV frame (common in Frigate's pipeline)
            is_yuv_format = False

            # YUV frames typically have different dimensions or are 2D
            if len(image.shape) == 2:
                # This is likely a YUV frame that needs conversion
                is_yuv_format = True
                logger.info(f"Detected 2D YUV frame with shape {image.shape}")

                # Save raw YUV frame (just for visualization)
                try:
                    yuv_path = f"{debug_dir}/yuv_raw_2d_{self.frame_count}.jpg"
                    cv2.imwrite(yuv_path, image)
                    logger.info(f"Saved raw 2D YUV frame to {yuv_path}")
                except Exception as e:
                    logger.error(f"Failed to save raw 2D YUV frame: {e}")

            elif len(image.shape) == 3 and image.shape[0] > image.shape[1] * 1.2:
                # Another way to detect YUV: height is ~1.5x width for I420 format
                is_yuv_format = True
                logger.info(f"Detected 3D YUV frame with shape {image.shape}")

                # Try to save a representation of the 3D YUV frame
                try:
                    # Take first channel for visualization
                    yuv_path = f"{debug_dir}/yuv_raw_3d_channel0_{self.frame_count}.jpg"
                    cv2.imwrite(yuv_path, image[:, :, 0])
                    logger.info(f"Saved raw 3D YUV frame (channel 0) to {yuv_path}")
                except Exception as e:
                    logger.error(f"Failed to save raw 3D YUV frame: {e}")

            # Check for zeros in the tensor (common sign of shared memory issues)
            if np.count_nonzero(image) == 0:
                logger.error(
                    f"TENSOR VALIDATION: Image contains all zeros! Shape: {image.shape}"
                )

                # Create a test pattern to verify image saving works
                test_pattern = np.zeros(image.shape, dtype=np.uint8)
                if len(test_pattern.shape) == 2:
                    # Grayscale test pattern
                    rows, cols = test_pattern.shape
                    test_pattern[rows // 4 : rows // 2, cols // 4 : cols // 2] = (
                        255  # White square
                    )
                    test_pattern[
                        rows // 2 : 3 * rows // 4, cols // 2 : 3 * cols // 4
                    ] = 128  # Gray square
                elif len(test_pattern.shape) == 3:
                    # Color test pattern
                    rows, cols = test_pattern.shape[:2]
                    if test_pattern.shape[2] == 3:
                        # Red square
                        test_pattern[
                            rows // 4 : rows // 2, cols // 4 : cols // 2, 0
                        ] = 255
                        # Green square
                        test_pattern[
                            rows // 2 : 3 * rows // 4, cols // 2 : 3 * cols // 4, 1
                        ] = 255
                        # Blue square
                        test_pattern[
                            rows // 4 : rows // 2, cols // 2 : 3 * cols // 4, 2
                        ] = 255

                test_pattern_path = f"{debug_dir}/test_pattern_{self.frame_count}.jpg"
                cv2.imwrite(test_pattern_path, test_pattern)
                logger.info(
                    f"TENSOR VALIDATION: Saved test pattern to {test_pattern_path}"
                )

                # Save memory buffer details to help diagnose shared memory issues
                logger.info(
                    f"MEMORY VALIDATION: Buffer memory address: {hex(image.__array_interface__['data'][0])}"
                )
                logger.info(
                    f"MEMORY VALIDATION: Buffer strides: {image.__array_interface__.get('strides', 'None')}"
                )
                logger.info(
                    f"MEMORY VALIDATION: Buffer readonly: {image.__array_interface__.get('readonly', False)}"
                )

            # Convert to RGB based on detected format
            if is_yuv_format:
                # Use Frigate's YUV to RGB conversion utility
                from frigate.util.image import yuv_region_2_bgr, yuv_region_2_rgb

                # Analyze YUV data for valid values
                logger.info(
                    f"YUV VALIDATION: Stats - min: {np.min(image)}, max: {np.max(image)}, mean: {np.mean(image)}"
                )
                # Check Y plane values (should be between 16-235 for valid video)
                if len(image.shape) == 2:
                    # For 2D YUV, analyze different sections
                    height = image.shape[0]
                    # Y plane is usually 2/3 of height
                    y_plane = image[: height // 3 * 2, :]
                    uv_plane = image[height // 3 * 2 :, :]
                    logger.info(
                        f"YUV VALIDATION: Y plane - min: {np.min(y_plane)}, max: {np.max(y_plane)}, mean: {np.mean(y_plane)}"
                    )
                    logger.info(
                        f"YUV VALIDATION: UV plane - min: {np.min(uv_plane)}, max: {np.max(uv_plane)}, mean: {np.mean(uv_plane)}"
                    )

                # Create a region covering the entire frame
                height = image.shape[0] // 3 * 2  # YUV height calculation
                width = image.shape[1]
                region = (0, 0, width, height)

                try:
                    logger.info(
                        f"Converting YUV frame to RGB for MediaPipe, YUV shape: {image.shape}, region: {region}"
                    )

                    # Try to save the YUV directly to BGR for visualization
                    try:
                        image_bgr_direct = yuv_region_2_bgr(image, region)
                        bgr_direct_path = (
                            f"{debug_dir}/yuv_to_bgr_direct_{self.frame_count}.jpg"
                        )
                        cv2.imwrite(bgr_direct_path, image_bgr_direct)
                        logger.info(
                            f"Saved direct YUV->BGR conversion to {bgr_direct_path}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to directly convert YUV to BGR: {e}")

                    # Now convert to RGB for MediaPipe processing
                    image_rgb = yuv_region_2_rgb(image, region)

                    # Save the RGB image (converted to BGR for OpenCV) after YUV conversion
                    rgb_after_yuv_path = (
                        f"{debug_dir}/rgb_after_yuv_conversion_{self.frame_count}.jpg"
                    )
                    cv2.imwrite(
                        rgb_after_yuv_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    )
                    logger.info(
                        f"Saved RGB (as BGR) after YUV conversion to {rgb_after_yuv_path}"
                    )

                    logger.info(f"Converted to RGB with shape: {image_rgb.shape}")

                except Exception as e:
                    logger.error(f"Failed to convert YUV to RGB: {e}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return np.zeros((20, 57), dtype=np.float32)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Standard BGR format, convert to RGB
                logger.info(f"Converting BGR to RGB, shape: {image.shape}")
                image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

                # Save the RGB image (converted to BGR for OpenCV) after BGR->RGB conversion
                rgb_after_bgr_path = (
                    f"{debug_dir}/rgb_after_bgr_conversion_{self.frame_count}.jpg"
                )
                cv2.imwrite(
                    rgb_after_bgr_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                )
                logger.info(
                    f"Saved RGB (as BGR) after BGR->RGB conversion to {rgb_after_bgr_path}"
                )

                logger.info(f"Converted to RGB with shape: {image_rgb.shape}")
            else:
                # Unknown format, just use as is
                logger.warning(
                    f"Unknown image format with shape {image.shape}, using as-is"
                )
                image_rgb = image.astype(np.uint8)

                # Try to save unknown format
                try:
                    unknown_format_path = (
                        f"{debug_dir}/unknown_format_{self.frame_count}.jpg"
                    )
                    cv2.imwrite(unknown_format_path, image_rgb)
                    logger.info(f"Saved unknown format image to {unknown_format_path}")
                except Exception as e:
                    logger.error(f"Failed to save unknown format image: {e}")

            # Save the final image that will be sent to MediaPipe (converted to BGR for OpenCV)
            final_input_path = (
                f"{debug_dir}/mediapipe_final_input_{self.frame_count}.jpg"
            )
            try:
                # Convert RGB to BGR for OpenCV's imwrite
                image_bgr_for_save = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(final_input_path, image_bgr_for_save)
                logger.info(
                    f"Saved final MediaPipe input (as BGR) to {final_input_path}"
                )
            except Exception as e:
                logger.error(f"Failed to save final MediaPipe input: {e}")

            # Process the image
            results = self.pose.process(image_rgb)
            logger.info(f"MediaPipe processing complete with results: {results}")

            # Also save the results visualization if landmarks are detected
            if results.pose_landmarks:
                try:
                    # Create a copy of the RGB image for visualization
                    vis_image = image_rgb.copy()
                    # Draw the pose landmarks on the image
                    self.mp_drawing.draw_landmarks(
                        vis_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
                    # Convert to BGR for saving
                    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    # Save the visualization
                    vis_path = f"{debug_dir}/pose_landmarks_{self.frame_count}.jpg"
                    cv2.imwrite(vis_path, vis_image_bgr)
                    logger.info(f"Saved pose landmarks visualization to {vis_path}")
                except Exception as e:
                    logger.error(f"Failed to save pose landmarks visualization: {e}")

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
        """Post-process MediaPipe pose results."""
        poses = []

        if results.pose_landmarks:
            height, width = image_shape[:2]

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # MediaPipe returns 33 landmarks, but we'll use the COCO 17 keypoints
            # Mapping from MediaPipe to COCO format
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

            for mp_idx, coco_idx in mp_to_coco_map.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    confidence = (
                        landmark.visibility
                    )  # MediaPipe uses visibility as confidence

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
            logger.info(f"Detected pose: {pose_output}")
            poses.append(pose_output)

        # Pad to fixed size (20 poses max)
        result = np.zeros((20, 57), dtype=np.float32)
        for i, pose in enumerate(poses[:20]):
            result[i] = pose

        return result
