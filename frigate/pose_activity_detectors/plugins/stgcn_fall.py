"""
STGCN Fall Detector Plugin

This module provides a Spatial Temporal Graph Convolutional Network based
fall detector that processes pose keypoints to detect falling actions.
"""

import logging
import os
from collections import deque
from typing import Tuple

import numpy as np
import torch

from frigate.events.pose_types import PoseActionTypeEnum
from frigate.pose_activity_detectors import register_detector
from frigate.pose_activity_detectors.base import PoseActivityDetector
from frigate.pose_detection.tensor_utils import (
    COCO_NUM_KEYPOINTS,
    KEYPOINT_DIMS,
    KINETICS_NUM_KEYPOINTS,
    ensure_keypoints_2d,
)

logger = logging.getLogger(__name__)

# Try to import TFLite interpreter following Frigate's pattern
try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ModuleNotFoundError:
        logger.warning("TFLite not available, fall detection will be disabled")
        Interpreter = None


def np_softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class STGCNFallDetector(PoseActivityDetector):
    """
    STGCN-based fall detector that processes keypoints directly.

    This detector uses a Spatial Temporal Graph Convolutional Network (ST-GCN)
    to detect falls from pose keypoints.
    """

    def __init__(self, model_path: str = "", num_threads: int = 3, **kwargs):
        """
        Initialize the STGCN fall detector.

        Args:
            model_path: Path to the TFLite model for fall detection
            num_threads: Number of threads to use for inference
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.initialized = False
        # Load confidence threshold from the camera/activity_detector config.
        # Default to 0.5 to preserve previous behaviour if not configured.
        self.confidence_threshold = float(kwargs.get("confidence_threshold", 0.5))

        # Define potential model paths to try - search in multiple locations
        self.model_paths = [
            model_path,
        ]

        # Filter out empty paths
        self.model_paths = [p for p in self.model_paths if p]

        logger.info(f"Searching for STGCN models in: {', '.join(self.model_paths)}")

        # Check if TFLite interpreter is available
        if Interpreter is None:
            logger.error("TFLite not available, fall detection will be disabled")
            return

        # Verify at least one model path
        if not self.model_paths:
            logger.error("No model paths specified, fall detection will be disabled")
            return

        # Try to initialize with each model path until one works
        for path in self.model_paths:
            if not os.path.exists(path):
                logger.debug(f"Fall detection model not found at {path}")
                continue

            try:
                logger.info(f"Attempting to load fall detection model from {path}")
                # Try to use EdgeTPU if device parameter is passed
                if "device" in kwargs:
                    try:
                        from tflite_runtime.interpreter import load_delegate

                        edge_tpu_delegate = load_delegate(
                            "libedgetpu.so.1.0",
                            {"device": kwargs["device"]},
                            # "libedgetpu.so.1",
                            # {"device": "usb"},
                        )
                        self.interpreter = Interpreter(
                            model_path=path, experimental_delegates=[edge_tpu_delegate]
                        )
                        logger.info(
                            f"Successfully loaded STGCN model on EdgeTPU: {path}"
                        )
                    except (ImportError, ValueError) as e:
                        logger.warning(
                            f"EdgeTPU load failed ({e}), falling back to CPU"
                        )
                        self.interpreter = Interpreter(
                            model_path=path, num_threads=num_threads
                        )
                else:
                    # Standard CPU inference
                    self.interpreter = Interpreter(
                        model_path=path, num_threads=num_threads
                    )
                    logger.info(f"Successfully loaded STGCN model on CPU: {path}")
                self.interpreter.allocate_tensors()

                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                # Get tensor shapes
                _, self.channels, self.window_size, self.num_points, self.num_people = (
                    self.input_details[0]["shape"]
                )
                self.input_index = self.input_details[0]["index"]
                # Allow an override for the effective window size for experiments.
                # If not provided, fall back to the model's window size.
                self.effective_window_size = int(
                    kwargs.get("effective_window_size", self.window_size)
                )
                if self.effective_window_size <= 0:
                    self.effective_window_size = int(self.window_size)
                # Clamp to model window size
                if self.effective_window_size > int(self.window_size):
                    logger.debug(
                        f"effective_window_size ({self.effective_window_size}) > model window_size ({self.window_size}), clamping"
                    )
                    self.effective_window_size = int(self.window_size)

                # Initialize pose history buffer
                self.reset()
                self.model_path = path  # Update model path to the one that worked
                self.initialized = True
                logger.info(
                    f"STGCN fall detector successfully initialized with model {path} window_size is {self.window_size} (effective={self.effective_window_size})"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to initialize STGCN fall detector with {path}: {e}"
                )
                continue

    def reset(self):
        """Reset the pose history buffer."""
        if (
            not hasattr(self, "channels")
            or not hasattr(self, "window_size")
            or not hasattr(self, "num_points")
            or not hasattr(self, "num_people")
        ):
            return

        initial_poses = torch.zeros(
            (self.window_size, self.channels, 1, self.num_points, self.num_people)
        )
        self.poses_history = deque(list(initial_poses), maxlen=int(self.window_size))
        self.history_idx = 0

    def add_to_history(self, pose_tensor: torch.Tensor):
        """
        Add a new pose to the history.

        Args:
            pose_tensor: Tensor of shape (channels, 1, num_points, num_people)
        """
        if not self.initialized:
            return

        if self.history_idx < self.window_size:
            self.poses_history[self.history_idx] = pose_tensor
        else:
            self.poses_history.append(pose_tensor)

        self.history_idx += 1

    def coco_to_kinetics(self, coco_pose):
        """
        Convert COCO keypoints (17, 3) to Kinetics keypoints (18, 3)

        Args:
            coco_pose (np.ndarray): COCO keypoints array of shape (17, 3)

        Returns:
            np.ndarray: Kinetics keypoints array of shape (18, 3)
        """
        kinetics_pose = np.zeros((18, 3), dtype=np.float32)

        # Direct mappings
        kinetics_pose[0] = coco_pose[0]  # Nose
        kinetics_pose[2] = coco_pose[6]  # R Shoulder
        kinetics_pose[3] = coco_pose[8]  # R Elbow
        kinetics_pose[4] = coco_pose[10]  # R Wrist
        kinetics_pose[5] = coco_pose[5]  # L Shoulder
        kinetics_pose[6] = coco_pose[7]  # L Elbow
        kinetics_pose[7] = coco_pose[9]  # L Wrist
        kinetics_pose[8] = coco_pose[12]  # R Hip
        kinetics_pose[9] = coco_pose[14]  # R Knee
        kinetics_pose[10] = coco_pose[16]  # R Ankle
        kinetics_pose[11] = coco_pose[11]  # L Hip
        kinetics_pose[12] = coco_pose[13]  # L Knee
        kinetics_pose[13] = coco_pose[15]  # L Ankle
        kinetics_pose[14] = coco_pose[2]  # R Eye
        kinetics_pose[15] = coco_pose[1]  # L Eye
        kinetics_pose[16] = coco_pose[4]  # R Ear
        kinetics_pose[17] = coco_pose[3]  # L Ear

        # Estimate neck (midpoint between shoulders)
        left_shoulder = coco_pose[5]
        right_shoulder = coco_pose[6]
        kinetics_pose[1, :2] = (left_shoulder[:2] + right_shoulder[:2]) / 2
        kinetics_pose[1, 2] = (
            left_shoulder[2] + right_shoulder[2]
        ) / 2  # average confidence

        return kinetics_pose

    def keypoints_to_tensor(
        self, keypoints: np.ndarray, frame_width: int = None, frame_height: int = None
    ) -> torch.Tensor:
        """
        Convert keypoints to tensor format expected by STGCN model.

        Optimized to minimize unnecessary copies and shape checks.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]
            frame_width: Width of the frame in pixels, used for normalization
            frame_height: Height of the frame in pixels, used for normalization

        Returns:
            torch.Tensor of shape (channels, 1, num_points, num_people)
        """
        if not self.initialized:
            logger.warning("STGCN detector not initialized, returning zero tensor")
            return torch.zeros((self.channels, 1, self.num_points, self.num_people))

        # Use helper to ensure keypoints are in canonical 2D shape
        keypoints_2d = ensure_keypoints_2d(keypoints, COCO_NUM_KEYPOINTS)

        # Log the incoming keypoints for debugging
        logger.debug(
            f"Raw keypoints shape: {keypoints_2d.shape}, non-zero coords: {np.count_nonzero(keypoints_2d[:, :2])}"
        )

        # Replace NaN values with zeros (in-place if already float32)
        if keypoints_2d.dtype != np.float32:
            keypoints_copy = keypoints_2d.astype(np.float32)
        else:
            keypoints_copy = keypoints_2d.copy()

        np.nan_to_num(keypoints_copy, copy=False)

        # Map COCO 17-keypoint format to Kinetics 18-keypoint format if needed
        if (
            keypoints_copy.shape[0] == COCO_NUM_KEYPOINTS
            and self.num_points == KINETICS_NUM_KEYPOINTS
        ):
            logger.debug(
                f"Converting 17 COCO keypoints to {self.num_points} STGCN keypoints"
            )
            keypoints_copy = self.coco_to_kinetics(keypoints_copy)
        # For other formats, pad or truncate to expected number of keypoints
        elif keypoints_copy.shape[0] != self.num_points:
            logger.debug(
                f"Keypoint count mismatch: got {keypoints_copy.shape[0]}, need {self.num_points}"
            )
            if keypoints_copy.shape[0] < self.num_points:
                # Pad with zeros if we don't have enough points
                pad_size = self.num_points - keypoints_copy.shape[0]
                keypoints_copy = np.vstack(
                    [
                        keypoints_copy,
                        np.zeros((pad_size, KEYPOINT_DIMS), dtype=np.float32),
                    ]
                )
            else:
                # Truncate if we have too many points
                keypoints_copy = keypoints_copy[: self.num_points]

        # Check if keypoints contain valid values (non-zero coordinates)
        non_zero_coords = np.count_nonzero(keypoints_copy[:, :2])
        if non_zero_coords < 5:  # Require at least 5 non-zero coordinates
            logger.debug(
                f"Too few non-zero coordinates ({non_zero_coords}), using default pose"
            )
            keypoints_copy = self._create_default_pose_keypoints()
            logger.debug("Using default pose keypoints")

        # Get useful statistics about keypoints for debugging
        x_values = keypoints_copy[:, 0]
        y_values = keypoints_copy[:, 1]
        conf_values = keypoints_copy[:, 2]
        logger.debug(
            f"Keypoint stats: x_range=[{np.min(x_values):.1f}, {np.max(x_values):.1f}], "
            f"y_range=[{np.min(y_values):.1f}, {np.max(y_values):.1f}], "
            f"avg_conf={np.mean(conf_values):.2f}"
        )

        # Create a normalized copy of the keypoints
        normalized_keypoints = keypoints_copy.copy()

        # Use frame dimensions if provided, otherwise use keypoint min/max
        if (
            frame_width is not None
            and frame_height is not None
            and frame_width > 0
            and frame_height > 0
        ):
            logger.debug(
                f"Normalizing keypoints using frame dimensions: {frame_width}x{frame_height}"
            )
            # Use absolute frame dimensions for normalization
            x_min, y_min = 0, 0
            x_range, y_range = frame_width, frame_height
        else:
            # Fall back to keypoint-based normalization
            logger.debug(
                "Frame dimensions not provided, using keypoint bounds for normalization"
            )
            x_min, x_max = np.min(x_values), np.max(x_values)
            y_min, y_max = np.min(y_values), np.max(y_values)

            # Ensure we have non-zero ranges and valid min/max to avoid division by zero
            # Use absolute dimensions if range is too small (prevents division by small numbers)
            x_range = max(100, x_max - x_min) if x_max > x_min else 100
            y_range = max(100, y_max - y_min) if y_max > y_min else 100

        # Normalize coordinates to [0, 1] range
        normalized_keypoints[:, 0] = (keypoints_copy[:, 0] - x_min) / x_range
        normalized_keypoints[:, 1] = (keypoints_copy[:, 1] - y_min) / y_range

        # Ensure confidence values are properly clipped to valid range [0, 1]
        normalized_keypoints[:, 2] = np.clip(keypoints_copy[:, 2], 0, 1)

        # Log the normalized keypoints statistics for debugging at debug level
        logger.debug(
            f"Normalized keypoint stats: "
            f"x_range=[{np.min(normalized_keypoints[:, 0]):.2f}, {np.max(normalized_keypoints[:, 0]):.2f}], "
            f"y_range=[{np.min(normalized_keypoints[:, 1]):.2f}, {np.max(normalized_keypoints[:, 1]):.2f}]"
        )

        # Create a tensor with the right shape
        try:
            # Initialize with zeros to ensure all values are set properly
            torch_pose = torch.zeros(
                (self.channels, 1, self.num_points, self.num_people)
            )

            # Convert the normalized keypoints to a tensor with explicit dtype
            keypoints_tensor = torch.from_numpy(normalized_keypoints.astype(np.float32))

            # Explicitly set each channel to ensure correct format
            # Center coordinates around 0 for better model performance
            torch_pose[0, 0, :, 0] = (
                keypoints_tensor[:, 0] - 0.5
            )  # X coords centered at 0
            torch_pose[1, 0, :, 0] = (
                keypoints_tensor[:, 1] - 0.5
            )  # Y coords centered at 0
            torch_pose[2, 0, :, 0] = keypoints_tensor[:, 2]  # Confidence values

            # Verify tensor has non-zero values
            non_zero = torch.count_nonzero(torch_pose).item()
            logger.debug(f"Created pose tensor with {non_zero} non-zero values")

            # Log a sample of the tensor values for debugging at debug level only
            logger.debug(f"Pose tensor sample: {torch_pose[:, 0, 0:3, 0]}")

            return torch_pose

        except Exception as e:
            logger.error(f"Error creating pose tensor: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return torch.zeros((self.channels, 1, self.num_points, self.num_people))

    def _create_default_pose_keypoints(self) -> np.ndarray:
        """
        Create a default pose keypoint set when valid keypoints aren't available.
        This creates a simple stick figure pose spread across the frame.

        Returns:
            np.ndarray: Array of default keypoints
        """
        # Create a basic human-like pose structure
        default_pose = np.zeros((self.num_points, 3))

        # We'll use a simplified model with key points for head, shoulders,
        # elbows, wrists, hips, knees, and ankles

        # Head (center top)
        default_pose[0] = [0.5, 0.1, 0.8]

        # Shoulders
        default_pose[1] = [0.4, 0.2, 0.8]  # Left shoulder
        default_pose[2] = [0.6, 0.2, 0.8]  # Right shoulder

        # Elbows
        default_pose[3] = [0.3, 0.3, 0.7]  # Left elbow
        default_pose[4] = [0.7, 0.3, 0.7]  # Right elbow

        # Wrists
        default_pose[5] = [0.25, 0.4, 0.6]  # Left wrist
        default_pose[6] = [0.75, 0.4, 0.6]  # Right wrist

        # Hips
        default_pose[7] = [0.45, 0.5, 0.8]  # Left hip
        default_pose[8] = [0.55, 0.5, 0.8]  # Right hip

        # Knees
        default_pose[9] = [0.4, 0.7, 0.7]  # Left knee
        default_pose[10] = [0.6, 0.7, 0.7]  # Right knee

        # Ankles
        default_pose[11] = [0.35, 0.9, 0.6]  # Left ankle
        default_pose[12] = [0.65, 0.9, 0.6]  # Right ankle

        # Fill the rest with default values if we need more points
        for i in range(13, self.num_points):
            default_pose[i] = [0.5, 0.5, 0.5]  # Center with medium confidence

        return default_pose

    def detect(
        self, keypoints: np.ndarray, frame_width: int = None, frame_height: int = None
    ) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect if the pose represents a fall.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]
            frame_width: Width of the frame in pixels, used for keypoint normalization
            frame_height: Height of the frame in pixels, used for keypoint normalization

        Returns:
            Tuple of (action_type, confidence)
        """
        if not self.initialized:
            logger.warning("Detector not initialized, returning default standing pose")
            return PoseActionTypeEnum.standing, 0.0

        try:
            # Convert keypoints to tensor with more robust processing
            pose_tensor = self.keypoints_to_tensor(keypoints, frame_width, frame_height)

            # Verify the tensor has valid data before adding to history
            non_zero = torch.count_nonzero(pose_tensor).item()
            if non_zero == 0:
                logger.warning("Tensor has all zeros, skipping detection")
                return PoseActionTypeEnum.standing, 0.3

            # Add to history
            self.add_to_history(pose_tensor)
            logger.debug(
                f"Added pose to history buffer, current size: {self.history_idx}/{self.window_size}"
            )

            # Determine how many valid frames we have in the buffer
            available_frames = min(self.history_idx, int(self.window_size))
            # Only run inference once we have at least `effective_window_size` recent frames
            effective_window = int(
                getattr(self, "effective_window_size", self.window_size)
            )
            if available_frames < effective_window:
                logger.debug(
                    f"Not enough history yet for effective window: {available_frames}/{effective_window}"
                )
                return PoseActionTypeEnum.standing, 0.0

            try:
                # Build an input tensor from the last-K frames (K = effective_window_size)
                poses_list = list(self.poses_history)
                logger.debug(
                    f"History buffer length: {len(poses_list)}, available_frames: {available_frames}"
                )

                last_k = min(effective_window, available_frames)
                pad_count = int(self.window_size) - last_k

                # Optimized tensor construction: pre-allocate and fill
                # Target shape: (1, channels, window_size, num_points, num_people)
                poses_np = np.zeros(
                    (
                        1,
                        self.channels,
                        int(self.window_size),
                        self.num_points,
                        self.num_people,
                    ),
                    dtype=np.float32,
                )

                # Fill from the tail of poses_list into the end of the window
                # Each pose tensor has shape (channels, 1, num_points, num_people)
                for i, pose_tensor in enumerate(poses_list[-last_k:]):
                    # Place at position pad_count + i (skip padding zone)
                    poses_np[0, :, pad_count + i, :, :] = pose_tensor[
                        :, 0, :, :
                    ].numpy()

                # Log tensor shape for debugging
                logger.debug(
                    f"Stacked pose tensor shape: {poses_np.shape} (last_k={last_k}, pad={pad_count})"
                )

                # Verify the numpy array has valid data
                if np.count_nonzero(poses_np) == 0:
                    logger.warning("Numpy array has all zeros, skipping inference")
                    return PoseActionTypeEnum.standing, 0.3

                # Diagnostic logging: show input tensor stats to help debug unexpected outputs
                try:
                    logger.debug(
                        f"STGCN input stats - shape: {poses_np.shape}, nonzero: {np.count_nonzero(poses_np)}, min: {np.min(poses_np):.6f}, max: {np.max(poses_np):.6f}, mean: {np.mean(poses_np):.6f}"
                    )
                except Exception:
                    logger.debug(f"STGCN input shape: {poses_np.shape}")

                # Run inference
                self.interpreter.set_tensor(self.input_index, poses_np)
                self.interpreter.invoke()

                # Get output
                output = self.interpreter.get_tensor(self.output_details[0]["index"])
                # Log raw model output at INFO so it's visible with current logger settings
                try:
                    logger.debug(f"Raw model output: {output}")
                except Exception:
                    logger.debug("Raw model output received (unable to format)")

                # Process output (softmax and get class/confidence)
                probs = np_softmax(output)
                # Log the softmax probabilities for debugging
                try:
                    logger.debug(f"STGCN softmax probs: {probs}")
                except Exception:
                    logger.debug("STGCN softmax probs received")

                confidence = float(np.max(probs))
                class_id = int(np.argmax(probs))

                logger.debug(
                    f"Processed output - class: {class_id}, confidence: {confidence:.4f}"
                )

                # Class 1 typically represents fall in these models
                # Map to appropriate PoseActionTypeEnum using configured confidence threshold
                if class_id == 1 and confidence > float(
                    getattr(self, "confidence_threshold", 0.5)
                ):
                    logger.info(f"Detected FALL with confidence: {confidence:.4f}")
                    return PoseActionTypeEnum.falling, confidence
                else:
                    logger.debug(f"Detected standing with confidence: {confidence:.4f}")
                    return PoseActionTypeEnum.standing, confidence

            except Exception as e:
                logger.error(f"Error during inference: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return PoseActionTypeEnum.standing, 0.3

        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return PoseActionTypeEnum.standing, 0.3


# Register this detector with the registry
register_detector("stgcn_fall", STGCNFallDetector)
