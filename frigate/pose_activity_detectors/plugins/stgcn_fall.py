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
        self.model_path = model_path

        if not model_path or not os.path.exists(model_path):
            logger.error(f"Fall detection model not found at {model_path}")
            self.initialized = False
            return

        if Interpreter is None:
            logger.error("TFLite not available, fall detection will be disabled")
            self.initialized = False
            return

        try:
            self.interpreter = Interpreter(
                model_path=model_path, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Get tensor shapes
            _, self.channels, self.window_size, self.num_points, self.num_people = (
                self.input_details[0]["shape"]
            )
            self.input_index = self.input_details[0]["index"]

            # Initialize pose history buffer
            self.reset()
            self.initialized = True
            logger.info(f"STGCN fall detector initialized with model {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize STGCN fall detector: {e}")
            self.initialized = False

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

    def keypoints_to_tensor(self, keypoints: np.ndarray) -> torch.Tensor:
        """
        Convert keypoints to tensor format expected by STGCN model.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]

        Returns:
            torch.Tensor of shape (channels, 1, num_points, num_people)
        """
        if not self.initialized or keypoints.shape[0] < self.num_points:
            # Return empty tensor if not initialized or not enough keypoints
            return torch.zeros((self.channels, 1, self.num_points, self.num_people))

        # Normalize keypoints to [0, 1] range
        # We don't have image dimensions, so we'll use the keypoint values themselves
        # to estimate the scale
        x_values = keypoints[:, 0]
        y_values = keypoints[:, 1]
        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)

        x_range = max(1, x_max - x_min)
        y_range = max(1, y_max - y_min)

        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, 0] = (keypoints[:, 0] - x_min) / x_range
        normalized_keypoints[:, 1] = (keypoints[:, 1] - y_min) / y_range

        # Convert to torch tensor and reshape
        torch_pose = (
            torch.from_numpy(normalized_keypoints)[
                ..., None, None
            ]  # (N_points, 3, 1, 1)
            .repeat(1, 1, 1, 2)  # -> (N_points, 3, 1, 2)
            .permute(1, 2, 0, 3)  # -> (3, 1, N_points, 2)
        )

        # Center the xy coordinates around zero and zero out the other people's coordinates
        torch_pose[:2] -= 0.5
        torch_pose[..., 1:] = 0

        return torch_pose

    def detect(self, keypoints: np.ndarray) -> Tuple[PoseActionTypeEnum, float]:
        """
        Detect if the pose represents a fall.

        Args:
            keypoints: NumPy array of shape (num_points, 3) where each row is [x, y, confidence]

        Returns:
            Tuple of (action_type, confidence)
        """
        if not self.initialized:
            return PoseActionTypeEnum.standing, 0.0

        # Convert keypoints to tensor
        pose_tensor = self.keypoints_to_tensor(keypoints)

        # Add to history
        self.add_to_history(pose_tensor)

        # Check if we have enough history
        if self.history_idx < self.window_size:
            return PoseActionTypeEnum.standing, 0.0

        # Stack the historical poses into a tensor
        poses_tensor = torch.stack(list(self.poses_history), dim=0).permute(
            2, 1, 0, 3, 4
        )

        # Convert to numpy for TFLite
        poses_np = poses_tensor.numpy().astype(np.float32)

        # Run inference
        self.interpreter.set_tensor(self.input_index, poses_np)
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Process output (softmax and get class/confidence)
        probs = np_softmax(output)
        confidence = float(np.max(probs))
        class_id = int(np.argmax(probs))

        # Class 1 typically represents fall in these models
        # Map to appropriate PoseActionTypeEnum
        if class_id == 1 and confidence > 0.5:
            logger.info(f"Detected fall with confidence: {confidence}")
            return PoseActionTypeEnum.falling, confidence

        else:
            logger.info(f"Detected standing with confidence: {confidence}")
            return PoseActionTypeEnum.standing, confidence


# Register this detector with the registry
register_detector("stgcn_fall", STGCNFallDetector)
