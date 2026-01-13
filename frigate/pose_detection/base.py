import datetime
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from multiprocessing import Queue, Value
from multiprocessing.synchronize import Event as MpEvent

import numpy as np

from frigate.comms.pose_detector_signaler import (
    PoseDetectorPublisher,
    PoseDetectorSubscriber,
)
from frigate.config import FrigateConfig
from frigate.const import PROCESS_PRIORITY_HIGH
from frigate.pose_detectors import create_pose_detector
from frigate.pose_detectors.detector_config import (
    BasePoseDetectorConfig,
    InputDTypeEnum,
    PoseModelConfig,
)
from frigate.util.builtin import EventsPerSecond, load_labels
from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory
from frigate.util.process import FrigateProcess

from .util import tensor_transform

logger = logging.getLogger(__name__)


class PoseDetector(ABC):
    @abstractmethod
    def detect(self, tensor_input, threshold: float = 0.4):
        pass


class BaseLocalPoseDetector(PoseDetector):
    def __init__(
        self,
        detector_config: BasePoseDetectorConfig = None,
        labels: str = None,
    ):
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels = {}
        else:
            self.labels = load_labels(labels)

        if detector_config and detector_config.model is not None:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)
            self.dtype = detector_config.model.input_dtype
        else:
            self.input_transform = None
            self.dtype = InputDTypeEnum.int
        logger.info(f"Pose detector input : {detector_config.model}")
        self.detect_api = create_pose_detector(detector_config)

    def _transform_input(self, tensor_input: np.ndarray) -> np.ndarray:
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)

        if self.dtype == InputDTypeEnum.float:
            tensor_input = tensor_input.astype(np.float32)
            tensor_input /= 255
        elif self.dtype == InputDTypeEnum.float_denorm:
            tensor_input = tensor_input.astype(np.float32)

        return tensor_input

    def detect(self, tensor_input: np.ndarray, threshold=0.4):
        poses = []

        raw_poses = self.detect_raw(tensor_input)

        for pose in raw_poses:
            # pose format: [person_id, confidence, keypoints...]
            if pose[1] < threshold:
                break
            poses.append(
                {
                    "person_id": int(pose[0]),
                    "confidence": float(pose[1]),
                    "keypoints": pose[2:].reshape(
                        -1, 3
                    ),  # reshape to (num_keypoints, 3) for x,y,confidence
                    "bbox": pose[-4:]
                    if len(pose) > 2
                    else None,  # bounding box if available
                }
            )
        self.fps.update()
        return poses


class LocalPoseDetector(BaseLocalPoseDetector):
    def detect_raw(self, tensor_input: np.ndarray, camera_name=None):
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.detect_raw(
            tensor_input=tensor_input, camera_name=camera_name
        )


class AsyncLocalPoseDetector(BaseLocalPoseDetector):
    def async_send_input(self, tensor_input: np.ndarray, connection_id: str):
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.send_input(connection_id, tensor_input)

    def async_receive_output(self):
        return self.detect_api.receive_output()


class PoseDetectorRunner(FrigateProcess):
    def __init__(
        self,
        name,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Value,
        start_time: Value,
        config: FrigateConfig,
        detector_config: BasePoseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict = {}
        self.frame_count = 0

    def create_output_shm(self, name: str):
        # Pose detection output: person_id, confidence, keypoints (17*3=51), bbox (4) = 57 floats max
        # Extract camera name if it already has a pose- prefix to avoid double-prefixing
        camera_name = name[5:] if name.startswith("pose-") else name
        out_shm = UntrackedSharedMemory(name=f"pose-out-{camera_name}", create=False)
        out_np = np.ndarray((20, 57), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)
        try:
            frame_manager = SharedMemoryFrameManager()
            pose_detector = LocalPoseDetector(detector_config=self.detector_config)
            detector_publisher = PoseDetectorPublisher()

            # Log model dimensions inside try block
            width = 320
            height = 320
            if self.detector_config.model:
                width = self.detector_config.model.width
                height = self.detector_config.model.height
            logger.info(f"detector model config height {height} width {width}")

            # Initialize frame counter after successful detector initialization
            self.frame_count += 1
        except Exception as e:
            logger.error(f"PoseDetectorRunner failed to start: {e}", exc_info=True)
            raise

        for name in self.cameras:
            try:
                self.create_output_shm(name)
            except Exception as e:
                logger.error(
                    f"Failed to create output SHM for {name}: {e}", exc_info=True
                )
                raise

        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Increment frame count for debugging - do this first to ensure unique filenames
            self.frame_count += 1

            # Ensure SHM for input exists and is correct size
            input_shape = (1, height, width, 3)

            # Now ensure consistent naming with RemotePoseDetector
            if connection_id.startswith("pose-"):
                shm_name = connection_id
            else:
                shm_name = f"pose-{connection_id}"

            # Debug the actual SHM name being used
            logger.debug(
                f"[POSE PROC] Accessing shared memory with name: {shm_name} for frame {self.frame_count}"
            )

            # Try to verify if this shared memory exists

            # Create and get frame with direct SHM access to match RemotePoseDetector
            input_frame = None

            # Define a function to force memory synchronization
            def force_memory_sync(shm_name, input_shape):
                try:
                    # Direct low-level shared memory access bypassing frame_manager cache
                    direct_shm = UntrackedSharedMemory(name=shm_name, create=False)
                    direct_array = np.ndarray(
                        input_shape, dtype=np.uint8, buffer=direct_shm.buf
                    )

                    # Force a memory refresh by toggling the writeable flag
                    direct_array.flags.writeable = False
                    direct_array.flags.writeable = True

                    # Create a fresh copy to ensure we're getting the latest data
                    result = direct_array.copy()

                    # Verify data
                    non_zero = np.count_nonzero(result)
                    logger.debug(
                        f"[POSE PROC] Force sync: shm={shm_name}, shape={result.shape}, non-zero={non_zero}"
                    )

                    direct_shm.close()
                    if non_zero > 0:
                        return result
                except Exception as e:
                    logger.error(f"[POSE PROC] Force sync failed: {e}")
                return None

            # Try different access methods for shared memory to avoid synchronization issues
            logger.debug(
                f"[POSE PROC] Accessing shared memory for frame {self.frame_count} with multiple methods"
            )

            # Method 1: Try using frame_manager first (may have caching issues)
            try:
                # frame_manager.create(shm_name, shape=input_shape, dtype=np.uint8)
                input_frame = frame_manager.get(shm_name, input_shape)
                logger.debug(
                    f"[POSE PROC] Method 1: Got frame from frame_manager with shape {input_frame.shape if input_frame is not None else 'None'}"
                )
            except Exception as e:
                logger.error(f"[POSE PROC] Method 1 error: {e}")

            # Check if frame has data
            if input_frame is not None:
                non_zero_count = np.count_nonzero(input_frame)
                logger.debug(
                    f"[POSE PROC] Frame manager frame has {non_zero_count} non-zero values"
                )
                if non_zero_count == 0:
                    # Frame exists but is empty - force synchronization
                    logger.debug(
                        "[POSE PROC] Frame manager returned empty frame, forcing sync"
                    )
                    input_frame = force_memory_sync(shm_name, input_shape)

            # Save the filename format as a timestamp-based value to prevent overwriting
            debug_filename = f"input_frame_manager_{self.name}_{self.frame_count}_{time.time():.6f}.jpg"

            import os

            import cv2

            debug_dir = os.path.join("/media/frigate", "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # Save the frame for debugging
            if input_frame is not None and np.count_nonzero(input_frame) > 0:
                # Remove batch dimension (squeeze from 4D to 3D if needed)
                if len(input_frame.shape) == 4:
                    frame_to_save = np.squeeze(input_frame, axis=0)
                else:
                    frame_to_save = input_frame

                # Convert to BGR format for OpenCV (assuming the input is RGB)
                try:
                    # bgr_frame = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(
                    #     f"{debug_dir}/{debug_filename}",
                    #     bgr_frame,
                    # )
                    logger.debug(
                        f"[SHM-ANALYSIS] Successfully wrote debug image to {debug_dir}/{debug_filename}"
                    )
                except Exception as e:
                    logger.error(
                        f"[SHM-ANALYSIS] Failed to convert and save frame: {e}"
                    )
                    # Fallback: try to save the original frame directly
                    try:
                        cv2.imwrite(
                            f"{debug_dir}/input_frame_raw_{self.name}_{self.frame_count}_{time.time():.6f}.jpg",
                            frame_to_save,
                        )
                        logger.debug("[SHM-ANALYSIS] Saved raw frame as fallback")
                    except Exception as e2:
                        logger.error(f"[SHM-ANALYSIS] Failed to save raw frame: {e2}")
            else:
                logger.error("[SHM-ANALYSIS] Input frame is empty or all zeros")
            logger.debug(
                f"[SHM-ANALYSIS] Frame Manager input frame {debug_dir}/{debug_filename}"
            )

            # Extract camera name from connection_id for debug purposes
            camera_name = (
                connection_id[5:]
                if connection_id.startswith("pose-")
                else connection_id
            )

            # detect and send the output
            self.start_time.value = datetime.datetime.now().timestamp()
            poses = pose_detector.detect_raw(input_frame, camera_name=camera_name)
            logger.debug(f"poses {poses} process...")
            duration = datetime.datetime.now().timestamp() - self.start_time.value
            frame_manager.close(shm_name)

            # Handle the case where connection_id may already have a pose- prefix
            camera_name = (
                connection_id[5:]
                if connection_id.startswith("pose-")
                else connection_id
            )
            output_key = (
                f"pose-{camera_name}"
                if not connection_id.startswith("pose-")
                else connection_id
            )

            # Ensure the output SHM exists
            if output_key not in self.outputs:
                self.create_output_shm(output_key)

            # Copy detection results to output SHM
            self.outputs[output_key]["np"][:] = poses[:]

            # Publish using the camera name (without pose- prefix)
            detector_publisher.publish(camera_name)
            self.start_time.value = 0.0

            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10

        detector_publisher.stop()
        logger.info("Exited pose detection process...")


class AsyncPoseDetectorRunner(FrigateProcess):
    def __init__(
        self,
        name,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Value,
        start_time: Value,
        config: FrigateConfig,
        detector_config: BasePoseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict = {}
        self._frame_manager: SharedMemoryFrameManager | None = None
        self._publisher: PoseDetectorPublisher | None = None
        self._detector: AsyncLocalPoseDetector | None = None
        self.send_times = deque()

    def create_output_shm(self, name: str):
        # Extract camera name if it already has a pose- prefix to avoid double-prefixing
        camera_name = name[5:] if name.startswith("pose-") else name
        out_shm = UntrackedSharedMemory(name=f"pose-out-{camera_name}", create=False)
        out_np = np.ndarray((20, 57), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def _detect_worker(self) -> None:
        logger.info("Starting Pose Detect Worker Thread")
        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Default dimensions if model is None
            width = 320
            height = 320
            if self.detector_config.model:
                width = self.detector_config.model.width
                height = self.detector_config.model.height

            # Ensure SHM for input exists and is correct size
            input_shape = (1, height, width, 3)
            # Use the correct shared memory name format that includes the 'pose-' prefix
            shm_name = f"pose-{connection_id}"
            self._frame_manager.create(shm_name, shape=input_shape, dtype=np.uint8)
            input_frame = self._frame_manager.get(shm_name, input_shape)

            if input_frame is None:
                logger.warning(f"Failed to get frame {shm_name} from SHM")
                continue

            # mark start time and send to accelerator
            self.send_times.append(time.perf_counter())
            self._detector.async_send_input(input_frame, connection_id)

    def _result_worker(self) -> None:
        logger.info("Starting Pose Result Worker Thread")
        while not self.stop_event.is_set():
            connection_id, poses = self._detector.async_receive_output()

            if not self.send_times:
                # guard; shouldn't happen if send/recv are balanced
                continue
            ts = self.send_times.popleft()
            duration = time.perf_counter() - ts

            # release input buffer
            shm_name = f"pose-{connection_id}"
            self._frame_manager.close(shm_name)

            # Handle the case where connection_id may already have a pose- prefix
            camera_name = (
                connection_id[5:]
                if connection_id.startswith("pose-")
                else connection_id
            )
            output_key = (
                f"pose-{camera_name}"
                if not connection_id.startswith("pose-")
                else connection_id
            )

            # Ensure the output SHM exists
            if output_key not in self.outputs:
                self.create_output_shm(output_key)

            # write results and publish
            if poses is not None:
                self.outputs[output_key]["np"][:] = poses[:]

            # Publish using the camera name (without pose- prefix)
            self._publisher.publish(camera_name)

            # update timers
            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10
            self.start_time.value = 0.0

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        self._frame_manager = SharedMemoryFrameManager()
        self._publisher = PoseDetectorPublisher()
        self._detector = AsyncLocalPoseDetector(detector_config=self.detector_config)

        for name in self.cameras:
            self.create_output_shm(name)

        t_detect = threading.Thread(target=self._detect_worker, daemon=True)
        t_result = threading.Thread(target=self._result_worker, daemon=True)
        t_detect.start()
        t_result.start()

        while not self.stop_event.is_set():
            time.sleep(0.5)

        self._publisher.stop()
        logger.info("Exited async pose detection process...")


class PoseDetectProcess:
    def __init__(
        self,
        name: str,
        detection_queue: Queue,
        cameras: list[str],
        config: FrigateConfig,
        detector_config: BasePoseDetectorConfig,
        stop_event: MpEvent,
    ):
        self.name = name
        self.cameras = cameras
        self.detection_queue = detection_queue
        self.avg_inference_speed = Value("d", 0.01)
        self.detection_start = Value("d", 0.0)
        self.detect_process: FrigateProcess | None = None
        self.config = config
        # Ensure detector_config.model is set, fallback to global pose_model if missing
        if getattr(detector_config, "model", None) is None and hasattr(
            config, "pose_model"
        ):
            detector_config.model = config.pose_model
        self.detector_config = detector_config
        self.stop_event = stop_event
        self.start_or_restart()

    def stop(self):
        # if the process has already exited on its own, just return
        if self.detect_process and self.detect_process.exitcode:
            return
        self.detect_process.terminate()
        logging.info("Waiting for pose detection process to exit gracefully...")
        self.detect_process.join(timeout=30)
        if self.detect_process.exitcode is None:
            logging.info("Pose detection process didn't exit. Force killing...")
            self.detect_process.kill()
            self.detect_process.join()
        logging.info("Pose detection process has exited...")

    def start_or_restart(self):
        self.detection_start.value = 0.0
        if (self.detect_process is not None) and self.detect_process.is_alive():
            self.stop()

        # Async path for MemryX and other async detectors
        if self.detector_config.type == "memryx":
            self.detect_process = AsyncPoseDetectorRunner(
                f"frigate.pose_detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        else:
            self.detect_process = PoseDetectorRunner(
                f"frigate.pose_detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        self.detect_process.start()


class RemotePoseDetector:
    def __init__(
        self,
        name: str,
        labels: dict[int, str],
        detection_queue: Queue,
        model_config: PoseModelConfig,
        stop_event: MpEvent,
    ):
        self.labels = labels
        self.name = name
        self.fps = EventsPerSecond()
        self.detection_queue = detection_queue
        self.stop_event = stop_event
        self.model_config = model_config

        # Extract camera name from detector name (remove "pose-" prefix if present)
        # This will be passed to the detector implementation for debugging purposes
        if self.name.startswith("pose-"):
            self.camera_name = self.name[5:]  # Skip the "pose-" prefix
        else:
            self.camera_name = self.name

        logger.info(f"Initializing RemotePoseDetector for camera: {self.camera_name}")

        try:
            # Try to create a new shared memory buffer of the required size
            # This way we can ensure it has the correct size
            buffer_size_needed = (
                1
                * model_config.height
                * model_config.width
                * 3
                * np.dtype(np.uint8).itemsize
            )

            # Add a safety margin to handle alignment and overhead
            buffer_size_with_margin = buffer_size_needed * 2

            # Minimum buffer size of 10MB to be safe
            final_buffer_size = max(buffer_size_with_margin, 10485760)  # 10MB

            logger.info(
                f"RemotePoseDetector creating buffer for {self.name}: "
                f"needed={buffer_size_needed} bytes, "
                f"allocating={final_buffer_size} bytes for safety, "
                f"dimensions={model_config.height}x{model_config.width}"
            )

            try:
                # Access existing shared memory created by app.py
                self.shm = UntrackedSharedMemory(name=self.name, create=False)
                logger.info(f"Connected to existing shared memory for {self.name}")
            except FileNotFoundError:
                logger.error(
                    f"Shared memory {self.name} not found. It should be created by app.py before camera startup."
                )
                raise
            actual_buffer_size = len(self.shm.buf)
            logger.info(
                f"Actual buffer size for {self.name}: {actual_buffer_size} bytes"
            )

            # Map the buffer to a numpy array of the correct dimensions
            self.np_shm = np.ndarray(
                (1, model_config.height, model_config.width, 3),
                dtype=np.uint8,
                buffer=self.shm.buf,
            )
        except Exception as e:
            logger.error(
                f"RemotePoseDetector failed to map input SHM for {self.name}: {e}",
                exc_info=True,
            )
            raise
        try:
            # Extract the camera name without the "pose-" prefix to avoid double prefixing
            if self.name.startswith("pose-"):
                camera_name = self.name[5:]  # Remove 'pose-' prefix
                output_name = f"pose-out-{camera_name}"
            else:
                output_name = f"pose-out-{self.name}"

            logger.info(f"Accessing output shared memory with name: {output_name}")

            self.out_shm = UntrackedSharedMemory(name=output_name, create=False)
            self.out_np_shm = np.ndarray(
                (20, 57), dtype=np.float32, buffer=self.out_shm.buf
            )
        except Exception as e:
            logger.error(
                f"RemotePoseDetector failed to map output SHM for {self.name}: {e}",
                exc_info=True,
            )
            raise

        try:
            # Initialize the detector subscriber with the name
            # The error indicates PoseDetectorSubscriber constructor requires detector_name
            self.detector_subscriber = PoseDetectorSubscriber(self.name)
            logger.info(
                f"Successfully initialized PoseDetectorSubscriber for {self.name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize PoseDetectorSubscriber for {self.name}: {e}",
                exc_info=True,
            )
            raise

    def detect(self, tensor_input, threshold=0.4):
        """Detect poses in the input tensor using the remote detector.

        Args:
            tensor_input: RGB or YUV image tensor
            threshold: Confidence threshold for pose detection

        Returns:
            List of detected poses
        """
        poses = []

        # Don't process if stop event is set
        if self.stop_event.is_set():
            return poses

        try:
            # Extract camera name for better logging
            camera_name = self.name[5:] if self.name.startswith("pose-") else self.name

            # Track frame count for debugging
            frame_count = getattr(self, "frame_count", 0) + 1
            setattr(self, "frame_count", frame_count)

            # Preprocess the input tensor to match model requirements
            tensor_input = self._preprocess_input_tensor(tensor_input)

            # Verify tensor has valid data before copying to shared memory
            if tensor_input is None or np.count_nonzero(tensor_input) == 0:
                logger.error(
                    f"Input tensor for camera {camera_name} is empty - skipping detection"
                )
                return poses

            # Create an explicit copy for memory consistency
            tensor_copy = tensor_input.copy()

            # Copy to shared memory and ensure memory sync
            self.np_shm[:] = tensor_copy
            self.np_shm.flags.writeable = False  # Force flush
            self.np_shm.flags.writeable = True

            # Signal pose detection process that frame is ready
            self.detection_queue.put(self.name)

            # Wait for pose detection results (with timeout)
            result = self.detector_subscriber.check_for_update(timeout=1)

            # Handle timeout case
            if result is None:
                logger.debug(
                    f"No pose detection result received (timeout) for camera {camera_name}"
                )
                return poses

            # Process detection results
            for pose_data in self.out_np_shm:
                if pose_data[1] < threshold:
                    break

                poses.append(
                    {
                        "person_id": int(pose_data[0]),
                        "confidence": float(pose_data[1]),
                        "keypoints": pose_data[2:53].reshape(
                            -1, 3
                        ),  # 17 keypoints * 3 = 51
                        "bbox": pose_data[53:57] if len(pose_data) > 53 else None,
                    }
                )

            self.fps.update()
            return poses
        except Exception as e:
            camera_name = self.name[5:] if self.name.startswith("pose-") else self.name
            logger.error(f"Error processing input tensor for camera {camera_name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return poses

    def _preprocess_input_tensor(self, tensor_input):
        """Preprocess input tensor for pose detection.

        Handles YUV to RGB conversion, resizing, and ensuring proper format.

        Args:
            tensor_input: Input image tensor

        Returns:
            Preprocessed tensor ready for pose detection
        """
        try:
            import cv2

            from frigate.util.image import yuv_region_2_rgb

            # Check if this is a YUV frame
            is_yuv = False
            if len(tensor_input.shape) == 2:  # 2D YUV format
                is_yuv = True
                height = tensor_input.shape[0]
                width = tensor_input.shape[1]
            elif (
                len(tensor_input.shape) == 3
                and tensor_input.shape[0] > tensor_input.shape[1] * 1.2
            ):
                # Another way to detect YUV: height is ~1.5x width for I420 format
                is_yuv = True
                height = tensor_input.shape[0]
                width = tensor_input.shape[1]

            # Convert YUV to RGB if needed
            if is_yuv:
                region = (0, 0, width, height)
                tensor_input = yuv_region_2_rgb(tensor_input, region)

            # Handle 2D grayscale inputs (expand to 3 channels)
            if len(tensor_input.shape) == 2:
                tensor_input = np.stack(
                    [tensor_input, tensor_input, tensor_input], axis=2
                )

            # Get input and buffer dimensions
            input_height, input_width = tensor_input.shape[:2]
            buffer_height = self.model_config.height
            buffer_width = self.model_config.width

            # Resize if needed
            if (input_height, input_width) != (buffer_height, buffer_width):
                tensor_input = cv2.resize(
                    tensor_input,
                    (buffer_width, buffer_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Ensure tensor has 3 RGB channels
            if len(tensor_input.shape) == 3 and tensor_input.shape[2] != 3:
                if tensor_input.shape[2] == 1:  # Single channel grayscale
                    tensor_input = np.repeat(tensor_input, 3, axis=2)
                else:  # Other channel count
                    channels = min(tensor_input.shape[2], 3)
                    new_tensor = np.zeros(
                        (buffer_height, buffer_width, 3), dtype=np.uint8
                    )
                    new_tensor[:, :, :channels] = tensor_input[:, :, :channels]
                    tensor_input = new_tensor

            # Add batch dimension for model input (NHWC format)
            if len(tensor_input.shape) == 3:
                tensor_input = np.expand_dims(tensor_input, axis=0)

            return tensor_input

        except Exception as e:
            camera_name = self.name[5:] if self.name.startswith("pose-") else self.name
            logger.error(
                f"Error preprocessing input tensor for camera {camera_name}: {e}"
            )
            return None

    def cleanup(self):
        self.detector_subscriber.stop()
        self.shm.unlink()
        self.out_shm.unlink()
