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
from frigate.util.image import UntrackedSharedMemory
from frigate.util.process import FrigateProcess

from .tensor_utils import (
    ensure_rgb_hwc,
    extract_keypoints_from_pose_output,
    is_yuv_frame,
)
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
            # pose format: [person_id, confidence, keypoints(51), bbox(4)]
            if pose[1] < threshold:
                break
            # Use extract_keypoints to get (17, 3) view without copy
            keypoints = extract_keypoints_from_pose_output(pose)
            poses.append(
                {
                    "person_id": int(pose[0]),
                    "confidence": float(pose[1]),
                    "keypoints": keypoints,
                    "bbox": pose[53:57] if len(pose) >= 57 else None,
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
            pose_detector = LocalPoseDetector(detector_config=self.detector_config)
            detector_publisher = PoseDetectorPublisher()

            # Log that we're using dynamic SHM format (no fixed model dimensions needed)
            logger.info(
                "PoseDetectorRunner using dynamic SHM format (variable frame sizes)"
            )

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

            # Now ensure consistent naming with RemotePoseDetector
            if connection_id.startswith("pose-"):
                shm_name = connection_id
            else:
                shm_name = f"pose-{connection_id}"

            # Get frame from shared memory using dynamic SHM format (with header)
            input_frame = None
            try:
                from frigate.pose_detection.shm_format import read_frame_from_shm

                direct_shm = UntrackedSharedMemory(name=shm_name, create=False)
                input_frame = read_frame_from_shm(direct_shm.buf)
                direct_shm.close()

                if input_frame is not None:
                    logger.debug(
                        f"[POSE PROC] Read dynamic frame {input_frame.shape} from SHM {shm_name}"
                    )
            except Exception as e:
                logger.error(f"[POSE PROC] Failed to get frame from SHM: {e}")

            if input_frame is None or np.count_nonzero(input_frame) == 0:
                continue

            # Extract camera name from connection_id for debug purposes
            camera_name = (
                connection_id[5:]
                if connection_id.startswith("pose-")
                else connection_id
            )

            # detect and send the output
            self.start_time.value = datetime.datetime.now().timestamp()
            poses = pose_detector.detect_raw(input_frame, camera_name=camera_name)
            duration = datetime.datetime.now().timestamp() - self.start_time.value

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
        logger.info("Starting Pose Detect Worker Thread (dynamic SHM)")
        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Use the correct shared memory name format that includes the 'pose-' prefix
            shm_name = f"pose-{connection_id}"

            # Read frame from dynamic SHM with header
            input_frame = None
            try:
                from frigate.pose_detection.shm_format import read_frame_from_shm

                direct_shm = UntrackedSharedMemory(name=shm_name, create=False)
                input_frame = read_frame_from_shm(direct_shm.buf)
                direct_shm.close()

                if input_frame is not None:
                    logger.debug(
                        f"[ASYNC POSE] Read dynamic frame {input_frame.shape} from SHM {shm_name}"
                    )
            except Exception as e:
                logger.error(f"[ASYNC POSE] Failed to get frame from SHM: {e}")
                continue

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

            # No need to release input buffer with dynamic SHM - it's reused

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

        # Track accelerator type and device info for stats
        self.accelerator_type: str = self._determine_accelerator_type()
        self.accelerator_device: str | None = getattr(detector_config, "device", None)
        self.model_type: str | None = None
        if detector_config.model:
            self.model_type = getattr(detector_config.model, "model_type", None)
            if self.model_type:
                self.model_type = (
                    str(self.model_type.value)
                    if hasattr(self.model_type, "value")
                    else str(self.model_type)
                )

        self.start_or_restart()

    def _determine_accelerator_type(self) -> str:
        """Determine the accelerator type from detector config."""
        # Check explicit accelerator field first
        if (
            hasattr(self.detector_config, "accelerator")
            and self.detector_config.accelerator
        ):
            return self.detector_config.accelerator

        # Infer from detector type
        detector_type = getattr(self.detector_config, "type", "cpu")
        accelerator_mapping = {
            "edgetpu": "edgetpu",
            "coral": "edgetpu",
            "memryx": "memryx",
            "tensorrt": "gpu",
            "gpu": "gpu",
            "cuda": "gpu",
            "hailo": "hailo",
            "rknn": "rockchip",
            "rocm": "rocm",
            "openvino": "openvino",
        }
        return accelerator_mapping.get(detector_type.lower(), "cpu")

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

        # Import dynamic SHM format
        from frigate.pose_detection.shm_format import HEADER_SIZE_BYTES

        try:
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
                f"RemotePoseDetector connected to dynamic SHM for {self.name}: "
                f"buffer_size={actual_buffer_size} bytes (header={HEADER_SIZE_BYTES})"
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
            tensor_input: RGB image tensor (variable size, no resize needed)
            threshold: Confidence threshold for pose detection

        Returns:
            List of detected poses
        """
        poses = []

        # Don't process if stop event is set
        if self.stop_event.is_set():
            return poses

        try:
            from frigate.pose_detection.shm_format import write_frame_to_shm

            # Preprocess: ensure RGB HWC format (no resize!)
            tensor_input = self._preprocess_input_tensor(tensor_input)

            # Verify tensor has valid data before copying to shared memory
            if tensor_input is None or tensor_input.size == 0:
                return poses

            # Write frame with header to shared memory (dynamic size, no resize)
            if not write_frame_to_shm(self.shm.buf, tensor_input):
                logger.warning(f"Frame too large for SHM buffer: {tensor_input.shape}")
                return poses

            # Signal pose detection process that frame is ready
            self.detection_queue.put(self.name)

            # Wait for pose detection results (with timeout)
            result = self.detector_subscriber.check_for_update(timeout=1)

            # Handle timeout case
            if result is None:
                return poses

            # Process detection results using optimized extraction
            for pose_data in self.out_np_shm:
                if pose_data[1] < threshold:
                    break

                # Use extract_keypoints for view-based access (no copy)
                keypoints = extract_keypoints_from_pose_output(pose_data)
                poses.append(
                    {
                        "person_id": int(pose_data[0]),
                        "confidence": float(pose_data[1]),
                        "keypoints": keypoints,
                        "bbox": pose_data[53:57] if len(pose_data) >= 57 else None,
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

        Handles YUV to RGB conversion and ensures proper HWC format.
        With dynamic SHM, no resizing is performed - the frame is sent at its
        original size for maximum efficiency.

        Args:
            tensor_input: Input image tensor

        Returns:
            Preprocessed tensor ready for pose detection with shape (H, W, 3)
        """
        try:
            from frigate.pose_detection.shm_format import (
                MAX_POSE_HEIGHT,
                MAX_POSE_WIDTH,
            )
            from frigate.util.image import yuv_region_2_rgb

            # Fast path: already in correct HWC RGB format
            if tensor_input.ndim == 3 and tensor_input.shape[2] == 3:
                h, w = tensor_input.shape[:2]
                if w <= MAX_POSE_WIDTH and h <= MAX_POSE_HEIGHT:
                    return tensor_input

            # Check YUV format using helper
            if is_yuv_frame(tensor_input):
                height, width = tensor_input.shape[:2]
                region = (0, 0, width, height)
                tensor_input = yuv_region_2_rgb(tensor_input, region)

            # Ensure HWC format (removes batch dim if present, handles grayscale)
            tensor_input = ensure_rgb_hwc(tensor_input)

            # No resize! Dynamic SHM handles variable sizes
            # Only check that it fits in the buffer
            h, w = tensor_input.shape[:2]
            if w > MAX_POSE_WIDTH or h > MAX_POSE_HEIGHT:
                # If frame is too large, downscale to fit
                import cv2

                scale = min(MAX_POSE_WIDTH / w, MAX_POSE_HEIGHT / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                tensor_input = cv2.resize(
                    tensor_input,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                logger.debug(
                    f"Frame resized from {w}x{h} to {new_w}x{new_h} to fit buffer"
                )

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
