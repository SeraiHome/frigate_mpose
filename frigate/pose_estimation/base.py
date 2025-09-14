import datetime
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from multiprocessing import Queue, Value
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frigate.comms.object_detector_signaler import (
    ObjectDetectorPublisher,
    ObjectDetectorSubscriber,
)
from frigate.config import FrigateConfig
from frigate.const import PROCESS_PRIORITY_HIGH
from frigate.detectors import create_detector
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    InputDTypeEnum,
    ModelConfig,
)
from frigate.util.builtin import EventsPerSecond, load_labels
from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory
from frigate.util.process import FrigateProcess

from .util import tensor_transform

logger = logging.getLogger(__name__)


# Pose keypoint structure
class PoseKeypoint:
    def __init__(self, x: float, y: float, confidence: float):
        self.x = x
        self.y = y
        self.confidence = confidence

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


class Pose:
    def __init__(
        self,
        keypoints: List[PoseKeypoint],
        pose_id: Optional[str] = None,
        confidence: float = 0.0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.keypoints = keypoints
        self.pose_id = pose_id
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
        }


class PoseDetector(ABC):
    @abstractmethod
    def detect_poses(self, tensor_input, threshold: float = 0.4) -> List[Pose]:
        pass


class BaseLocalPoseDetector(PoseDetector):
    def __init__(
        self,
        detector_config: BaseDetectorConfig = None,
        labels: str = None,
    ):
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels = {}
        else:
            self.labels = load_labels(labels)

        if detector_config:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)
            self.dtype = detector_config.model.input_dtype
        else:
            self.input_transform = None
            self.dtype = InputDTypeEnum.int

        self.detect_api = create_detector(detector_config)

    def _transform_input(self, tensor_input: np.ndarray) -> np.ndarray:
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)

        if self.dtype == InputDTypeEnum.float:
            tensor_input = tensor_input.astype(np.float32)
            tensor_input /= 255
        elif self.dtype == InputDTypeEnum.float_denorm:
            tensor_input = tensor_input.astype(np.float32)

        return tensor_input

    def detect_poses(self, tensor_input: np.ndarray, threshold=0.4) -> List[Pose]:
        poses = []

        raw_poses = self.detect_raw(tensor_input)

        for pose_data in raw_poses:
            # Process pose data based on the model output format
            # This is a placeholder - actual implementation depends on model
            if pose_data.get("confidence", 0) < threshold:
                continue
            
            keypoints = []
            for kp in pose_data.get("keypoints", []):
                keypoints.append(PoseKeypoint(kp[0], kp[1], kp[2]))
            
            pose = Pose(
                keypoints=keypoints,
                confidence=pose_data.get("confidence", 0),
                bbox=pose_data.get("bbox"),
            )
            poses.append(pose)
        
        self.fps.update()
        return poses

    @abstractmethod
    def detect_raw(self, tensor_input: np.ndarray):
        pass


class LocalPoseDetector(BaseLocalPoseDetector):
    def detect_raw(self, tensor_input: np.ndarray):
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.detect_raw(tensor_input=tensor_input)


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
        detector_config: BaseDetectorConfig,
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

    def create_output_shm(self, name: str):
        # Create shared memory for pose outputs
        # Structure: max_poses * (pose_confidence + bbox(4) + num_keypoints * (x, y, confidence))
        # Assuming max 5 poses, 17 keypoints (COCO format)
        max_poses = 5
        keypoints_per_pose = 17
        values_per_keypoint = 3  # x, y, confidence
        values_per_pose = 1 + 4 + keypoints_per_pose * values_per_keypoint  # confidence + bbox + keypoints
        
        out_shm = UntrackedSharedMemory(name=f"pose-out-{name}", create=True)
        out_np = np.ndarray((max_poses, values_per_pose), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        frame_manager = SharedMemoryFrameManager()
        pose_detector = LocalPoseDetector(detector_config=self.detector_config)
        detector_publisher = ObjectDetectorPublisher()

        for name in self.cameras:
            self.create_output_shm(name)

        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue
            input_frame = frame_manager.get(
                connection_id,
                (
                    1,
                    self.detector_config.model.height,
                    self.detector_config.model.width,
                    3,
                ),
            )

            if input_frame is None:
                logger.warning(f"Failed to get frame {connection_id} from SHM")
                continue

            # detect and send the output
            self.start_time.value = datetime.datetime.now().timestamp()
            raw_poses = pose_detector.detect_raw(input_frame)
            duration = datetime.datetime.now().timestamp() - self.start_time.value
            frame_manager.close(connection_id)

            if connection_id not in self.outputs:
                self.create_output_shm(connection_id)

            # Convert pose data to numpy array format
            # This is a placeholder - actual format depends on model output
            self.outputs[connection_id]["np"][:] = 0  # Clear previous data
            # TODO: Fill with actual pose data
            
            detector_publisher.publish(f"pose-{connection_id}")
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
        detector_config: BaseDetectorConfig,
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
        self._publisher: ObjectDetectorPublisher | None = None
        self._detector: AsyncLocalPoseDetector | None = None
        self.send_times = deque()

    def create_output_shm(self, name: str):
        # Same as PoseDetectorRunner
        max_poses = 5
        keypoints_per_pose = 17
        values_per_keypoint = 3
        values_per_pose = 1 + 4 + keypoints_per_pose * values_per_keypoint
        
        out_shm = UntrackedSharedMemory(name=f"pose-out-{name}", create=True)
        out_np = np.ndarray((max_poses, values_per_pose), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def _detect_worker(self) -> None:
        logger.info("Starting Pose Detect Worker Thread")
        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            input_frame = self._frame_manager.get(
                connection_id,
                (
                    1,
                    self.detector_config.model.height,
                    self.detector_config.model.width,
                    3,
                ),
            )

            if input_frame is None:
                logger.warning(f"Failed to get frame {connection_id} from SHM")
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
            self._frame_manager.close(connection_id)

            if connection_id not in self.outputs:
                self.create_output_shm(connection_id)

            # write results and publish
            if poses is not None:
                self.outputs[connection_id]["np"][:] = 0  # Clear previous data
                # TODO: Fill with actual pose data
            self._publisher.publish(f"pose-{connection_id}")

            # update timers
            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10
            self.start_time.value = 0.0

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        self._frame_manager = SharedMemoryFrameManager()
        self._publisher = ObjectDetectorPublisher()
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
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ):
        self.name = name
        self.cameras = cameras
        self.detection_queue = detection_queue
        self.avg_inference_speed = Value("d", 0.01)
        self.detection_start = Value("d", 0.0)
        self.detect_process: FrigateProcess | None = None
        self.config = config
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

        # Async path for MemryX
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
        model_config: ModelConfig,
        stop_event: MpEvent,
    ):
        self.labels = labels
        self.name = name
        self.fps = EventsPerSecond()
        self.detection_queue = detection_queue
        self.stop_event = stop_event
        self.shm = UntrackedSharedMemory(name=f"pose-{self.name}", create=True)
        self.np_shm = np.ndarray(
            (1, model_config.height, model_config.width, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        
        # Output shared memory for poses
        max_poses = 5
        keypoints_per_pose = 17
        values_per_keypoint = 3
        values_per_pose = 1 + 4 + keypoints_per_pose * values_per_keypoint
        
        self.out_shm = UntrackedSharedMemory(name=f"pose-out-{self.name}", create=True)
        self.out_np_shm = np.ndarray((max_poses, values_per_pose), dtype=np.float32, buffer=self.out_shm.buf)
        self.detector_subscriber = ObjectDetectorSubscriber(f"pose-{name}")

    def detect_poses(self, tensor_input, threshold=0.4) -> List[Pose]:
        poses = []

        if self.stop_event.is_set():
            return poses

        # copy input to shared memory
        self.np_shm[:] = tensor_input[:]
        self.detection_queue.put(f"pose-{self.name}")
        result = self.detector_subscriber.check_for_update()

        # if it timed out
        if result is None:
            return poses

        # Parse pose data from shared memory
        # TODO: Implement actual parsing based on the data format
        for pose_data in self.out_np_shm:
            if pose_data[0] < threshold:  # confidence check
                break
            
            # Extract keypoints and bbox from pose_data
            # This is a placeholder - actual implementation depends on data format
            
        self.fps.update()
        return poses

    def cleanup(self):
        self.detector_subscriber.stop()
        self.shm.unlink()
        self.out_shm.unlink()