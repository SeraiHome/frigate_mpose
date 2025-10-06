import logging
from multiprocessing import shared_memory
from multiprocessing.synchronize import Event as MpEvent

from frigate.comms.inter_process import InterProcessCommunicator

logger = logging.getLogger(__name__)


class PoseDetectorPublisher(InterProcessCommunicator):
    def __init__(self) -> None:
        self.stop_event: MpEvent = None

    def publish(self, payload: str) -> None:
        """Publish pose detection results."""
        topic = f"pose_detections/{payload}"
        logger.debug(f"Publishing pose detection result: {topic}")
        super().publish(topic, "", False)

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()


class PoseDetectorSubscriber(InterProcessCommunicator):
    def __init__(self, detector_name) -> None:
        self.detector_name = detector_name
        topic = f"pose_detections/{detector_name}"
        super().__init__(topic)

    def check_for_update(self, timeout: float = 0.1) -> str:
        """Check for pose detection updates."""
        return super().check_for_update(timeout=timeout)