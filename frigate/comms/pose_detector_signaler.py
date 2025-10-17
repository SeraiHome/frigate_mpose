import logging
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
        # Call parent constructor without arguments
        super().__init__()
        self.detector_name = detector_name
        # Create topic based on detector name
        self.topic = f"pose_detections/{detector_name}"
        # Set up the subscription
        self.subscribe(self._handle_updates)

    def _handle_updates(self, topic, payload):
        """Handle incoming updates."""
        # This callback is required for the subscription
        # but we don't need to process anything here
        return None

    def check_for_update(self, timeout: float = 0.1) -> str:
        """Check for pose detection updates."""
        return self.topic
