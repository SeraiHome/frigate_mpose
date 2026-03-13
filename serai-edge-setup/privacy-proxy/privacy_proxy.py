"""Privacy proxy: generates skeleton RTSP stream for Frigate recording.

Subscribes to pose keypoints via MQTT, renders skeleton frames,
serves raw video over TCP for go2rtc to pull via exec: source.
On fall events, switches to pass-through of real camera frames.

Usage:
    python privacy_proxy.py --config /config/config.yaml
"""

import argparse
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import av
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import yaml

from skeleton_renderer import render_skeleton_bgr

logger = logging.getLogger("privacy-proxy")


@dataclass
class CameraState:
    """Per-camera privacy proxy state."""

    name: str
    width: int
    height: int
    fps: int
    real_rtsp_url: str
    tcp_port: int  # Port for raw frame TCP server

    # Privacy state
    override_until: float = 0.0
    latest_keypoints: Optional[list] = None
    keypoints_lock: threading.Lock = field(default_factory=threading.Lock)

    # Pre-event buffer: (timestamp, jpeg_bytes) tuples at native camera fps
    # 300 frames @ ~30fps ≈ 10 seconds of real footage
    pre_event_buffer: deque = field(default_factory=lambda: deque(maxlen=300))

    # Latest decoded real camera frame (set by pre-event capture thread)
    latest_real_frame: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_override_active(self) -> bool:
        return time.time() < self.override_until


class PrivacyProxy:
    """Main privacy proxy service."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.cameras: dict[str, CameraState] = {}
        for cam_cfg in config.get("cameras", []):
            cam = CameraState(
                name=cam_cfg["name"],
                width=cam_cfg.get("width", 1280),
                height=cam_cfg.get("height", 720),
                fps=cam_cfg.get("fps", 5),
                real_rtsp_url=cam_cfg["real_rtsp_url"],
                tcp_port=cam_cfg["tcp_port"],
            )
            self.cameras[cam.name] = cam

        self.pre_event_seconds = config.get("pre_event_seconds", 10)
        self.post_event_seconds = config.get("post_event_seconds", 60)
        self.recordings_dir = config.get("recordings_dir", "/media/frigate/recordings")

        # MQTT
        self.mqtt_host = config.get("mqtt_host", "localhost")
        self.mqtt_port = config.get("mqtt_port", 1883)
        # Support both paho-mqtt v1 and v2
        try:
            self.mqtt_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id="privacy-proxy"
            )
        except (AttributeError, TypeError):
            self.mqtt_client = mqtt.Client(client_id="privacy-proxy")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message

    def _on_mqtt_connect(self, client, userdata, flags, rc, *args):
        logger.info(f"MQTT connected (rc={rc})")
        for cam_name in self.cameras:
            client.subscribe(f"frigate/{cam_name}/pose_keypoints", qos=0)
            client.subscribe(f"frigate/{cam_name}/privacy_override", qos=1)
        logger.info(f"Subscribed to {len(self.cameras)} camera topics")

    def _on_mqtt_message(self, client, userdata, msg, *args):
        parts = msg.topic.split("/")
        if len(parts) < 3:
            return

        cam_name = parts[1]
        msg_type = parts[2]
        cam = self.cameras.get(cam_name)
        if not cam:
            return

        if msg_type == "pose_keypoints":
            try:
                data = json.loads(msg.payload)
                with cam.keypoints_lock:
                    cam.latest_keypoints = data.get("poses", [])
            except json.JSONDecodeError:
                pass

        elif msg_type == "privacy_override":
            try:
                data = json.loads(msg.payload)
                duration = data.get("duration_seconds", self.post_event_seconds)
                cam.override_until = time.time() + duration
                logger.warning(
                    f"Privacy override ACTIVATED for {cam_name} "
                    f"({duration}s, real frames)"
                )
            except json.JSONDecodeError:
                pass

    def _generate_frames(self, cam: CameraState):
        """Generator: yield BGR frames (skeleton or real) for a camera.

        During override, first flushes the pre-event buffer (10s of real
        footage at native fps) then continues with live real frames.
        FFmpeg assigns sequential PTS at the declared framerate, so the
        flushed frames play back at correct speed.
        """
        frame_interval = 1.0 / cam.fps
        pre_event_flushed = False

        while True:
            start = time.time()

            if cam.is_override_active:
                if not pre_event_flushed:
                    pre_event_flushed = True
                    # Flush pre-event buffer: decode JPEGs and yield as
                    # raw frames. FFmpeg PTS assignment makes them play
                    # back at native fps (e.g. 300 frames = 10s @ 30fps).
                    buf_frames = list(cam.pre_event_buffer)
                    cam.pre_event_buffer.clear()
                    logger.info(
                        f"Privacy override: flushing {len(buf_frames)} "
                        f"pre-event frames for {cam.name}"
                    )
                    for jpeg_bytes in buf_frames:
                        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            yield img

                # --- PASS-THROUGH: latest real frame from capture thread ---
                frame = cam.latest_real_frame
                if frame is not None:
                    yield frame
                else:
                    yield np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            else:
                # --- SKELETON MODE: render from keypoints ---
                if pre_event_flushed:
                    pre_event_flushed = False
                    logger.info(
                        f"Privacy override ended for {cam.name}, back to skeleton"
                    )

                with cam.keypoints_lock:
                    keypoints = cam.latest_keypoints

                if keypoints:
                    frame = render_skeleton_bgr(keypoints, cam.width, cam.height)
                else:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

                yield frame

            # Maintain target FPS
            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_tcp_client(self, conn, addr, cam: CameraState):
        """Handle a single TCP client, sending raw BGR24 frames."""
        logger.info(f"go2rtc connected for {cam.name} from {addr}")
        try:
            for frame in self._generate_frames(cam):
                conn.sendall(frame.tobytes())
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"go2rtc disconnected for {cam.name}: {e}")
        finally:
            conn.close()

    def _run_tcp_server(self, cam: CameraState):
        """Serve raw BGR24 frames over TCP for go2rtc exec: source to pull.

        Accepts multiple connections; each new connection is handled in its
        own thread so go2rtc reconnects are never blocked.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", cam.tcp_port))
        server.listen(2)
        logger.info(
            f"TCP frame server for {cam.name} listening on port {cam.tcp_port} "
            f"({cam.width}x{cam.height} BGR24 @ {cam.fps}fps)"
        )

        while True:
            conn, addr = server.accept()
            t = threading.Thread(
                target=self._handle_tcp_client,
                args=(conn, addr, cam),
                daemon=True,
            )
            t.start()

    def _run_pre_event_capture(self, cam: CameraState):
        """Continuously capture real frames into ring buffer for pre-event footage.

        Also sets cam.latest_real_frame so override passthrough can use it
        without opening a separate RTSP connection.  Keeps running during
        override so live frames stay current.
        """
        backoff = 5
        max_backoff = 120

        while True:
            try:
                container = av.open(
                    cam.real_rtsp_url,
                    options={"rtsp_transport": "tcp"},
                )
                backoff = 5  # Reset on successful connection
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[0] != cam.height or img.shape[1] != cam.width:
                        img = cv2.resize(img, (cam.width, cam.height))

                    # Always update latest frame (used during override)
                    cam.latest_real_frame = img

                    # Only buffer for pre-event when NOT in override
                    if not cam.is_override_active:
                        _, jpeg = cv2.imencode(
                            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        cam.pre_event_buffer.append(jpeg.tobytes())
                container.close()
            except Exception as e:
                logger.warning(
                    f"Pre-event capture error for {cam.name}: {e} "
                    f"(retry in {backoff}s)"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def run(self):
        """Start the privacy proxy."""
        # Wait for dependent services (go2rtc, Frigate) to be ready
        logger.info("Waiting 10s for go2rtc/Frigate to initialize...")
        time.sleep(10)

        # Retry MQTT connection on startup (broker may not be ready yet)
        while True:
            try:
                self.mqtt_client.connect(self.mqtt_host, self.mqtt_port)
                break
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"MQTT not ready ({e}), retrying in 5s...")
                time.sleep(5)
        self.mqtt_client.loop_start()

        threads = []
        for cam in self.cameras.values():
            # TCP server thread (replaces FFmpeg RTSP push)
            t = threading.Thread(
                target=self._run_tcp_server,
                args=(cam,),
                name=f"tcp-{cam.name}",
                daemon=True,
            )
            t.start()
            threads.append(t)

            # Pre-event real frame capture thread
            t2 = threading.Thread(
                target=self._run_pre_event_capture,
                args=(cam,),
                name=f"pre-event-{cam.name}",
                daemon=True,
            )
            t2.start()
            threads.append(t2)

        logger.info(f"Privacy proxy running for {len(self.cameras)} cameras")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.mqtt_client.loop_stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/config/config.yaml")
    args = parser.parse_args()

    proxy = PrivacyProxy(args.config)
    proxy.run()
