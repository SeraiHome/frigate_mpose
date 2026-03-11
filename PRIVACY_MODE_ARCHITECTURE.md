# Privacy-Preserving Video Architecture

This document describes the privacy mode system for Serai's Frigate fork. It ensures that **real camera footage of the senior is never stored or transmitted** during normal operation. Instead, all outputs (web UI, recordings, video calls) show pose skeleton visualizations on a black background. When a fall is detected, the system temporarily switches to real camera footage for a configurable event window.

---

## Table of Contents

1. [Design Goals](#1-design-goals)
2. [The Recording Problem](#2-the-recording-problem)
3. [Solution Overview](#3-solution-overview)
4. [Component 1: SHM Frame Substitution (Frigate)](#4-component-1-shm-frame-substitution-frigate)
5. [Component 2: Privacy Proxy Sidecar](#5-component-2-privacy-proxy-sidecar)
6. [Component 3: MQTT Topics & Config](#6-component-3-mqtt-topics--config)
7. [Output Matrix](#7-output-matrix)
8. [Compute Budget](#8-compute-budget)
9. [Implementation Guide](#9-implementation-guide)
10. [Testing Plan](#10-testing-plan)
11. [Future: Reconstructed Avatars](#11-future-reconstructed-avatars)

---

## 1. Design Goals

| Goal | Detail |
|---|---|
| **Privacy by default** | No real RGB frames stored or transmitted during normal operation |
| **Fall detection unaffected** | ST-GCN operates on COCO keypoint coordinates, not pixels — detection pipeline always sees real frames |
| **Event override** | On fall detection, real footage captured for a configurable window (pre-event buffer + post-event) |
| **Per-camera control** | Some cameras (e.g., front door) can remain in real mode; bedroom cameras get privacy mode |
| **Minimal Frigate changes** | ~100 lines of changes inside Frigate; bulk of logic in an external sidecar |
| **No ML duplication** | Privacy proxy uses MQTT keypoint data from Frigate — no second MediaPipe or ST-GCN instance |

---

## 2. The Recording Problem

Frigate runs **two independent FFmpeg processes** per camera:

```
                         +------------------------------+
                         |        Real Camera (RTSP)     |
                         +-------------+----------------+
                                       |
                       +---------------+---------------+
                       |                               |
                 +-----v------+                  +-----v------+
                 |  DETECT     |                  |   RECORD    |
                 |  FFmpeg     |                  |   FFmpeg    |
                 |  (substream)|                  | (mainstream)|
                 +-----+------+                  +-----+------+
                       |                               |
                 Shared Memory                    Direct to disk
                 (we CAN modify)                  (we CANNOT modify)
                       |                               |
                 +-----v------+                  +-----v------+
                 | Pose detect |                  |  MP4 segs   |
                 | Motion det  |                  | /tmp/cache/ |
                 | Web UI out  |                  | -> storage  |
                 +-------------+                  +-------------+
```

**Detect stream**: Frames land in shared memory (`SharedMemoryFrameManager`). We can substitute skeleton frames here. This affects the web UI (JSMPEG), birdseye, and snapshots.

**Record stream**: A separate FFmpeg process reads directly from the camera RTSP source and writes MP4 segments to disk. These frames **never pass through the detect pipeline**. We cannot intercept them from within Frigate's code.

**Key files involved:**
- `frigate/video.py` — `capture_frames()` writes to SHM; `process_frames()` reads from SHM
- `frigate/output/camera.py` — reads SHM frames for JSMPEG web output
- `frigate/output/birdseye.py` — reads SHM frames for multi-camera composite
- `frigate/record/maintainer.py` — manages MP4 segments from the record FFmpeg (independent of SHM)

This means **modifying SHM frames only covers web UI, not recordings**. For recordings, we need to control what the record FFmpeg reads.

---

## 3. Solution Overview

Insert a **privacy proxy** between the real camera and Frigate's record stream, using go2rtc (already running on every Serai edge device) as the RTSP router:

```
                     +------------------+
                     |  Real Camera     |
                     |  rtsp://cam/main |
                     +--------+---------+
                              |
                     +--------v---------+
                     |     go2rtc       |
                     |  "cam_real"      |  <-- real source, always available
                     +--+----------+---+
                        |          |
           +------------v--+  +---v--------------------------+
           | Frigate DETECT |  |  Privacy Proxy (Python)      |
           | reads cam_real |  |                              |
           | (needs real    |  |  Normal: MQTT keypoints      |
           |  frames for    |  |    -> skeleton on black      |
           |  ML accuracy)  |  |    -> RTSP output            |
           +-------+--------+  |                              |
                   |           |  Fall event: pass-through    |
                   |           |    real frames from cam_real |
         publishes |           +-------------+----------------+
         keypoints |                         |
         via MQTT  |            +------------v--------------+
                   +----------->|  Frigate RECORD            |
                               |  reads cam_priv            |
                               |  (skeleton normally,       |
                               |   real RGB during events)  |
                               +----------------------------+
```

**Three components:**

| # | Component | Where | Lines | What it does |
|---|-----------|-------|-------|-------------|
| 1 | SHM frame substitution | `frigate/video.py` + new `privacy_renderer.py` | ~100 | Replaces detect frames with skeleton render for web UI |
| 2 | Privacy proxy sidecar | New Docker service in edge compose | ~250 | Generates skeleton RTSP stream for recordings + LiveKit |
| 3 | MQTT topics + config | `integration.py` + Frigate config schema | ~50 | Publishes keypoints; receives privacy override commands |

---

## 4. Component 1: SHM Frame Substitution (Frigate)

### What it does

After pose detection runs on each frame, replace the frame in shared memory with a skeleton-rendered version. Everything downstream (web UI, JSMPEG, birdseye, snapshots) sees skeletons instead of real footage.

### New file: `frigate/pose_detection/privacy_renderer.py`

```python
"""Privacy-preserving skeleton renderer.

Renders COCO 17-keypoint pose skeletons on a black background.
Returns YUV I420 frames compatible with Frigate's SharedMemoryFrameManager.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# COCO 17-keypoint skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # head: nose-eyes-ears
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# Color palette for skeleton parts (BGR)
COLORS = {
    "head": (255, 200, 50),      # light blue
    "torso": (50, 255, 50),      # green
    "arms": (50, 200, 255),      # orange
    "legs": (255, 50, 200),      # purple
    "keypoint": (255, 255, 255), # white
}

# Map edge index to body part for coloring
EDGE_COLORS = (
    [COLORS["head"]] * 4 +
    [COLORS["torso"]] * 1 +
    [COLORS["arms"]] * 4 +
    [COLORS["torso"]] * 2 +
    [COLORS["torso"]] * 1 +
    [COLORS["legs"]] * 4
)

# Confidence threshold for drawing a keypoint/edge
MIN_CONFIDENCE = 0.3


def render_skeleton_bgr(
    tracked_poses: Dict,
    width: int,
    height: int,
    keypoint_radius: int = 5,
    line_thickness: int = 3,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render pose skeletons on a solid background.

    Args:
        tracked_poses: Dict of tracked pose objects, each with .keypoints
                       as list of (x, y, confidence) tuples in COCO 17-point format.
        width: Frame width in pixels.
        height: Frame height in pixels.
        keypoint_radius: Radius of keypoint circles.
        line_thickness: Thickness of skeleton lines.
        bg_color: Background color in BGR.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for pose in tracked_poses.values():
        kps = pose.keypoints  # [(x, y, confidence), ...]

        # Draw skeleton edges
        for idx, (i, j) in enumerate(SKELETON_EDGES):
            if kps[i][2] > MIN_CONFIDENCE and kps[j][2] > MIN_CONFIDENCE:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                color = EDGE_COLORS[idx]
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        # Draw keypoint circles
        for kp in kps:
            if kp[2] > MIN_CONFIDENCE:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(canvas, pt, keypoint_radius, COLORS["keypoint"], -1, cv2.LINE_AA)

    return canvas


def render_skeleton_yuv(
    tracked_poses: Dict,
    frame_shape: Tuple[int, int],
    keypoint_radius: int = 5,
    line_thickness: int = 3,
) -> np.ndarray:
    """Render pose skeletons and return as YUV I420 frame.

    This format is compatible with Frigate's SharedMemoryFrameManager.

    Args:
        tracked_poses: Dict of tracked pose objects.
        frame_shape: (height, width) of the frame.

    Returns:
        YUV I420 numpy array of shape (height * 1.5, width).
    """
    height, width = frame_shape[0], frame_shape[1]
    bgr = render_skeleton_bgr(tracked_poses, width, height, keypoint_radius, line_thickness)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv


def render_no_detection_yuv(frame_shape: Tuple[int, int]) -> np.ndarray:
    """Return a black YUV frame (no poses detected).

    Used when privacy mode is on but no poses are visible — shows pure black
    rather than the real camera frame.
    """
    height, width = frame_shape[0], frame_shape[1]
    black = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.cvtColor(black, cv2.COLOR_BGR2YUV_I420)
```

### Modification: `frigate/video.py`

In `process_frames()`, after the pose detection call (around line ~1084), add:

```python
# --- PRIVACY MODE: Replace SHM frame with skeleton render ---
from frigate.pose_detection.privacy_renderer import render_skeleton_yuv, render_no_detection_yuv

# After pose detection runs and tracked_poses is available:
if camera_config.pose.privacy_mode:
    privacy_override_active = self._check_privacy_override(current_time)

    if not privacy_override_active:
        if tracked_poses:
            skeleton_frame = render_skeleton_yuv(
                tracked_poses,
                (frame_shape[0], frame_shape[1]),
            )
        else:
            skeleton_frame = render_no_detection_yuv(
                (frame_shape[0], frame_shape[1]),
            )

        # Overwrite the frame in shared memory
        current_frame = frame_manager.get(frame_name, frame_shape_yuv)
        if current_frame is not None:
            current_frame[:] = skeleton_frame
# --- END PRIVACY MODE ---
```

Add a helper method for the privacy override check:

```python
def _check_privacy_override(self, current_time: float) -> bool:
    """Check if privacy override is active (real frames requested during event)."""
    if not hasattr(self, '_privacy_override_until'):
        self._privacy_override_until = 0.0
    return current_time < self._privacy_override_until
```

### Modification: `frigate/config/camera/pose.py`

Add two new config fields to the pose config schema:

```python
# Add to PoseConfig class:
privacy_mode: bool = Field(
    default=False,
    title="Privacy mode: show skeleton instead of real camera in web UI and snapshots.",
)
publish_keypoints: bool = Field(
    default=False,
    title="Publish pose keypoints to MQTT for the privacy proxy sidecar.",
)
```

---

## 5. Component 2: Privacy Proxy Sidecar

### What it does

A lightweight Python service that generates an RTSP stream of skeleton-rendered frames. Frigate's record stream reads this instead of the real camera. On fall events, it switches to passing through real camera frames.

### Architecture

```
+-------------------+        +-----------------+        +------------------+
| Local Mosquitto   |  MQTT  | Privacy Proxy   |  RTSP  | go2rtc           |
| frigate/{cam}/    +------->| (per camera)    +------->| "cam_priv"       |
|   pose_keypoints  |        |                 |        |                  |
|   privacy_override|        | Normal: render  |        | Frigate RECORD   |
+-------------------+        |   skeleton      |        |   reads this     |
                             | Event: read     |        |                  |
+-------------------+  RTSP  |   cam_real,     |        | LiveKit edge     |
| go2rtc "cam_real" +------->|   pass through  |        |   reads this     |
+-------------------+        +-----------------+        +------------------+
```

### Service structure

```
serai-edge-setup/privacy-proxy/
├── privacy_proxy.py        # Main entry point (~250 lines)
├── skeleton_renderer.py    # Shared rendering code (from privacy_renderer.py)
├── config.yaml             # Per-camera privacy settings
├── Dockerfile
├── requirements.txt
└── README.md
```

### `privacy_proxy.py` — Core Logic

```python
"""Privacy proxy: generates skeleton RTSP stream for Frigate recording.

Subscribes to pose keypoints via MQTT, renders skeleton frames,
outputs as RTSP stream via go2rtc. On fall events, switches to
pass-through of real camera frames.

Usage:
    python privacy_proxy.py --config /config/config.yaml
"""

import argparse
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
    real_rtsp_url: str              # go2rtc cam_real URL
    output_rtsp_url: str            # Where to publish cam_priv

    # Privacy state
    override_until: float = 0.0     # Timestamp when override expires
    latest_keypoints: Optional[list] = None
    keypoints_lock: threading.Lock = field(default_factory=threading.Lock)

    # Pre-event buffer (ring buffer of real JPEG frames)
    pre_event_buffer: deque = field(default_factory=lambda: deque(maxlen=150))
    # 150 frames @ 5fps = 30 seconds

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
                output_rtsp_url=cam_cfg["output_rtsp_url"],
            )
            self.cameras[cam.name] = cam

        self.pre_event_seconds = config.get("pre_event_seconds", 30)
        self.post_event_seconds = config.get("post_event_seconds", 60)

        # MQTT
        self.mqtt_host = config.get("mqtt_host", "localhost")
        self.mqtt_port = config.get("mqtt_port", 1883)
        self.mqtt_client = mqtt.Client(client_id="privacy-proxy")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        logger.info(f"MQTT connected (rc={rc})")
        for cam_name in self.cameras:
            # Subscribe to keypoints from Frigate
            client.subscribe(f"frigate/{cam_name}/pose_keypoints", qos=0)
            # Subscribe to privacy override commands
            client.subscribe(f"frigate/{cam_name}/privacy_override", qos=1)
        logger.info(f"Subscribed to {len(self.cameras)} camera topics")

    def _on_mqtt_message(self, client, userdata, msg):
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

                # Flush pre-event buffer to sidecar file
                self._write_pre_event_buffer(cam)
            except json.JSONDecodeError:
                pass

    def _write_pre_event_buffer(self, cam: CameraState):
        """Write buffered real frames to a sidecar video file."""
        if not cam.pre_event_buffer:
            return

        output_path = f"/media/frigate/pre_event_{cam.name}_{int(time.time())}.mp4"
        try:
            container = av.open(output_path, mode="w")
            stream = container.add_stream("h264", rate=cam.fps)
            stream.width = cam.width
            stream.height = cam.height

            for frame_bytes in cam.pre_event_buffer:
                img = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()
            logger.info(f"Pre-event buffer written: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write pre-event buffer: {e}")

    def _generate_frames(self, cam: CameraState):
        """Generator: yield BGR frames (skeleton or real) for a camera."""
        real_stream = None
        frame_interval = 1.0 / cam.fps

        while True:
            start = time.time()

            if cam.is_override_active:
                # --- PASS-THROUGH MODE: real frames ---
                if real_stream is None:
                    try:
                        real_stream = av.open(cam.real_rtsp_url)
                    except Exception as e:
                        logger.error(f"Cannot open {cam.real_rtsp_url}: {e}")
                        yield np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                        time.sleep(frame_interval)
                        continue

                try:
                    for frame in real_stream.decode(video=0):
                        img = frame.to_ndarray(format="bgr24")
                        if img.shape[0] != cam.height or img.shape[1] != cam.width:
                            img = cv2.resize(img, (cam.width, cam.height))
                        yield img
                        break
                except (av.AVError, StopIteration):
                    real_stream = None
                    yield np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            else:
                # --- SKELETON MODE: render from keypoints ---
                if real_stream is not None:
                    real_stream.close()
                    real_stream = None
                    logger.info(f"Privacy override ended for {cam.name}, back to skeleton")

                with cam.keypoints_lock:
                    keypoints = cam.latest_keypoints

                if keypoints:
                    # Build a mock tracked_poses dict for the renderer
                    mock_poses = {}
                    for i, pose in enumerate(keypoints):
                        kps = pose.get("keypoints", [])
                        if len(kps) >= 17:
                            mock_poses[i] = type("Pose", (), {
                                "keypoints": [(k[0], k[1], k[2]) for k in kps]
                            })()
                    frame = render_skeleton_bgr(mock_poses, cam.width, cam.height)
                else:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

                yield frame

            # Maintain target FPS
            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_camera_output(self, cam: CameraState):
        """Run RTSP output for one camera via FFmpeg pipe to go2rtc."""
        import subprocess

        logger.info(f"Starting RTSP output for {cam.name} -> {cam.output_rtsp_url}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{cam.width}x{cam.height}",
            "-r", str(cam.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            cam.output_rtsp_url,
        ]

        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            for frame in self._generate_frames(cam):
                # Buffer real frames for pre-event capture (JPEG compressed)
                if not cam.is_override_active:
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    cam.pre_event_buffer.append(jpeg.tobytes())

                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()
        except BrokenPipeError:
            logger.error(f"FFmpeg pipe broken for {cam.name}")
        finally:
            proc.stdin.close()
            proc.wait()

    def _run_pre_event_capture(self, cam: CameraState):
        """Continuously capture real frames into the ring buffer for pre-event footage."""
        while True:
            if not cam.is_override_active:
                try:
                    container = av.open(cam.real_rtsp_url)
                    for frame in container.decode(video=0):
                        if cam.is_override_active:
                            break
                        img = frame.to_ndarray(format="bgr24")
                        if img.shape[0] != cam.height or img.shape[1] != cam.width:
                            img = cv2.resize(img, (cam.width, cam.height))
                        _, jpeg = cv2.imencode(
                            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        cam.pre_event_buffer.append(jpeg.tobytes())
                        time.sleep(1.0 / cam.fps)
                    container.close()
                except Exception as e:
                    logger.error(f"Pre-event capture error for {cam.name}: {e}")
                    time.sleep(1.0)
            else:
                time.sleep(0.5)

    def run(self):
        """Start the privacy proxy."""
        # Connect MQTT
        self.mqtt_client.connect(self.mqtt_host, self.mqtt_port)
        self.mqtt_client.loop_start()

        # Start per-camera threads
        threads = []
        for cam in self.cameras.values():
            # RTSP output thread
            t = threading.Thread(
                target=self._run_camera_output,
                args=(cam,),
                name=f"output-{cam.name}",
                daemon=True,
            )
            t.start()
            threads.append(t)

            # Pre-event capture thread
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
```

### `config.yaml`

```yaml
# Privacy proxy configuration
mqtt_host: localhost
mqtt_port: 1883

pre_event_seconds: 30    # Ring buffer duration for pre-event real frames
post_event_seconds: 60   # How long to show real frames after fall detected

cameras:
  - name: living_room
    width: 1280
    height: 720
    fps: 5
    real_rtsp_url: rtsp://go2rtc:1984/living_room_real
    output_rtsp_url: rtsp://localhost:8554/living_room_priv

  - name: bedroom
    width: 1280
    height: 720
    fps: 5
    real_rtsp_url: rtsp://go2rtc:1984/bedroom_real
    output_rtsp_url: rtsp://localhost:8554/bedroom_priv
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "privacy_proxy.py", "--config", "/config/config.yaml"]
```

### `requirements.txt`

```
opencv-python-headless>=4.8
paho-mqtt>=1.6
av>=12.0
pyyaml>=6.0
numpy>=1.24
```

### Docker Compose addition (edge `compose.yaml`)

```yaml
services:
  privacy-proxy:
    build: ./privacy-proxy
    restart: unless-stopped
    volumes:
      - ./privacy-proxy/config.yaml:/config/config.yaml:ro
      - frigate_media:/media/frigate   # For pre-event buffer writes
    depends_on:
      - mosquitto
      - go2rtc
    network_mode: host  # Needs access to go2rtc and Mosquitto on localhost
```

---

## 6. Component 3: MQTT Topics & Config

### New MQTT Topics

Add keypoint publishing to `frigate/pose_detection/integration.py`:

```python
# In PoseDetectionIntegration, after poses are tracked:

def _publish_keypoints(self, camera_name: str, tracked_poses: dict, frame_time: float):
    """Publish pose keypoints to MQTT for the privacy proxy."""
    if not self.config.pose.publish_keypoints:
        return

    poses_data = []
    for pose_id, pose in tracked_poses.items():
        poses_data.append({
            "id": str(pose_id),
            "keypoints": [
                [float(kp[0]), float(kp[1]), float(kp[2])]
                for kp in pose.keypoints
            ],
        })

    payload = json.dumps({
        "timestamp": frame_time,
        "frame_width": self.config.frame_shape[1],
        "frame_height": self.config.frame_shape[0],
        "poses": poses_data,
    })

    self.mqtt_client.publish(
        f"frigate/{camera_name}/pose_keypoints",
        payload,
        qos=0,  # Fire-and-forget for performance
        retain=False,
    )
```

### Privacy Override Topic

Published by the escalation engine (config-agent) when a fall is detected:

```
Topic:   frigate/{camera_name}/privacy_override
QoS:     1 (at least once)
Payload: {
    "mode": "real",              // "real" = show real frames, "skeleton" = revert
    "duration_seconds": 60,      // How long to show real frames
    "event_id": "abc-123",       // Alert ID for correlation
    "triggered_at": 1710000000.0 // Unix timestamp
}
```

The escalation engine already subscribes to `frigate/events` for fall detection. Adding the privacy override publish is a one-line addition:

```python
# In config-agent escalation_engine.py, when fall detected:
mqtt_client.publish(
    f"frigate/{camera_name}/privacy_override",
    json.dumps({"mode": "real", "duration_seconds": 60, "event_id": alert_id}),
    qos=1,
)
```

### Frigate Camera Config

```yaml
cameras:
  living_room:
    ffmpeg:
      inputs:
        # Real source for detection (ML always sees real frames)
        - path: rtsp://go2rtc:1984/living_room_real
          roles: [detect]
        # Privacy proxy output for recording
        - path: rtsp://go2rtc:1984/living_room_priv
          roles: [record]
    pose:
      enabled: true
      privacy_mode: true         # NEW: skeleton substitution in web UI
      publish_keypoints: true    # NEW: publish keypoints for proxy
      detector:
        type: mediapipe_task
        model_complexity: 2
      activity_detector:
        type: stgcn_fall
        model_path: /config/models/stgcn_fall.tflite
        confidence_threshold: 0.5

  # Example: front door camera — NO privacy mode
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://go2rtc:1984/front_door
          roles: [detect, record]
    pose:
      enabled: false
      privacy_mode: false
```

### go2rtc Config

```yaml
streams:
  # Real camera sources
  living_room_real:
    - rtsp://192.168.1.100/stream1
  bedroom_real:
    - rtsp://192.168.1.101/stream1
  front_door:
    - rtsp://192.168.1.102/stream1

  # Privacy proxy skeleton streams
  living_room_priv:
    - rtsp://privacy-proxy:8554/living_room_priv
  bedroom_priv:
    - rtsp://privacy-proxy:8554/bedroom_priv
```

---

## 7. Output Matrix

What each output shows in each mode:

| Output | Normal (no event) | Fall Detected (event window) | Privacy Disabled |
|---|---|---|---|
| **Frigate web UI** | Skeleton on black (SHM substitution) | Real RGB (override active) | Real RGB |
| **Frigate recordings** | Skeleton on black (from `cam_priv`) | Real RGB (proxy pass-through) | Real RGB |
| **Frigate snapshots** | Skeleton (from SHM) | Real RGB | Real RGB |
| **LiveKit video call** | Skeleton (edge reads `cam_priv`) | Real RGB (proxy switches) | Real RGB |
| **Birdseye composite** | Skeleton (reads SHM) | Real RGB | Real RGB |
| **Home Assistant** | Skeleton (via Frigate integration) | Real RGB | Real RGB |
| **Pre-event sidecar** | N/A | Real RGB (30s buffer dump) | N/A |

---

## 8. Compute Budget

### No ML Duplication

The privacy proxy does **NOT** run MediaPipe or ST-GCN. It only:
1. Receives 17 keypoint coordinates via MQTT (~500 bytes JSON per frame)
2. Draws lines and circles on a black canvas (OpenCV, sub-millisecond)
3. Encodes the skeleton frame to RTSP via FFmpeg

The fall detection pipeline (MediaPipe + ST-GCN) runs **only once**, inside Frigate's detect process, on real frames.

### CPU Estimate (RPi5, per camera)

| Process | What | CPU % |
|---|---|---|
| Frigate detect FFmpeg | Decode camera substream | ~10-15% |
| MediaPipe pose detection | Per-frame inference | ~20-30% |
| ST-GCN fall classification | Temporal window inference | ~5-10% |
| **Privacy proxy (skeleton mode)** | **MQTT receive + OpenCV draw + FFmpeg encode** | **~1-3%** |
| Frigate record FFmpeg | Record from `cam_priv` | ~5% |
| **Total** | | **~41-63%** |

The privacy proxy adds **1-3% CPU** during normal operation. During fall events (pass-through mode, ~60s), it adds ~5-10% for RTSP decode + re-encode, but this is short-lived.

### Memory

| Component | RAM |
|---|---|
| Privacy proxy base | ~30 MB |
| Pre-event ring buffer (30s @ 5fps, 720p JPEG) | ~50 MB per camera |
| **Total (2 cameras)** | **~130 MB** |

---

## 9. Implementation Guide

### Step-by-step

| Step | What | Where | Depends on |
|---|---|---|---|
| **1** | Create `privacy_renderer.py` | `frigate/pose_detection/` | Nothing |
| **2** | Add `privacy_mode` and `publish_keypoints` to pose config schema | `frigate/config/camera/pose.py` | Nothing |
| **3** | Add SHM substitution in `process_frames()` | `frigate/video.py` | Steps 1, 2 |
| **4** | Add keypoint MQTT publishing in pose integration | `frigate/pose_detection/integration.py` | Step 2 |
| **5** | Test web UI shows skeletons (SHM substitution only) | Frigate dev environment | Steps 1-4 |
| **6** | Build privacy proxy sidecar | `serai-edge-setup/privacy-proxy/` | Step 4 |
| **7** | Configure go2rtc dual streams (`cam_real` + `cam_priv`) | `go2rtc.yaml` | Step 6 |
| **8** | Update Frigate config: detect reads `cam_real`, record reads `cam_priv` | `config.yml` | Steps 6, 7 |
| **9** | Add privacy proxy to edge Docker Compose | `compose.yaml` | Steps 6-8 |
| **10** | Add privacy override publish to escalation engine | `config-agent/escalation_engine.py` | Step 6 |
| **11** | End-to-end test: normal mode + fall event override | Edge device | Steps 1-10 |

### Estimated Effort

| Component | Lines | Time |
|---|---|---|
| `privacy_renderer.py` | ~100 | 1 day |
| Frigate config schema changes | ~10 | 0.5 day |
| SHM substitution in `video.py` | ~30 | 0.5 day |
| Keypoint MQTT publish in `integration.py` | ~30 | 0.5 day |
| Privacy proxy sidecar | ~250 | 3 days |
| go2rtc + Frigate config + Docker Compose | ~30 | 0.5 day |
| Escalation engine override publish | ~5 | 0.5 day |
| Testing + validation | — | 2-3 days |
| **Total** | **~455** | **~8-10 days** |

---

## 10. Testing Plan

### Unit Tests

| Test | What | How |
|---|---|---|
| Skeleton rendering | Verify correct keypoints drawn, edges connected | Pass known keypoints, assert non-zero pixels at expected positions |
| YUV conversion | Verify output shape matches Frigate SHM format | `assert yuv.shape == (h * 3 // 2, w)` |
| No-detection frame | Verify all-black frame returned | `assert np.all(frame == 0)` |
| Config parsing | Verify `privacy_mode` and `publish_keypoints` fields | Load config YAML, check values |

### Integration Tests

| Test | What | Expected |
|---|---|---|
| SHM substitution | Enable privacy mode, check Frigate web UI | JSMPEG stream shows skeleton on black |
| Recording privacy | Check recorded MP4 segments | Frames show skeleton, not real camera |
| Fall override | Trigger fall event via MQTT | Recording switches to real frames for duration |
| Override expiry | Wait for duration to expire | Recording reverts to skeleton |
| Pre-event buffer | Trigger fall, check sidecar file | MP4 with 30s of real pre-event footage |
| LiveKit call | Initiate call during normal mode | Video shows skeleton |
| LiveKit call + fall | Call during fall event | Video shows real frames |

### Validation with Test Videos

Use the Le2i fall detection dataset videos (already in `config/Lei2/`):
1. Run with privacy mode enabled
2. Verify fall is still detected (ST-GCN operates on keypoints, not pixels)
3. Verify skeleton rendering matches pose positions
4. Verify override switches to real footage within 1 frame

---

## 11. Future: Reconstructed Avatars

The skeleton renderer is a single function (`render_skeleton_bgr`). It can be swapped for more sophisticated visualizations without changing any pipeline code:

| Phase | Visual | Complexity | Renderer change |
|---|---|---|---|
| **v1 (POC)** | Stick-figure skeleton (colored lines + dots on black) | Trivial | Current implementation |
| **v2 (MVP)** | Styled skeleton with glow effects, head circle, body outline | Low | Add `cv2.GaussianBlur` overlay, ellipse for head |
| **v3 (GA)** | 2D cartoon avatar driven by keypoints | Medium | Map keypoints to sprite positions (e.g., a simple humanoid SVG) |
| **v4 (Post-GA)** | 3D mesh avatar (SMPL-lite) rendered via OpenGL | High | Use `pyrender` or `trimesh` for offscreen rendering |

The privacy proxy's RTSP pipeline, MQTT integration, and Frigate hooks remain identical across all visual styles — only the renderer function changes.

---

## File Reference

| File | Purpose |
|---|---|
| `frigate/pose_detection/privacy_renderer.py` | **NEW** — Skeleton rendering (YUV + BGR) |
| `frigate/pose_detection/integration.py` | **MODIFY** — Add keypoint MQTT publishing |
| `frigate/video.py` | **MODIFY** — Add SHM frame substitution |
| `frigate/config/camera/pose.py` | **MODIFY** — Add `privacy_mode`, `publish_keypoints` fields |
| `serai-edge-setup/privacy-proxy/privacy_proxy.py` | **NEW** — Sidecar service |
| `serai-edge-setup/privacy-proxy/config.yaml` | **NEW** — Proxy config |
| `serai-edge-setup/privacy-proxy/Dockerfile` | **NEW** — Container build |
| `serai-edge-setup/config-agent/escalation_engine.py` | **MODIFY** — Add privacy override publish |
| `serai-edge-setup/go2rtc.yaml` | **MODIFY** — Add `cam_real` + `cam_priv` streams |
| `serai-edge-setup/compose.yaml` | **MODIFY** — Add privacy-proxy service |
| `config.yml` (Frigate) | **MODIFY** — Dual inputs (detect: real, record: priv) |
