# Privacy-Preserving Video Architecture

This document describes the privacy mode system for Serai's Frigate fork. **Real camera footage of the senior is never stored or transmitted** during normal operation. All outputs (web UI, birdseye, recordings) show pose skeleton visualizations. When a fall is detected, the system temporarily switches to real camera footage for a configurable event window.

---

## Table of Contents

1. [Design Goals](#1-design-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [The Three Privacy Layers](#3-the-three-privacy-layers)
4. [Fall Detection Override Flow](#4-fall-detection-override-flow)
5. [Stream Configuration](#5-stream-configuration)
6. [Output Matrix](#7-output-matrix)
7. [Key Files](#8-key-files)
8. [Dev Environment](#9-dev-environment)

---

## 1. Design Goals

| Goal | Detail |
|---|---|
| **Privacy by default** | No real RGB frames stored or transmitted during normal operation |
| **Fall detection unaffected** | ST-GCN operates on COCO keypoint coordinates, not pixels -- detection pipeline always sees real frames |
| **Event override** | On fall detection, real footage captured for a configurable window (pre-event buffer + post-event) |
| **Per-camera control** | Some cameras can remain in real mode; privacy cameras get skeleton substitution |
| **No ML duplication** | Privacy proxy uses MQTT keypoint data from Frigate -- no second MediaPipe or ST-GCN instance |

---

## 2. Architecture Overview

```
                     +------------------+
                     |  Real Camera     |
                     |  (iPhone RTSP)   |
                     +--------+---------+
                              |
                     +--------v---------+
                     |     go2rtc       |
                     |  "iphone_real"   |  <-- real source, always available
                     +--+----------+---+
                        |          |
           +------------v--+  +---v--------------------------+
           | Frigate DETECT |  |  Privacy Proxy Sidecar       |
           | reads          |  |                              |
           | iphone_real    |  |  1. MQTT keypoints -> render |
           | (real frames   |  |     skeleton BGR24 frames    |
           |  for ML)       |  |  2. Serve over TCP :9000     |
           +-------+--------+  |  3. Pre-event JPEG buffer    |
                   |           |  4. go2rtc API switching      |
                   |           +---+-----------+--------------+
          SHM substitution         |           |
          (skeleton YUV)    +------v---+  +----v-----------+
                   |        | Frigate  |  | go2rtc API     |
                   v        | RECORD   |  | PUT /api/      |
           +------+------+  | reads TCP|  | streams?name=  |
           | Web UI      |  | :9000    |  | iphone         |
           | Birdseye    |  +----------+  +----------------+
           | Snapshots   |                  |
           | (JSMPEG)    |           +------v--------+
           +-------------+           | Live View     |
                                     | (WebRTC/MSE)  |
                                     | switched on   |
                                     | override      |
                                     +---------------+
```

---

## 3. The Three Privacy Layers

### Layer 1: SHM Frame Substitution (Frigate-internal)

**Covers:** JSMPEG live view, birdseye, snapshots, API image endpoints

After pose detection runs on each real frame in `process_frames()`, the frame in shared memory is replaced with a skeleton render. Everything downstream that reads SHM sees skeletons.

- **File:** `frigate/video.py` ~line 1270
- **Renderer:** `frigate/pose_detection/privacy_renderer.py` (YUV I420 output)
- **Background modes:**
  - `black` (default): solid black canvas
  - `scene`: uses `motion_detector.avg_frame` (grayscale running average of static room, zero extra cost)
- **Override:** Uses `CameraMetrics.privacy_override_until` shared `mp.Value` (cross-process safe between main process and camera subprocess)

### Layer 2: Privacy Proxy TCP Stream (Recording)

**Covers:** Frigate recordings (MP4 segments)

Frigate runs two independent FFmpeg processes per camera. The record FFmpeg reads directly from its input source, bypassing SHM entirely. To control what gets recorded, we insert the privacy proxy between the camera and Frigate's record stream.

- **Transport:** Raw BGR24 frames over TCP socket at 30fps (native camera fps)
- **Frigate reads:** `tcp://privacy-proxy:9000` with `-f rawvideo -pix_fmt bgr24 -video_size 1280x720 -framerate 30`
- **Frigate encodes:** `-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 30 -an`
- **Normal mode:** Proxy renders skeleton frames from MQTT keypoints
- **Override mode:** Proxy flushes pre-event JPEG buffer (10s at native fps) into TCP stream, then passes through live `latest_real_frame` from its RTSP capture thread
- **Pre-event buffer:** 300 JPEG frames (~10s at 30fps, ~15MB), flushed on override for full-quality fall footage

### Layer 3: go2rtc Live View Switching (WebRTC/MSE)

**Covers:** Live view via WebRTC/MSE during override

Normally, no `iphone` stream exists in go2rtc, so Frigate's live view falls back to JSMPEG (which reads SHM skeletons). During override, the proxy dynamically creates an `iphone` stream in go2rtc pointing to the real camera, enabling WebRTC/MSE live view of real footage.

- **API:** `PUT http://go2rtc:1984/api/streams?name={cam}&src=...`
- **Normal:** `iphone` stream points to dead URL (`rtsp://0.0.0.0:0/privacy`) -- WebRTC fails, JSMPEG fallback shows SHM skeletons
- **Override:** `iphone` stream points to `rtsp://127.0.0.1:8554/iphone_real` -- WebRTC shows real camera
- **Watchdog:** Proxy polls every 2s, switches back to privacy when override expires
- **Config:** `go2rtc_api_url` in proxy config

---

## 4. Fall Detection Override Flow

```
Fall detected by ST-GCN (in Frigate pose pipeline)
        |
        v
PoseConsumer (frigate/track/pose_consumer.py)
        |
        +---> Sets CameraMetrics.privacy_override_until (shared mp.Value)
        |     -> SHM stops substituting skeletons (Layer 1)
        |     -> Birdseye/JSMPEG show real frames
        |
        +---> Publishes MQTT frigate/{cam}/privacy_override
              -> Privacy proxy receives override
              |
              +---> Flushes pre-event buffer into TCP stream (Layer 2)
              |     -> Recording captures 10s of pre-fall footage
              |
              +---> Switches to live real frame passthrough (Layer 2)
              |     -> Recording continues with real frames
              |
              +---> PUT go2rtc API: iphone -> iphone_real (Layer 3)
                    -> WebRTC live view shows real camera

        ... post_event_seconds later (default 60s) ...

Override expires
        |
        +---> SHM resumes skeleton substitution (Layer 1)
        +---> TCP stream resumes skeleton rendering (Layer 2)
        +---> Proxy PUT go2rtc API: iphone -> dead URL (Layer 3)
              -> WebRTC fails, JSMPEG fallback -> SHM skeletons
```

**Debouncing:** PoseConsumer tracks `_privacy_override_until` per camera to prevent re-triggering during an active override window.

---

## 5. Stream Configuration

### go2rtc (inside Frigate devcontainer)

```yaml
go2rtc:
  streams:
    # Real camera -- used by detect role + proxy pre-event capture
    # NOT named "iphone" -- prevents WebRTC from serving real frames
    iphone_real:
      - rtsp://192.168.1.71:8554/live.sdp
```

The `iphone` stream is **not** defined in config -- it is dynamically managed by the proxy via go2rtc API during override.

### Frigate Camera Config

```yaml
cameras:
  iphone:
    ffmpeg:
      inputs:
        # Real source for detection (ML always sees real frames)
        - path: rtsp://127.0.0.1:8554/iphone_real
          input_args: -rtsp_transport tcp
          roles:
            - detect
        # Privacy proxy output for recording
        - path: tcp://privacy-proxy:9000
          input_args: -f rawvideo -pix_fmt bgr24 -video_size 1280x720 -framerate 30
          roles:
            - record
      output_args:
        record: -f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1
          -strftime 1 -c:v libx264 -preset ultrafast -tune zerolatency
          -pix_fmt yuv420p -g 30 -an
    detect:
      enabled: false  # Pose detection runs, but object detection disabled
    pose:
      enabled: true
      privacy_mode: true
      privacy_background: scene
      publish_keypoints: true
      confidence_threshold: 0.1
    record:
      enabled: true
      alerts:
        retain: { days: 1, mode: all }
      detections:
        retain: { days: 1, mode: all }
```

### Privacy Proxy Config (`config.dev.yaml`)

```yaml
mqtt_host: mqtt
mqtt_port: 1883
pre_event_seconds: 30
post_event_seconds: 60
go2rtc_api_url: http://frigate-devcontainer:1984

cameras:
  - name: iphone
    width: 1280
    height: 720
    fps: 30
    real_rtsp_url: rtsp://frigate-devcontainer:8554/iphone_real
    tcp_port: 9000
```

---

## 6. Output Matrix

| Output | Normal (no event) | Fall Detected (override) | Privacy Disabled |
|---|---|---|---|
| **JSMPEG live view** | Skeleton (SHM substitution) | Real RGB (SHM override via mp.Value) | Real RGB |
| **WebRTC/MSE live** | Skeleton (JSMPEG fallback, no go2rtc stream) | Real RGB (go2rtc API switch) | Real RGB |
| **Birdseye** | Skeleton (SHM substitution) | Real RGB (SHM override via mp.Value) | Real RGB |
| **Recordings** | Skeleton (proxy TCP) | Real RGB (pre-event flush + passthrough) | Real RGB |
| **Snapshots** | Skeleton (SHM substitution) | Real RGB (SHM override) | Real RGB |

---

## 7. Key Files

| File | Purpose |
|---|---|
| `frigate/pose_detection/privacy_renderer.py` | Skeleton rendering (YUV for SHM + BGR for proxy) |
| `frigate/video.py` ~L670, ~L1270 | `_check_privacy_override()` + SHM frame substitution |
| `frigate/camera/__init__.py` | `privacy_override_until` shared mp.Value in CameraMetrics |
| `frigate/track/pose_consumer.py` | MQTT keypoint publishing + fall override trigger |
| `frigate/config/camera/pose.py` | `privacy_mode`, `privacy_background`, `publish_keypoints` fields |
| `frigate/record/maintainer.py` | Patched to handle `detect.enabled: false` cameras |
| `frigate/app.py` | Passes `camera_metrics` to PoseConsumer |
| `serai-edge-setup/privacy-proxy/privacy_proxy.py` | Sidecar: TCP server + go2rtc API + pre-event capture |
| `serai-edge-setup/privacy-proxy/skeleton_renderer.py` | Standalone skeleton renderer for proxy |
| `serai-edge-setup/privacy-proxy/config.dev.yaml` | Proxy dev config |
| `config/config.yml` | Frigate config with dual-stream camera setup |
| `docker-compose.yml` | Privacy proxy service + shared volume |

---

## 8. Dev Environment

| What | Detail |
|---|---|
| **go2rtc** | Runs inside devcontainer (s6), RTSP on :8554, API on :1984 |
| **MQTT** | `mqtt` container on port 1883 |
| **Proxy rebuild** | `docker compose up -d --build privacy-proxy` (from host) |
| **Frigate launch** | F5 in VSCode (debugpy, preLaunchTask kills stale FFmpeg) |
| **Stale FFmpeg** | Old ffmpeg processes can survive restarts and compete for proxy TCP. preLaunchTask in `.vscode/tasks.json` handles cleanup. |
| **Recording maintainer** | Patched to skip "unprocessed segments" check when `detect.enabled: false` |
| **go2rtc regen** | After config changes: `sudo python3 /usr/local/go2rtc/create_config.py && sudo pkill go2rtc` |
