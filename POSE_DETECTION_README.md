# Pose Detection & Activity Detection — SeraiHome Fork

This document describes the pose detection and activity detection subsystem implemented in this Frigate fork. The implementation follows Frigate's existing patterns while introducing **per-camera parallel processing** to preserve temporal coherence—a critical requirement for sequence-based activity models.

---

## Why Per-Camera Parallel Detectors? (Critical Architecture Decision)

**The original Frigate object detection pipeline uses a single detector process that ingests frames from all cameras through a shared queue.** This approach works well for stateless object detection where each frame is processed independently.

**However, for pose-based activity detection (e.g., fall detection with ST-GCN), this architecture fails:**

- Sequence models like ST-GCN require a **contiguous temporal history** of frames from a single camera
- When frames from multiple cameras are interleaved in a single queue, the per-camera temporal context is destroyed
- Activity detectors cannot maintain consistent per-camera pose histories when frames arrive out of order

**This fork enforces per-camera isolation:**

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Camera A   │────▶│ pose-detection-queue │────▶│ PoseDetectProcess A │
│  frames     │     │     (Camera A)       │     │  + Activity Detector│
└─────────────┘     └──────────────────────┘     └─────────────────────┘

┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Camera B   │────▶│ pose-detection-queue │────▶│ PoseDetectProcess B │
│  frames     │     │     (Camera B)       │     │  + Activity Detector│
└─────────────┘     └──────────────────────┘     └─────────────────────┘
```

Each camera gets:

- Its own `mp.Queue` for frame delivery
- Its own `PoseDetectProcess` running in a separate OS process
- Its own shared memory buffers (`pose-<camera>` and `pose-out-<camera>`)
- Its own activity detector instance with isolated temporal history

This guarantees:

- **Correct frame ordering** per camera
- **Consistent history lengths** for activity detectors
- **Easier debugging** (SHM names clearly tied to camera names)
- **Better resource isolation** (one slow camera doesn't block others)

---

## Architecture Overview

### Directory Structure

```
frigate/
├── pose_detection/           # Core pose detection pipeline
│   ├── base.py               # PoseDetectProcess, PoseDetectorRunner, RemotePoseDetector
│   ├── integration.py        # Per-camera pose detection integration
│   ├── tensor_utils.py       # Centralized tensor constants and helpers (NEW)
│   ├── util.py               # Pose utilities (bbox, similarity, formatting)
│   └── fall_processor/       # Storage of local models and weights
│
├── pose_detectors/           # Pose detector implementations
│   ├── detector_config.py    # BasePoseDetectorConfig, PoseModelConfig
│   └── plugins/
│       ├── cpu_pose.py       # CPU-based detector
│       ├── mediapipe_pose.py # MediaPipe legacy API
│       ├── mediapipe_task_pose.py  # MediaPipe Tasks API (recommended)
│       └── yolo_pose.py      # YOLO pose via ONNX Runtime
│
├── pose_activity_detectors/  # Activity detection from pose sequences
│   ├── base.py               # PoseActivityDetector base class
│   ├── detector_config.py    # Activity detector config classes
│   └── plugins/
│       ├── heuristic.py      # Rule-based activity detection
│       └── stgcn_fall.py     # ST-GCN fall detector (TFLite)
│
├── config/camera/pose.py     # Per-camera PoseConfig
├── track/                    # Pose tracking and consumers
└── app.py                    # Process lifecycle (start_pose_detectors)
```

---

## Tensor Handling & Optimizations

This fork introduces a centralized tensor utilities module ([tensor_utils.py](frigate/pose_detection/tensor_utils.py)) to eliminate redundant shape checks and reshaping operations across the codebase.

### Canonical Data Formats

| Data Type          | Shape          | Description                                       |
| ------------------ | -------------- | ------------------------------------------------- |
| COCO Keypoints     | `(17, 3)`      | x, y, confidence per keypoint                     |
| Kinetics Keypoints | `(18, 3)`      | Used by some ST-GCN models                        |
| Pose Output Row    | `(57,)`        | `[person_id, confidence, keypoints(51), bbox(4)]` |
| Pose Output Batch  | `(20, 57)`     | Up to 20 detected poses                           |
| Input Frame        | `(1, H, W, 3)` | NHWC format in shared memory                      |

### Constants (defined in `tensor_utils.py`)

```python
# Keypoint format
COCO_NUM_KEYPOINTS = 17
KINETICS_NUM_KEYPOINTS = 18
KEYPOINT_DIMS = 3  # x, y, confidence

# Pose output layout
POSE_OUTPUT_SIZE = 57
POSE_PERSON_ID_IDX = 0
POSE_CONFIDENCE_IDX = 1
POSE_KEYPOINTS_START = 2
POSE_KEYPOINTS_END = 53      # 2 + 17*3
POSE_BBOX_START = 53
POSE_BBOX_END = 57

MAX_POSES = 20
```

### Helper Functions

| Function                               | Purpose                           | Performance Note            |
| -------------------------------------- | --------------------------------- | --------------------------- |
| `ensure_keypoints_2d()`                | Normalize keypoints to `(17, 3)`  | Returns view when possible  |
| `extract_keypoints_from_pose_output()` | Extract keypoints from raw output | Uses slicing (no copy)      |
| `create_pose_output()`                 | Build standardized pose output    | Pre-allocated array         |
| `create_pose_output_batch()`           | Create empty batch array          | Single allocation           |
| `is_yuv_frame()`                       | Detect YUV format from shape      | Eliminates duplicate checks |
| `ensure_rgb_hwc()`                     | Convert to RGB HWC format         | Handles batch dimension     |
| `ensure_batch_nhwc()`                  | Add batch dimension if needed     | Minimal overhead            |

### Optimizations Applied

1. **ST-GCN Activity Detector**: Replaced `torch.stack().squeeze().unsqueeze()` chain with direct NumPy pre-allocation and single `torch.from_numpy()` conversion
2. **Keypoint Extraction**: Uses array slicing to return views instead of copies
3. **YUV Detection**: Centralized logic eliminates 6+ duplicate implementations
4. **MediaPipe Postprocessing**: Combined bbox calculation with keypoint extraction loop
5. **Class-level Constants**: Moved mapping dictionaries outside methods
6. **Frame Copy Elimination**: Removed unnecessary `rgb_frame.copy()` before detection
7. **Shared Memory Buffer Reuse**: Uses `np.copyto()` for direct buffer writes instead of intermediate copies
8. **Debug Logging Reduction**: Removed verbose debug logging in hot paths

---

## Runtime Performance Tuning

### Frame Skipping (`skip_frames`)

Skip N frames between pose detections to reduce CPU/GPU load:

```yaml
pose:
  skip_frames: 2 # Process every 3rd frame (2 skipped + 1 processed)
```

| Setting          | Effective FPS (at 30fps input) | Use Case                    |
| ---------------- | ------------------------------ | --------------------------- |
| `skip_frames: 0` | 30 fps                         | Real-time activity tracking |
| `skip_frames: 1` | 15 fps                         | Balanced performance        |
| `skip_frames: 2` | 10 fps                         | Low-power devices           |
| `skip_frames: 4` | 6 fps                          | Very constrained hardware   |

**Note**: Higher skip values may miss fast activities (falls happen in ~0.5-1s). For fall detection, `skip_frames: 1` or `2` is recommended.

### Motion ROI Cropping (`use_motion_roi`)

Only process the region containing detected motion, reducing inference area:

```yaml
pose:
  use_motion_roi: true # Crop to motion bounding box before detection
```

**Benefits**:

- **2-5x speedup** on large frames where motion is localized
- Reduces GPU memory bandwidth
- Particularly effective for high-resolution cameras

**Caveats**:

- May miss poses at frame edges if they're outside the motion region
- Adds 10% padding around motion region to capture full poses

**When to use**:

- High-resolution cameras (1080p+) with mostly static scenes
- Localized activity monitoring (e.g., doorway, specific area)
- Resource-constrained hardware

**When to avoid**:

- Full-frame activity monitoring
- Very active scenes with motion everywhere

### Dynamic Shared Memory (No Resize)

The pose detection pipeline uses **dynamic shared memory** that eliminates the need to resize frames to a fixed model input size. This provides significant performance benefits:

```
BEFORE (Fixed 320×320 SHM):
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐
│ ROI Crop        │ --> │ cv2.resize   │ --> │ Fixed 320×320 │ --> MediaPipe
│ (e.g., 200×200) │     │ to 320×320   │     │ SHM buffer    │
└─────────────────┘     └──────────────┘     └───────────────┘

AFTER (Dynamic SHM):
┌─────────────────┐     ┌──────────────────────────────┐
│ ROI Crop        │ --> │ Dynamic SHM (header + data)  │ --> MediaPipe
│ (e.g., 200×200) │     │ [16B header][200×200×3 data] │     (any size)
└─────────────────┘     └──────────────────────────────┘
```

**Benefits**:

- **No cv2.resize()** - eliminates resize computation entirely
- **Better quality** - no interpolation artifacts from upscaling small ROIs
- **Memory efficient** - buffer sized per-camera, not global max
- **MediaPipe handles variable sizes natively** - internal resize is optimized

---

## Shared Memory Layout

Pose detection uses dynamic shared memory with a header-based format that allows variable-size frames without resizing. This eliminates the computational overhead of resizing and preserves frame quality.

### Dynamic SHM Format

```
┌────────────────────────────────────────────────────────┐
│ Header (16 bytes): 4 x int32                           │
│   [0] width  - actual frame width in pixels            │
│   [1] height - actual frame height in pixels           │
│   [2] stride - row stride in bytes (width * 3 for RGB) │
│   [3] flags  - reserved for future use                 │
├────────────────────────────────────────────────────────┤
│ Pixel data (up to max_width × max_height × 3 bytes)    │
│   RGB format, row-major order                          │
└────────────────────────────────────────────────────────┘
```

### Per-Camera Buffer Sizing

The SHM buffer is sized **per-camera based on the camera's detect dimensions**, not a global maximum. This significantly reduces memory usage:

| Camera Detect Size | Buffer Size | vs 1080p Max    |
| ------------------ | ----------- | --------------- |
| 640×480            | ~922 KB     | **85% smaller** |
| 1280×720           | ~2.8 MB     | **55% smaller** |
| 1920×1080          | ~6.2 MB     | baseline        |

**Example**: A system with 4 cameras at 640×480 uses ~3.7 MB total instead of ~24.8 MB.

### Input Buffer: `pose-<camera_name>`

- **Size**: `16 + (detect_width × detect_height × 3)` bytes per camera
- **Format**: Header (16 bytes) + RGB pixel data (variable size)
- **Created by**: `app.py:start_pose_detectors()`
- **Written by**: `RemotePoseDetector.detect()` via `write_frame_to_shm()`
- **Read by**: `PoseDetectorRunner` via `read_frame_from_shm()`

### Output Buffer: `pose-out-<camera_name>`

- **Size**: `20 * 57 * 4 = 4,560 bytes`
- **Format**: `(20, 57)` float32 array
- **Row format**: `[person_id(1), confidence(1), keypoints(51), bbox(4)]`
- **Created by**: `app.py:start_pose_detectors()`
- **Written by**: Pose detector process
- **Read by**: Integration layer for tracking

---

## Configuration Reference

### Global Pose Detectors (`pose_detectors:`)

Define named detector configurations that can be shared across cameras:

```yaml
pose_detectors:
  mediapipe_task:
    type: mediapipe_task
    model_complexity: 2 # 0: Lite, 1: Full, 2: Heavy
    num_poses: 1 # Max poses to detect per frame
    min_detection_confidence: 0.3
    min_pose_presence_confidence: 0.5
    min_tracking_confidence: 0.5
    running_mode: video # "image", "video", or "live_stream"
    num_threads: 3 # CPU threads for inference
    # model_path: /path/to/custom.task  # Optional custom model
```

**Supported detector types:**

- `mediapipe_task` — MediaPipe Tasks API (recommended)
- `mediapipe` — MediaPipe legacy API
- `yolo_pose` — YOLO pose via ONNX Runtime
- `cpu` — Generic CPU-based detector

### Global Pose Model (`pose_model:`)

Default model configuration used when detector-specific settings are not provided:

```yaml
pose_model:
  width: 320 # Model input width
  height: 320 # Model input height
  model_type: mediapipe_task # Model architecture type
  input_tensor: nhwc # Input tensor format: nhwc, nchw
  input_pixel_format: rgb # Pixel format: rgb, bgr, yuv
  input_dtype: int # Data type: int, float, float_denorm
  num_keypoints: 17 # COCO format (don't change)
  confidence_threshold: 0.4 # Detection threshold
  keypoint_threshold: 0.3 # Per-keypoint threshold
```

### Per-Camera Pose Configuration (`cameras.<name>.pose:`)

> **✅ Standalone Pose Detection Supported (More Efficient)**
>
> Pose detection can now work **without** object detection enabled. When `detect.enabled: false` and `pose.enabled: true`:
>
> 1. **Pose detections are injected as tracked objects**: Detected poses are converted to "person" detections and fed directly into the object tracker
> 2. **Motion-based filtering applies**: Pose detection only runs when motion is detected (performance optimization)
> 3. **Actions become sub-labels**: Pose actions (e.g., "falling") are attached as sub-labels to the pose-based person objects
> 4. **Objects appear in Frigate UI**: Tracked pose-based persons show up in the debug view and can trigger events/recordings
>
> **⚡ Performance Benefits of Standalone Mode:**
>
> Running pose detection without object detection is **more efficient** for single-purpose deployments:
>
> | Resource          | With Object Detection     | Standalone Pose Only | Savings                 |
> | ----------------- | ------------------------- | -------------------- | ----------------------- |
> | GPU/CPU Inference | Object model + Pose model | Pose model only      | ~50% compute            |
> | Memory            | ~2-4GB (both models)      | ~1-2GB (pose only)   | ~50% RAM                |
> | Latency           | Two inference steps       | One inference step   | Lower latency           |
> | Power             | Higher                    | Lower                | Better for edge devices |
>
> **Use standalone mode when:**
>
> - You only need fall detection or activity monitoring (not general object detection)
> - Running on resource-constrained hardware (Raspberry Pi, Coral, etc.)
> - You want dedicated pose/activity cameras without full NVR object tracking
>
> **Note**: The `publish_to_detected_objects` setting controls whether poses are published when `detect.enabled: true`. When `detect.enabled: false`, poses are automatically injected into the tracker regardless of this setting.

```yaml
cameras:
  my_camera:
    detect:
      enabled: false # Object detection disabled = more efficient for pose-only use
    pose:
      enabled: true # Enable pose detection
      confidence_threshold: 0.4 # Min detection confidence
      keypoint_threshold: 0.3 # Min keypoint confidence
      skip_frames: 0 # Skip N frames between detections (0 = process every frame)
      use_motion_roi: false # Crop to motion region before pose detection
      publish_to_detected_objects: true # Recommended; controls pose visibility when detect.enabled: true
      mask: "" # Detection mask (polygon)
      required_zones: [] # Zones required for events

      # Actions to track (filtered by what detector actually produces)
      actions:
        - standing
        - walking
        - sitting
        - lying
        - falling

      # Actions that trigger snapshots (✓ IMPLEMENTED)
      snapshot_actions:
        - falling

      # Actions that trigger recording retention (✓ IMPLEMENTED)
      record_actions:
        - falling

      # Per-action filters
      filters:
        falling:
          min_score: 0.6
          threshold: 0.5
          min_area: 0
          max_area: 24000000

      # Activity detector configuration
      activity_detector:
        type: stgcn_fall # "stgcn_fall" or "heuristic"
        model_path: /path/to/model.tflite
        confidence_threshold: 0.5
        num_threads: 3
        effective_window_size: 40 # Must be <= model window size
        device: usb # EdgeTPU device (optional)
```

> **⚠️ Important: Action Detection Capabilities**
>
> The available actions depend on which activity detector you use:
>
> | Detector     | Actions Detected                          | Notes                                 |
> | ------------ | ----------------------------------------- | ------------------------------------- |
> | `stgcn_fall` | `falling`, `standing`                     | Binary classifier (fall vs. non-fall) |
> | `heuristic`  | `standing`, `sitting`, `lying`, `walking` | Rule-based, no `falling` detection    |
>
> Actions like `waving`, `pointing`, `running`, `jumping` are defined in the enum but **not currently implemented** by any detector. Configure only actions that your chosen detector can actually produce.

### snapshot_actions and record_actions (Implementation Status)

These features **are implemented** in the codebase:

1. **`snapshot_actions`**: When a detected action matches an entry in this list:

   - `TrackedPoseProcessor.should_save_pose_snapshot()` returns `True`
   - `PoseConsumer` sets `has_snapshot = True` on matched tracked objects
   - Triggers the standard Frigate snapshot pipeline

2. **`record_actions`**: When a detected action matches an entry in this list:
   - `TrackedPoseProcessor.should_retain_pose_recording()` returns `True`
   - `PoseConsumer` sets `has_clip = True` on matched tracked objects
   - Triggers recording retention via the standard Frigate recording pipeline

**Code paths:**

- [pose_processing.py#L335-L372](frigate/track/pose_processing.py) — `should_save_pose_snapshot()` and `should_retain_pose_recording()`
- [pose_consumer.py#L540-L560](frigate/track/pose_consumer.py) — Sets flags on matched objects

### Key Configuration Options Explained

| Option                                    | Default     | Description                                 |
| ----------------------------------------- | ----------- | ------------------------------------------- |
| `pose.enabled`                            | `false`     | Enable pose detection for this camera       |
| `pose.fps`                                | `5`         | Pose detection framerate (keep low for CPU) |
| `pose.confidence_threshold`               | `0.4`       | Min confidence to accept a pose             |
| `pose.keypoint_threshold`                 | `0.3`       | Min confidence for individual keypoints     |
| `pose.publish_to_detected_objects`        | `true`      | Bridge poses to object pipeline             |
| `pose.required_zones`                     | `[]`        | Zones where poses must appear for events    |
| `activity_detector.type`                  | `heuristic` | Activity detector type                      |
| `activity_detector.effective_window_size` | varies      | Temporal window for sequence models         |

---

## Activity Detectors

### Available Detectors and Their Capabilities

| Detector     | Type                    | Actions Produced                          | Use Case                        |
| ------------ | ----------------------- | ----------------------------------------- | ------------------------------- |
| `stgcn_fall` | Sequence model (TFLite) | `falling`, `standing`                     | Fall detection for elderly care |
| `heuristic`  | Rule-based              | `standing`, `sitting`, `lying`, `walking` | Basic posture classification    |

### Defined Actions (PoseActionTypeEnum)

The following actions are defined in `frigate/events/pose_types.py`:

```python
class PoseActionTypeEnum(str, Enum):
    standing = "standing"   # ✓ heuristic, ✓ stgcn_fall (default)
    sitting = "sitting"     # ✓ heuristic only
    lying = "lying"         # ✓ heuristic only
    falling = "falling"     # ✓ stgcn_fall only
    walking = "walking"     # ✓ heuristic only
    running = "running"     # ✗ NOT IMPLEMENTED
    jumping = "jumping"     # ✗ NOT IMPLEMENTED
    waving = "waving"       # ✗ NOT IMPLEMENTED
    pointing = "pointing"   # ✗ NOT IMPLEMENTED
    custom = "custom"       # For future extensions
```

> **Note**: Only configure actions in `actions`, `snapshot_actions`, and `record_actions` that your chosen detector can actually produce. Unimplemented actions will never trigger.

---

## Activity Detector: ST-GCN Fall Detection

The ST-GCN plugin (`stgcn_fall.py`) implements **binary fall detection** using Spatial Temporal Graph Convolutional Networks.

**Output**: Returns either `falling` (class 1) or `standing` (class 0) based on model confidence threshold.

### Input Requirements

- **Model input shape**: `(1, C, T, V, M)` where:
  - `C=3` (x, y, confidence channels)
  - `T=window_size` (temporal frames, e.g., 40 or 60)
  - `V=18` (Kinetics keypoint count)
  - `M=1` (single person)

### COCO to Kinetics Mapping

The detector automatically maps COCO 17 keypoints to Kinetics 18 keypoints:

- Adds a "neck" keypoint (midpoint between shoulders)
- Reorders keypoints to match Kinetics skeleton graph

### Troubleshooting Dimension Errors

If you see TFLite dimension mismatch errors:

1. Verify `activity_detector.model_path` points to a valid TFLite model
2. Ensure `effective_window_size` ≤ model's trained window size
3. Check model expects Kinetics-18 format input

---

## Activity Detector: Heuristic

The heuristic plugin (`heuristic.py`) uses **rule-based analysis** of keypoint positions to classify postures.

**Output**: Returns one of `standing`, `sitting`, `lying`, or `walking` based on:

- Body height (shoulder-to-hip distance)
- Head position relative to hips
- Leg spread (ankle distance)

### Configuration Options

```yaml
activity_detector:
  type: heuristic
  body_height_threshold: 50 # Pixels; below this = lying
  leg_spread_threshold: 100 # Pixels; above this = walking
  confidence_threshold: 0.5 # Min keypoint confidence
```

### Limitations

- **No fall detection**: Use `stgcn_fall` for falls
- **No temporal analysis**: Each frame analyzed independently
- **Sensitive to camera angle**: Works best with side or frontal views

---

## Debugging & Logging

### Recommended Logger Configuration

```yaml
logger:
  default: info
  logs:
    frigate.pose_detection: debug
    frigate.pose_detection.base: debug
    frigate.pose_detection.integration: debug
    frigate.pose_detectors: debug
    frigate.pose_detectors.plugins: debug
    frigate.pose_activity_detectors.plugins.stgcn_fall: debug
    frigate.track.tracked_pose: debug
    frigate.track.pose_consumer: debug
```

### Checking Shared Memory

```bash
# List pose-related shared memory segments
ls -la /dev/shm/pose-*

# Expected files per camera:
# /dev/shm/pose-<camera_name>       (input buffer, 10MB)
# /dev/shm/pose-out-<camera_name>   (output buffer, ~4.5KB)
```

### Common Issues

| Symptom                  | Likely Cause               | Solution                                        |
| ------------------------ | -------------------------- | ----------------------------------------------- |
| Empty SHM frames         | Frame producer not writing | Check ffmpeg pipeline and frame dimensions      |
| No poses detected        | Threshold too high         | Lower `confidence_threshold` to 0.1-0.3         |
| Activity detector errors | Wrong model format         | Verify TFLite model and `effective_window_size` |
| High CPU usage           | FPS too high               | Reduce `pose.fps` to 2-5                        |

---

## Performance Best Practices

1. **Use standalone mode for pose-only cameras**: Set `detect.enabled: false` with `pose.enabled: true` to skip object detection entirely — this reduces compute by ~50% and is ideal for dedicated fall detection or activity monitoring cameras
2. **Keep pose FPS low**: 2-5 FPS is typically sufficient for activity detection
3. **Use hardware acceleration**: EdgeTPU for TFLite models, GPU for ONNX
4. **Filter by zones**: Use `required_zones` to reduce processing
5. **Disable object bridge**: Set `publish_to_detected_objects: false` if not needed when running with object detection enabled
6. **Tune window size**: Smaller `effective_window_size` = faster inference

---

## Architecture: Action Labeling

Pose actions are integrated with Frigate's object detection pipeline by attaching them as **sub-labels to person detections**.

### How Action Labeling Works

1. **Pose Detection** (`video.py` lines 1043-1120):

   - When a pose is detected, it's converted to a detection dict with:
     - `label: "person"` — canonical main label for object pipeline compatibility
     - `sub_label: (action, confidence)` — e.g., `("falling", 0.85)`
   - This ensures pose detections work with existing object filters and thresholds

2. **Object Matching** (`pose_consumer.py` lines 480-520):

   - `PoseConsumer` matches detected poses to existing tracked objects by centroid distance
   - When matched, it attaches the action to the tracked object:
     ```python
     matched_obj.obj_data["sub_label"] = (action, confidence)
     matched_obj.obj_data["action"] = action
     matched_obj.obj_data["action_confidence"] = confidence
     ```

3. **Event Storage** (`events/maintainer.py` line 346, 371):
   - Pose events use the action as the label: `Event.label: event_data.get("action", "pose")`
   - Actions are also stored in `Event.sub_label` for filtering

**Code paths:**

- [video.py#L1073-1097](frigate/video.py) — Builds detection dict with `label="person"`, `sub_label=(action, confidence)`
- [pose_consumer.py#L510-520](frigate/track/pose_consumer.py) — Attaches action to matched tracked objects
- [events/maintainer.py#L346](frigate/events/maintainer.py) — Stores action as event label

---

## Motion Filtering for Pose Detection (✓ IMPLEMENTED)

Motion detection filtering **is now applied** to pose detection. Frames are only sent for pose detection when motion is detected in the scene.

### How It Works

In `integration.py:detect_poses()`:

```python
def detect_poses(self, frame, frame_time, motion_boxes, regions):
    # ...

    # Performance optimization: skip pose detection if no motion detected
    if not motion_boxes and not regions:
        logger.debug(f"Skipping pose detection for {camera_name}: no motion or regions")
        return []

    # ... rest of detection logic
```

### Performance Benefits

- **Reduced CPU/GPU load**: No pose detection on static scenes
- **Lower power consumption**: Critical for edge deployments
- **Better resource allocation**: Detection only when activity is likely

### When Pose Detection Runs

Pose detection is triggered when:

1. Motion boxes are detected by the motion detector, OR
2. Regions exist from tracked objects

This aligns with how object detection regions are computed in `video.py`.

---

## Runtime Details

### Process Lifecycle (`app.py`)

1. `start_pose_detectors()` iterates over cameras with `pose.enabled: true`
2. For each camera:
   - Creates dedicated `mp.Queue` in `self.pose_detection_queues[camera_name]`
   - Allocates input SHM: `pose-<camera_name>` (10MB)
   - Allocates output SHM: `pose-out-<camera_name>` (20×57×4 bytes)
   - Spawns `PoseDetectProcess` bound to that single camera
3. `TrackedPoseProcessor` and `PoseConsumer` handle downstream processing

### Key Classes

| Class                   | Location                                        | Responsibility                     |
| ----------------------- | ----------------------------------------------- | ---------------------------------- |
| `PoseDetectProcess`     | `pose_detection/base.py`                        | Manages detector process lifecycle |
| `PoseDetectorRunner`    | `pose_detection/base.py`                        | Runs synchronous detection loop    |
| `RemotePoseDetector`    | `pose_detection/base.py`                        | SHM-based detector interface       |
| `CameraPoseIntegration` | `pose_detection/integration.py`                 | Per-camera pose tracking           |
| `PoseActivityDetector`  | `pose_activity_detectors/base.py`               | Activity detector base class       |
| `STGCNFallDetector`     | `pose_activity_detectors/plugins/stgcn_fall.py` | Fall detection                     |

---

## Testing

```bash
# Run pose-related tests
pytest test_matching_logic.py -v

# Check for Python syntax errors
python -m py_compile frigate/pose_detection/*.py frigate/pose_detectors/plugins/*.py

# Lint
ruff check frigate/pose_detection frigate/pose_detectors frigate/pose_activity_detectors
```

---

## Privacy Mode

Privacy mode ensures that **real camera footage of monitored individuals is never stored or transmitted** during normal operation. All visual outputs show pose skeleton overlays instead. When a configured action (e.g., a fall) is detected, the system temporarily switches to real footage for a configurable event window.

This is designed for elderly care deployments where continuous video surveillance raises privacy concerns, but fall detection must still capture real footage for caregiver review.

### How It Works

Privacy mode operates through **three independent layers**, each covering a different output path:

```
Real Camera (RTSP)
       │
       ▼
    go2rtc ─────────────────────────────────────┐
       │                                         │
       ▼                                         ▼
  Frigate DETECT                          Privacy Proxy Sidecar
  (always sees real frames)               (MQTT keypoints → skeleton)
       │                                         │
       ▼                                         │
  ┌─ Layer 1: SHM Substitution ─┐    ┌─ Layer 2: TCP Stream ───┐
  │ Replaces frame in shared     │    │ Serves skeleton BGR24   │
  │ memory with skeleton render  │    │ frames over TCP :9000   │
  │                              │    │ for Frigate record      │
  │ Covers:                      │    │ FFmpeg to encode        │
  │  • JSMPEG live view          │    │                         │
  │  • Birdseye                  │    │ Covers:                 │
  │  • Snapshots                 │    │  • MP4 recordings       │
  │  • API image endpoints       │    │                         │
  └──────────────────────────────┘    └─────────────────────────┘
                                               │
                                    ┌─ Layer 3: go2rtc API ─────┐
                                    │ Dynamically switches the  │
                                    │ go2rtc stream source      │
                                    │ between dead URL (privacy)│
                                    │ and real RTSP (override)  │
                                    │                           │
                                    │ Covers:                   │
                                    │  • WebRTC/MSE live view   │
                                    └───────────────────────────┘
```

**Layer 1 (SHM)** is built into Frigate. After pose detection runs on the real frame, the shared memory contents are replaced with a skeleton render. Everything downstream that reads SHM sees skeletons.

**Layer 2 (Proxy TCP)** is a sidecar container. Frigate's record FFmpeg reads from the proxy's TCP socket instead of the camera directly. The proxy renders skeleton frames from MQTT keypoints published by Frigate.

**Layer 3 (go2rtc API)** is also handled by the proxy sidecar. Normally the go2rtc stream for the camera points to a dead URL, so WebRTC/MSE fails and Frigate falls back to JSMPEG (which reads SHM skeletons). During override, the proxy switches the go2rtc stream to the real camera via the REST API.

### Privacy Override (Event Window)

When a configured action is detected (e.g., `falling` in `record_actions`), all three layers switch to real footage simultaneously:

1. **SHM override**: `CameraMetrics.privacy_override_until` shared mp.Value is set — `process_frames()` in the camera subprocess skips skeleton substitution
2. **MQTT signal**: `frigate/{camera}/privacy_override` published — proxy flushes its pre-event JPEG buffer (real footage from before the event) into the TCP stream, then passes through live real frames
3. **go2rtc switch**: Proxy calls `PUT /api/streams` to point the camera's go2rtc stream at the real RTSP source — WebRTC/MSE shows real video

After `privacy_override_seconds` (default 60s), all layers revert to skeleton mode automatically.

**Trigger condition**: The override fires when ALL of these are true:
- `privacy_mode: true` on the camera
- The detected action is in `record_actions` (e.g., `falling`)
- No override is already active (debounced per camera)

### Configuration Reference

```yaml
cameras:
  my_camera:
    detect:
      enabled: false          # Standalone pose mode (no object detection)

    pose:
      enabled: true
      privacy_mode: true      # Enable skeleton substitution
      privacy_background: scene  # "black" or "scene"
      privacy_override_seconds: 60  # Real-frame window after trigger
      publish_keypoints: true  # Required for proxy sidecar (Layer 2+3)

      # Actions that trigger privacy override + recording retention
      record_actions:
        - falling

      # Actions that trigger snapshots (independent of override)
      snapshot_actions:
        - falling

      # Activity detector (produces the actions)
      activity_detector:
        type: stgcn_fall
        model_path: /path/to/model.tflite
        confidence_threshold: 0.5

      # ROI cropping — disable for privacy cameras (see caveats below)
      use_motion_roi: false

    record:
      enabled: true
      alerts:
        retain: { days: 1, mode: all }   # mode: all required when detect: false
      detections:
        retain: { days: 1, mode: all }

    ffmpeg:
      inputs:
        # Real source for detection (ML always sees real frames)
        - path: rtsp://127.0.0.1:8554/camera_real
          input_args: -rtsp_transport tcp
          roles: [detect]
        # Privacy proxy TCP for recording (skeleton or real on override)
        - path: tcp://privacy-proxy:9000
          input_args: -f rawvideo -pix_fmt bgr24 -video_size 1280x720 -framerate 30
          roles: [record]
      output_args:
        record: >-
          -f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1
          -strftime 1 -c:v libx264 -preset ultrafast -tune zerolatency
          -pix_fmt yuv420p -g 30 -an
```

### Privacy Mode Settings

| Setting | Default | Description |
|---|---|---|
| `privacy_mode` | `false` | Enable skeleton substitution across all layers |
| `privacy_background` | `black` | Background behind skeletons: `black` (solid) or `scene` (grayscale room from motion detector average frame — zero extra cost) |
| `privacy_override_seconds` | `60` | Seconds of real footage shown after a trigger action is detected. Applies to SHM, proxy TCP, and go2rtc live view |
| `publish_keypoints` | `false` | Publish COCO keypoints to MQTT. **Required** for the proxy sidecar to render skeletons for recordings and live view switching |
| `record_actions` | — | Actions that trigger both recording retention AND privacy override. This is the link between fall detection and the override |
| `use_motion_roi` | `false` | Crop to motion region before pose detection. **Recommended `false` for privacy cameras** — dynamic ROI size changes cause frame-size jitter in skeleton renders |

### Recording & Detect Mode Interactions

| `detect.enabled` | `pose.enabled` | `record.mode` | Notes |
|---|---|---|---|
| `false` | `true` | `all` | **Recommended for privacy cameras.** Pose-only, no object detection overhead. Must use `mode: all` because `mode: motion` requires detect. |
| `true` | `true` | `motion` | Object detection + pose detection. More resource-intensive. Motion mode works because detect produces motion events. |
| `false` | `true` | `motion` | **Will not record.** Motion mode requires detect to produce events. |

### Output Matrix

| Output | Normal (privacy) | Override (event) | Privacy disabled |
|---|---|---|---|
| JSMPEG live view | Skeleton (SHM) | Real RGB (SHM override) | Real RGB |
| WebRTC/MSE live | Skeleton (JSMPEG fallback) | Real RGB (go2rtc switch) | Real RGB |
| Birdseye | Skeleton (SHM) | Real RGB (SHM override) | Real RGB |
| Recordings | Skeleton (proxy TCP) | Real RGB (pre-event flush + passthrough) | Real RGB |
| Snapshots | Skeleton (SHM) | Real RGB (SHM override) | Real RGB |

### Privacy Proxy Sidecar

The proxy is a separate container that handles Layers 2 and 3. It:
- Subscribes to `frigate/{camera}/pose_keypoints` via MQTT
- Renders skeleton frames (BGR24) and serves them over a TCP socket
- Maintains a pre-event ring buffer of real JPEG frames (configurable, default 10s at native fps)
- On override: flushes buffer into TCP stream, passes through live RTSP frames, switches go2rtc
- Monitors override expiry and restores privacy mode automatically

**Proxy config** (`config.yaml`):

```yaml
mqtt_host: mqtt
mqtt_port: 1883
pre_event_seconds: 30    # Ring buffer duration (seconds of real footage before event)
post_event_seconds: 60   # Should match privacy_override_seconds in Frigate config
go2rtc_api_url: http://frigate:1984

cameras:
  - name: my_camera
    width: 1280
    height: 720
    fps: 30              # Native camera fps for TCP stream
    real_rtsp_url: rtsp://frigate:8554/camera_real
    tcp_port: 9000
```

### Limitations

**Privacy proxy requires a live RTSP stream.** The proxy captures real frames via RTSP for the pre-event buffer and override passthrough. This means:

- **Local video files (e.g., Lei2 test dataset) cannot be used with the proxy.** The proxy's pre-event capture thread will fail to connect and retry indefinitely (with backoff). The TCP skeleton stream still works (renders from MQTT keypoints), but override passthrough will show black frames instead of real footage.
- **Layer 1 (SHM substitution) works with any input source**, including local files. You can test skeleton rendering in the web UI with test videos — only recording override is affected.
- **To test the full override flow**, you need a live RTSP camera source (or an RTSP simulator like `mediamtx` serving a looped video file).

**`use_motion_roi: true` not recommended for privacy cameras.** Motion ROI causes the detection crop region to change size frame-to-frame, which produces jitter in skeleton renders. Use `use_motion_roi: false` for stable privacy output.

**go2rtc stream naming convention.** The real camera stream in go2rtc must be named `{camera}_real` (e.g., `iphone_real`). The camera's main stream name (e.g., `iphone`) is dynamically managed by the proxy — do NOT define it in go2rtc config.

### Key Files

| File | Layer | Purpose |
|---|---|---|
| `frigate/config/camera/pose.py` | — | `privacy_mode`, `privacy_background`, `privacy_override_seconds`, `publish_keypoints` fields |
| `frigate/video.py` ~L670, ~L1270 | 1 | `_check_privacy_override()` + SHM frame substitution |
| `frigate/pose_detection/privacy_renderer.py` | 1 | Skeleton rendering (YUV for SHM) |
| `frigate/camera/__init__.py` | 1 | `privacy_override_until` shared mp.Value in CameraMetrics |
| `frigate/track/pose_consumer.py` | 1+2+3 | Override trigger: sets SHM mp.Value + publishes MQTT |
| `serai-edge-setup/privacy-proxy/privacy_proxy.py` | 2+3 | Sidecar: TCP server + go2rtc API + pre-event buffer |
| `serai-edge-setup/privacy-proxy/skeleton_renderer.py` | 2 | Standalone skeleton renderer for proxy |

---

## Notes for Contributors

1. **COCO 17-keypoint layout is a hard constant** — do not make this user-configurable without updating all code paths
2. **Use `tensor_utils.py` helpers** — avoid inline shape checks and reshaping
3. **Per-camera isolation is required** — do not merge camera queues
4. **Document model shapes** — when adding detectors, specify input/output tensor formats
5. **Test with multiple cameras** — ensure temporal isolation is preserved
