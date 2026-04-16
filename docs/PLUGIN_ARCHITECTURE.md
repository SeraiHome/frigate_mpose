# Pose detection plugin architecture + per-camera parallelism

A proposal for discussion with upstream Frigate maintainers.

## 1. Intent

This document describes a pose detection subsystem built on top of Frigate and asks a single question: **could this architecture, or parts of it, be reintegrated into Frigate as one or more upstream PRs?**

The fork that implements this is in development and stabilizing on a single-camera Pi 5 testbed, with ongoing multi-camera validation work. The pose layer is opt-in per camera and disabled by default. The design tries to mirror patterns Frigate already uses rather than inventing new ones, and to touch as few existing files as possible. The remaining touches — a small set of extension points — are the subject of this document.

**Version note.** This fork is based on Frigate 0.17.0 (pre-release, circa February 2026). It has not been rebased onto current upstream `dev`. The pose subsystem touches a small number of existing Frigate files (`app.py`, `video.py`, `config/config.py`, `camera/__init__.py`, `track/object_processing.py`, `comms/dispatcher.py`, `comms/detection_publisher.py`, `const.py`) — these edits are against the 0.17.0-era versions and may need adaptation for the current upstream tree. The new files (everything under `frigate/pose_*`, `frigate/track/pose_*`, `frigate/config/camera/pose.py`) are additive and largely independent of upstream version drift.

We're not asking for a merge commitment. We're asking whether this architecture is a shape upstream Frigate would consider hosting, and if so, which extension points would be acceptable as standalone PRs. If any of the answers are "out of scope", that's useful too — it tells us whether to keep this as a fork or invest in the merge path.

---

## 2. Background — why per-camera parallelism

This section matters more than any other in the document. It explains the single architectural decision that forced most of the rest. If we get nothing else upstreamed, the conversation about *why* this decision was necessary is worth having.

### 2a. The first attempt followed the existing detector pool model

Frigate's object detector architecture is elegant and well-proven: a pool of one or more detector workers pulls frames off a shared queue, each frame is an independent classification problem, and the tracker holds all the temporal state. When we first started integrating pose detection, we copied the model — a single shared `pose_detection_queue` feeding a pool of pose detector workers, each of which pulled any camera's next frame off the queue and ran MediaPipe on it.

That first attempt worked fine with one camera. The moment we enabled pose detection on a second camera, it broke.

Everything else — per-camera processes, per-camera SHM segments, the pose plugin layer, the integration patterns — follows from it.

### 2b. Why the shared pool pattern breaks for pose detection — but not for every layer that comes after it

Object detection is a **stateless** per-frame classification problem. Pose detection feeding temporal activity classification is **stateful**. But the fork carries two different kinds of temporal state, with different requirements for how they can be shared across processes. Conflating them leads to over-claiming that nothing in the pose layer can pool. The precise story:

**Kind 1 — implicit in-interpreter state (pose detection).** Pose detection in MediaPipe's VIDEO mode holds an implicit ROI hint from frame N-1 *inside the interpreter*, used to refine the landmarker on frame N. This state is not exposed, not serializable, and cannot be moved between processes. If frame N and frame N+1 from the same camera land on different pool workers, each worker is doing cold re-detection on every frame and the "video tracking" mode gains nothing over image mode. **This is why pose detection genuinely cannot share a pool** — and it's the rationale behind the per-camera `PoseDetectProcess` model described in §2c.

**Kind 2 — explicit per-track state (activity classification).** A sliding-window activity classifier — fall detection, gesture recognition, loitering, any behavior analysis — needs an ordered uninterrupted window of keypoints from one subject on one camera. But that window is **owned by Python code, not hidden inside an interpreter**. When the classifier stores its per-track state as an explicit dict keyed by `(camera, pose_id)`, a single worker process can hold windows for any number of camera/track combinations without cross-contamination. Two conditions make this safe:

1. **Per-`(camera, pose_id)` keying.** Camera A's `track_5` and camera B's `track_5` live in separate buffer entries and never touch each other.
2. **FIFO per `(camera, pose_id)`.** Each camera's `TrackedPoseProcessor` publishes to a single queue in frame order, so successive calls for the same track arrive in order. Cross-camera interleaving is irrelevant — each per-track buffer grows temporally regardless of what other cameras are doing.

Kind 2 temporal state **can safely centralize into a shared worker pool**. Kind 1 cannot. Recognizing that distinction is what lets the fork put pose detection in per-camera processes and activity classification in a shared pool — getting the right guarantee at the right layer, with the minimum of each.

### 2c. Per-camera processes are the minimum correct design

The fix is to make the per-camera decomposition explicit at the process boundary. One dedicated pose detector process per pose-enabled camera, spawned at startup, with its own input queue and its own shared memory segments. Frames stay in order. Each worker holds its own MediaPipe tracker state. Each activity classifier sees a clean continuous window of keypoints from a single subject on a single camera.

As a secondary benefit (but not the motivation), the per-camera design also unlocks per-camera hardware accelerator assignment — see §5 on HW acceleration.

### 2d. What this means for upstream

The per-camera pose architecture is not parallel to Frigate's detector pool as an alternative — it's **additive**. The existing detector pool continues to be correct for stateless object detection. What's missing upstream is a place to host stateful, per-camera pipelines that can't share workers. This document's proposal is to formalize that place as a small set of extension points.

---

## 3. What the subsystem does now

Two new responsibilities, cleanly separated into two plugin layers that mirror the existing `frigate/detectors/` plugin architecture (details in §4):

1. **Pose detectors** — given a frame, return keypoints. Implementations wrap MediaPipe Pose (two API variants), YOLOv8-Pose, and a CPU fallback. Runs in a dedicated per-camera process (§2b Kind 1).
2. **Pose activity detectors** — given a temporal window of keypoints for a tracked subject, return an action label. Per-track state is keyed by `(camera, pose_id)`, which lets the same classifier instance serve multiple cameras concurrently. Two deployment modes are supported: **per-camera instance** (one classifier per `TrackedPoseProcessor`, simplest, runs in each per-camera process) or **shared pool** (one dedicated worker process drains a shared request queue, see §5b). Choice is a single config field on the camera. Reference implementations include a stateless heuristic and a sliding-window neural classifier plugin.

The two layers are independently configurable per camera. Pose results flow into Frigate's existing object tracker either as `sub_label` enrichment on person detections (when object detection is enabled) or as a synthetic detection source (when object detection is disabled). Event rows, review segments, MQTT output, and clip retention all flow through Frigate's existing pipeline with no parallel machinery.

Pose detection is **gated by Frigate's existing motion detection** — if no motion boxes in a frame, the pose pipeline does not run for that frame. This reuses Frigate's motion gate without modification and makes pose detection effectively free on static scenes.

---

## 4. Plugin architecture — mirrors what Frigate already does

Frigate's detector plugin layer at [`frigate/detectors/detector_types.py:17-37`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/detectors/detector_types.py) uses `pkgutil.iter_modules` to walk a plugins directory at import, discovers subclasses via `DetectionApi.__subclasses__()`, builds a `type_key` → class lookup dict, and generates a dynamic `DetectorTypeEnum` for Pydantic config discrimination.

The fork's pose detector plugin layer at [`frigate/pose_detectors/detector_types.py:17-35`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_detectors/detector_types.py) is **literally the same pattern**, line for line — same `pkgutil.iter_modules` walk, same `__subclasses__()` discovery, same `StrEnum` from `type_key`, same Pydantic discriminated-union config. The base class is minimal:

```python
class PoseDetectionApi(ABC):
    type_key: str
    supported_models: List[PoseModelTypeEnum]

    @abstractmethod
    def __init__(self, detector_config, camera_name=None): ...

    @abstractmethod
    def detect_raw(self, tensor_input): ...

    def postprocess_poses(self, raw_output, threshold=0.4):
        """Standardized output: list of {person_id, confidence, keypoints(17×3), bbox}"""
```

New implementations drop into `frigate/pose_detectors/plugins/` with a unique `type_key`. Current implementations: `mediapipe`, `mediapipe_task`, `yolo_pose`, `cpu`.

The pose activity detector layer uses an explicit `register_detector(name, class)` call at module import (rather than subclass discovery) because activity detectors carry non-trivial state that plugin authors want to initialize explicitly. The registry is a ~50-line `DETECTOR_REGISTRY` dict in [`frigate/pose_activity_detectors/__init__.py:20-73`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_activity_detectors/__init__.py). The base class carries a per-track state contract:

```python
def detect(
    self,
    keypoints: np.ndarray,             # (17, 3) current frame
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    pose_id: Optional[str] = None,     # stable track id for per-track state
    camera: Optional[str] = None,      # camera name — state is keyed by (camera, pose_id)
) -> Tuple[PoseActionTypeEnum, float]:
    ...

def forget(self, pose_id: str, camera: Optional[str] = None) -> None:
    """Drop any per-track state held for (camera, pose_id). Default no-op."""
```

`(camera, pose_id)` is plumbed through the contract so stateful classifiers can keep per-track sliding windows without interleaving keypoints from distinct people, even across cameras that share one classifier instance via a pool. Stateless classifiers ignore both arguments. `forget()` is called by the integration layer when a track expires so plugins can release memory.

**Call sites are split by responsibility.** The fork separates tracking-state updates from classification calls: `TrackedPose.update()` only advances tracking bookkeeping (keypoints, bbox, history, `time_since_update`), while `TrackedPose.classify()` is the single site that triggers `PoseActivityDetector.detect()`. The `pose_processing.py` consumer loop gates `classify()` on `time_since_update == 0` so stale keypoints never reach the classifier. This split prevents duplicate classification calls when a pose is updated through multiple code paths within the same frame — a subtle bug we shipped briefly and now gate against structurally.

**Why two layers.** Keypoint extraction (what the body looks like right now) and activity classification (what the body has been doing over time) have different compute profiles, different update cadences when new models land, and different plugin lifecycles. Separating them lets operators mix-and-match — swap the pose detector without re-qualifying the activity classifier, or swap the classifier without re-tuning the pose detector.

**Config in YAML:**

```yaml
pose_detectors:
  mediapipe_task:
    type: mediapipe_task
    model_complexity: 2
    num_poses: 1
    running_mode: video

# Optional shared activity classifier pools. Each entry spawns one dedicated
# worker process that serves classification requests from any camera
# referencing this pool by name. State is keyed per (camera, pose_id) inside
# the worker. Omit this section for per-camera-local classifiers.
pose_activity_detectors:
  activity_pool:
    type: heuristic
    confidence_threshold: 0.5

cameras:
  my_camera:
    pose:
      enabled: true
      # Per-camera inline classifier (default, recommended for CPU):
      activity_detector:
        type: heuristic
        confidence_threshold: 0.5
      # For shared-accelerator deployments, route to a pool instead:
      # activity_detector_pool: activity_pool
```

---

## 5. Hardware acceleration and the two-layer parallelism model

The pose subsystem runs on two architectural layers that have different parallelism requirements:

- **Layer 1 — pose detection (per-camera).** Frame-ordered, stateful in the interpreter, one process per camera (§2b Kind 1). Each process holds its own TFLite/ONNX session; accelerator assignment is per-camera.
- **Layer 2 — activity classification.** Per-track sliding-window inference with explicit state keyed by `(camera, pose_id)` (§2b Kind 2). Two deployment modes: **per-camera inline** (default, recommended for CPU — each camera gets its own classifier instance in its own thread, TFLite releases the GIL so N cameras = N concurrent inferences) or **shared pool** (opt-in for shared-accelerator deployments — one worker process centralizes inference on a single device). Choice is a single config field on each camera.

[`frigate/pose_detectors/detector_config.py:178-189`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_detectors/detector_config.py) exposes the plumbing for Layer 1, and a parallel `BasePoseActivityDetectorConfig` does the same for Layer 2:

```python
class BasePoseDetectorConfig(FrigateBaseModel):
    type: str
    accelerator: Optional[str]   # e.g. "coral", "cuda", "gpu"
    device: Optional[str]         # e.g. "usb", "pci:0"
    # ...
```

### 5a. Layer 1 — pose detection

| Plugin | Hardware acceleration |
|---|---|
| `yolo_pose` | **GPU (CUDA) via ONNX Runtime** — wired and tested. `device: "gpu"` adds the `CUDAExecutionProvider` to the session ([`yolo_pose.py:31-56`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_detectors/plugins/yolo_pose.py)). YOLO-pose is also a plausible candidate for Coral acceleration via `edgetpu_compiler`, since the model is a standard TFLite detector and the compiler readily handles it. Coral quantization of yolo_pose is on the roadmap but not yet validated end-to-end. |
| `mediapipe_task` | **CPU only today.** The MediaPipe Tasks Python API exposes `BaseOptions.delegate` for GPU only; no Coral delegate is present in the enum.  |
| `mediapipe` (legacy) | CPU only. |
| `cpu` | CPU only by design. |

A deployment with N pose-enabled cameras spawns N independent `PoseDetectProcess` instances, each holding its own interpreter. Each can be assigned to a different accelerator device via config. Per-camera interpreter isolation means there are no cross-camera locking or context-switching issues at this layer.

### 5b. Layer 2 — activity classification

**Default: per-camera inline.** Each camera's `TrackedPoseProcessor` thread holds its own classifier instance. Classification runs synchronously after each `TrackedPose.classify()` call with zero IPC overhead. TFLite releases the GIL during inference, so N cameras produce N genuinely concurrent classifier calls in the same process. This is the recommended mode for CPU-only deployments.

**Opt-in: shared pool for hardware accelerator sharing.** When multiple cameras need to share a single hardware accelerator device (e.g. one Coral USB TPU), the pool pattern centralizes inference into one dedicated worker process holding one interpreter and one device delegate. The pool is implemented in [`frigate/pose_activity_detectors/pool.py`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_activity_detectors/pool.py). The pool is NOT recommended for CPU-only deployments — a single worker serializes what inline mode runs concurrently, causing queue backpressure under multi-camera load.

Architectural shape:

```
┌──────────────────────────────────────────┐    ┌────────────────────────────┐
│ Per-camera PoseDetectProcess (camera 1)  │    │                            │
│   MediaPipe → keypoints → TrackedPose    │    │ PoseActivityDetectProcess  │
│                    │                     │    │ (shared, one per pool)     │
│   TrackedPoseProcessor                   │    │                            │
│     on classify() for a fresh frame:     │───►│   one activity classifier  │
│       submit (camera, pose_id, kps) ─────┼────┤   one TFLite interpreter   │
│       returns last-known action/conf     │◄───┤   one device delegate      │
└──────────────────────────────────────────┘    │                            │
                                                │   per-track state keyed    │
┌──────────────────────────────────────────┐    │   by (camera, pose_id)     │
│ Per-camera PoseDetectProcess (camera 2)  │───►│                            │
│   ... same as camera 1 ...               │◄───┤                            │
└──────────────────────────────────────────┘    └────────────────────────────┘
             ... up to N cameras ...
```

**Request pattern is fire-and-forget async.** Each camera's `PoolActivityDetectorProxy` publishes a request to the pool's input queue and immediately returns the track's last-known `(action, confidence)` from a shared cache. A small listener thread in each `TrackedPoseProcessor` drains results from the pool's output queue and updates the cache, so the next `classify()` call returns fresh data — at worst one frame behind. The camera thread never blocks on classifier inference, even under contention.

**Key properties:**

- **One worker, many cameras.** The classifier process is a first-class Frigate process spawned from `frigate/app.py`, lifetime tied to the app. `pose_activity_detectors:` is a top-level config section mirroring the existing `pose_detectors:` sibling.
- **Per-(camera, pose_id) state isolation.** The worker's `_per_track` dict is keyed by the tuple so cross-camera track id collisions don't corrupt buffers. Each per-track window grows in temporal order regardless of cross-camera interleaving — see §2b Kind 2.
- **Single accelerator across cameras.** One TFLite interpreter, one device delegate, no cross-process device locking. This is the cleanest shape for a single Coral USB stick serving multiple cameras.
- **Graceful degradation.** If the pool fails to load its accelerator delegate, it falls back to CPU in the same worker. If the pool is not configured at all, cameras fall back to per-camera local classifiers. Zero downgrade for users who don't opt in.

### 5c. Pool implementation status

| Aspect | Status |
|---|---|
| Pool process spawn + config + request/response queues | **Implemented** |
| `(camera, pose_id)` per-track state keying | **Implemented** |
| Fire-and-forget submit + listener thread + last-known cache | **Implemented** |
| CPU activity classifier in the pool | **Implemented and validated** on the Lei2 fall dataset (sensitivity preserved from single-camera baseline) |
| Coral / accelerator activity classifier in the pool | **Blocked on ML work** — see §5d |
| Multi-device pools (multiple pool entries, each on a different device) | Architecture supports it; not exercised |

### 5d. Coral / accelerator acceleration for the classifier pool

Blocked on model deliverables. Getting the current activity classifier onto an Edge TPU requires either quantization-aware retraining or a hardware-friendly replacement architecture; that work lives on the ML side, not the Frigate side. Once a compatible model lands, flipping the pool to the target accelerator is a single config field change on the pool entry — the architecture is in place.

### 5e. Device contention between object detection and activity classification

Frigate's existing object detector pool and the activity classifier pool are **separate processes**, each loading their own interpreter and device delegate independently. On most accelerator APIs this is fine — CUDA, OpenVINO, and RKNN runtimes all support multiple processes sharing a device via the driver's scheduler, each with its own context.

**Coral (Edge TPU) is the exception.** `libedgetpu.so.1.0` acquires exclusive USB device access when a process loads the delegate. Two separate pool processes — one for object detection, one for activity classification — each calling `load_delegate("libedgetpu.so.1.0", {"device": "usb:0"})` would contend for the same physical device. The library was not designed for cross-process sharing of a single USB device.

For single-Coral deployments where both object detection and activity classification need TPU acceleration, the correct architectural solution would be: **one process, one delegate, multiple workloads** — merging the object detector pool and the activity classifier pool into a single "inference service" process that multiplexes different models through one interpreter. That's a deeper change than the current pool-per-workload pattern and is not implemented today. Workarounds for current hardware:

- **Two Coral devices** — `usb:0` for object detection, `usb:1` for activity classification. Each pool gets its own device. Works today with the existing config surface.
- **Coral for objects, CPU inline for classification** — activity classifiers on keypoints are lightweight enough to run on CPU without bottlenecking (204 bytes input, <1ms for heuristic, ~80ms for neural). The CPU path is the default and doesn't contend with Coral at all.
- **YOLO-pose in the existing detector pool** — a YOLO-pose model produces both bounding boxes and keypoints in one inference pass on one Coral device. No second pool needed. The activity classifier then runs downstream on CPU using the keypoints from the YOLO-pose output. This is the most efficient single-Coral architecture for combined object + pose detection.

### 5f. Multi-device scale-out

The pool design trivially extends to N pools on N devices. Define multiple pool entries under `pose_activity_detectors:`, each with its own `device:` string, and map cameras to pools as desired. Accelerator *auto-discovery* (enumerating available devices at startup and assigning one per pool automatically) is not implemented; device strings are set per pool in config today.

---

## 6. Per-camera process implementation

One dedicated pose detector process per pose-enabled camera, spawned at startup from [`frigate/app.py:449-465`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/app.py):

```python
for camera_name in pose_enabled_cameras:
    detector_name = next(iter(self.config.pose_detectors))
    pose_detector_config = self.config.pose_detectors[detector_name]
    detector_instance_name = f"{detector_name}_{camera_name}"
    self.pose_detectors[detector_instance_name] = PoseDetectProcess(
        detector_instance_name,
        self.pose_detection_queues[camera_name],  # Camera-specific queue
        [camera_name],                             # Only process frames from this camera
        self.config,
        pose_detector_config,
        self.stop_event,
    )
```

Each process receives its own queue, reads frames from its own SHM segment, writes results back to its own output SHM segment.

**Frame bridge (input).** Shared memory segment named `pose-{camera_name}`, sized per-camera based on the camera's `detect.width × detect.height`. Layout documented in [`frigate/pose_detection/shm_format.py:1-54`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_detection/shm_format.py):

```
┌────────────────────────────────────────────────────────┐
│ Header (16 bytes): 4 × int32                           │
│   width, height, stride, flags                         │
├────────────────────────────────────────────────────────┤
│ Pixel data — RGB, row-major                            │
└────────────────────────────────────────────────────────┘
```

Dynamic sizing — the pose detector runs at the camera's native resolution (quality matters for keypoints on far-from-camera subjects). The header reports actual dimensions to the consumer.

**Result bridge (output).** Shared memory segment `pose-out-{camera_name}` holding a fixed-shape `(20, 57)` float32 array — up to 20 detections per frame, each encoded as `[person_id, confidence, 17×3 keypoints, 4 bbox]`. Created in [`frigate/pose_detection/base.py:140-146`](https://github.com/SeraiHome/frigate_mpose/blob/dev/frigate/pose_detection/base.py).

Memory usage scales linearly with the number of pose-enabled cameras, which matches Frigate's existing per-camera resource model.

---

## 7. Integration with Frigate's object pipeline

Two patterns, chosen per camera via the `pose.detect_persons` config flag.

### 7a. Pose as enrichment (`detect_persons: true`, default)

Frigate's object detector runs normally and produces person tracked objects. Pose results attach to them as metadata:

```python
matched_obj.obj_data["sub_label"] = (action, action_confidence)
matched_obj.obj_data["action"] = action
```

The IoU match between the pose bbox and the person tracked_object happens in the pose consumer thread in the main Frigate process. Frigate's next tracker update callback fires an MQTT message carrying the mutated `obj_data`, so downstream consumers see `after.action == "<classified_action>"` on the standard event stream.

No new event type, no new DB namespace, no new MQTT topic — `sub_label` is a pattern Frigate already supports for face recognition, bird classification, and license plate recognition.

### 7b. Pose as primary detection source (`detect_persons: false`)

Used on pose-only cameras where the standard object detector is turned off entirely (common on edge devices minimizing CPU). Without a standard person detection source, there's no person tracked_object for pose to enrich. So the fork synthesizes one directly from each `TrackedPose` and injects it into the standard tracker as if it came from the object detector:

```python
detections[stable_id] = {
    "id": stable_id,
    "label": "person",
    "false_positive": False,
    "score": max(float(pose.confidence), 0.7),
    "box": list(box),
    # ... centroid, region, area, ratio, frame_time
}
```

The injected detection flows through the standard tracker → `TrackedObject` → event row → review segment → MQTT output. A pose-driven event on such a camera produces exactly one DB `Event` row with a standard tracker id, exactly one MQTT `new → update → end` lifecycle, exactly one review segment, and exactly one downstream trigger. Indistinguishable from an object-detector-driven event from the consumer's point of view.

The injection site is currently ~50 lines in `frigate/video.py`. It's the main place where the fork still touches an existing file, and the natural candidate for refactoring into an extension point (§8).

---

## 8. Minimal extension points we'd like Frigate to provide

For the pose layer to land cleanly on top of unmodified upstream Frigate — i.e. as a plugin that touches zero existing files — four small extension points would need to exist. None exist today. Each is small and each has uses beyond pose detection.

| # | Extension | File | Purpose | Scope |
|---|---|---|---|---|
| **1** | Plugin config discovery | `config/camera/camera.py` | Discover plugin-provided config classes via entrypoint instead of hardcoded imports | ~80 LoC |
| **2** | Detection source registry | `video.py` | Accept detection inputs from registered sources beyond the standard object detector | ~150 LoC |
| **3** | Object metadata subscription API | `track/object_processing.py` | Formal `subscribe_object_metadata(type, callback)` surface for enrichments to mutate `obj_data.sub_label` via a published contract | ~120 LoC |
| **4** | `review.alerts.required_sub_labels` filter | `config/camera/review.py`, `review/maintainer.py` | Gate alert-severity review segments on whether the tracked object has a specific sub_label value | ~10 LoC |

Extension **4** is the smallest and most general — it benefits the face recognition, bird classification, and license plate recognition features Frigate already ships. Today users of those features either accept alert noise from every tracked object or filter downstream. A ~10 line change in the review maintainer fixes that regardless of whether any of the other extension points land.

Extensions **1-3** are the ones that would let a pose plugin drop in without touching existing files. Scope is modest (~350 LoC upstream total).

---

## 9. Questions and answers

Common questions we've collected while sharing earlier drafts of this document.

**"Pose detection is out of scope for Frigate's object-centric focus."**
Pose as a primary Frigate feature, possibly. But the ask here isn't "merge pose into Frigate" — it's "accept the four small extension points that would let a pose plugin attach cleanly". Three of them benefit the classifier features Frigate already ships. The fourth (detection source registry) enables any non-standard detection pipeline, not just pose.

**"Plugin architecture adds maintenance burden."**
The pose detector plugin layer is byte-for-byte a copy of the existing `frigate/detectors/` plugin pattern. Same `pkgutil.iter_modules` walk, same `__subclasses__()` discovery, same config-discriminator pattern. Zero new architectural concepts — the maintenance burden is what Frigate already carries for its detector plugins, replicated for a new layer.

**"Per-camera processes don't fit Frigate's shared-pool model."**
The pose subsystem actually uses *both* models, at different layers (§5). Pose detection itself is per-camera because MediaPipe's VIDEO-mode ROI continuity lives inside the interpreter and can't be moved between processes (§2b Kind 1). Activity classification is a shared pool because its per-track sliding-window state is explicit Python state that can be keyed by `(camera, pose_id)` without contamination (§2b Kind 2). The layering gets the right guarantee at the right place, and Frigate's SHM management, camera resource model, and process supervision already handle both shapes — the pose layer uses the existing machinery for both.

**"Shared memory management becomes complex with N × 2 segments."**
Each camera already has several SHM segments in current Frigate (frame storage, motion, etc). Pose adds two more per pose-enabled camera (input frame, output detection array). Sized dynamically per-camera, opt-in, scales linearly with enabled features.

**"Direct injection into `video.py` for pose-as-primary-detection-source is invasive."**
Today it's ~50 lines in `video.py`. That's exactly why extension point #2 (detection source registry) is proposed — it factors the pattern out into a registration API so the ~50 lines move into the plugin package and `video.py` acquires a single iteration over registered sources.

**"Motion-gated pose detection could miss slow events."**
Pose detection inherits Frigate's existing motion gate behavior. Slow events that don't trigger motion also don't trigger any current Frigate feature. Operators who need lower motion thresholds tune them the same way they do for any other Frigate feature — no new configuration surface.

**"What if both the object detector and the activity classifier want the same Coral?"**
On most accelerators (CUDA, OpenVINO, RKNN) multiple processes can share the device via the driver's scheduler — no conflict. Coral is the exception: `libedgetpu` acquires exclusive USB device access per process, so two pool processes loading delegates on the same `usb:0` would contend. For single-Coral deployments the cleanest answer is YOLO-pose in the existing detector pool (one model, one Coral, both boxes and keypoints in one pass), with the activity classifier running downstream on CPU. For multi-Coral deployments, each pool gets its own device via the `device:` config field. See §5e for the full analysis.

**"Running neural activity classifiers on every frame is expensive."**
Activity classifiers range from trivial (heuristic rule-based, handful of joint angle computations) to non-trivial (sliding-window neural nets). The plugin architecture accommodates both. Motion gating ensures the pipeline is quiet on empty scenes. Per-camera processes keep one camera's activity classifier from blocking another's. Per-camera accelerator assignment (§5) spreads the compute load across devices. The opt-in-per-camera design means users who don't need this pay nothing.

**"This is a big patch."**
It is — but ~95% of it is new files under new directories. Only the four extension point proposals touch existing files. After those land, the pose implementation itself could be distributed as a separate package (`pip install frigate-pose-plugin`) that plugs in via entrypoints and never touches the Frigate tree.

---

## 10. What we're asking

Two-part ask:

**Part A (principles):** does the per-camera plugin architecture described above sound like a shape Frigate would consider accepting? Is the rationale in §2 something Frigate sees value in formalizing, or does the shared pool model already cover the cases Frigate cares about?

**Part B (extension points):** of the four extension points in §8, which would you accept as standalone PRs? `review.alerts.required_sub_labels` is the smallest and most immediately useful (10 lines, benefits existing features), and might be a good first PR to validate cooperation before the larger ones.

We're open to:
- Shipping `review.alerts.required_sub_labels` first as a standalone PR regardless of the broader conversation
- A collaborative RFC issue on the detection source registry and metadata subscription API if those are the right shape
- Restructuring any of the proposed extension points based on your architectural preferences
- Hearing "out of scope" on any piece and adjusting our plans accordingly

---

## 11. What's not in this document

This document describes the generic substrate only. The specific pose and activity classifier implementations we use in production — model architectures, training data, tuning parameters, evaluation results — are not in scope here. If the architecture conversation goes well and there's interest in specific use cases, we're happy to cover those separately. The point of this document is to scope what could potentially be upstreamed, not to pitch a particular downstream application.

---

## 12. File reference

Every file path in this document points at source you can open and verify. If you spot a claim that doesn't match the code, it's a bug in this document.

| Topic | File |
|---|---|
| Existing Frigate detector plugin discovery (mirrored by the pose layer) | `frigate/detectors/detector_types.py:17-37` |
| Pose detector plugin discovery | `frigate/pose_detectors/detector_types.py:17-35` |
| Pose detector base class | `frigate/pose_detectors/detection_api.py:22-77` |
| Pose detector implementations | `frigate/pose_detectors/plugins/` |
| Pose activity detector base class | `frigate/pose_activity_detectors/base.py:19-74` |
| Pose activity detector registry | `frigate/pose_activity_detectors/__init__.py:20-73` |
| Pose activity detector implementations | `frigate/pose_activity_detectors/plugins/` |
| Per-camera pose process spawn | `frigate/app.py:449-465` |
| Input frame SHM layout | `frigate/pose_detection/shm_format.py:1-54` |
| Output detection SHM layout | `frigate/pose_detection/base.py:140-146` |
| Accelerator/device config fields | `frigate/pose_detectors/detector_config.py:178-189` |
| YOLO-Pose GPU path | `frigate/pose_detectors/plugins/yolo_pose.py:31-56` |
| MediaPipe Tasks GPU delegate (disabled) | `frigate/pose_detectors/plugins/mediapipe_task_pose.py:139-154` |
| Motion gate for pose detection | `frigate/pose_detection/integration.py:176` |
| Pose-as-primary detection injection site | `frigate/video.py:1181-1258` |
| Pose sub_label enrichment site | `frigate/track/pose_consumer.py:1144-1157` |
