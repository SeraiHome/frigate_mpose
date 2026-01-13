<!-- Copilot / AI Agent instructions for contributors and automated agents -->

# Frigate (SeraiHome fork) — AI agent guidance

This file gives concise, actionable context for AI coding agents working in this repository. Keep answers tight and reference the files listed below when making changes.

**Big Picture**:

- **Purpose**: Frigate is an NVR focused on realtime object and pose detection with multiprocessing for performance. Core runtime is in `frigate/` and services are started by `frigate/app.py` (class `FrigateApp`).
- **Runtime model**: The application launches multiple OS-level processes and threads (detectors, recording, embeddings, output, review). Communication is via multiprocessing `Queue`, shared memory (UntrackedSharedMemory), and internal communicator classes in `frigate/comms/`.

**Key directories & files** (use these as entry points):

- `frigate/app.py` — main orchestration, process lifecycle, and shared-memory/queue setup (see `start_detectors`, `start_pose_detectors`).
- `frigate/pose_detection/` — pose subsystem; `integration.py` shows per-camera detector integration and expected shared-memory names (`pose-<camera_name>`).
- `frigate/detectors/` and `frigate/object_detection` — object detector processes and configs.
- `frigate/comms/` — MQTT, WebSocket, ZMQ and cross-process dispatchers; use these for messaging and integration patterns.
- `config/` — runtime configuration (`config.yml`, `config.yml.example`). Tests and local dev often rely on values here.
- `migrations/` — DB migration scripts; database stored under the configured `CONFIG_DIR` (see `frigate/app.py` DB initialization).
- `pyproject.toml` — lint configuration (ruff).

**Patterns and conventions to follow**:

- Multiprocessing-first: prefer using existing `Queue`s and shared memory helpers (`frigate/util/image.py` and `UntrackedSharedMemory`) rather than passing large arrays between threads.
- One pose detector per camera: pose detectors are per-camera processes/instances — name SHM `pose-<camera_name>` and per-camera queues in `FrigateApp.pose_detection_queues`.
- Process lifecycle: start processes in `FrigateApp.start_*` methods and ensure they are stopped/cleaned up in `FrigateApp.stop()` (terminate/join, close SHM, clear queues).
- DB usage: the code binds Peewee models to a `SqliteVecQueueDatabase` in `app.py` — create migrations by following the existing `migrations/` style.
- Logging: prefer `logger = logging.getLogger(__name__)` and structured messages. Use existing log levels and avoid noisy prints.

**Build / test / dev workflows** (concrete commands)

- Lint: `ruff .` (configured via `pyproject.toml`).
- Run tests: `pytest` (project uses standard pytest layout; run targeted tests like `pytest test_matching_logic.py`).
- Run locally (dev): use the Makefile or Docker compose in repo root. Typical quick dev run:
  - `docker-compose up --build` (or use platform-specific `docker/` subfolders for accelerator images)
  - For direct dev runs, run Python with a config: `python -m frigate --config config/config.yml` (match how containers invoke it).

**What to change and where — concrete examples**

- Adding a new pose detector:
  - Add model config under `frigate/pose_detectors/` and a matching entry in the parsed config.
  - Ensure `RemotePoseDetector` (used by `pose_detection/integration.py`) is initialized with SHM name `pose-<camera_name>` and matched buffer sizes (see `start_pose_detectors` in `frigate/app.py`).
- Adding telemetry or new communicator: extend `frigate/comms/` and register the communicator in `FrigateApp.init_dispatcher()` so it participates in `Dispatcher` flows.

**Integration & external dependencies**

- External runtime integrations: `ffmpeg` (see `ffmpeg_presets.py`), MQTT, ONVIF (PTZ), and optional accelerators (see `docker/` and `docker/*` folders). Use the `frigate/config` paths to discover runtime expectations.
- Native resources: uses `/dev/shm` for PID/state files and shared memory. Tests that simulate SHM should clean up created segments.

**Quick rules for PRs / AI edits**

- Keep changes minimal and local: edit a few files and run targeted tests. Don't reformat unrelated files.
- When touching process startup or shutdown, update `frigate/app.py` and ensure `stop()` cleans resources added.
- When changing config keys, update `config/config.yml.example` and any parsing/validation code.

If anything here is unclear or you want extra examples (e.g., adding a new detector plugin or a step-by-step dev run using a specific accelerator folder under `docker/`), tell me which area to expand and I will iterate.
