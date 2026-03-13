"""Integration tests for the privacy proxy sidecar.

Tests the core logic without requiring MQTT broker, go2rtc, or real cameras.
Run with: python test_privacy_proxy.py
"""

import json
import sys
import threading
import time

import cv2
import numpy as np

from skeleton_renderer import render_skeleton_bgr
from privacy_proxy import CameraState, PrivacyProxy

# ── Test data ──────────────────────────────────────────────────────────────

MOCK_KEYPOINTS = [
    [320, 100, 0.9],  # nose
    [310, 95, 0.8],   # left eye
    [330, 95, 0.8],   # right eye
    [300, 100, 0.7],  # left ear
    [340, 100, 0.7],  # right ear
    [280, 180, 0.9],  # left shoulder
    [360, 180, 0.9],  # right shoulder
    [250, 280, 0.8],  # left elbow
    [390, 280, 0.8],  # right elbow
    [230, 360, 0.7],  # left wrist
    [410, 360, 0.7],  # right wrist
    [300, 380, 0.9],  # left hip
    [340, 380, 0.9],  # right hip
    [290, 500, 0.8],  # left knee
    [350, 500, 0.8],  # right knee
    [280, 620, 0.7],  # left ankle
    [360, 620, 0.7],  # right ankle
]

MOCK_POSE_PAYLOAD = {
    "timestamp": time.time(),
    "frame_width": 640,
    "frame_height": 720,
    "poses": [{"id": "1", "keypoints": MOCK_KEYPOINTS}],
}


# ── Tests ──────────────────────────────────────────────────────────────────

def test_skeleton_renderer_basic():
    """Skeleton renderer produces correct output shape with non-zero pixels."""
    poses = [{"keypoints": MOCK_KEYPOINTS}]
    frame = render_skeleton_bgr(poses, 640, 720)
    assert frame.shape == (720, 640, 3), f"Wrong shape: {frame.shape}"
    assert frame.dtype == np.uint8
    assert np.count_nonzero(frame) > 0, "Frame is all black — no skeleton drawn"
    print("  PASS: skeleton renderer basic")


def test_skeleton_renderer_no_poses():
    """Empty pose list produces all-black frame."""
    frame = render_skeleton_bgr([], 640, 720)
    assert frame.shape == (720, 640, 3)
    assert np.count_nonzero(frame) == 0, "Frame should be all black"
    print("  PASS: skeleton renderer no poses")


def test_skeleton_renderer_low_confidence():
    """Keypoints below confidence threshold are not drawn."""
    low_conf_kps = [[x, y, 0.1] for x, y, _ in MOCK_KEYPOINTS]
    poses = [{"keypoints": low_conf_kps}]
    frame = render_skeleton_bgr(poses, 640, 720)
    assert np.count_nonzero(frame) == 0, "Low-confidence keypoints should not be drawn"
    print("  PASS: skeleton renderer low confidence")


def test_camera_state_override():
    """CameraState override flag works correctly."""
    cam = CameraState(
        name="test",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    assert not cam.is_override_active, "Override should not be active initially"

    cam.override_until = time.time() + 5.0
    assert cam.is_override_active, "Override should be active after setting future time"

    cam.override_until = time.time() - 1.0
    assert not cam.is_override_active, "Override should expire after time passes"
    print("  PASS: camera state override")


def test_keypoints_lock_thread_safety():
    """Keypoints can be safely written and read from multiple threads."""
    cam = CameraState(
        name="test",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    errors = []

    def writer():
        for i in range(100):
            with cam.keypoints_lock:
                cam.latest_keypoints = [{"id": str(i), "keypoints": MOCK_KEYPOINTS}]

    def reader():
        for _ in range(100):
            with cam.keypoints_lock:
                kps = cam.latest_keypoints
                if kps is not None:
                    try:
                        _ = kps[0]["keypoints"]
                    except Exception as e:
                        errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"
    print("  PASS: keypoints lock thread safety")


def test_skeleton_mode_frame_generation():
    """Simulate skeleton mode: MQTT keypoints → rendered frame."""
    cam = CameraState(
        name="test",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    # Simulate MQTT message setting keypoints
    payload = json.dumps(MOCK_POSE_PAYLOAD)
    data = json.loads(payload)
    with cam.keypoints_lock:
        cam.latest_keypoints = data.get("poses", [])

    # Simulate frame generation (skeleton mode, not override)
    assert not cam.is_override_active

    with cam.keypoints_lock:
        keypoints = cam.latest_keypoints

    assert keypoints is not None and len(keypoints) > 0

    # Render skeleton frame
    frame = render_skeleton_bgr(keypoints, cam.width, cam.height)
    assert frame.shape == (720, 640, 3)
    assert np.count_nonzero(frame) > 0, "Skeleton frame should have drawn pixels"

    # Verify it could be encoded (simulate FFmpeg pipe)
    raw_bytes = frame.tobytes()
    assert len(raw_bytes) == 720 * 640 * 3, f"Wrong byte count: {len(raw_bytes)}"
    print("  PASS: skeleton mode frame generation")


def test_pre_event_buffer():
    """Pre-event ring buffer stores and retrieves JPEG frames."""
    cam = CameraState(
        name="test",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    # Simulate buffering 10 frames
    for i in range(10):
        fake_frame = np.full((720, 640, 3), i * 25, dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", fake_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        cam.pre_event_buffer.append(jpeg.tobytes())

    assert len(cam.pre_event_buffer) == 10

    # Verify frames can be decoded back
    for frame_bytes in cam.pre_event_buffer:
        img = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert img is not None
        assert img.shape == (720, 640, 3)

    print("  PASS: pre-event buffer")


def test_pre_event_buffer_maxlen():
    """Ring buffer respects maxlen (default 150)."""
    cam = CameraState(
        name="test",
        width=64,
        height=48,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    # Fill beyond maxlen
    for i in range(200):
        cam.pre_event_buffer.append(b"fake_frame")

    assert len(cam.pre_event_buffer) == 150, f"Expected 150, got {len(cam.pre_event_buffer)}"
    print("  PASS: pre-event buffer maxlen")


def test_override_activates_and_expires():
    """Full cycle: normal → override → back to normal."""
    cam = CameraState(
        name="test",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    # Normal mode
    assert not cam.is_override_active

    # Simulate fall event → override for 0.2s
    cam.override_until = time.time() + 0.2
    assert cam.is_override_active

    # Wait for expiry
    time.sleep(0.3)
    assert not cam.is_override_active, "Override should have expired"
    print("  PASS: override activates and expires")


def test_mqtt_message_parsing():
    """Simulate MQTT message parsing for keypoints and override."""
    cam = CameraState(
        name="living_room",
        width=640,
        height=720,
        fps=5,
        real_rtsp_url="rtsp://fake",
        output_rtsp_url="rtsp://fake_out",
    )

    # Test keypoint message parsing
    payload = json.dumps(MOCK_POSE_PAYLOAD)
    data = json.loads(payload)
    with cam.keypoints_lock:
        cam.latest_keypoints = data.get("poses", [])

    assert len(cam.latest_keypoints) == 1
    assert len(cam.latest_keypoints[0]["keypoints"]) == 17

    # Test override message parsing
    override_payload = json.dumps({
        "mode": "real",
        "duration_seconds": 60,
        "event_id": "test-123",
        "triggered_at": time.time(),
    })
    override_data = json.loads(override_payload)
    duration = override_data.get("duration_seconds", 60)
    cam.override_until = time.time() + duration
    assert cam.is_override_active

    print("  PASS: MQTT message parsing")


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_skeleton_renderer_basic,
        test_skeleton_renderer_no_poses,
        test_skeleton_renderer_low_confidence,
        test_camera_state_override,
        test_keypoints_lock_thread_safety,
        test_skeleton_mode_frame_generation,
        test_pre_event_buffer,
        test_pre_event_buffer_maxlen,
        test_override_activates_and_expires,
        test_mqtt_message_parsing,
    ]

    print(f"\nRunning {len(tests)} privacy proxy tests...\n")
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
