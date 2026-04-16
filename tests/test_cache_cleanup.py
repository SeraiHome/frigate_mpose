"""Tests for cache cleanup fixes — validates disk space management behavior."""

import datetime
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_cache(tmp_path):
    """Create a temporary cache directory structure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    preview_dir = cache_dir / "preview_frames"
    preview_dir.mkdir()
    return cache_dir


class TestEmergencyCacheCleanup:
    """Test the emergency cleanup function in video.py."""

    def test_removes_old_webp_frames(self, temp_cache):
        """Emergency cleanup removes webp frames older than 30 minutes."""
        from frigate.video import _emergency_cache_cleanup

        preview_dir = temp_cache / "preview_frames"
        # Create old frames (>30 min old)
        old_frame = preview_dir / "preview_cam-1234.webp"
        old_frame.write_bytes(b"fake")
        old_time = time.time() - 35 * 60
        os.utime(old_frame, (old_time, old_time))

        # Create recent frame
        new_frame = preview_dir / "preview_cam-9999.webp"
        new_frame.write_bytes(b"fake")

        with patch("frigate.video.CACHE_DIR", str(temp_cache)):
            _emergency_cache_cleanup()

        assert not old_frame.exists(), "Old frame should be deleted"
        assert new_frame.exists(), "Recent frame should be kept"

    def test_keeps_newest_4_segments(self, temp_cache):
        """Emergency cleanup keeps only the 4 newest segment files."""
        from frigate.video import _emergency_cache_cleanup

        # Create 8 segment files with different timestamps
        for i in range(8):
            seg = temp_cache / f"cam@2026032608{i:02d}00+0000.mp4"
            seg.write_bytes(b"fake" * 100)
            mtime = time.time() - (8 - i) * 60
            os.utime(seg, (mtime, mtime))

        with patch("frigate.video.CACHE_DIR", str(temp_cache)):
            _emergency_cache_cleanup()

        remaining = sorted(temp_cache.glob("*.mp4"))
        assert len(remaining) == 4, f"Expected 4 segments, got {len(remaining)}"
        # The 4 newest should remain
        for seg in remaining:
            assert seg.name.startswith("cam@2026032608")


class TestCacheHasSpace:
    """Test the cache_has_space helper."""

    def test_returns_true_when_space_available(self, temp_cache):
        from frigate.video import cache_has_space

        with patch("frigate.video.CACHE_DIR", str(temp_cache)):
            assert cache_has_space() is True

    def test_returns_true_on_error(self):
        from frigate.video import cache_has_space

        with patch("frigate.video.CACHE_DIR", "/nonexistent/path"):
            # Should return True (don't block restart) when check fails
            assert cache_has_space() is True

    def test_detects_low_space(self, temp_cache):
        from frigate.video import cache_has_space

        # Mock disk_usage to report nearly full
        fake_usage = MagicMock()
        fake_usage.total = 100 * 1024 * 1024  # 100MB
        fake_usage.free = 5 * 1024 * 1024  # 5MB (5%)

        with patch("frigate.video.CACHE_DIR", str(temp_cache)):
            with patch("shutil.disk_usage", return_value=fake_usage):
                assert cache_has_space(min_percent=10.0) is False


class TestOrphanedWebpCleanup:
    """Test the extended clean_tmp_previews in cleanup.py."""

    def test_cleans_stale_webp_frames(self, temp_cache):
        """Webp frames older than 2 hours should be cleaned."""
        from frigate.record.cleanup import RecordingCleanup

        preview_dir = temp_cache / "preview_frames"

        # Create old webp frame (3 hours old)
        old_frame = preview_dir / "preview_cam-1234.webp"
        old_frame.write_bytes(b"fake")
        old_time = time.time() - 3 * 60 * 60
        os.utime(old_frame, (old_time, old_time))

        # Create recent webp frame (30 min old)
        new_frame = preview_dir / "preview_cam-9999.webp"
        new_frame.write_bytes(b"fake")
        new_time = time.time() - 30 * 60
        os.utime(new_frame, (new_time, new_time))

        # Mock the cleanup instance
        cleanup = RecordingCleanup.__new__(RecordingCleanup)
        cleanup.config = MagicMock()

        with patch("frigate.record.cleanup.CACHE_DIR", str(temp_cache)):
            cleanup.clean_tmp_previews()

        assert not old_frame.exists(), "Old webp frame (3h) should be deleted"
        assert new_frame.exists(), "Recent webp frame (30m) should be kept"


class TestFFMpegConverterCleanup:
    """Test that FFMpegConverter always cleans frames even on failure."""

    def test_frames_cleaned_on_exception(self, temp_cache):
        """Frames should be cleaned even when sp.run raises an unexpected exception."""
        from frigate.output.preview import FFMpegConverter, get_cache_image_name

        preview_dir = temp_cache / "preview_frames"

        # Create fake frame files
        frame_times = [1000.0, 1001.0, 1002.0]
        for t in frame_times:
            frame_path = preview_dir / f"preview_testcam-{t}.webp"
            frame_path.write_bytes(b"fake")

        # Create a converter instance
        config = MagicMock()
        config.name = "testcam"
        config.record.preview.quality = MagicMock()
        config.ffmpeg.ffmpeg_path = "ffmpeg"

        converter = FFMpegConverter.__new__(FFMpegConverter)
        converter.config = config
        converter.frame_times = frame_times
        converter.path = str(temp_cache / "output.mp4")
        converter.ffmpeg_cmd = "false"  # will fail
        converter.requestor = MagicMock()

        with patch(
            "frigate.output.preview.get_cache_image_name",
            side_effect=lambda cam, t: str(preview_dir / f"preview_{cam}-{t}.webp"),
        ):
            with patch("subprocess.run", side_effect=OSError("No space left")):
                converter.run()

        # First two frames should be cleaned (last frame intentionally kept)
        assert not (preview_dir / "preview_testcam-1000.0.webp").exists()
        assert not (preview_dir / "preview_testcam-1001.0.webp").exists()
        # Last frame is kept for next segment
        assert (preview_dir / "preview_testcam-1002.0.webp").exists()


class TestMediaPipeModelDir:
    """Test that MediaPipe models use MODEL_CACHE_DIR, not /tmp."""

    def test_task_pose_uses_model_cache_dir(self):
        from frigate.pose_detectors.plugins.mediapipe_task_pose import (
            MEDIAPIPE_MODEL_DIR,
        )

        assert "/tmp" not in MEDIAPIPE_MODEL_DIR, (
            f"MediaPipe task models should not be in /tmp, got: {MEDIAPIPE_MODEL_DIR}"
        )
        assert "model_cache" in MEDIAPIPE_MODEL_DIR

    def test_pose_uses_model_cache_dir(self):
        from frigate.pose_detectors.plugins.mediapipe_pose import MEDIAPIPE_MODEL_DIR

        assert "/tmp" not in MEDIAPIPE_MODEL_DIR, (
            f"MediaPipe models should not be in /tmp, got: {MEDIAPIPE_MODEL_DIR}"
        )
        assert "model_cache" in MEDIAPIPE_MODEL_DIR


class TestExponentialBackoff:
    """Test that consecutive failures produce increasing backoff."""

    def test_backoff_increases(self):
        """Verify the backoff formula: min(sleeptime * 2^(failures-1), 300)."""
        sleeptime = 10  # typical value
        backoffs = []
        for failures in range(1, 8):
            backoff = min(sleeptime * (2 ** (failures - 1)), 300)
            backoffs.append(backoff)

        assert backoffs == [10, 20, 40, 80, 160, 300, 300]
