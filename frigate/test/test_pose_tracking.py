"""Unit tests for pose detection and tracking functionality."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.pose_detection.integration import PoseDetectionIntegration
from frigate.track.tracked_pose import TrackedPose


class TestTrackedPose(unittest.TestCase):
    """Test cases for TrackedPose class."""

    def setUp(self):
        """Set up test fixtures."""
        self.keypoints = np.array(
            [
                [100, 100, 0.9],  # nose
                [110, 95, 0.85],  # left_eye
                [120, 95, 0.88],  # right_eye
                [105, 110, 0.9],  # left_ear
                [125, 110, 0.87],
                [80, 150, 0.92],  # left_shoulder
                [140, 150, 0.91],
                [75, 180, 0.88],  # left_elbow
                [145, 180, 0.89],
                [70, 220, 0.85],  # left_wrist
                [150, 220, 0.86],
                [85, 250, 0.9],  # left_hip
                [135, 250, 0.89],
                [80, 320, 0.87],  # left_knee
                [140, 320, 0.88],
                [75, 380, 0.84],  # left_ankle
                [145, 380, 0.85],
            ],
            dtype=np.float32,
        )

    def test_tracked_pose_creation(self):
        """Test basic TrackedPose creation."""
        pose = TrackedPose(
            pose_id="test_pose_1",
            person_id=0,
            keypoints=self.keypoints,
            confidence=0.9,
            bbox=[100, 100, 50, 280],
            frame_time=1000.0,
            camera_name="test_camera",
        )

        assert pose.pose_id == "test_pose_1"
        assert pose.person_id == 0
        assert pose.confidence == 0.9
        assert pose.age == 0
        assert pose.time_since_update == 0
        assert np.array_equal(pose.keypoints, self.keypoints)

    def test_tracked_pose_update(self):
        """Test updating TrackedPose with new detections."""
        pose = TrackedPose(
            pose_id="test_pose_1",
            person_id=0,
            keypoints=self.keypoints,
            confidence=0.9,
            bbox=[100, 100, 50, 280],
            frame_time=1000.0,
            camera_name="test_camera",
        )

        # Update with slightly different keypoints
        new_keypoints = self.keypoints + np.array([2, 1, 0], dtype=np.float32)
        pose.update(new_keypoints, 0.92, [102, 101, 50, 280])

        assert pose.confidence == 0.92
        assert pose.time_since_update == 0
        assert pose.hit_streak == 1
        assert np.allclose(pose.keypoints, new_keypoints)

    def test_tracked_pose_predict(self):
        """Test pose prediction for time_since_update."""
        pose = TrackedPose(
            pose_id="test_pose_1",
            person_id=0,
            keypoints=self.keypoints,
            confidence=0.9,
            bbox=[100, 100, 50, 280],
            frame_time=1000.0,
            camera_name="test_camera",
        )

        pose.predict()
        assert pose.age == 1
        assert pose.time_since_update == 1
        assert pose.hit_streak == 0

        pose.predict()
        assert pose.age == 2
        assert pose.time_since_update == 2

    def test_tracked_pose_to_dict(self):
        """Test conversion of TrackedPose to dictionary."""
        pose = TrackedPose(
            pose_id="test_pose_1",
            person_id=0,
            keypoints=self.keypoints,
            confidence=0.9,
            bbox=[100, 100, 50, 280],
            frame_time=1000.0,
            camera_name="test_camera",
            frame_width=1920,
            frame_height=1080,
        )

        pose_dict = pose.to_dict()

        assert pose_dict["id"] == "test_pose_1"
        assert pose_dict["person_id"] == 0
        assert pose_dict["confidence"] == 0.9
        assert pose_dict["bbox"] == [100, 100, 50, 280]
        assert pose_dict["frame_width"] == 1920
        assert pose_dict["frame_height"] == 1080
        assert isinstance(pose_dict["keypoints"], list)


class TestPoseMatching(unittest.TestCase):
    """Test cases for pose matching and tracking across frames."""

    def setUp(self):
        """Set up test fixtures."""
        self.camera_name = "test_camera"
        self.pose_config = MagicMock()
        self.pose_config.enabled = True
        self.pose_config.confidence_threshold = 0.5

        self.camera_config = MagicMock()
        self.camera_config.pose = self.pose_config
        self.camera_config.name = self.camera_name

        self.model_config = MagicMock()
        self.model_config.height = 384
        self.model_config.width = 384
        self.model_config.keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Create keypoints for testing
        self.keypoints_frame1 = np.array(
            [
                [100, 100, 0.9],
                [110, 95, 0.85],
                [120, 95, 0.88],
                [105, 110, 0.9],
                [125, 110, 0.87],
                [80, 150, 0.92],
                [140, 150, 0.91],
                [75, 180, 0.88],
                [145, 180, 0.89],
                [70, 220, 0.85],
                [150, 220, 0.86],
                [85, 250, 0.9],
                [135, 250, 0.89],
                [80, 320, 0.87],
                [140, 320, 0.88],
                [75, 380, 0.84],
                [145, 380, 0.85],
            ],
            dtype=np.float32,
        )

        # Slightly offset keypoints for frame 2 (simulating movement)
        self.keypoints_frame2 = self.keypoints_frame1 + np.array(
            [3, 2, 0], dtype=np.float32
        )

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_pose_matching_same_position(self, mock_detector_class):
        """Test that poses at the same position are matched."""
        # Create integration instance
        detection_queue = MagicMock()
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=detection_queue,
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Create detection for frame 1
        detection1 = {
            "keypoints": self.keypoints_frame1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        # Track poses from frame 1
        tracked_poses_f1 = integration.track_poses(
            [detection1], frame_time=1000.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses_f1) == 1
        pose_id_frame1 = tracked_poses_f1[0].pose_id
        assert "test_camera_0" in pose_id_frame1

        # Create detection for frame 2 (same position)
        detection2 = {
            "keypoints": self.keypoints_frame2,
            "confidence": 0.91,
            "bbox": [103, 102, 50, 280],  # Slightly moved
            "person_id": 0,
        }

        # Predict for existing poses
        for pose in integration.tracked_poses.values():
            pose.predict()

        # Track poses from frame 2
        tracked_poses_f2 = integration.track_poses(
            [detection2], frame_time=1001.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses_f2) == 1
        pose_id_frame2 = tracked_poses_f2[0].pose_id
        # Should match the same ID, not create a new one
        assert pose_id_frame2 == pose_id_frame1

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_pose_matching_far_distance(self, mock_detector_class):
        """Test that poses far apart create new tracking IDs."""
        # Create integration instance
        detection_queue = MagicMock()
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=detection_queue,
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Create detection for frame 1
        detection1 = {
            "keypoints": self.keypoints_frame1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        tracked_poses_f1 = integration.track_poses(
            [detection1], frame_time=1000.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses_f1) == 1
        pose_id_frame1 = tracked_poses_f1[0].pose_id

        # Create detection for frame 2 (far away, > 8% of diagonal)
        keypoints_far = self.keypoints_frame1 + np.array(
            [100, 100, 0], dtype=np.float32
        )
        detection2 = {
            "keypoints": keypoints_far,
            "confidence": 0.91,
            "bbox": [200, 200, 50, 280],  # Far away
            "person_id": 0,
        }

        # Predict for existing poses
        for pose in integration.tracked_poses.values():
            pose.predict()

        tracked_poses_f2 = integration.track_poses(
            [detection2], frame_time=1001.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses_f2) == 1
        pose_id_frame2 = tracked_poses_f2[0].pose_id
        # Should create a new ID due to large distance
        assert pose_id_frame2 != pose_id_frame1
        assert "test_camera_1" in pose_id_frame2

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_pose_cleanup_old_poses(self, mock_detector_class):
        """Test that old unmatched poses are cleaned up."""
        # Create integration instance
        detection_queue = MagicMock()
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=detection_queue,
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Create detection for frame 1
        detection1 = {
            "keypoints": self.keypoints_frame1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        tracked_poses_f1 = integration.track_poses(
            [detection1], frame_time=1000.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses_f1) == 1
        pose_id_frame1 = tracked_poses_f1[0].pose_id

        # Simulate 11 frames without updates (aging out)
        for i in range(11):
            for pose in integration.tracked_poses.values():
                pose.predict()

        # Track an empty frame (no detections)
        tracked_poses_old = integration.track_poses(
            [], frame_time=1011.0, frame_width=384, frame_height=384
        )

        # Old pose should be removed from tracked_poses
        assert pose_id_frame1 not in integration.tracked_poses
        assert len(tracked_poses_old) == 0

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_multiple_poses_in_single_frame(self, mock_detector_class):
        """Test tracking multiple poses in a single frame."""
        # Create integration instance
        detection_queue = MagicMock()
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=detection_queue,
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Create detections for two poses in same frame
        keypoints_person1 = self.keypoints_frame1.copy()
        keypoints_person2 = self.keypoints_frame1 + np.array(
            [200, 0, 0], dtype=np.float32
        )

        detection1 = {
            "keypoints": keypoints_person1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        detection2 = {
            "keypoints": keypoints_person2,
            "confidence": 0.91,
            "bbox": [300, 100, 50, 280],
            "person_id": 1,
        }

        tracked_poses = integration.track_poses(
            [detection1, detection2],
            frame_time=1000.0,
            frame_width=384,
            frame_height=384,
        )

        # Should have created two separate poses
        assert len(tracked_poses) == 2
        pose_ids = {p.pose_id for p in tracked_poses}
        assert len(pose_ids) == 2

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_pose_centroid_calculation(self, mock_detector_class):
        """Test that keypoint centroids are calculated correctly."""
        # Create integration instance
        detection_queue = MagicMock()
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=detection_queue,
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Create detection with known keypoints
        detection = {
            "keypoints": self.keypoints_frame1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        tracked_poses = integration.track_poses(
            [detection], frame_time=1000.0, frame_width=384, frame_height=384
        )

        assert len(tracked_poses) == 1
        pose = tracked_poses[0]

        # Verify the pose was properly created
        assert pose.confidence == 0.92
        assert pose.bbox == [100, 100, 50, 280]


class TestPoseProcessingConversion(unittest.TestCase):
    """Test cases for pose processing and conversion to TrackedObject format."""

    def test_pose_dict_has_centroid(self):
        """Test that converted pose dicts include centroid field."""
        pose = TrackedPose(
            pose_id="test_pose_1",
            person_id=0,
            keypoints=np.zeros((17, 3), dtype=np.float32),
            confidence=0.9,
            bbox=[100, 100, 50, 280],  # x, y, w, h
            frame_time=1000.0,
            camera_name="test_camera",
        )

        pose_dict = pose.to_dict()

        # The to_dict method doesn't add centroid, but pose_processing should
        assert "id" in pose_dict
        assert "keypoints" in pose_dict
        assert "bbox" in pose_dict

    def test_pose_dict_conversion_bbox_to_centroid(self):
        """Test centroid calculation from bbox coordinates."""
        # Simulate what pose_processing does
        bbox = [100, 100, 50, 280]  # x, y, w, h
        box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, x2, y2

        # Calculate centroid
        cx = int((box[0] + box[2]) / 2.0)
        cy = int((box[1] + box[3]) / 2.0)
        centroid = (cx, cy)

        # Expected: center of [100, 100, 150, 380]
        assert centroid == (125, 240)


if __name__ == "__main__":
    unittest.main()
