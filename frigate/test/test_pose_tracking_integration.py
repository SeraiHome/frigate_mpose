"""Integration tests for pose detection pipeline."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.pose_detection.integration import PoseDetectionIntegration


class TestPoseDetectionIntegration(unittest.TestCase):
    """Integration tests for the complete pose detection pipeline."""

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

    def _create_keypoints(self, offset_x=0, offset_y=0):
        """Create a test keypoint array with optional offset."""
        return np.array(
            [
                [100 + offset_x, 100 + offset_y, 0.9],
                [110 + offset_x, 95 + offset_y, 0.85],
                [120 + offset_x, 95 + offset_y, 0.88],
                [105 + offset_x, 110 + offset_y, 0.9],
                [125 + offset_x, 110 + offset_y, 0.87],
                [80 + offset_x, 150 + offset_y, 0.92],
                [140 + offset_x, 150 + offset_y, 0.91],
                [75 + offset_x, 180 + offset_y, 0.88],
                [145 + offset_x, 180 + offset_y, 0.89],
                [70 + offset_x, 220 + offset_y, 0.85],
                [150 + offset_x, 220 + offset_y, 0.86],
                [85 + offset_x, 250 + offset_y, 0.9],
                [135 + offset_x, 250 + offset_y, 0.89],
                [80 + offset_x, 320 + offset_y, 0.87],
                [140 + offset_x, 320 + offset_y, 0.88],
                [75 + offset_x, 380 + offset_y, 0.84],
                [145 + offset_x, 380 + offset_y, 0.85],
            ],
            dtype=np.float32,
        )

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_multi_frame_pose_tracking_sequence(self, mock_detector_class):
        """Test pose tracking across multiple frames with realistic motion."""
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=MagicMock(),
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Frame 1: Person detected
        kp_f1 = self._create_keypoints(offset_x=0, offset_y=0)
        det_f1 = {
            "keypoints": kp_f1,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }
        poses_f1 = integration.track_poses(
            [det_f1], frame_time=1000.0, frame_width=384, frame_height=384
        )
        assert len(poses_f1) == 1
        pose_id_f1 = poses_f1[0].pose_id
        assert poses_f1[0].confidence == 0.92
        assert poses_f1[0].time_since_update == 0

        # Frame 2: Same person moves slightly
        for pose in integration.tracked_poses.values():
            pose.predict()

        kp_f2 = self._create_keypoints(offset_x=5, offset_y=2)
        det_f2 = {
            "keypoints": kp_f2,
            "confidence": 0.90,
            "bbox": [105, 102, 50, 280],
            "person_id": 0,
        }
        poses_f2 = integration.track_poses(
            [det_f2], frame_time=1001.0, frame_width=384, frame_height=384
        )
        assert len(poses_f2) == 1
        assert poses_f2[0].pose_id == pose_id_f1  # Same pose ID
        assert poses_f2[0].confidence == 0.90
        assert poses_f2[0].hit_streak == 1  # Reset after predict
        assert poses_f2[0].age == 2  # Incremented during predict

        # Frame 3: Person moves more
        for pose in integration.tracked_poses.values():
            pose.predict()

        kp_f3 = self._create_keypoints(offset_x=10, offset_y=4)
        det_f3 = {
            "keypoints": kp_f3,
            "confidence": 0.91,
            "bbox": [110, 104, 50, 280],
            "person_id": 0,
        }
        poses_f3 = integration.track_poses(
            [det_f3], frame_time=1002.0, frame_width=384, frame_height=384
        )
        assert len(poses_f3) == 1
        assert poses_f3[0].pose_id == pose_id_f1  # Still same ID
        assert poses_f3[0].hit_streak > 0  # Has been updated

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_multiple_people_tracking(self, mock_detector_class):
        """Test tracking multiple people across frames."""
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=MagicMock(),
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        # Frame 1: Two people detected
        kp_person1_f1 = self._create_keypoints(offset_x=50, offset_y=0)
        kp_person2_f1 = self._create_keypoints(offset_x=250, offset_y=0)

        det_p1_f1 = {
            "keypoints": kp_person1_f1,
            "confidence": 0.92,
            "bbox": [50, 100, 50, 280],
            "person_id": 0,
        }
        det_p2_f1 = {
            "keypoints": kp_person2_f1,
            "confidence": 0.89,
            "bbox": [250, 100, 50, 280],
            "person_id": 1,
        }

        poses_f1 = integration.track_poses(
            [det_p1_f1, det_p2_f1], frame_time=1000.0, frame_width=384, frame_height=384
        )
        assert len(poses_f1) == 2
        pose_ids_f1 = {p.pose_id for p in poses_f1}
        assert len(pose_ids_f1) == 2

        # Frame 2: Both people present, slightly moved
        for pose in integration.tracked_poses.values():
            pose.predict()

        kp_person1_f2 = self._create_keypoints(offset_x=55, offset_y=3)
        kp_person2_f2 = self._create_keypoints(offset_x=255, offset_y=2)

        det_p1_f2 = {
            "keypoints": kp_person1_f2,
            "confidence": 0.91,
            "bbox": [55, 103, 50, 280],
            "person_id": 0,
        }
        det_p2_f2 = {
            "keypoints": kp_person2_f2,
            "confidence": 0.88,
            "bbox": [255, 102, 50, 280],
            "person_id": 1,
        }

        poses_f2 = integration.track_poses(
            [det_p1_f2, det_p2_f2], frame_time=1001.0, frame_width=384, frame_height=384
        )
        assert len(poses_f2) == 2
        pose_ids_f2 = {p.pose_id for p in poses_f2}
        # Both poses should maintain their IDs
        assert pose_ids_f2 == pose_ids_f1

    @patch("frigate.pose_detection.integration.RemotePoseDetector")
    def test_pose_to_dict_conversion(self, mock_detector_class):
        """Test conversion of tracked poses to dictionary format."""
        tracked_poses_queue = MagicMock()
        stop_event = MagicMock()
        stop_event.is_set.return_value = False

        integration = PoseDetectionIntegration(
            camera_name=self.camera_name,
            config=self.camera_config,
            model_config=self.model_config,
            detection_queue=MagicMock(),
            tracked_poses_queue=tracked_poses_queue,
            stop_event=stop_event,
        )

        kp = self._create_keypoints()
        detection = {
            "keypoints": kp,
            "confidence": 0.92,
            "bbox": [100, 100, 50, 280],
            "person_id": 0,
        }

        poses = integration.track_poses(
            [detection], frame_time=1000.0, frame_width=384, frame_height=384
        )

        assert len(poses) == 1
        pose = poses[0]

        # Convert to dict
        pose_dict = pose.to_dict()

        # Verify required fields are present
        assert "id" in pose_dict
        assert "person_id" in pose_dict
        assert "keypoints" in pose_dict
        assert "confidence" in pose_dict
        assert "bbox" in pose_dict
        assert "frame_time" in pose_dict
        assert "action" in pose_dict
        assert "age" in pose_dict
        assert "hit_streak" in pose_dict
        assert "time_since_update" in pose_dict

        # Verify values
        assert pose_dict["confidence"] == 0.92
        assert pose_dict["person_id"] == 0
        assert pose_dict["frame_time"] == 1000.0


if __name__ == "__main__":
    unittest.main()
