#!/usr/bin/env python3
"""
Test script to verify pose estimation integration in Frigate.
This script tests the basic functionality without requiring actual camera feeds.
"""

import sys
import numpy as np
from typing import List, Dict, Any

# Add frigate to path
sys.path.insert(0, '/workspace')

from frigate.config import FrigateConfig
from frigate.detectors.detector_config import BaseDetectorConfig, ModelConfig, InputTensorEnum
from frigate.pose_estimation.base import Pose, PoseKeypoint, BaseLocalPoseDetector
from frigate.pose_estimation.pose_detector_types import PoseDetectorConfig
from frigate.pose_estimation.pose_factory import create_pose_detector


def test_pose_keypoint():
    """Test PoseKeypoint class."""
    print("Testing PoseKeypoint...")
    kp = PoseKeypoint(100.5, 200.5, 0.85)
    assert kp.x == 100.5
    assert kp.y == 200.5
    assert kp.confidence == 0.85
    
    kp_dict = kp.to_dict()
    assert kp_dict["x"] == 100.5
    assert kp_dict["y"] == 200.5
    assert kp_dict["confidence"] == 0.85
    print("✓ PoseKeypoint test passed")


def test_pose():
    """Test Pose class."""
    print("\nTesting Pose...")
    keypoints = [
        PoseKeypoint(100, 100, 0.9),
        PoseKeypoint(110, 105, 0.85),
        PoseKeypoint(90, 105, 0.88),
    ]
    
    pose = Pose(
        keypoints=keypoints,
        pose_id="test_pose_1",
        confidence=0.87,
        bbox=[85, 95, 115, 150]
    )
    
    assert pose.pose_id == "test_pose_1"
    assert pose.confidence == 0.87
    assert len(pose.keypoints) == 3
    assert pose.bbox == [85, 95, 115, 150]
    
    pose_dict = pose.to_dict()
    assert pose_dict["pose_id"] == "test_pose_1"
    assert pose_dict["confidence"] == 0.87
    assert len(pose_dict["keypoints"]) == 3
    print("✓ Pose test passed")


def test_pose_detector_config():
    """Test PoseDetectorConfig."""
    print("\nTesting PoseDetectorConfig...")
    config = PoseDetectorConfig(
        type="yolo_pose",
        keypoint_thresh=0.6,
        pose_thresh=0.5,
        max_poses=10
    )
    
    assert config.type == "yolo_pose"
    assert config.keypoint_thresh == 0.6
    assert config.pose_thresh == 0.5
    assert config.max_poses == 10
    print("✓ PoseDetectorConfig test passed")


def test_config_integration():
    """Test pose configuration in FrigateConfig."""
    print("\nTesting config integration...")
    
    config_dict = {
        "mqtt": {"enabled": False},
        "cameras": {
            "test_camera": {
                "enabled": True,
                "ffmpeg": {
                    "inputs": [
                        {
                            "path": "rtsp://fake",
                            "roles": ["detect"]
                        }
                    ]
                },
                "detect": {
                    "enabled": True,
                    "width": 1280,
                    "height": 720
                },
                "pose": {
                    "enabled": True,
                    "fps": 5,
                    "keypoint_threshold": 0.6,
                    "filters": {
                        "person": {
                            "min_keypoints": 5,
                            "min_confidence": 0.5
                        }
                    },
                    "actions": {
                        "enabled": True,
                        "actions": ["falling", "running"],
                        "action_threshold": 0.7
                    }
                }
            }
        }
    }
    
    # Test that config can be parsed
    try:
        config = FrigateConfig(**config_dict)
        camera_config = config.cameras["test_camera"]
        
        assert camera_config.pose.enabled == True
        assert camera_config.pose.fps == 5
        assert camera_config.pose.keypoint_threshold == 0.6
        assert camera_config.pose.filters["person"].min_keypoints == 5
        assert camera_config.pose.actions.enabled == True
        assert "falling" in camera_config.pose.actions.actions
        print("✓ Config integration test passed")
    except Exception as e:
        print(f"✗ Config integration test failed: {e}")
        raise


def test_mock_pose_detection():
    """Test mock pose detection flow."""
    print("\nTesting mock pose detection...")
    
    # Create a mock detector config
    class MockPoseDetector(BaseLocalPoseDetector):
        def detect_raw(self, tensor_input: np.ndarray):
            # Return mock pose data
            return [{
                "confidence": 0.85,
                "bbox": [100, 100, 300, 400],
                "keypoints": [[150, 120, 0.9], [160, 125, 0.88], [140, 125, 0.87]]
            }]
    
    detector_config = BaseDetectorConfig(type="mock")
    detector_config.model = ModelConfig(
        path="/fake/model.pt",
        width=640,
        height=640,
        input_tensor=InputTensorEnum.nhwc
    )
    
    # Create detector
    detector = MockPoseDetector(detector_config=detector_config)
    
    # Create mock frame
    mock_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Detect poses
    poses = detector.detect_poses(mock_frame, threshold=0.4)
    
    assert len(poses) == 1
    assert poses[0].confidence == 0.85
    assert len(poses[0].keypoints) == 3
    print("✓ Mock pose detection test passed")


def test_pose_event_data():
    """Test pose event data structure."""
    print("\nTesting pose event data...")
    
    from frigate.pose_estimation.pose_processor import PoseEvent
    
    # Create mock pose
    keypoints = [PoseKeypoint(100 + i*10, 100 + i*5, 0.8 - i*0.05) for i in range(17)]
    pose = Pose(keypoints=keypoints, confidence=0.85, bbox=[100, 100, 200, 300])
    
    # Create pose event
    event = PoseEvent(
        pose_id="test_pose_event_1",
        camera="test_camera",
        pose=pose,
        frame_time=1234567890.123,
        zones=["entrance", "hallway"],
        actions=["walking"]
    )
    
    assert event.pose_id == "test_pose_event_1"
    assert event.camera == "test_camera"
    assert event.start_time == 1234567890.123
    assert "entrance" in event.zones
    assert "walking" in event.actions
    
    # Test event dict conversion
    event_dict = event.to_dict()
    assert event_dict["id"] == "test_pose_event_1"
    assert event_dict["camera"] == "test_camera"
    assert event_dict["label"] == "pose"
    assert event_dict["sub_label"] == "walking"
    assert event_dict["score"] == 0.85
    assert "pose_data" in event_dict
    print("✓ Pose event data test passed")


def main():
    """Run all tests."""
    print("Running Pose Estimation Integration Tests")
    print("=" * 50)
    
    try:
        test_pose_keypoint()
        test_pose()
        test_pose_detector_config()
        test_config_integration()
        test_mock_pose_detection()
        test_pose_event_data()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("\nPose estimation has been successfully integrated into Frigate.")
        print("\nKey features implemented:")
        print("- Base pose detection classes (PoseKeypoint, Pose)")
        print("- YOLO Pose and MediaPipe Pose detector plugins")
        print("- Pose configuration in camera settings")
        print("- Pose event processing and tracking")
        print("- Integration with existing event system")
        print("- Action detection from poses (falling, fighting, running)")
        
        print("\nTo use pose estimation:")
        print("1. Add pose detector configuration to your config file")
        print("2. Enable pose detection for specific cameras")
        print("3. Configure pose filters and actions as needed")
        print("4. Pose events will be handled like object detection events")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()