# Pose Estimation Integration for Frigate

## Overview

This document describes the pose estimation feature that has been integrated into Frigate, following the same architecture pattern as object detection.

## What Has Been Implemented

### 1. Core Pose Estimation Module (`/frigate/pose_estimation/`)

- **base.py**: Core classes and interfaces
  - `PoseKeypoint`: Represents a single keypoint (x, y, confidence)
  - `Pose`: Represents a complete pose with keypoints and bounding box
  - `PoseDetector`: Abstract base class for pose detectors
  - `LocalPoseDetector`: Local pose detection implementation
  - `RemotePoseDetector`: Remote pose detection via shared memory
  - `PoseDetectorRunner`: Process runner for pose detection
  - `PoseDetectProcess`: Manages pose detection processes

- **util.py**: Utility functions for tensor transformations

- **plugins/**: Detector implementations
  - `yolo_pose.py`: YOLO Pose detector implementation
  - `mediapipe_pose.py`: MediaPipe Pose detector implementation

- **pose_detector_types.py**: Type definitions and enums for pose detectors

- **pose_factory.py**: Factory for creating pose detector instances

- **pose_processor.py**: Event processing for pose detections
  - `PoseEvent`: Represents a pose detection event
  - `PoseTracker`: Tracks poses across frames
  - `PoseActionDetector`: Detects actions from poses (falling, fighting, running)
  - `PoseEventProcessor`: Processes pose events and sends to event system

- **pose_integration.py**: Integration with video processing pipeline
  - `detect_poses()`: Detect poses in frame regions
  - `create_pose_detector()`: Create pose detector for camera
  - `filter_pose_detections()`: Filter poses based on configuration
  - `convert_object_to_pose()`: Convert person objects to poses
  - `PoseDetectionIntegration`: Main integration class

### 2. Configuration Updates

- **Added to FrigateConfig** (`/frigate/config/config.py`):
  - `pose_detectors`: Dictionary of pose detector configurations
  - `pose_model`: Pose model configuration
  - Validation logic for pose detectors

- **New Camera Configuration** (`/frigate/config/camera/pose.py`):
  - `PoseConfig`: Complete pose detection configuration per camera
  - `PoseFilterConfig`: Filtering options for poses
  - `PoseActionConfig`: Action detection configuration

- **Camera Config Integration**: Added `pose` field to `CameraConfig`

### 3. Event System Updates

- **Event Types** (`/frigate/events/types.py`):
  - Added `EventTypeEnum.pose` for pose events

- **Event Maintainer** (`/frigate/events/maintainer.py`):
  - Added `handle_pose_detection()` method
  - Pose events are handled similarly to object detection events
  - Stores pose data, actions, and keypoints in the database

### 4. Video Processing Integration

- **Updated video.py**:
  - Imports pose integration module
  - Added pose detection queues to `CameraTracker`
  - Creates `PoseDetectionIntegration` instance when pose is enabled
  - Processes poses after object detection in the frame loop
  - Handles cleanup of pose resources

## Configuration Example

```yaml
# Pose detector configuration
pose_detectors:
  yolo_pose:
    type: yolo_pose
    model_path: /models/yolov8n-pose.pt
    device: 0  # GPU device or 'cpu'

# Pose model settings
pose_model:
  width: 640
  height: 640
  model_type: yolo-generic
  input_tensor: nhwc
  input_pixel_format: rgb

# Camera configuration
cameras:
  front_door:
    # ... existing config ...
    
    # Pose detection configuration
    pose:
      enabled: true
      fps: 5
      max_disappeared: 25
      
      # Tracking configuration
      track:
        - person
      
      # Filtering
      filters:
        person:
          min_keypoints: 5
          min_confidence: 0.4
          min_area: 1000
          
      # Action detection
      actions:
        enabled: true
        actions:
          - falling
          - fighting
          - running
        action_threshold: 0.7
        
      # Other options
      objects_as_poses: false  # Convert person detections to poses
      keypoint_threshold: 0.5
      required_zones: []
```

## Key Features

### 1. Multiple Detector Support
- YOLO Pose: Fast and accurate pose estimation
- MediaPipe Pose: Lightweight pose estimation with 3D landmarks

### 2. Action Detection
- Falling detection: Analyzes torso orientation
- Fighting detection: Detects raised arms
- Running detection: Analyzes leg positions

### 3. Pose Tracking
- Tracks poses across frames
- Handles pose disappearance and reappearance
- Maintains pose IDs for consistent tracking

### 4. Event Integration
- Pose events stored in the same database as object events
- Supports clips, snapshots, and recordings
- Works with existing review and alert systems

### 5. Zone Support
- Pose detection can be limited to specific zones
- Events track which zones poses have entered

### 6. Object-to-Pose Conversion
- Can convert detected person objects to poses
- Useful when pose model is not available

## Architecture

The pose estimation system follows the same architecture as object detection:

1. **Detection Process**: Separate process for pose detection using shared memory
2. **Event Processing**: Pose events are processed like object events
3. **Database Storage**: Poses stored in the events table with type "pose"
4. **Integration**: Seamlessly integrates with existing Frigate features

## Usage

1. **Add Pose Model**: Place your pose model file in the appropriate directory
2. **Configure Detectors**: Add pose detector configuration to your config file
3. **Enable for Cameras**: Enable pose detection for specific cameras
4. **Configure Actions**: Set up action detection if needed
5. **Review Events**: Pose events appear in the Frigate UI like object events

## Implementation Notes

- The current action detection is basic and uses simple heuristics
- Real implementations should use proper pose models for action recognition
- The pose tracking algorithm is simplified and could be enhanced
- Integration points are designed to be non-invasive to existing functionality

## Future Enhancements

1. More sophisticated action recognition models
2. Support for additional pose models (OpenPose, PoseNet, etc.)
3. 3D pose estimation support
4. Pose-based analytics and statistics
5. Custom action definitions via configuration
6. Pose similarity matching for specific poses

## Testing

A test script (`test_pose_integration.py`) is provided to verify the implementation:
- Tests core classes (PoseKeypoint, Pose)
- Tests configuration integration
- Tests mock pose detection flow
- Tests event data structures

The implementation is ready for integration with actual pose models and can be extended based on specific requirements.