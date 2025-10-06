# Pose Detection Integration for Frigate

This document describes the pose detection implementation that has been integrated into Frigate, following the same patterns as the existing object detection system.

## Overview

The pose detection system allows Frigate to detect and track human poses in real-time, identify pose actions (standing, sitting, walking, etc.), and trigger events based on pose activities. This is implemented as a parallel system to object detection, with its own detectors, processing pipeline, and event system.

## Architecture

### Core Components

1. **Pose Detection Module** (`/frigate/pose_detection/`)
   - `base.py` - Core pose detection classes and process management
   - `util.py` - Utility functions for pose processing and analysis

2. **Pose Detectors** (`/frigate/pose_detectors/`)
   - Plugin-based system supporting multiple pose detection backends
   - `cpu_pose.py` - CPU-based pose detection (placeholder)
   - `yolo_pose.py` - YOLO pose detection using ONNX Runtime
   - `mediapipe_pose.py` - MediaPipe pose detection

3. **Pose Tracking** (`/frigate/track/`)
   - `tracked_pose.py` - Pose tracking and action analysis
   - `pose_processing.py` - Pose event processing and management

4. **Configuration** (`/frigate/config/camera/pose.py`)
   - Pose detection settings per camera
   - Action filtering and thresholds
   - Snapshot and recording triggers

5. **Events** (`/frigate/events/`)
   - `pose_types.py` - Pose-specific event types and actions
   - Updated `maintainer.py` to handle pose events

## Supported Pose Detectors

### 1. YOLO Pose (Recommended)
- **Type**: `yolo_pose`
- **Backend**: ONNX Runtime
- **Models**: YOLOv8-pose, YOLOv11-pose
- **Performance**: High speed, good accuracy
- **Hardware**: CPU/GPU support

```yaml
pose_detectors:
  yolo_pose:
    type: yolo_pose
    device: cpu  # or gpu
```

### 2. MediaPipe Pose
- **Type**: `mediapipe`
- **Backend**: MediaPipe
- **Models**: MediaPipe Pose (Lite, Full, Heavy)
- **Performance**: Good speed, excellent accuracy
- **Hardware**: CPU optimized

```yaml
pose_detectors:
  mediapipe:
    type: mediapipe
    model_complexity: 1
    min_detection_confidence: 0.5
    min_tracking_confidence: 0.5
```

### 3. CPU Pose (Testing Only)
- **Type**: `cpu`
- **Backend**: Placeholder implementation
- **Purpose**: Testing and development

## Configuration

### Global Pose Detection Settings

```yaml
# Pose detectors configuration
pose_detectors:
  yolo_pose:
    type: yolo_pose
    device: cpu

# Global pose model configuration
pose_model:
  width: 640
  height: 480
  model_type: yolo_pose
  path: /path/to/yolov8n-pose.onnx
  confidence_threshold: 0.4
  keypoint_threshold: 0.3
  num_keypoints: 17
```

### Per-Camera Pose Configuration

```yaml
cameras:
  front_door:
    pose:
      enabled: true
      confidence_threshold: 0.5
      keypoint_threshold: 0.3
      fps: 2  # Pose detection FPS (should be <= camera detect fps)
      
      # Actions to track
      actions:
        - standing
        - walking
        - sitting
        - waving
        - pointing
      
      # Actions that trigger snapshots
      snapshot_actions:
        - waving
        - pointing
      
      # Actions that trigger recording retention
      record_actions:
        - walking
        - running
        - jumping
      
      # Action-specific filters
      filters:
        waving:
          min_score: 0.6
          threshold: 0.5
      
      # Required zones for pose detection
      required_zones:
        - entrance
```

## Pose Actions

The system can detect and classify the following pose actions:

- **standing** - Person standing upright
- **sitting** - Person in sitting position
- **lying** - Person lying down
- **walking** - Person in walking motion
- **running** - Person in running motion
- **jumping** - Person jumping
- **waving** - Person waving (hand raised)
- **pointing** - Person pointing
- **custom** - User-defined actions

## Event Integration

Pose detection events are integrated into Frigate's existing event system:

### Event Types
- `tracked_pose` - Pose detection events
- Events are stored in the same database with type "pose"
- MQTT messages published to `pose_events` topic

### Event Data Structure
```json
{
  "id": "pose_123456789",
  "person_id": 0,
  "action": "waving",
  "action_confidence": 0.85,
  "confidence": 0.92,
  "keypoints": [[x1, y1, conf1], [x2, y2, conf2], ...],
  "bbox": [x, y, width, height],
  "zones": ["entrance"],
  "has_snapshot": true,
  "has_clip": false
}
```

## Installation and Setup

### 1. Install Dependencies

For YOLO Pose:
```bash
pip install onnxruntime  # CPU
pip install onnxruntime-gpu  # GPU
```

For MediaPipe:
```bash
pip install mediapipe
```

### 2. Download Models

For YOLO Pose:
- Download YOLOv8-pose or YOLOv11-pose ONNX models
- Place in accessible directory
- Update `pose_model.path` in configuration

### 3. Configure Cameras

Add pose detection configuration to your camera settings as shown in the examples above.

### 4. Restart Frigate

The pose detection system will start automatically when Frigate starts, provided at least one camera has pose detection enabled.

## Performance Considerations

### Resource Usage
- Pose detection is computationally intensive
- Recommended to run at lower FPS than object detection (2-5 FPS)
- Consider using GPU acceleration for YOLO models
- MediaPipe is well-optimized for CPU usage

### Memory Requirements
- Each pose detector process requires additional memory
- Shared memory buffers are created for pose detection data
- Monitor system resources when enabling multiple cameras

### Optimization Tips
1. **Lower FPS**: Set pose detection FPS to 2-3 for most use cases
2. **Zone Filtering**: Use `required_zones` to limit processing area
3. **Action Filtering**: Only track actions you need
4. **Model Selection**: Choose appropriate model complexity for your hardware

## Troubleshooting

### Common Issues

1. **Pose Detection Not Starting**
   - Check that at least one camera has `pose.enabled: true`
   - Verify pose detector configuration
   - Check logs for initialization errors

2. **Poor Detection Quality**
   - Adjust `confidence_threshold` and `keypoint_threshold`
   - Ensure adequate lighting in detection area
   - Check model path and validity

3. **High CPU Usage**
   - Reduce pose detection FPS
   - Use GPU acceleration if available
   - Limit detection to specific zones

4. **Missing Dependencies**
   - Install required packages (onnxruntime, mediapipe)
   - Check package versions compatibility

### Logs and Debugging

Pose detection logs are available in Frigate's standard logging:
- Process startup: `frigate.pose_detector:*`
- Detection results: `frigate.pose_detection.*`
- Event processing: `frigate.events.maintainer`

Enable debug logging for detailed information:
```yaml
logger:
  logs:
    frigate.pose_detection: debug
    frigate.pose_detectors: debug
```

## API Integration

Pose detection events are accessible through Frigate's existing API endpoints:

- **Events API**: `/api/events` (filter by type="pose")
- **WebSocket**: Real-time pose detection updates
- **MQTT**: `frigate/events` topic with pose event data

## Future Enhancements

Potential areas for future development:

1. **Additional Models**: Support for more pose detection models
2. **Custom Actions**: User-defined pose action detection
3. **Gesture Recognition**: More sophisticated gesture analysis
4. **Multi-Person Tracking**: Improved handling of multiple people
5. **Integration with Home Assistant**: Dedicated pose detection entities

## Contributing

When contributing to the pose detection system:

1. Follow the existing code patterns from object detection
2. Add tests for new pose detectors
3. Update documentation for new features
4. Ensure compatibility with existing Frigate functionality

## License

This pose detection implementation follows the same license as Frigate.