"""Shared memory format constants for dynamic-size pose detection frames.

The SHM buffer layout for pose detection input:
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

Buffer sizing: The SHM buffer is sized per-camera based on the camera's
detect dimensions (detect.width × detect.height), not a global maximum.
This reduces memory usage significantly for lower-resolution cameras.
"""

import numpy as np

# Header format constants
HEADER_SIZE_BYTES = 16  # 4 x int32
HEADER_DTYPE = np.int32
HEADER_NUM_FIELDS = 4

# Header field indices
HEADER_WIDTH_IDX = 0
HEADER_HEIGHT_IDX = 1
HEADER_STRIDE_IDX = 2
HEADER_FLAGS_IDX = 3

# Fallback maximum dimensions (used if camera dimensions not available)
MAX_POSE_WIDTH = 1920
MAX_POSE_HEIGHT = 1080
PIXEL_CHANNELS = 3  # RGB

# Fallback buffer size (only used as default)
MAX_PIXEL_DATA_SIZE = MAX_POSE_WIDTH * MAX_POSE_HEIGHT * PIXEL_CHANNELS
POSE_SHM_BUFFER_SIZE = HEADER_SIZE_BYTES + MAX_PIXEL_DATA_SIZE


def calculate_shm_buffer_size(width: int, height: int) -> int:
    """Calculate the SHM buffer size needed for a given frame dimension.

    Args:
        width: Maximum frame width in pixels
        height: Maximum frame height in pixels

    Returns:
        Total buffer size in bytes (header + pixel data)
    """
    pixel_data_size = width * height * PIXEL_CHANNELS
    return HEADER_SIZE_BYTES + pixel_data_size


def write_frame_to_shm(shm_buf, frame: np.ndarray) -> bool:
    """Write a frame to shared memory with header metadata.

    Args:
        shm_buf: Shared memory buffer (must be at least POSE_SHM_BUFFER_SIZE bytes)
        frame: RGB frame as numpy array with shape (H, W, 3)

    Returns:
        True if successful, False if frame too large
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")

    height, width = frame.shape[:2]

    if width > MAX_POSE_WIDTH or height > MAX_POSE_HEIGHT:
        return False

    stride = width * PIXEL_CHANNELS

    # Write header
    header = np.ndarray(
        (HEADER_NUM_FIELDS,), dtype=HEADER_DTYPE, buffer=shm_buf[:HEADER_SIZE_BYTES]
    )
    header[HEADER_WIDTH_IDX] = width
    header[HEADER_HEIGHT_IDX] = height
    header[HEADER_STRIDE_IDX] = stride
    header[HEADER_FLAGS_IDX] = 0

    # Write pixel data
    pixel_size = height * width * PIXEL_CHANNELS
    pixel_view = np.ndarray(
        (height, width, PIXEL_CHANNELS),
        dtype=np.uint8,
        buffer=shm_buf[HEADER_SIZE_BYTES : HEADER_SIZE_BYTES + pixel_size],
    )
    np.copyto(pixel_view, frame)

    return True


def read_frame_from_shm(shm_buf) -> np.ndarray | None:
    """Read a frame from shared memory using header metadata.

    Args:
        shm_buf: Shared memory buffer

    Returns:
        RGB frame as numpy array with shape (H, W, 3), or None if invalid
    """
    # Read header
    header = np.ndarray(
        (HEADER_NUM_FIELDS,), dtype=HEADER_DTYPE, buffer=shm_buf[:HEADER_SIZE_BYTES]
    )
    width = int(header[HEADER_WIDTH_IDX])
    height = int(header[HEADER_HEIGHT_IDX])

    if width <= 0 or height <= 0 or width > MAX_POSE_WIDTH or height > MAX_POSE_HEIGHT:
        return None

    # Read pixel data
    pixel_size = height * width * PIXEL_CHANNELS
    pixel_view = np.ndarray(
        (height, width, PIXEL_CHANNELS),
        dtype=np.uint8,
        buffer=shm_buf[HEADER_SIZE_BYTES : HEADER_SIZE_BYTES + pixel_size],
    )

    # Return a copy to avoid issues with buffer reuse
    return pixel_view.copy()


def get_frame_dimensions_from_shm(shm_buf) -> tuple[int, int] | None:
    """Read just the frame dimensions from shared memory header.

    Args:
        shm_buf: Shared memory buffer

    Returns:
        Tuple of (width, height) or None if invalid
    """
    header = np.ndarray(
        (HEADER_NUM_FIELDS,), dtype=HEADER_DTYPE, buffer=shm_buf[:HEADER_SIZE_BYTES]
    )
    width = int(header[HEADER_WIDTH_IDX])
    height = int(header[HEADER_HEIGHT_IDX])

    if width <= 0 or height <= 0:
        return None

    return (width, height)
