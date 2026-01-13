"""Debug script to test pose matching logic."""

import numpy as np

# Test the matching thresholds
frame_width = 384
frame_height = 384
diag = (frame_width**2 + frame_height**2) ** 0.5

print(f"Frame diagonal: {diag:.2f}")
print(f"3% threshold: {diag * 0.03:.2f}")
print(f"8% threshold: {diag * 0.08:.2f}")

# Test centroid distances
kp1 = np.array(
    [
        [100, 100, 0.9],
        [110, 95, 0.85],
        [120, 95, 0.88],
    ]
    + [[50 + i * 10, 150 + i * 10, 0.9] for i in range(14)],
    dtype=np.float32,
)

kp2 = kp1 + np.array([5, 2, 0], dtype=np.float32)  # Slight movement

vis1 = kp1[:, 2] > 0.1
vis2 = kp2[:, 2] > 0.1

centroid1 = np.mean(kp1[vis1, :2], axis=0)
centroid2 = np.mean(kp2[vis2, :2], axis=0)

dist = np.linalg.norm(centroid2 - centroid1)
print(f"\nCentroid distance (small movement): {dist:.2f}")
print(f"Normalized distance: {dist / diag:.4f}")
print(f"Will match at 3% threshold? {dist / diag < 0.03}")

# Larger movement
kp3 = kp1 + np.array([50, 50, 0], dtype=np.float32)
centroid3 = np.mean(kp3[vis1, :2], axis=0)
dist2 = np.linalg.norm(centroid3 - centroid1)
print(f"\nCentroid distance (large movement): {dist2:.2f}")
print(f"Normalized distance: {dist2 / diag:.4f}")
print(f"Will match at 3% threshold? {dist2 / diag < 0.03}")
