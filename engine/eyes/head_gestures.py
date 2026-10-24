"""Nod = quick pitch dip and return; shake = yaw back and forth; tilts. Per-user thresholds.

Route: Khang and Hung (Hung leads)
Publishes: Gesture
Subscribes: HeadPose
APIs: NumPy over the last second of head angles
"""

class GestureDetector:
    def update(self, pose) -> "Gesture | None":
        raise NotImplementedError
