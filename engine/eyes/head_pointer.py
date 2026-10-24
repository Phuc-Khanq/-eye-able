"""Fallback cursor driven by head angles, for when gaze is too jittery.

Route: Khang and Hung (Hung leads)
Publishes: GazePoint
Subscribes: HeadPose
APIs: NumPy
"""

def to_cursor(pose) -> tuple[float, float]:
    raise NotImplementedError
