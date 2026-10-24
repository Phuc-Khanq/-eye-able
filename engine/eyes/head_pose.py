"""Yaw, pitch and roll from the facial transformation matrix.

Route: Khang and Hung (Hung leads)
Publishes: HeadPose
Subscribes: FaceFrame
APIs: NumPy, scipy.spatial.transform.Rotation.from_matrix
"""

def compute(face) -> "HeadPose":
    raise NotImplementedError
