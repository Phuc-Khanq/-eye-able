"""Yaw, pitch and roll from the facial transformation matrix.

Route: Khang and Hung (Hung leads)
Publishes: HeadPose
Subscribes: FaceFrame
APIs: NumPy, scipy.spatial.transform.Rotation.from_matrix

Angles are in degrees, in the subject's own terms, and 0 when facing the camera:
  yaw   > 0 = head turned to the subject's right (toward the right edge of the screen)
  pitch > 0 = chin down
  roll  > 0 = head tilted toward the subject's right shoulder
MediaPipe's matrix maps the canonical face into camera space (x right in the image, y up,
z toward the camera). The signs below follow from that; check them once on the laptop with
`python -m tests.head_bars` and flip a sign here if one runs the wrong way.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from engine.core.types import FaceFrame, HeadPose

# Raw rotations about camera x, y, z -> pitch, yaw, roll in the subject's terms.
PITCH_SIGN = 1.0   # chin down turns the face normal toward -y: positive rotation about x
YAW_SIGN = -1.0    # turning to the subject's right moves the face normal toward image left (-x)
ROLL_SIGN = 1.0    # tilting to the subject's right moves the top of the head toward image left


NEUTRAL_FRAMES = 15   # set_neutral uses the median of this many recent frames, so one bad frame can't skew it


def angle_diff(a: float, b: float) -> float:
    """a - b in degrees, wrapped to -180..180 so an angle near the wrap point can't jump by 360."""
    return (a - b + 180.0) % 360.0 - 180.0


def raw_angles(head_matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees, not yet relative to any neutral pose."""
    r = np.asarray(head_matrix, dtype=np.float64)[:3, :3]
    # The matrix can carry a little scale; normalize each column so it is a pure rotation.
    norms = np.linalg.norm(r, axis=0)
    if np.any(norms < 1e-9):
        return 0.0, 0.0, 0.0
    rx, ry, rz = Rotation.from_matrix(r / norms).as_euler("xyz", degrees=True)
    return float(YAW_SIGN * ry), float(PITCH_SIGN * rx), float(ROLL_SIGN * rz)


class HeadPoseEstimator:
    """Head angles relative to a neutral pose, so a user who sits a little turned still reads 0."""

    def __init__(self) -> None:
        self.neutral = (0.0, 0.0, 0.0)
        self.last: HeadPose | None = None
        self._recent: deque[tuple[float, float, float]] = deque(maxlen=NEUTRAL_FRAMES)

    def set_neutral(self, face: FaceFrame | None = None) -> None:
        """Call while the user looks at the screen center, e.g. at the calibration center dot.
        Uses the median of the last NEUTRAL_FRAMES frames (plus `face`, if given)."""
        samples = list(self._recent)
        if face is not None:
            samples.append(raw_angles(face.head_matrix))
        if samples:
            self.neutral = tuple(float(np.median([s[i] for s in samples])) for i in range(3))

    def reset_neutral(self) -> None:
        self.neutral = (0.0, 0.0, 0.0)

    def compute(self, face: FaceFrame) -> HeadPose:
        raw = raw_angles(face.head_matrix)
        self._recent.append(raw)
        n = self.neutral
        self.last = HeadPose(t_ms=face.t_ms, yaw=angle_diff(raw[0], n[0]), pitch=angle_diff(raw[1], n[1]),
                             roll=angle_diff(raw[2], n[2]))
        return self.last


def compute(face: FaceFrame) -> HeadPose:
    """One-off helper with no neutral offset. Use HeadPoseEstimator in the live loop."""
    yaw, pitch, roll = raw_angles(face.head_matrix)
    return HeadPose(t_ms=face.t_ms, yaw=yaw, pitch=pitch, roll=roll)
