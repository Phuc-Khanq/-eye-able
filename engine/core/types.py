"""Data types every module shares. Changing a field means telling the whole team."""
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceFrame:
    t_ms: int                       # capture time
    width: int                      # frame size in pixels
    height: int
    landmarks: np.ndarray           # (478, 3), MediaPipe normalized x, y, z
    blendshapes: dict[str, float]   # e.g. {"eyeBlinkLeft": 0.03, ...}
    head_matrix: np.ndarray         # (4, 4) facial transformation matrix


@dataclass
class EyeFeatures:
    t_ms: int
    h: float      # along the eye, 0 at corner A, 1 at corner B
    v: float      # below the corner line, divided by eye width
    lid: float    # lid gap divided by eye width
    usable: bool  # False on blinks or bad frames
    blend_h: float = 0.0  # MediaPipe's eye-direction score, + = toward the image's right (about -1 to 1)
    blend_v: float = 0.0  # MediaPipe's eye-direction score, + = looking down (about -1 to 1)


@dataclass
class HeadPose:
    t_ms: int
    yaw: float    # degrees
    pitch: float
    roll: float


@dataclass
class GazePoint:
    t_ms: int
    x: float          # screen pixels (physical)
    y: float
    confidence: float  # 0 to 1


@dataclass
class Gesture:
    t_ms: int
    type: str  # "nod" | "shake" | "tilt_left" | "tilt_right"


@dataclass
class TargetRequest:
    gaze_x: int  # screen pixels (physical)
    gaze_y: int
    label: str   # e.g. "Send"


@dataclass
class TargetResult:
    label: str
    found: bool
    x: int = -1     # screen pixels (physical); -1 when not found
    y: int = -1
    ms: float = 0.0  # time UI-Venus took
