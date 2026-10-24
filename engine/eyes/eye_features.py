"""Turns eye landmarks into h, v and lid, each divided by eye width.

Route: Khang and Hung (Khang leads)
Publishes: EyeFeatures
Subscribes: FaceFrame
APIs: NumPy; blendshapes eyeBlinkLeft / eyeBlinkRight for blink skipping

Per eye, with corner A (image left) and corner B (image right):
  h   = how far along A->B the iris center sits, divided by eye width (0 at A, 1 at B)
  v   = how far below the A->B line the iris center sits, divided by eye width (positive = down)
  lid = gap between upper and lower lid, divided by eye width
Dividing by width cancels leaning in and out; measuring along the eye's own line cancels head tilt.
"""
from dataclasses import dataclass

import numpy as np

from engine.config import settings
from engine.core.types import EyeFeatures, FaceFrame

# Indices checked on a real face with the installed MediaPipe: A is left of B in the image for both
# eyes, and each iris center sits between its own corners. Don't mirror frames before MediaPipe.
# "Right"/"left" are the subject's own sides, like MediaPipe's blendshape names.
RIGHT_EYE = {"a": 33, "b": 133, "iris": 468, "upper": 159, "lower": 145}   # left side of the image
LEFT_EYE = {"a": 362, "b": 263, "iris": 473, "upper": 386, "lower": 374}   # right side of the image


@dataclass
class EyeMeasure:
    h: float
    v: float
    lid: float
    width_px: float


def measure_eye(points_px: np.ndarray, eye: dict) -> EyeMeasure | None:
    """points_px: (N, 2) landmark positions in pixels."""
    a, b = points_px[eye["a"]], points_px[eye["b"]]
    ab = b - a
    width = float(np.hypot(*ab))
    if width < 1e-6:
        return None
    u = ab / width                    # along the eye
    n = np.array([-u[1], u[0]])       # at a right angle, pointing down in the image
    d = points_px[eye["iris"]] - a
    lid = float(np.hypot(*(points_px[eye["upper"]] - points_px[eye["lower"]])))
    return EyeMeasure(h=float(d @ u) / width, v=float(d @ n) / width, lid=lid / width, width_px=width)


class EyeFeatureExtractor:
    """Keeps the last good value so blinks don't make the cursor jump."""

    def __init__(self, blink_threshold: float | None = None) -> None:
        self.blink_threshold = settings.blink_threshold if blink_threshold is None else blink_threshold
        self.last: EyeFeatures | None = None
        self.right: EyeMeasure | None = None
        self.left: EyeMeasure | None = None
        self.blinking = False

    def compute(self, face: FaceFrame) -> EyeFeatures:
        # Convert to pixels first; the frame isn't square, so normalized units would stretch ratios.
        points_px = face.landmarks[:, :2] * np.array([face.width, face.height], dtype=np.float32)
        self.right = measure_eye(points_px, RIGHT_EYE)
        self.left = measure_eye(points_px, LEFT_EYE)
        bs = face.blendshapes
        self.blinking = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) > self.blink_threshold

        if self.blinking or self.right is None or self.left is None:
            if self.last is None:
                return EyeFeatures(t_ms=face.t_ms, h=float("nan"), v=float("nan"), lid=float("nan"), usable=False)
            return EyeFeatures(t_ms=face.t_ms, h=self.last.h, v=self.last.v, lid=self.last.lid, usable=False)

        # Average the two eyes, trusting the wider one more (the far eye looks narrower when you turn).
        wr, wl = self.right.width_px, self.left.width_px
        total = wr + wl
        feats = EyeFeatures(
            t_ms=face.t_ms,
            h=(self.right.h * wr + self.left.h * wl) / total,
            v=(self.right.v * wr + self.left.v * wl) / total,
            lid=(self.right.lid * wr + self.left.lid * wl) / total,
            usable=True,
        )
        self.last = feats
        return feats


def compute(face: FaceFrame) -> EyeFeatures:
    """One-off helper without blink memory. Use EyeFeatureExtractor in the live loop."""
    return EyeFeatureExtractor().compute(face)
