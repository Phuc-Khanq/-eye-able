"""Turns eye landmarks into h, v and lid, each divided by eye width.

Route: Khang and Hung (Khang leads)
Publishes: EyeFeatures
Subscribes: FaceFrame
APIs: NumPy; blendshapes eyeBlinkLeft / eyeBlinkRight for blink skipping

Per eye, with corner A (image left) and corner B (image right):
  h   = how far along A->B the iris center sits, divided by eye width (0 at A, 1 at B)
  v   = how far below the A->B line the iris center sits, divided by eye width (positive = down)
  lid = gap between upper and lower lid, measured across the eye, divided by eye width
Dividing by width cancels leaning in and out; measuring along the eye's own line cancels head tilt.

Experiments (compare them live in tests/eye_bars.py):
  iris_mode "center"  uses MediaPipe's iris center point
  iris_mode "ring"    averages the center with the 4 iris-edge points, which may jitter less
  blend_h / blend_v   MediaPipe's own eye-direction scores (eyeLook* blendshapes), same directions
"""
from dataclasses import dataclass

import numpy as np

from engine.config import settings
from engine.core.types import EyeFeatures, FaceFrame

# Indices checked on a real face with the installed MediaPipe: A is left of B in the image for both
# eyes, each iris center sits between its own corners, and each ring circles its own center.
# Don't mirror frames before MediaPipe. "Right"/"left" are the subject's own sides, like MediaPipe's
# blendshape names.
RIGHT_EYE = {"a": 33, "b": 133, "iris": 468, "upper": 159, "lower": 145,     # left side of the image
             "ring": (469, 470, 471, 472)}
LEFT_EYE = {"a": 362, "b": 263, "iris": 473, "upper": 386, "lower": 374,     # right side of the image
            "ring": (474, 475, 476, 477)}


@dataclass
class EyeMeasure:
    h: float
    v: float
    lid: float
    width_px: float


def measure_eye(points_px: np.ndarray, eye: dict, iris_mode: str = "center") -> EyeMeasure | None:
    """points_px: (N, 2) landmark positions in pixels."""
    a, b = points_px[eye["a"]], points_px[eye["b"]]
    ab = b - a
    width = float(np.hypot(*ab))
    if width < 1e-6:
        return None
    if iris_mode == "ring":
        iris = points_px[[eye["iris"], *eye["ring"]]].mean(axis=0)
    else:
        iris = points_px[eye["iris"]]
    u = ab / width                    # along the eye
    n = np.array([-u[1], u[0]])       # at a right angle, pointing down in the image
    d = iris - a
    # Lid gap measured across the eye only (along n). The two lid points aren't exactly above each
    # other, so a straight-line distance would never reach 0 for a closed eye.
    lid = max(0.0, float((points_px[eye["lower"]] - points_px[eye["upper"]]) @ n))
    return EyeMeasure(h=float(d @ u) / width, v=float(d @ n) / width, lid=lid / width, width_px=width)


def blendshape_gaze(bs: dict) -> tuple[float, float]:
    """MediaPipe's eye-direction scores turned into image directions, each about -1 to 1.
    blend_h > 0 means looking toward the image's right (h goes up); blend_v > 0 means looking down."""
    g = lambda name: bs.get(name, 0.0)  # noqa: E731
    # Subject's left eye looking out, or right eye looking in, both point toward the image's right.
    blend_h = ((g("eyeLookOutLeft") + g("eyeLookInRight")) - (g("eyeLookInLeft") + g("eyeLookOutRight"))) / 2
    blend_v = ((g("eyeLookDownLeft") + g("eyeLookDownRight")) - (g("eyeLookUpLeft") + g("eyeLookUpRight"))) / 2
    return blend_h, blend_v


class EyeFeatureExtractor:
    """Keeps the last good value so blinks don't make the cursor jump.

    While a blink is detected (either eyeBlink score above the threshold), compute() returns the
    LAST OPEN-EYE values with usable=False. So the returned lid never shows a closed eye; that's on
    purpose. For debugging, `lid_now` and `blink_score` always describe the current frame.
    """

    def __init__(self, blink_threshold: float | None = None, iris_mode: str = "center") -> None:
        self.blink_threshold = settings.blink_threshold if blink_threshold is None else blink_threshold
        self.iris_mode = iris_mode
        self.last: EyeFeatures | None = None
        self.right: EyeMeasure | None = None
        self.left: EyeMeasure | None = None
        self.blinking = False
        self.blink_score = float("nan")  # higher of the two eyeBlink scores, this frame
        self.lid_now = float("nan")      # lid this frame, even during a blink
        self.blend_h = float("nan")
        self.blend_v = float("nan")

    def compute(self, face: FaceFrame) -> EyeFeatures:
        # Convert to pixels first; the frame isn't square, so normalized units would stretch ratios.
        points_px = face.landmarks[:, :2] * np.array([face.width, face.height], dtype=np.float32)
        self.right = measure_eye(points_px, RIGHT_EYE, self.iris_mode)
        self.left = measure_eye(points_px, LEFT_EYE, self.iris_mode)
        bs = face.blendshapes
        self.blend_h, self.blend_v = blendshape_gaze(bs)
        self.blink_score = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        self.blinking = self.blink_score > self.blink_threshold
        if self.right is not None and self.left is not None:
            wr, wl = self.right.width_px, self.left.width_px
            self.lid_now = (self.right.lid * wr + self.left.lid * wl) / (wr + wl)
        else:
            self.lid_now = float("nan")

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
