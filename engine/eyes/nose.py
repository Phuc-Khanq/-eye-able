"""The nose as a second way to read the head's turn: where the nose tip sits between the cheeks and between the
forehead and the chin.

Route: Khang and Hung (Khang leads)
Publishes: NoseAngles (to the pointer, through the eyes service)
Subscribes: FaceFrame landmarks, HeadPose (only to learn the scale)
APIs: MediaPipe face landmarks 1 (nose tip), 234 and 454 (the cheeks), 10 (forehead), 152 (chin); NumPy

Why: the head pointer is a relative "mouse": it only knows how far the head turned since the last frame, so after
a while of turning left and right the cursor can stop short of an edge (Khang, Oct 24 night). The facial
transformation matrix gives the head angles; the nose gives them again, from the picture itself, and it does not
flatten at big turns. The pointer uses the nose only in the outer part of a turn (pointer.py, "head-nose"), to
carry the cursor the last stretch to the edge the head is pointing at. Everything in the middle is unchanged.

How: across the face, the nose tip's offset from the middle of the cheeks, divided by the face width, is about
K * sin(yaw); up and down, its offset from the middle of forehead and chin, divided by the face height, is about
K * sin(pitch). K depends on the face and the camera, so it is learned while you turn a little (5 degrees or more
by the matrix): the median of (ratio change) / sin(matrix angle). Until it has 20 samples, a typical value is used.
The ratios are measured against the same neutral as the head pose (set_neutral, at the calibration or recenter).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from engine.core.types import FaceFrame, HeadPose
from engine.eyes.smoothing import OneEuroFilter

NOSE_TIP, CHEEK_A, CHEEK_B, FOREHEAD, CHIN = 1, 234, 454, 10, 152

K_YAW_DEFAULT = 0.55      # typical: nose offset / half... see the module doc; replaced by the learned median
K_PITCH_DEFAULT = 0.40
K_RANGE = (0.2, 1.2)      # a learned K outside this is a bad sample (the angle was tiny or the tracker glitched)
LEARN_MIN_DEG = (5.0, 4.0)   # yaw, pitch: below this the angle is too small to tell K from noise
LEARN_SAMPLES = 20        # samples before the learned K is used
LEARN_KEEP = 300          # recent samples kept (the median of these)
MAX_ANGLE_DEG = 50.0      # beyond this the nose angle is clamped (the face is nearly in profile)
MIN_FACE_PX = 0.05        # a face narrower or shorter than this fraction of the picture isn't measured
FILTER_CUTOFF = 0.8       # One Euro on the nose angles (Hz): the pointer only needs the slow, outer part of a turn
FILTER_BETA = 0.2


@dataclass
class NoseAngles:
    t_ms: int
    yaw: float            # degrees, > 0 = turned to the subject's right (same sign as HeadPose)
    pitch: float          # > 0 = chin down
    learned: bool         # the scale was learned from this face (else a typical value)


class NoseTracker:
    def __init__(self) -> None:
        self.neutral = (0.0, 0.0)               # ratios at the neutral pose
        self._last_ratio: tuple[float, float] | None = None
        self._k = (deque(maxlen=LEARN_KEEP), deque(maxlen=LEARN_KEEP))    # learned K samples (yaw, pitch)
        self._fy = OneEuroFilter(FILTER_CUTOFF, FILTER_BETA)
        self._fp = OneEuroFilter(FILTER_CUTOFF, FILTER_BETA)
        self._recent: deque[tuple[float, float]] = deque(maxlen=15)
        self.last: NoseAngles | None = None

    # ---------------------------------------------------------------- ratios
    @staticmethod
    def ratios(landmarks: np.ndarray) -> tuple[float, float] | None:
        """(across, down) ratios of the nose tip in the face, or None if the face is too small or flat."""
        lm = np.asarray(landmarks)
        if lm.shape[0] <= max(CHEEK_A, CHEEK_B, CHIN):
            return None
        nx, ny = float(lm[NOSE_TIP][0]), float(lm[NOSE_TIP][1])
        width = float(lm[CHEEK_B][0] - lm[CHEEK_A][0])
        height = float(lm[CHIN][1] - lm[FOREHEAD][1])
        if abs(width) < MIN_FACE_PX or abs(height) < MIN_FACE_PX:
            return None
        across = (nx - (float(lm[CHEEK_A][0]) + float(lm[CHEEK_B][0])) / 2) / width
        down = (ny - (float(lm[FOREHEAD][1]) + float(lm[CHIN][1])) / 2) / height
        return across, down

    # ---------------------------------------------------------------- neutral
    def set_neutral(self) -> None:
        """Call together with HeadPoseEstimator.set_neutral: the median of the last frames' ratios."""
        if self._recent:
            self.neutral = (float(np.median([r[0] for r in self._recent])),
                            float(np.median([r[1] for r in self._recent])))
            self._fy.reset()
            self._fp.reset()

    def k(self, axis: int) -> float:
        samples = self._k[axis]
        if len(samples) >= LEARN_SAMPLES:
            return float(np.median(samples))
        return (K_YAW_DEFAULT, K_PITCH_DEFAULT)[axis]

    @property
    def learned(self) -> bool:
        return len(self._k[0]) >= LEARN_SAMPLES

    # ---------------------------------------------------------------- per frame
    def update(self, face: FaceFrame, pose: HeadPose) -> NoseAngles | None:
        r = self.ratios(face.landmarks)
        if r is None:
            return None
        self._recent.append(r)
        da, dd = r[0] - self.neutral[0], r[1] - self.neutral[1]
        # learn K from the matrix angles while the head is turned enough to tell it from noise.
        # Across: turning to the subject's right (yaw > 0) moves the nose toward image left, so ratio falls.
        s_yaw, s_pitch = math.sin(math.radians(pose.yaw)), math.sin(math.radians(pose.pitch))
        if abs(pose.yaw) >= LEARN_MIN_DEG[0]:
            k = -da / s_yaw
            if K_RANGE[0] <= k <= K_RANGE[1]:
                self._k[0].append(k)
        if abs(pose.pitch) >= LEARN_MIN_DEG[1]:
            k = dd / s_pitch
            if K_RANGE[0] <= k <= K_RANGE[1]:
                self._k[1].append(k)
        yaw = -math.degrees(math.asin(min(max(da / self.k(0), -0.999), 0.999)))
        pitch = math.degrees(math.asin(min(max(dd / self.k(1), -0.999), 0.999)))
        yaw = min(max(yaw, -MAX_ANGLE_DEG), MAX_ANGLE_DEG)
        pitch = min(max(pitch, -MAX_ANGLE_DEG), MAX_ANGLE_DEG)
        t = face.t_ms / 1000.0
        self.last = NoseAngles(face.t_ms, float(self._fy(t, yaw)), float(self._fp(t, pitch)), self.learned)
        return self.last
