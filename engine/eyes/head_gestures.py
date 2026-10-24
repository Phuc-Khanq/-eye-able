"""Nod = quick pitch dip and return; shake = yaw back and forth; tilts. Per-user thresholds.

Route: Khang and Hung (Hung leads)
Publishes: Gesture
Subscribes: HeadPose
APIs: NumPy over the last second of head angles

A nod or shake is judged against where the head was when the motion started, so slow drift
and a user who sits turned don't matter. Only quick, deliberate motions count:
  nod   = pitch leaves the start by the nod threshold (down, or up for users who nod that way) and comes
          back within NOD_WINDOW_MS, while yaw stays fairly still
  shake = yaw swings past the shake threshold on both sides of the start within SHAKE_WINDOW_MS,
          while pitch stays fairly still
  tilt  = roll held past the tilt threshold for TILT_HOLD_MS (tilt_right = toward the right shoulder)
After any gesture the detector stays quiet for COOLDOWN_MS so one motion fires once.
"""
from collections import deque

import numpy as np

from engine.core.types import Gesture, HeadPose

# Defaults; each user tunes them with tests/head_bars.py and the console settings panel.
NOD_DEG = 8.0             # pitch dip that counts as a nod
SHAKE_DEG = 8.0           # yaw swing each way that counts toward a shake
TILT_DEG = 15.0           # roll held this far counts as a tilt
TILT_HOLD_MS = 400
COOLDOWN_MS = 800         # ignore new gestures right after one

NOD_WINDOW_MS = 800       # a nod longer than this is just looking down
NOD_MIN_MS = 120          # shorter than this is tracking noise
SHAKE_WINDOW_MS = 1200
HISTORY_MS = 1200
RETURN_FRACTION = 0.4     # "came back" = within this share of the threshold from the start
CROSS_AXIS_FRACTION = 0.8  # the other axis may move at most this share of the threshold


class GestureDetector:
    def __init__(self, nod_deg: float | None = None, shake_deg: float | None = None,
                 tilt_deg: float | None = None, tilt_hold_ms: int | None = None,
                 cooldown_ms: int | None = None) -> None:
        self.nod_deg = NOD_DEG if nod_deg is None else nod_deg
        self.shake_deg = SHAKE_DEG if shake_deg is None else shake_deg
        self.tilt_deg = TILT_DEG if tilt_deg is None else tilt_deg
        self.tilt_hold_ms = TILT_HOLD_MS if tilt_hold_ms is None else tilt_hold_ms
        self.cooldown_ms = COOLDOWN_MS if cooldown_ms is None else cooldown_ms
        self.history: deque[HeadPose] = deque()
        self.quiet_until_ms = -1
        self.tilt_since_ms: int | None = None
        self.tilt_side = ""

    def set_thresholds(self, nod_deg: float | None = None, shake_deg: float | None = None,
                       tilt_deg: float | None = None) -> None:
        """For the console's settings panel and per-user tuning."""
        if nod_deg is not None:
            self.nod_deg = nod_deg
        if shake_deg is not None:
            self.shake_deg = shake_deg
        if tilt_deg is not None:
            self.tilt_deg = tilt_deg

    def reset(self) -> None:
        self.history.clear()
        self.tilt_since_ms = None
        self.tilt_side = ""

    def update(self, pose: HeadPose) -> Gesture | None:
        """Feed every HeadPose; returns a Gesture on the frame it completes, else None."""
        self.history.append(pose)
        while self.history and pose.t_ms - self.history[0].t_ms > HISTORY_MS:
            self.history.popleft()
        if pose.t_ms < self.quiet_until_ms:
            return None

        gesture = self._tilt(pose) or self._nod() or self._shake()
        if gesture is not None:
            self.quiet_until_ms = pose.t_ms + self.cooldown_ms
            self.reset()
        return gesture

    def _window(self, ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now = self.history[-1].t_ms
        recent = [p for p in self.history if now - p.t_ms <= ms]
        t = np.array([p.t_ms for p in recent], dtype=np.float64)
        yaw = np.array([p.yaw for p in recent], dtype=np.float64)
        pitch = np.array([p.pitch for p in recent], dtype=np.float64)
        roll = np.array([p.roll for p in recent], dtype=np.float64)
        return t, yaw, pitch, roll

    def _nod(self) -> Gesture | None:
        t, yaw, pitch, _ = self._window(NOD_WINDOW_MS)
        if len(t) < 4:
            return None
        start = pitch[0]
        offset = pitch - start
        peak = int(np.argmax(np.abs(offset)))
        if abs(offset[peak]) < self.nod_deg or peak in (0, len(t) - 1):
            return None
        if abs(offset[-1]) > RETURN_FRACTION * self.nod_deg:
            return None  # hasn't come back yet
        if t[-1] - t[0] < NOD_MIN_MS:
            return None
        if np.ptp(yaw) > CROSS_AXIS_FRACTION * max(self.nod_deg, abs(offset[peak])):
            return None  # that was a look around, not a nod
        return Gesture(t_ms=int(t[-1]), type="nod")

    def _shake(self) -> Gesture | None:
        t, yaw, pitch, _ = self._window(SHAKE_WINDOW_MS)
        if len(t) < 6:
            return None
        offset = yaw - yaw[0]
        # Count swings: each time yaw passes the threshold on the opposite side from the last one.
        sides = []
        for v in offset:
            side = 1 if v > self.shake_deg else -1 if v < -self.shake_deg else 0
            if side and (not sides or sides[-1] != side):
                sides.append(side)
        if len(sides) < 2:
            return None
        if abs(offset[-1]) > RETURN_FRACTION * self.shake_deg * 2:
            return None  # wait until the head is back near the middle
        if np.ptp(pitch) > CROSS_AXIS_FRACTION * 2 * self.shake_deg:
            return None
        return Gesture(t_ms=int(t[-1]), type="shake")

    def _tilt(self, pose: HeadPose) -> Gesture | None:
        side = "tilt_right" if pose.roll > self.tilt_deg else "tilt_left" if pose.roll < -self.tilt_deg else ""
        if not side:
            self.tilt_since_ms = None
            self.tilt_side = ""
            return None
        if side != self.tilt_side:
            self.tilt_side = side
            self.tilt_since_ms = pose.t_ms
            return None
        if pose.t_ms - self.tilt_since_ms >= self.tilt_hold_ms:
            return Gesture(t_ms=pose.t_ms, type=side)
        return None
