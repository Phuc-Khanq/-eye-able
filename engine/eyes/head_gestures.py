"""Nod = quick pitch dip and return; shake = yaw back and forth; tilts. Per-user thresholds.

Route: Khang and Hung (Hung leads)
Publishes: Gesture
Subscribes: HeadPose
APIs: NumPy over the last second of head angles

Only quick, deliberate motions count, judged from the motion itself rather than from where the
head was a fixed time ago, so a slow drift or a head that settles at a new height doesn't matter:
  nod   = from a still head, pitch moves nod_deg away, fast, then comes back at least 0.6 * nod_deg
          while yaw stays fairly still. Looking down and staying, or a slow glance at a phone, is too
          slow, never comes back, or didn't start from a still head, so it isn't a nod.
  shake = yaw has two turning points at least 1.8 * shake_deg apart, each swing fast. Looking one way
          and back has only one turning point, so it isn't a shake.
  tilt  = roll held past tilt_deg for tilt_hold_ms (tilt_right = toward the right shoulder). It lets go
          only when roll drops under 70% of tilt_deg, so a noisy frame can't restart the hold.
Each pose goes through a 3-frame median first, which removes one-frame tracking glitches. A gap of
more than MAX_GAP_MS (the face was lost) clears the history, so two motions are never stitched
together. After any gesture the detector stays quiet for COOLDOWN_MS so one motion fires once.
"""
from collections import deque

import numpy as np

from engine.core.types import Gesture, HeadPose

# Defaults; each user tunes them with tests/head_bars.py and the console settings panel.
NOD_DEG = 8.0             # pitch move that counts as a nod
SHAKE_DEG = 8.0           # half of the yaw swing that counts as a shake (the swing is 2x this)
TILT_DEG = 15.0           # roll held this far counts as a tilt
TILT_HOLD_MS = 400
COOLDOWN_MS = 800         # ignore new gestures right after one

NOD_WINDOW_MS = 1000      # the whole nod, down and back, happens within this
NOD_MIN_MS = 120          # shorter than this is tracking noise
NOD_MAX_RISE_MS = 600     # the first half of the nod takes at most this long
NOD_MIN_SPEED_DPS = 25.0  # peak speed of the first half, degrees per second
SHAKE_WINDOW_MS = 1200
SHAKE_SWING_FACTOR = 1.8  # a swing is this many times shake_deg, turning point to turning point
SHAKE_MIN_SPEED_DPS = 22.0  # average speed of every swing
HISTORY_MS = 1200
MAX_GAP_MS = 150          # longer without a pose = face lost
MEDIAN_FRAMES = 3
LEAD_IN_FRAMES = 3        # a nod starts from a head that was still for at least this many frames
LEAD_IN_FRACTION = 0.5    # ...within this share of nod_deg
RETURN_FRACTION = 0.6     # a nod must come back this share of nod_deg
RETURN_OF_RISE = 0.4      # ...and this share of how far it went
CROSS_AXIS_FRACTION = 0.8  # the other axis may move at most this share of the threshold
TILT_RELEASE = 0.7        # a tilt ends when roll falls under this share of tilt_deg


def turning_points(x: np.ndarray, swing: float) -> list[tuple[int, float]]:
    """Peaks and valleys of x that differ from the previous one by at least `swing`.
    A peak is confirmed once x has fallen `swing` from it, so small wiggles don't count. The
    resting level at the start isn't a turn, so the first extreme only counts if x really moved
    to it. The extreme x is turning away from right now also counts once it has come back
    swing / 4, so a shake is seen as the head swings back, not a whole swing later."""
    x0 = float(x[0])
    hi = lo = x0
    hi_i = lo_i = 0
    direction = 0  # 1 rising, -1 falling, 0 not known yet
    out: list[tuple[int, float]] = []
    for i, v in enumerate(x):
        v = float(v)
        if v > hi:
            hi, hi_i = v, i
        if v < lo:
            lo, lo_i = v, i
        if direction >= 0 and hi - v >= swing:
            if hi_i > 0 and (direction != 0 or abs(hi - x0) >= swing / 2):
                out.append((hi_i, hi))
            direction, lo, lo_i, hi, hi_i = -1, v, i, v, i
        elif direction <= 0 and v - lo >= swing:
            if lo_i > 0 and (direction != 0 or abs(lo - x0) >= swing / 2):
                out.append((lo_i, lo))
            direction, hi, hi_i, lo, lo_i = 1, v, i, v, i
    if out:
        last = float(x[-1])
        if direction == -1 and last - lo >= swing / 4 and out[-1][1] - lo >= swing:
            out.append((lo_i, lo))
        elif direction == 1 and hi - last >= swing / 4 and hi - out[-1][1] >= swing:
            out.append((hi_i, hi))
    return out


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
        self._recent: deque[HeadPose] = deque(maxlen=MEDIAN_FRAMES)
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
        self._recent.clear()
        self.tilt_since_ms = None
        self.tilt_side = ""

    def update(self, pose: HeadPose) -> Gesture | None:
        """Feed every HeadPose; returns a Gesture on the frame it completes, else None."""
        if self.history and pose.t_ms - self.history[-1].t_ms > MAX_GAP_MS:
            self.reset()   # the face was lost; don't stitch two motions together
        self._recent.append(pose)
        self.history.append(HeadPose(
            t_ms=pose.t_ms,
            yaw=float(np.median([p.yaw for p in self._recent])),
            pitch=float(np.median([p.pitch for p in self._recent])),
            roll=float(np.median([p.roll for p in self._recent])),
        ))
        while self.history and pose.t_ms - self.history[0].t_ms > HISTORY_MS:
            self.history.popleft()
        if pose.t_ms < self.quiet_until_ms:
            return None

        gesture = self._tilt(self.history[-1]) or self._nod() or self._shake()
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
        if len(t) < 5:
            return None
        for sign in (1.0, -1.0):          # chin down first, then chin up
            p = sign * pitch
            k = int(np.argmax(p))
            if k == 0 or k == len(p) - 1:
                continue
            rise = p[k] - p[:k].min()
            fall = p[k] - p[k + 1:].min()
            if rise < self.nod_deg or fall < RETURN_FRACTION * self.nod_deg or fall < RETURN_OF_RISE * rise:
                continue
            # s = where the head left its resting level (20% of the way up), not just the lowest noisy sample
            s = int(np.nonzero(p[:k] <= p[k] - 0.8 * rise)[0][-1])
            if s < LEAD_IN_FRAMES or np.ptp(p[:s + 1]) > LEAD_IN_FRACTION * self.nod_deg:
                continue                  # no still head before it: the move began earlier (a slow glance)
            if t[k] - t[s] > NOD_MAX_RISE_MS or t[-1] - t[s] < NOD_MIN_MS:
                continue
            dt = np.maximum(np.diff(t[s:k + 1]), 1.0)
            if np.max(np.diff(p[s:k + 1]) / dt) * 1000.0 < NOD_MIN_SPEED_DPS:
                continue                  # too slow: looking down at something, not a nod
            if np.ptp(yaw[s:]) > CROSS_AXIS_FRACTION * max(self.nod_deg, rise):
                continue                  # that was a look around, not a nod
            return Gesture(t_ms=int(t[-1]), type="nod")
        return None

    def _shake(self) -> Gesture | None:
        t, yaw, pitch, _ = self._window(SHAKE_WINDOW_MS)
        if len(t) < 6:
            return None
        points = turning_points(yaw, SHAKE_SWING_FACTOR * self.shake_deg)
        if len(points) < 2:
            return None
        for (a, ya), (b, yb) in zip(points, points[1:]):
            if abs(yb - ya) / max(t[b] - t[a], 1.0) * 1000.0 < SHAKE_MIN_SPEED_DPS:
                return None               # a slow sweep across the screen, not a shake
        if np.ptp(pitch[points[0][0]:]) > CROSS_AXIS_FRACTION * 2 * self.shake_deg:
            return None
        return Gesture(t_ms=int(t[-1]), type="shake")

    def _tilt(self, pose: HeadPose) -> Gesture | None:
        side = "tilt_right" if pose.roll > self.tilt_deg else "tilt_left" if pose.roll < -self.tilt_deg else ""
        if not side and self.tilt_side:
            release = TILT_RELEASE * self.tilt_deg
            if (pose.roll > release) if self.tilt_side == "tilt_right" else (pose.roll < -release):
                side = self.tilt_side     # still holding, just a little under the bar
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
