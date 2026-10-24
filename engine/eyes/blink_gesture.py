"""A long, deliberate blink = click (the same as an eyebrow raise).

Route: Khang and Hung (Khang leads)
Publishes: Gesture (type "blink"), through the eyes service
Subscribes: FaceFrame.blendshapes, HeadPose.pitch
APIs: MediaPipe blendshapes eyeBlinkLeft, eyeBlinkRight

Both eyes must close (the lower of the two eyeBlink scores counts, so a wink or one squinting eye never does;
left_gain / right_gain scale a weak eye's score first)
for MIN_CLOSED_MS to MAX_CLOSED_MS, and it fires when they open again. A normal blink is shorter, and resting
the eyes is longer.

Relative, not fixed, thresholds. Looking at the lower half of the screen tips the head down and the upper lids
drop, so the "open" score rises (about 0.1 looking ahead, 0.4 or more looking down). With fixed thresholds a
blink while looking down never counted as "open again" and never fired (Khang's live test, Oct 24). So:
  open level   learned all the time from frames with the eyes open, per head-pitch band (PITCH_BIN_DEG): looking
               down has its own open level, ready the moment the head goes there
  closed       the score rises RISE above the open level (and over START_ABS)
  open again   the score falls back most of the way (RELEASE) from its peak toward the open level it started from,
               or RELEASE_DROP under its peak
  not a close  the score rose but never really shut within PLATEAU_MS (lids dropping as you look down), or the
               head moved more than MAX_PITCH_SWING: dropped at once, and those frames teach the open level
  a click      it lasted MIN_CLOSED_MS to MAX_CLOSED_MS, went at least CONFIRM_RISE over the open level and over
               CLOSED_ABS (a squint or droopy lids that never really close don't count), and the head stayed
               fairly still (a glance down with the head moving is the head pointer, not a blink)
`held` (the eyes service freezes the cursor) needs the eyes really shut (over CLOSED_ABS) and the head still, so
moving the cursor down never freezes it.
A frame with no face, or a gap over MAX_GAP_MS, drops a close in progress, so two halves never make a click.
tests/blink_record.py records your own scores at the top, middle and bottom of the screen and
tests/blink_replay.py replays them through this detector and suggests values.
"""
from __future__ import annotations

from collections import deque

from engine.core.types import Gesture

RISE = 0.20               # this far over the open level starts a close
START_ABS = 0.35          # ...and at least this high
CONFIRM_RISE = 0.30       # a click went at least this far over the open level
CLOSED_ABS = 0.55         # ...and at least this high: really shut
RELEASE = 0.45            # open again once back under open + RELEASE * (peak - open)...
RELEASE_DROP = 0.28       # ...or RELEASE_DROP under the peak, whichever comes first (looking down, the open
                          # level can be learned late, and the eyes must still count as open again)
PLATEAU_MS = 300          # a "close" that hasn't really shut (over CLOSED_ABS) by then was a glance down: dropped
MIN_CLOSED_MS = 400       # a deliberate close; a normal blink is shorter...
MAX_CLOSED_MS = 1200      # ...and anything longer is the user resting their eyes
MAX_PITCH_SWING = 10.0    # degrees the head may move up or down during a close
FREEZE_PITCH_SWING = 5.0  # ...and still enough for the cursor freeze
FREEZE_AFTER_MS = 150     # shut this long: hold the cursor still
COOLDOWN_MS = 800
MAX_GAP_MS = 500          # a webcam can slow to 8 fps in dim light; more than this drops the close
PITCH_BIN_DEG = 4.0       # open levels are kept per band of head pitch this wide
BIN_FRAMES = 60           # recent open frames kept per band (about 2 s of looking there)
BIN_MIN = 8               # a band needs this many before its own level is used
RECENT_FRAMES = 30        # the fallback: the last second of open frames, any pitch
OPEN_PERCENTILE = 0.25
DEFAULT_OPEN = 0.10
CLOSE_ABOVE = START_ABS   # the old name, for scripts that print it

# old fixed thresholds (before Oct 24 PM), kept for blink_replay.py's comparison
OLD_CLOSE_ABOVE = 0.55
OLD_OPEN_BELOW = 0.35


def _pct(values, q: float) -> float:
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


class BlinkDetector:
    def __init__(self, min_closed_ms: int = MIN_CLOSED_MS, max_closed_ms: int = MAX_CLOSED_MS,
                 rise: float = RISE, closed_abs: float = CLOSED_ABS,
                 left_gain: float = 1.0, right_gain: float = 1.0) -> None:
        self.left_gain = max(float(left_gain), 0.1)    # a weak eye's score is scaled up before the two are compared
        self.right_gain = max(float(right_gain), 0.1)
        self.min_closed_ms = min_closed_ms
        self.max_closed_ms = max_closed_ms
        self.rise = rise
        self.closed_abs = closed_abs
        self._bins: dict[int, deque] = {}
        self._recent: deque[float] = deque(maxlen=RECENT_FRAMES)
        self._last_t: int | None = None
        self._quiet_until = 0
        self._drop()
        self.score: float | None = None      # the lower of the two eyes, for the live view
        self.open_level = DEFAULT_OPEN       # the open level for the current head pitch, for the live view

    @property
    def held(self) -> bool:
        """Shut long enough to be a click in the making (and the head still), not too long to be one."""
        return (self._since is not None and self.score is not None and self.score >= self.closed_abs
                and FREEZE_AFTER_MS <= self.closed_ms <= self.max_closed_ms
                and self._swing() <= FREEZE_PITCH_SWING)

    def reset(self) -> None:
        self._drop()
        self._last_t = None

    def _drop(self) -> None:
        self._since: int | None = None
        self.closed_ms = 0
        self._ref = DEFAULT_OPEN
        self._peak = 0.0
        self._pitch_lo = self._pitch_hi = None
        self._frames: list[tuple[float, float | None]] = []   # (score, pitch) during this close

    def _swing(self) -> float:
        return 0.0 if self._pitch_lo is None else self._pitch_hi - self._pitch_lo

    def _open_level(self, pitch: float | None) -> float:
        if pitch is not None:
            b = int(pitch // PITCH_BIN_DEG)
            near = [s for k in (b - 1, b, b + 1) for s in self._bins.get(k, ())]
            if len(self._bins.get(b, ())) >= BIN_MIN:
                return _pct(near, OPEN_PERCENTILE)
        if len(self._recent) >= BIN_MIN:
            return _pct(self._recent, OPEN_PERCENTILE)
        return DEFAULT_OPEN

    def _learn(self, score: float, pitch: float | None) -> None:
        self._recent.append(score)
        if pitch is not None:
            b = int(pitch // PITCH_BIN_DEG)
            self._bins.setdefault(b, deque(maxlen=BIN_FRAMES)).append(score)

    def update(self, t_ms: int, blendshapes: dict[str, float], pitch: float | None = None) -> Gesture | None:
        bs = blendshapes or {}
        left, right = bs.get("eyeBlinkLeft"), bs.get("eyeBlinkRight")
        if left is None or right is None:
            self._drop()
            return None
        if self._last_t is not None and t_ms - self._last_t > MAX_GAP_MS:
            self._drop()
        self._last_t = t_ms
        # both eyes: a wink or a squint never counts. Each eye's score is scaled by its own gain first (capped
        # at 1), so a weak eye that only reaches 0.5 when shut can still count; an eye that is open stays low.
        score = min(min(float(left) * self.left_gain, 1.0), min(float(right) * self.right_gain, 1.0))
        self.score = score
        open_level = self._open_level(pitch)
        self.open_level = open_level
        if self._since is None:
            if score >= max(open_level + self.rise, START_ABS) and t_ms >= self._quiet_until:
                self._since, self._ref, self._peak = t_ms, open_level, score
                self.closed_ms = 0
                self._frames = [(score, pitch)]
                if pitch is not None:
                    self._pitch_lo = self._pitch_hi = pitch
            else:
                self._learn(score, pitch)        # an open frame: it teaches the open level
            return None
        self.closed_ms = t_ms - self._since
        self._peak = max(self._peak, score)
        self._frames.append((score, pitch))
        if pitch is not None:
            self._pitch_lo = pitch if self._pitch_lo is None else min(self._pitch_lo, pitch)
            self._pitch_hi = pitch if self._pitch_hi is None else max(self._pitch_hi, pitch)
        if ((self._peak < self.closed_abs and self.closed_ms >= PLATEAU_MS) or self._swing() > MAX_PITCH_SWING):
            frames = self._frames                # lids dropped as the eyes or head went down: not a close,
            self._drop()                         # and all of it teaches the open level for this pitch
            for s, p in frames:
                self._learn(s, p)
            return None
        if score > max(self._ref + RELEASE * (self._peak - self._ref), self._peak - RELEASE_DROP):
            return None                          # still shut (resting eyes keep going past max_closed_ms)
        lasted, ref, peak, swing = self.closed_ms, self._ref, self._peak, self._swing()
        self._drop()
        self._learn(score, pitch)
        if (self.min_closed_ms <= lasted <= self.max_closed_ms and peak >= max(ref + CONFIRM_RISE, self.closed_abs)
                and swing <= MAX_PITCH_SWING):
            self._quiet_until = t_ms + COOLDOWN_MS
            return Gesture(t_ms=t_ms, type="blink")
        return None
