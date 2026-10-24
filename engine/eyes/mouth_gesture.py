"""Open the mouth and close it again = a gesture (the right click in Pointer mode).

Route: Khang and Hung (Khang leads)
Publishes: Gesture (type "mouth_open"), through the eyes service
Subscribes: FaceFrame.blendshapes
APIs: MediaPipe blendshape jawOpen

It fires when the mouth closes after being held open (jawOpen OPEN_ABOVE over the resting level) for
MIN_OPEN_MS to MAX_OPEN_MS, without closing in between. That shape tells it apart from:
  talking   each syllable opens the jaw for well under MIN_OPEN_MS, and between syllables it is
            at most half open; a held open stays wide for WIDE_SHARE of the time
  a yawn    stays open for seconds: longer than MAX_OPEN_MS, so it never fires
Firing on the close, not the open, is what lets a yawn be told apart. A long sung vowel can still pass;
the mode manager ignores the gesture while the mic hears speech.
`held` turns true once the mouth has been open FREEZE_AFTER_MS, so the eyes service can freeze the
cursor (opening the jaw tips the head a little) and click where the cursor was when it opened.
The resting level is a low percentile of the last few seconds, so a mouth that rests a little open works.
A frame with no face, or a gap over MAX_GAP_MS, drops an open in progress.
"""
from __future__ import annotations

from collections import deque

from engine.core.types import Gesture

OPEN_ABOVE = 0.35         # jawOpen this far over the resting level counts as open (0 closed to about 1)
CLOSE_BELOW = 0.15        # and back under rest + this counts as closed again
MIN_OPEN_MS = 300         # open at least this long (a syllable is shorter)...
MAX_OPEN_MS = 1500        # ...and at most this long (a yawn is longer)
WIDE_SHARE = 0.7          # and wide (over rest + WIDE_FRACTION * OPEN_ABOVE) for this share of it: fast talk
WIDE_FRACTION = 0.75      # can hover half open for a while, but not wide open
FREEZE_AFTER_MS = 150     # open this long: hold the cursor still
COOLDOWN_MS = 800
MAX_GAP_MS = 400
REST_FRAMES = 150         # about 5 s
REST_MIN_FRAMES = 15      # don't fire until the resting level is known
REST_PERCENTILE = 0.2


class MouthOpenDetector:
    def __init__(self, open_above: float = OPEN_ABOVE, min_open_ms: int = MIN_OPEN_MS,
                 max_open_ms: int = MAX_OPEN_MS) -> None:
        self.open_above = open_above
        self.min_open_ms = min_open_ms
        self.max_open_ms = max_open_ms
        self._levels: deque[float] = deque(maxlen=REST_FRAMES)
        self._open_since: int | None = None
        self._frames = self._wide = 0     # frames in this open, and how many of them wide
        self._last_t: int | None = None
        self._quiet_until = 0
        self.rest: float | None = None
        self.jaw: float | None = None     # latest jawOpen, for the live view
        self.open_ms = 0                  # how long the current open has lasted

    @property
    def held(self) -> bool:
        """Open long enough to be a gesture in the making, not too long to be one."""
        return self._open_since is not None and FREEZE_AFTER_MS <= self.open_ms <= self.max_open_ms

    def reset(self) -> None:
        self._levels.clear()
        self.rest = None
        self._drop()

    def _drop(self) -> None:
        self._open_since = None
        self.open_ms = 0
        self._frames = self._wide = 0

    def update(self, t_ms: int, blendshapes: dict[str, float]) -> Gesture | None:
        jaw = (blendshapes or {}).get("jawOpen")
        if jaw is None:
            self._drop()
            return None
        if self._last_t is not None and t_ms - self._last_t > MAX_GAP_MS:
            self._drop()
        self._last_t = t_ms
        jaw = float(jaw)
        self.jaw = jaw
        self._levels.append(jaw)
        ordered = sorted(self._levels)
        self.rest = ordered[int(REST_PERCENTILE * (len(ordered) - 1))]
        if len(self._levels) < REST_MIN_FRAMES:
            return None
        if self._open_since is None:
            if jaw >= self.rest + self.open_above and t_ms >= self._quiet_until:
                self._open_since = t_ms
                self._frames = self._wide = 1
            return None
        self.open_ms = t_ms - self._open_since
        if jaw > self.rest + CLOSE_BELOW:
            self._frames += 1
            self._wide += jaw >= self.rest + WIDE_FRACTION * self.open_above
            return None                   # still open (a yawn keeps going here, past max_open_ms)
        lasted, wide = self.open_ms, self._wide >= WIDE_SHARE * max(self._frames, 1)
        self._drop()
        if wide and self.min_open_ms <= lasted <= self.max_open_ms:
            self._quiet_until = t_ms + COOLDOWN_MS
            return Gesture(t_ms=t_ms, type="mouth_open")
        return None
