"""A long, deliberate blink = click (the same as an eyebrow raise).

Route: Khang and Hung (Khang leads)
Publishes: Gesture (type "blink"), through the eyes service
Subscribes: FaceFrame.blendshapes
APIs: MediaPipe blendshapes eyeBlinkLeft, eyeBlinkRight

Both eyes must be closed (each eyeBlink score at or above CLOSE_ABOVE) for MIN_CLOSED_MS to MAX_CLOSED_MS.
It fires when the eyes open again. That shape tells it apart from:
  a normal blink   0.1 to 0.3 s, so shorter than MIN_CLOSED_MS
  resting the eyes longer than MAX_CLOSED_MS, so it never fires
  winking / squinting  one eye only: both must be closed (the lower of the two scores counts)
It does not move the head, which is why it is a better click than the eyebrow raise for the head pointer.
`held` turns true once the eyes have been closed FREEZE_AFTER_MS, so the eyes service can freeze the cursor
(closing the eyes can tip the head a little) and click where the cursor was when they closed.
A frame with no face, or a gap over MAX_GAP_MS, drops a close in progress, so two halves never make a click.
"""
from __future__ import annotations

from engine.core.types import Gesture

CLOSE_ABOVE = 0.55        # both eyeBlink scores this high = closed (open is about 0.05 to 0.25, shut about 0.8)
OPEN_BELOW = 0.35         # and back under this = open again (a gap between the two stops flicker)
MIN_CLOSED_MS = 400       # a deliberate close; a normal blink is shorter...
MAX_CLOSED_MS = 1200      # ...and anything longer is the user resting their eyes
FREEZE_AFTER_MS = 150     # closed this long: hold the cursor still
COOLDOWN_MS = 800
MAX_GAP_MS = 500          # a webcam can slow to 8 fps in dim light; more than this drops the close


class BlinkDetector:
    def __init__(self, min_closed_ms: int = MIN_CLOSED_MS, max_closed_ms: int = MAX_CLOSED_MS,
                 close_above: float = CLOSE_ABOVE) -> None:
        self.min_closed_ms = min_closed_ms
        self.max_closed_ms = max_closed_ms
        self.close_above = close_above
        self._closed_since: int | None = None
        self._last_t: int | None = None
        self._quiet_until = 0
        self.score: float | None = None      # the lower of the two eyes, for the live view
        self.closed_ms = 0                   # how long the current close has lasted

    @property
    def held(self) -> bool:
        """Closed long enough to be a click in the making, not too long to be one."""
        return self._closed_since is not None and FREEZE_AFTER_MS <= self.closed_ms <= self.max_closed_ms

    def reset(self) -> None:
        self._drop()
        self._last_t = None

    def _drop(self) -> None:
        self._closed_since = None
        self.closed_ms = 0

    def update(self, t_ms: int, blendshapes: dict[str, float]) -> Gesture | None:
        bs = blendshapes or {}
        left, right = bs.get("eyeBlinkLeft"), bs.get("eyeBlinkRight")
        if left is None or right is None:
            self._drop()
            return None
        if self._last_t is not None and t_ms - self._last_t > MAX_GAP_MS:
            self._drop()
        self._last_t = t_ms
        score = min(float(left), float(right))   # both eyes: a wink or a squint never counts
        self.score = score
        if self._closed_since is None:
            if score >= self.close_above and t_ms >= self._quiet_until:
                self._closed_since = t_ms
                self.closed_ms = 0
            return None
        self.closed_ms = t_ms - self._closed_since
        if score > OPEN_BELOW:
            return None                           # still closed (resting eyes keep going past max_closed_ms)
        lasted = self.closed_ms
        self._drop()
        if self.min_closed_ms <= lasted <= self.max_closed_ms:
            self._quiet_until = t_ms + COOLDOWN_MS
            return Gesture(t_ms=t_ms, type="blink")
        return None
