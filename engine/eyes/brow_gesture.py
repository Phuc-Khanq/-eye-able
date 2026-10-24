"""Eyebrow raise = yes / click. Replaces the nod as the click signal, since a nod moves the cursor.

Route: Khang and Hung (Hung leads)
Publishes: Gesture (type "brow_raise")
Subscribes: FaceFrame.blendshapes
APIs: MediaPipe blendshapes browInnerUp, browOuterUpLeft, browOuterUpRight, eyeBlinkLeft/Right

score = max(browInnerUp, mean of the two outer scores), through a 3-frame median.
rest  = running median of the score while the brows are down, so naturally high brows still work.
A raise fires once when the score stays above rest + threshold for hold_ms. It can fire again only
after the score drops under rest + 0.7 * threshold and the cooldown has passed, so a long hold is one
click (same idea as the held tilt). Blink frames are skipped, because a blink pulls the brows around.
A frame with no face, or a gap of more than MAX_GAP_MS, drops any raise in progress, so two halves never
make a click. The resting level is kept, unless the face was gone for REST_FORGET_MS (maybe a new person).
MAX_GAP_MS allows a few dropped frames even on a webcam that slows to 8 fps in dim light.
"""
from collections import deque

import numpy as np

from engine.core.types import Gesture

# Defaults; each user tunes them with tests/head_bars.py and the console settings panel.
BROW_THRESHOLD = 0.25     # score above the resting level that counts as raised (scores are 0 to 1)
BROW_HOLD_MS = 250        # held at least this long
COOLDOWN_MS = 800
RELEASE_FRACTION = 0.7    # raised ends when the score falls under rest + this share of the threshold
REST_GATE_FRACTION = 0.5  # only scores under rest + this share of the threshold update the resting level
REST_FRAMES = 90          # about 3 seconds of brows-down frames
REST_MIN_FRAMES = 10      # don't fire until the resting level is known
BLINK_MAX = 0.5           # eyeBlinkLeft/Right above this = blink frame, skipped
MAX_GAP_MS = 400
REST_FORGET_MS = 3000
MEDIAN_FRAMES = 3


def brow_score(blendshapes: dict[str, float]) -> float | None:
    """The raise score for one frame, or None when MediaPipe gave no brow scores."""
    inner = blendshapes.get("browInnerUp")
    left, right = blendshapes.get("browOuterUpLeft"), blendshapes.get("browOuterUpRight")
    if inner is None or left is None or right is None:
        return None
    return max(float(inner), (float(left) + float(right)) / 2)


class BrowRaiseDetector:
    def __init__(self, threshold: float = BROW_THRESHOLD, hold_ms: int = BROW_HOLD_MS,
                 cooldown_ms: int = COOLDOWN_MS):
        self.threshold = threshold
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.quiet_until_ms = 0
        self._rest_scores: deque[float] = deque(maxlen=REST_FRAMES)
        self._recent: deque[float] = deque(maxlen=MEDIAN_FRAMES)
        self._last_t: int | None = None
        self._above_since: int | None = None
        self._latched = False
        self.score: float | None = None  # latest smoothed score, for the live view
        self.rest: float | None = None   # resting level, None until known

    def set_thresholds(self, threshold: float | None = None, hold_ms: int | None = None) -> None:
        """For the console's settings panel and per-user tuning."""
        if threshold is not None:
            self.threshold = threshold
        if hold_ms is not None:
            self.hold_ms = hold_ms

    def reset(self) -> None:
        """Forget everything, including the resting level."""
        self._rest_scores.clear()
        self._last_t = None
        self.rest = None
        self._drop_raise()

    def _drop_raise(self) -> None:
        self._recent.clear()
        self._above_since = None
        self._latched = False
        self.score = None

    @property
    def raised(self) -> bool:
        """Above the threshold now: holding toward a raise, or already fired and not yet released."""
        return self._above_since is not None or self._latched

    def update(self, t_ms: int, blendshapes: dict[str, float]) -> Gesture | None:
        """Feed every frame's blendshapes; returns a Gesture on the frame a raise completes, else None."""
        raw = brow_score(blendshapes) if blendshapes else None
        if raw is None:
            self._drop_raise()   # no face this frame
            return None
        if self._last_t is not None and t_ms - self._last_t > REST_FORGET_MS:
            self.reset()
        elif self._last_t is not None and t_ms - self._last_t > MAX_GAP_MS:
            self._drop_raise()
        self._last_t = t_ms
        blink = max(blendshapes.get("eyeBlinkLeft", 0.0), blendshapes.get("eyeBlinkRight", 0.0))
        if blink > BLINK_MAX:
            return None   # skipped: neither starts, breaks nor finishes a raise

        self._recent.append(raw)
        s = float(np.median(self._recent))
        self.score = s
        if len(self._rest_scores) < REST_MIN_FRAMES:
            self._rest_scores.append(s)   # learning the resting level; assume the brows are down
            self.rest = float(np.median(self._rest_scores))
            return None
        if s < self.rest + REST_GATE_FRACTION * self.threshold:
            self._rest_scores.append(s)
            self.rest = float(np.median(self._rest_scores))

        if self._latched:
            if s < self.rest + RELEASE_FRACTION * self.threshold:
                self._latched = False
            return None
        if s < self.rest + RELEASE_FRACTION * self.threshold:
            self._above_since = None
            return None
        if s < self.rest + self.threshold:
            return None   # in between: keep the hold running if it started, don't start one
        if self._above_since is None:
            self._above_since = t_ms
        if t_ms - self._above_since >= self.hold_ms and t_ms >= self.quiet_until_ms:
            self._above_since = None
            self._latched = True
            self.quiet_until_ms = t_ms + self.cooldown_ms
            return Gesture(t_ms=t_ms, type="brow_raise")
        return None
