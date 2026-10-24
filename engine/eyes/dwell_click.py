"""Dwell = left click: rest the cursor on a spot for dwell_ms and it clicks there, once.

Route: Khang and Hung (Hung leads)
Publishes: nothing (the eyes service turns the click point into topics.DWELL_DONE and dwell progress)
Subscribes: nothing; the eyes service feeds it the cursor every frame
APIs: plain Python

A stop starts where the cursor slows down. Its anchor is the average of its first ANCHOR_SETTLE_MS, so
the last bit of the move doesn't count as where it stopped. Moving more than radius_px from the anchor
starts a new stop. After dwell_ms in one stop, it clicks at the median of every position in the stop,
the middle of where the cursor rested rather than wherever the last frame was.

Midas touch (a desktop is full of things that click): every click, of any kind, disarms the dwell
until the cursor has moved REARM_PX away from that spot. So a parked cursor clicks once, not every
second, and the menu that a right click opens under the cursor doesn't get clicked 1 s later. The eyes
service also turns it off, and disarms it at the cursor when it comes back on, whenever a click there
would be wrong (other modes, a question or preview waiting, UI-Venus searching, a raise in progress).
"""
from __future__ import annotations

import math
import statistics

DWELL_MS = 1000
DWELL_RADIUS_PX = 30.0    # the head pointer is steady; 60 px was for gaze
REARM_PX = 45.0           # after a click, move this far from it before the next dwell can start
ANCHOR_SETTLE_MS = 150
MAX_GAP_MS = 400          # no cursor for longer (face lost): the stop starts over


class DwellClicker:
    def __init__(self, radius_px: float = DWELL_RADIUS_PX, dwell_ms: int = DWELL_MS) -> None:
        self.radius_px = radius_px
        self.dwell_ms = dwell_ms
        self.enabled = True
        self.progress = 0.0     # 0 to 1 in the current stop, for the ring around the cursor
        self._stop: list[tuple[int, float, float]] = []
        self._anchor: tuple[float, float] | None = None
        self._disarmed_at: tuple[float, float] | None = None

    @property
    def armed(self) -> bool:
        return self.enabled and self._disarmed_at is None

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self.reset()

    def reset(self) -> None:
        """Forget the current stop (progress back to 0). The disarm stays."""
        self._stop.clear()
        self._anchor = None
        self.progress = 0.0

    def disarm(self, x: float, y: float) -> None:
        """Nothing clicks until the cursor has moved REARM_PX away from (x, y)."""
        self._disarmed_at = (x, y)
        self.reset()

    def update(self, t_ms: int, x: float, y: float) -> tuple[float, float] | None:
        """Feed every cursor position; returns the click point on the frame the dwell completes, else None."""
        if not self.enabled:
            self.reset()
            return None
        if self._disarmed_at is not None:
            if math.hypot(x - self._disarmed_at[0], y - self._disarmed_at[1]) <= REARM_PX:
                return None
            self._disarmed_at = None
        if self._stop and t_ms - self._stop[-1][0] > MAX_GAP_MS:
            self.reset()
        if self._anchor is None or math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.radius_px:
            self._stop = [(t_ms, x, y)]
            self._anchor = (x, y)
            self.progress = 0.0
            return None
        self._stop.append((t_ms, x, y))
        t0 = self._stop[0][0]
        if t_ms - t0 < ANCHOR_SETTLE_MS:
            self._anchor = (statistics.fmean(p[1] for p in self._stop), statistics.fmean(p[2] for p in self._stop))
        self.progress = min(1.0, (t_ms - t0) / self.dwell_ms)
        if self.progress < 1.0:
            return None
        at = (statistics.median(p[1] for p in self._stop), statistics.median(p[2] for p in self._stop))
        self.disarm(*at)
        return at
