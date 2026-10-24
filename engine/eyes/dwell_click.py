"""Dwell = left click: rest the cursor on a spot for dwell_ms and it clicks there, once.

Route: Khang and Hung (Hung leads)
Publishes: nothing (the eyes service turns the click point into topics.DWELL_DONE and dwell progress)
Subscribes: nothing; the eyes service feeds it the cursor every frame
APIs: plain Python

A stop starts where the cursor slows down. Its anchor is the average of its first ANCHOR_SETTLE_MS, and
a frame faster than SETTLE_SPEED_PXS in that time starts the stop again, so the end of the move doesn't
count as where it stopped. Moving more than radius_px from the anchor
starts a new stop. After dwell_ms in one stop, it clicks at the median of every position in the stop,
the middle of where the cursor rested rather than wherever the last frame was.

Only a stop the cursor travelled to counts: over the ARRIVE_MS before it stopped, the cursor was at
least ARRIVE_PX away for at least half the time. Reading moves the head, so a parked cursor creeps
10 to 50 px at a time and settles again; on Hung's laptop that clicked 10 times in a minute of reading.
Real dwell clicks came after a move of 100 px or more. Replaying that recording with this rule and the
40 px radius: 0 reading clicks (was 10), and 9 dwell clicks in the dwell test (was 7). Check a new log
with: python -m tests.brow_click --replay calibration\brow_logs\<file>.csv

Midas touch (a desktop is full of things that click): every click, of any kind, disarms the dwell
until the cursor has moved REARM_PX away from that spot. So a parked cursor clicks once, not every
second, and the menu that a right click opens under the cursor doesn't get clicked 1 s later. The eyes
service also turns it off, and disarms it at the cursor when it comes back on, whenever a click there
would be wrong (other modes, a question or preview waiting, UI-Venus searching, a raise in progress).
"""
from __future__ import annotations

import math
import statistics
from collections import deque

DWELL_MS = 1000
DWELL_RADIUS_PX = 40.0    # 30 px broke real stops on a 640x480 webcam; 60 px was for gaze
REARM_PX = 100.0          # after a click, move this far from it before the next dwell can start
ARRIVE_PX = 80.0          # a stop counts only if the cursor came from this far...
ARRIVE_MS = 2000          # ...for at least half of this long before it stopped
ANCHOR_SETTLE_MS = 150
SETTLE_SPEED_PXS = 300.0  # still arriving if a frame moves faster than this while the anchor settles
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
        self._arrived: bool | None = None   # this stop was travelled to; None until the anchor settles
        self._trail: deque[tuple[int, float, float]] = deque()   # every position over the last ARRIVE_MS

    @property
    def armed(self) -> bool:
        return self.enabled and self._disarmed_at is None

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._trail.clear()
        self.reset()

    def reset(self) -> None:
        """Forget the current stop (progress back to 0). The disarm stays."""
        self._stop.clear()
        self._anchor = None
        self._arrived = None
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
        self._trail.append((t_ms, x, y))   # kept through a lost face: where the cursor was still counts
        while t_ms - self._trail[0][0] > ARRIVE_MS + ANCHOR_SETTLE_MS:
            self._trail.popleft()
        if self._disarmed_at is not None:
            if math.hypot(x - self._disarmed_at[0], y - self._disarmed_at[1]) <= REARM_PX:
                return None
            self._disarmed_at = None
        if self._stop and t_ms - self._stop[-1][0] > MAX_GAP_MS:
            self.reset()
        arriving = False
        if self._stop and t_ms - self._stop[0][0] < ANCHOR_SETTLE_MS:
            lt, lx, ly = self._stop[-1]
            arriving = math.hypot(x - lx, y - ly) > SETTLE_SPEED_PXS * max(t_ms - lt, 1) / 1000
        if (arriving or self._anchor is None
                or math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.radius_px):
            self._stop = [(t_ms, x, y)]
            self._anchor = (x, y)
            self._arrived = None
            self.progress = 0.0
            return None
        self._stop.append((t_ms, x, y))
        t0 = self._stop[0][0]
        if t_ms - t0 < ANCHOR_SETTLE_MS:
            self._anchor = (statistics.fmean(p[1] for p in self._stop), statistics.fmean(p[2] for p in self._stop))
            return None
        if self._arrived is None:
            ax, ay = self._anchor
            before = [math.hypot(px - ax, py - ay) for pt, px, py in self._trail if t0 - ARRIVE_MS <= pt < t0]
            self._arrived = bool(before) and statistics.median(before) >= ARRIVE_PX
        if not self._arrived:
            return None   # the cursor crept here; it wasn't moved here
        self.progress = min(1.0, (t_ms - t0) / self.dwell_ms)
        if self.progress < 1.0:
            return None
        at = (statistics.median(p[1] for p in self._stop), statistics.median(p[2] for p in self._stop))
        self.disarm(*at)
        return at
