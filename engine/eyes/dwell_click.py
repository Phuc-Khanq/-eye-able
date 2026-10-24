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
with: python -m tests.brow_click --replay calibration/brow_logs/<file>.csv

Midas touch (a desktop is full of things that click): every click, of any kind, disarms the dwell
until the cursor has moved REARM_PX away from that spot. So a parked cursor clicks once, not every
second, and the menu that a right click opens under the cursor doesn't get clicked 1 s later. The eyes
service also turns it off, and disarms it at the cursor when it comes back on, whenever a click there
would be wrong (other modes, a question or preview waiting, UI-Venus searching, a raise in progress).

Only clickable things (gate=True, with `over` from engine/eyes/clickable.py): resting on blank space or
text never starts a dwell. On something clickable the stop is looser, since holding a head pointer
within 40 px for a second is hard: RECT_RADIUS_PX when the element's rectangle is known and the cursor
is in it, HAND_RADIUS_PX with only the hand cursor to go on (more could reach the next link). A drop in
clickability shorter than GRACE_MS (the cursor wobbling over a button's edge) holds the progress instead
of starting over, moving onto another element starts a new stop, and the click point is kept inside the
element's rectangle. A stop lost to a longer drop starts over but needn't be travelled to again.
With no `over` and gate=False (the default) it works exactly as before.
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
RECT_RADIUS_PX = 100.0    # on a clickable element whose rectangle is known, the cursor inside it
HAND_RADIUS_PX = 60.0     # on something clickable known only from the hand cursor
RECT_MARGIN_PX = 12.0     # "inside the rectangle" allows this much outside it
RECT_CHANGE_PX = 6.0      # a rectangle further off than this on any side is another element
GRACE_MS = 250            # clickable again within this long: the stop goes on (edge jitter)
CLAMP_INSET_PX = 2.0      # the click point stays at least this far inside the rectangle


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
        self._rect: tuple[float, float, float, float] | None = None   # the element this stop is on
        self._ok_at: int | None = None     # last frame the gate saw something clickable, in this stop
        self._held_ms = 0                  # time in this stop spent in the grace, not counted as resting
        self._last_t: int | None = None
        self._lost: tuple[int, float, float] | None = None   # t, anchor of an arrived stop the gate dropped

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
        self._rect = None
        self._ok_at = None
        self._held_ms = 0
        self.progress = 0.0

    def disarm(self, x: float, y: float) -> None:
        """Nothing clicks until the cursor has moved REARM_PX away from (x, y)."""
        self._disarmed_at = (x, y)
        self.reset()

    def update(self, t_ms: int, x: float, y: float, over=None, gate: bool = False) -> tuple[float, float] | None:
        """Feed every cursor position; returns the click point on the frame the dwell completes, else None.

        over: what is under the cursor (clickable.Over: .clickable, .rect as left, top, right, bottom),
        or None when unknown. gate=True: only a stop on something clickable can click."""
        last_t, self._last_t = self._last_t, t_ms
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
        clickable = over is not None and bool(over.clickable)
        rect = getattr(over, "rect", None) if clickable else None
        if gate and not clickable:
            if self._stop and self._ok_at is not None and t_ms - self._ok_at <= GRACE_MS:
                if last_t is not None:
                    self._held_ms += t_ms - last_t   # progress holds while it's off the edge
                return None
            if self._arrived and self._anchor is not None:
                self._lost = (t_ms, *self._anchor)   # it was travelled to; coming back soon still counts
            self.reset()
            return None
        if gate:
            self._ok_at = t_ms
        if rect is not None and self._rect is not None and _moved(rect, self._rect):
            self.reset()                             # onto another element: a new stop
        radius = self.radius_px
        if rect is not None and _inside(x, y, rect, RECT_MARGIN_PX):
            radius = max(radius, RECT_RADIUS_PX)
        elif clickable:
            radius = max(radius, HAND_RADIUS_PX)
        arriving = False
        if self._stop and t_ms - self._stop[0][0] < ANCHOR_SETTLE_MS:
            lt, lx, ly = self._stop[-1]
            arriving = math.hypot(x - lx, y - ly) > SETTLE_SPEED_PXS * max(t_ms - lt, 1) / 1000
        if (arriving or self._anchor is None
                or math.hypot(x - self._anchor[0], y - self._anchor[1]) > radius):
            self._stop = [(t_ms, x, y)]
            self._anchor = (x, y)
            self._arrived = None
            self._rect = rect
            self._ok_at = t_ms if gate else None
            self._held_ms = 0
            self.progress = 0.0
            return None
        self._stop.append((t_ms, x, y))
        if rect is not None:
            self._rect = rect
        t0 = self._stop[0][0]
        if t_ms - t0 < ANCHOR_SETTLE_MS:
            self._anchor = (statistics.fmean(p[1] for p in self._stop), statistics.fmean(p[2] for p in self._stop))
            return None
        if self._arrived is None and self._lost is not None:
            lt, lx, ly = self._lost
            self._lost = None
            if t0 - lt <= ARRIVE_MS and math.hypot(self._anchor[0] - lx, self._anchor[1] - ly) <= RECT_RADIUS_PX:
                self._arrived = True                 # back on the spot the edge wobble lost
        if self._arrived is None:
            ax, ay = self._anchor
            before = [math.hypot(px - ax, py - ay) for pt, px, py in self._trail if t0 - ARRIVE_MS <= pt < t0]
            self._arrived = bool(before) and statistics.median(before) >= ARRIVE_PX
        if not self._arrived:
            return None   # the cursor crept here; it wasn't moved here
        self.progress = min(1.0, max(0.0, t_ms - t0 - self._held_ms) / self.dwell_ms)
        if self.progress < 1.0:
            return None
        at = (statistics.median(p[1] for p in self._stop), statistics.median(p[2] for p in self._stop))
        if self._rect is not None:
            at = _clamp(at, self._rect)
        self.disarm(*at)
        return at


def _inside(x: float, y: float, rect, margin: float = 0.0) -> bool:
    left, top, right, bottom = rect
    return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin


def _moved(a, b) -> bool:
    return any(abs(p - q) > RECT_CHANGE_PX for p, q in zip(a, b))


def _clamp(at: tuple[float, float], rect) -> tuple[float, float]:
    """The point, moved inside the rectangle if it rested just outside (a small button's edge)."""
    left, top, right, bottom = rect
    inset_x = min(CLAMP_INSET_PX, (right - left) / 2)
    inset_y = min(CLAMP_INSET_PX, (bottom - top) / 2)
    return (min(max(at[0], left + inset_x), right - inset_x), min(max(at[1], top + inset_y), bottom - inset_y))
