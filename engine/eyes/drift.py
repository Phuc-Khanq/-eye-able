"""One-look recenter: fixes gaze drift after the user shifts in their chair, without recalibrating.

Route: Khang and Hung (Hung leads)
Publishes: corrected gaze (used inside the cursor loop)
Subscribes: raw gaze predictions from the calibrated GazeModel
APIs: NumPy

How it works: show a mark at a known spot (the screen center), the user looks at it and nods.
The median of the raw gaze predictions just before the nod, compared with the mark, is the drift.
From then on that offset is added to every prediction. A new recenter replaces the old offset.
Collect the uncorrected predictions (before apply()), or offsets would stack.
"""
from collections import deque

import numpy as np

COLLECT_MS = 600       # use this much gaze just before the nod
MIN_SAMPLES = 6        # usable frames needed in that window
MAX_OFFSET_PX = 400    # more than this means the user wasn't looking at the mark, or needs a recalibration


class OffsetCorrector:
    def __init__(self) -> None:
        self.dx = 0.0
        self.dy = 0.0
        self.collecting = False
        self.samples: deque[tuple[int, float, float]] = deque()

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return x + self.dx, y + self.dy

    def begin(self) -> None:
        """Start collecting while the user looks at the mark."""
        self.collecting = True
        self.samples.clear()

    def add(self, t_ms: int, raw_x: float, raw_y: float) -> None:
        """Feed uncorrected gaze predictions while collecting."""
        if not self.collecting:
            return
        self.samples.append((t_ms, raw_x, raw_y))
        while self.samples and t_ms - self.samples[0][0] > 4 * COLLECT_MS:
            self.samples.popleft()

    def finish(self, target_x: float, target_y: float, until_ms: int | None = None) -> str:
        """Use the gaze up to until_ms (the moment the nod began). Returns a message for the user;
        the offset only changes when the result looks sane."""
        self.collecting = False
        end = until_ms if until_ms is not None else (self.samples[-1][0] if self.samples else 0)
        window = [(x, y) for t, x, y in self.samples if end - COLLECT_MS <= t <= end]
        self.samples.clear()
        if len(window) < MIN_SAMPLES:
            return "Recenter failed: not enough steady gaze. Look at the mark a moment longer, then nod."
        mx, my = np.median(np.array(window), axis=0)
        dx, dy = target_x - float(mx), target_y - float(my)
        if np.hypot(dx, dy) > MAX_OFFSET_PX:
            return (f"Recenter skipped: gaze was {np.hypot(dx, dy):.0f} px from the mark. "
                    "Look right at the mark, or recalibrate with gaze_dot.")
        self.dx, self.dy = dx, dy
        return f"Recentered: offset ({dx:+.0f}, {dy:+.0f}) px."

    def cancel(self) -> None:
        self.collecting = False
        self.samples.clear()

    def reset(self) -> None:
        self.cancel()
        self.dx = self.dy = 0.0
