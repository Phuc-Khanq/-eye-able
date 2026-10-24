"""Look and say: gaze picks the area, UI-Venus finds the named element, a nod clicks it.

Route: Khang and Hung (Khang leads)
Publishes: TargetRequest -> TargetResult (through ScreenTarget), click and cursor-move actions
Subscribes: gaze samples (cursor points from the gaze cursor), requests with a label, nods
APIs: no camera, no model of its own: pass any finder with submit(TargetRequest) -> Future[TargetResult]
      (ScreenTarget in the app, a fake in tests). Nothing is saved.

Flow:  idle -> request(label)    take where you were looking just before the request
            -> searching         UI-Venus looks in the box around that point (zoom, then wide)
            -> confirm           cursor sits on the found element; nod (confirm) clicks, cancel/timeout drops it
Self-correction: every confirmed click tells us where the gaze really was (the element you meant), so a
shift (dx, dy) is learned from it and added to later gaze points. This follows the slow drift between
calibrations. It is capped and only learns from clicks UI-Venus found close to where the gaze said.
"""
import statistics
from collections import deque
from concurrent.futures import Future

from engine.core.types import TargetRequest, TargetResult

LOOK_BACK_S = 0.9       # use gaze from this long before the request...
LOOK_SKIP_S = 0.15      # ...up to this long before it (the head and eyes move as people start to speak)
CONFIRM_S = 6.0         # a found target waits this long for a nod
SHIFT_CAP_PX = 250.0    # never learn a bigger shift than this
LEARN_RATE = 0.5        # share of each click's error added to the shift
MAX_LEARN_ERR_PX = 450  # ignore clicks UI-Venus found farther than this from the gaze: probably a different element


class LookClick:
    def __init__(self, finder, screen_w: int, screen_h: int, auto: bool = False,
                 confirm_s: float = CONFIRM_S, learn: bool = True) -> None:
        self.finder = finder
        self.W, self.H = screen_w, screen_h
        self.auto = auto                      # click as soon as the element is found, no nod needed
        self.confirm_s = confirm_s
        self.learn = learn
        self.state = "idle"                   # idle | searching | confirm
        self.shift = (0.0, 0.0)
        self.samples: deque[tuple[float, float, float]] = deque()
        self.label = ""
        self.anchor: tuple[float, float] | None = None
        self.target: tuple[int, int] | None = None
        self._future: Future | None = None
        self._deadline = 0.0
        self.stats = {"asked": 0, "found": 0, "clicked": 0, "missed": 0, "canceled": 0}

    # -- gaze ----------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        """While True the cursor stays put: it is waiting on UI-Venus or on your nod."""
        return self.state != "idle"

    def add_gaze(self, now: float, x: float, y: float) -> None:
        self.samples.append((now, x, y))
        while self.samples and now - self.samples[0][0] > 3.0:
            self.samples.popleft()

    def shifted(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x + self.shift[0], 0), self.W - 1), min(max(y + self.shift[1], 0), self.H - 1))

    def _anchor(self, now: float) -> tuple[float, float] | None:
        window = [(x, y) for t, x, y in self.samples if now - LOOK_BACK_S <= t <= now - LOOK_SKIP_S]
        if not window and self.samples:
            window = [self.samples[-1][1:]]
        if not window:
            return None
        return self.shifted(statistics.median(p[0] for p in window), statistics.median(p[1] for p in window))

    # -- requests ------------------------------------------------------------------------------
    def request(self, label: str, now: float) -> str:
        label = label.strip()
        if not label:
            return "Empty label."
        if self.state == "searching":
            return "Still searching; wait for the answer."
        anchor = self._anchor(now)
        if anchor is None:
            return "No gaze yet; look at the screen and try again."
        self.label, self.anchor, self.target = label, anchor, None
        self.state = "searching"
        self.stats["asked"] += 1
        self._future = self.finder.submit(TargetRequest(gaze_x=int(round(anchor[0])), gaze_y=int(round(anchor[1])), label=label))
        return f"Looking for '{label}' around ({anchor[0]:.0f}, {anchor[1]:.0f})..."

    def poll(self, now: float) -> list[tuple]:
        """Call every frame. Returns actions: ("say", text), ("move", x, y), ("click", x, y)."""
        out: list[tuple] = []
        if self.state == "searching" and self._future is not None and self._future.done():
            fut, self._future = self._future, None
            try:
                res: TargetResult = fut.result()
            except Exception as e:  # model or screen problems must not crash the cursor
                self.state = "idle"
                self.stats["missed"] += 1
                return [("say", f"UI-Venus failed: {type(e).__name__}: {e}")]
            if not res.found:
                self.state = "idle"
                self.stats["missed"] += 1
                return [("say", f"'{self.label}' not found near your gaze ({res.ms:.0f} ms). Look closer to it and try again.")]
            self.stats["found"] += 1
            self.target = (res.x, res.y)
            dist = ((res.x - self.anchor[0]) ** 2 + (res.y - self.anchor[1]) ** 2) ** 0.5
            out.append(("move", res.x, res.y))
            if self.auto:
                out.append(("say", f"Found '{self.label}' at ({res.x}, {res.y}), {dist:.0f} px from your gaze, {res.ms:.0f} ms. Clicking."))
                out.append(self._do_click())
            else:
                self.state, self._deadline = "confirm", now + self.confirm_s
                out.append(("say", f"Found '{self.label}' at ({res.x}, {res.y}), {dist:.0f} px from your gaze, {res.ms:.0f} ms. Nod to click, X to cancel."))
        elif self.state == "confirm" and now > self._deadline:
            self.state, self.target = "idle", None
            self.stats["canceled"] += 1
            out.append(("say", "No nod, dropped it."))
        return out

    def confirm(self) -> tuple | None:
        """The nod. Returns ("click", x, y) when a found element was waiting, otherwise None."""
        return self._do_click() if self.state == "confirm" and self.target else None

    def cancel(self) -> str | None:
        if self.state == "idle":
            return None
        was = self.state
        self.state, self.target, self._future = "idle", None, None  # a late answer is ignored
        self.stats["canceled"] += 1
        return f"Canceled ({was})."

    def _do_click(self) -> tuple:
        x, y = self.target
        err = (x - self.anchor[0], y - self.anchor[1])
        if self.learn and (err[0] ** 2 + err[1] ** 2) ** 0.5 <= MAX_LEARN_ERR_PX:
            sx = min(max(self.shift[0] + LEARN_RATE * err[0], -SHIFT_CAP_PX), SHIFT_CAP_PX)
            sy = min(max(self.shift[1] + LEARN_RATE * err[1], -SHIFT_CAP_PX), SHIFT_CAP_PX)
            self.shift = (sx, sy)
        self.state, self.target = "idle", None
        self.stats["clicked"] += 1
        return ("click", x, y)
