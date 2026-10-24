"""Is something clickable under the cursor? For dwell: only buttons, links, tabs and the like start one.

Route: Khang and Hung (Khang leads)
Publishes: nothing; the eyes service asks over_at(x, y, now) every frame
Subscribes: nothing
APIs: Windows only. Tier 1: user32 GetCursorInfo through ctypes (no packages). Tier 2, optional:
      Windows UI Automation through the uiautomation package (pip install uiautomation)

Tier 1, the cursor shape: browsers show the hand on anything with cursor: pointer, so the hand means a
link or a button. A background thread reads it about 30 times a second; cheap, no new packages.
Tier 2, UI Automation: native buttons, the taskbar, title-bar buttons, menus, desktop icons and text
boxes (which show an I-beam, not the hand), plus the element's rectangle, which lets dwell use a looser
radius and keep the click inside the element. ControlFromPoint can take 50 to 300 ms in Chrome, so it
runs on its own thread and only asks again when the cursor has moved MOVE_PX or every REQUERY_S.
Without the package, tier 1 alone runs. An element of our own process (the dwell meter) is skipped.

over_at() never blocks: it combines the latest answers and returns None (unknown) when neither is
fresh or was taken near (x, y). Off Windows, make_probe() returns None and dwell works as before.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass

POLL_S = 0.033            # the hand cursor, about 30 times a second
FRESH_S = 0.4             # an answer older than this is unknown...
NEAR_PX = 60.0            # ...and so is one taken further than this from where the cursor is now
UIA_FRESH_S = 0.8         # UI Automation is slower to answer; its rectangle says where it holds
RECT_MARGIN_PX = 12.0
MOVE_PX = 4.0             # ask UI Automation again when the cursor has moved this far...
REQUERY_S = 0.5           # ...or this long after the last answer
MAX_PARENTS = 3           # a link's text is the element under the cursor; the link is its parent
MAX_W, MAX_H = 1000, 700  # bigger "buttons" are page areas (a whole list, an editor), not targets
IDC_HAND = 32649
CLICKABLE_TYPES = {
    "ButtonControl", "HyperlinkControl", "MenuItemControl", "TabItemControl", "ListItemControl",
    "TreeItemControl", "CheckBoxControl", "RadioButtonControl", "SplitButtonControl", "ComboBoxControl",
    "EditControl",
}


@dataclass(frozen=True)
class Over:
    clickable: bool
    rect: tuple[int, int, int, int] | None = None   # left, top, right, bottom in physical pixels


@dataclass(frozen=True)
class _Seen:
    t: float
    x: float
    y: float
    over: Over


class ClickableProbe:
    """Two background threads (hand cursor, UI Automation) and a non-blocking over_at().

    read_cursor() -> (x, y, hand: bool) or None, and ui_at(x, y) -> Over | None | "own" ("own": our own
    window, keep the last answer) are the Windows calls; tests pass fakes and call poll_* directly."""

    def __init__(self, read_cursor=None, ui_at=None, clock=time.monotonic, uia_context=None) -> None:
        self.read_cursor = read_cursor or win_cursor
        self.ui_at = ui_at
        self.clock = clock
        self._uia_context = uia_context
        self._hand: _Seen | None = None
        self._uia: _Seen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------- answers
    def over_at(self, x: float, y: float, now: float | None = None) -> Over | None:
        """What is under (x, y), from the latest answers; None when nothing recent was seen near it."""
        now = self.clock() if now is None else now
        with self._lock:
            hand, uia = self._hand, self._uia
        hand_ok = (hand is not None and now - hand.t <= FRESH_S
                   and math.hypot(x - hand.x, y - hand.y) <= NEAR_PX)
        uia_ok = uia is not None and now - uia.t <= UIA_FRESH_S
        if uia_ok:
            if uia.over.rect is not None:
                uia_ok = _inside(x, y, uia.over.rect, RECT_MARGIN_PX)
            else:
                uia_ok = math.hypot(x - uia.x, y - uia.y) <= NEAR_PX
        if uia_ok and uia.over.clickable:
            return uia.over                       # with its rectangle
        if hand_ok and hand.over.clickable:
            return hand.over
        if hand_ok or uia_ok:
            return Over(False)
        return None

    def poll_hand(self) -> None:
        got = self.read_cursor()
        if got is not None:
            x, y, hand = got
            with self._lock:
                self._hand = _Seen(self.clock(), float(x), float(y), Over(bool(hand)))

    def poll_uia(self) -> None:
        """Ask UI Automation about the element under the real cursor, if it moved or the answer is old."""
        with self._lock:
            hand, last = self._hand, self._uia
        if hand is None or self.ui_at is None:
            return
        now = self.clock()
        if (last is not None and now - last.t < REQUERY_S
                and math.hypot(hand.x - last.x, hand.y - last.y) < MOVE_PX):
            return
        got = self.ui_at(int(hand.x), int(hand.y))
        if got == "own":
            return                                # our own meter window: keep what we had
        with self._lock:
            self._uia = _Seen(self.clock(), hand.x, hand.y, got if isinstance(got, Over) else Over(False))

    # ---------------------------------------------------------- threads
    def start(self) -> None:
        self._stop.clear()
        self._threads = [threading.Thread(target=self._loop, args=(self.poll_hand, None), name="clickable", daemon=True)]
        if self.ui_at is not None:
            self._threads.append(threading.Thread(target=self._loop, args=(self.poll_uia, self._uia_context),
                                                  name="clickable-uia", daemon=True))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1)

    def _loop(self, poll, context) -> None:
        def run():
            while not self._stop.wait(POLL_S):
                try:
                    poll()
                except Exception:   # a COM hiccup or a window closing under the cursor: try again
                    time.sleep(0.2)
        if context is None:
            run()
        else:
            with context():
                run()


def _inside(x: float, y: float, rect, margin: float) -> bool:
    left, top, right, bottom = rect
    return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin


# ------------------------------------------------------------ Windows calls
_user32 = None
_hand_cursor = None


def win_cursor():
    """(x, y, hand) for the real cursor, physical pixels once os_input.make_dpi_aware() has run."""
    global _user32, _hand_cursor
    import ctypes
    from ctypes import wintypes

    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hCursor", ctypes.c_void_p), ("ptScreenPos", wintypes.POINT)]

    if _user32 is None:
        _user32 = ctypes.windll.user32
        _user32.LoadCursorW.restype = ctypes.c_void_p
        _user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _hand_cursor = _user32.LoadCursorW(None, ctypes.c_void_p(IDC_HAND))
    info = CURSORINFO()
    info.cbSize = ctypes.sizeof(CURSORINFO)
    if not _user32.GetCursorInfo(ctypes.byref(info)):
        return None
    pt = wintypes.POINT()
    if not _user32.GetCursorPos(ctypes.byref(pt)):   # the same moment; ptScreenPos can be unscaled
        pt = info.ptScreenPos
    hand = bool(info.flags & 1) and info.hCursor is not None and info.hCursor == _hand_cursor
    return pt.x, pt.y, hand


def uia_lookup():
    """(ui_at, context) for UI Automation, or (None, None) if the uiautomation package isn't installed."""
    try:
        import uiautomation as auto
    except Exception:
        return None, None
    own = os.getpid()

    def ui_at(x: int, y: int):
        c = auto.ControlFromPoint(x, y)
        if c is None:
            return Over(False)
        if c.ProcessId == own:
            return "own"
        for _ in range(MAX_PARENTS + 1):
            if c is None:
                break
            if c.ControlTypeName in CLICKABLE_TYPES and c.IsEnabled:
                r = c.BoundingRectangle
                rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
                w, h = rect[2] - rect[0], rect[3] - rect[1]
                if 0 < w <= MAX_W and 0 < h <= MAX_H:
                    return Over(True, rect)
                return Over(False)
            c = c.GetParentControl()
        return Over(False)

    return ui_at, auto.UIAutomationInitializerInThread


def make_probe(use_uia: bool = True):
    """(probe, note): a ClickableProbe on Windows, else None. note says when UI Automation is missing."""
    if sys.platform != "win32":
        return None, ""
    ui_at, context = uia_lookup() if use_uia else (None, None)
    note = ""
    if use_uia and ui_at is None:
        note = ("Dwell: only links and buttons that show the hand cursor count "
                "(pip install uiautomation adds native buttons and text boxes)")
    return ClickableProbe(win_cursor, ui_at, uia_context=context), note
