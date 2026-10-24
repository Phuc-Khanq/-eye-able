"""The dwell ring at the cursor, drawn by the engine itself, so it shows without the Electron UI.

Route: Khang and Hung (Khang leads)
Publishes: one status warning if the window can't be made
Subscribes: topics.TO_UI, read only: dwell (progress, ready) and target_searching
APIs: tkinter on its own thread (every Tk call stays on it); Windows ctypes to make the window
      click-through, never focused and out of the taskbar

What it shows:
  hidden    not ready and nothing filling
  ready     a thin bright ring: resting on something clickable, dwell armed
  filling   a bright arc fills clockwise over a dark track, from the top
  done      a full ring, briefly, on the click
High contrast for low vision: a yellow arc with a black outline reads on light and dark pages.
The window is a small square with a key color made transparent; clicks go through it (WS_EX_TRANSPARENT)
and it never takes focus (WS_EX_NOACTIVATE), so pynput's clicks land on the app underneath. It hides
while UI-Venus searches, so the screenshot doesn't see the ring over a button.
"""
from __future__ import annotations

import sys
import threading
import time

from engine.core import messages, topics

SIZE = 72                 # window px; the ring sits around the cursor's tip
RING_W = 8                # filling arc width
READY_W = 3               # the "this clicks" ring
POLL_MS = 16              # about 60 Hz: follow the cursor and redraw
DONE_S = 0.25             # how long the full ring flashes on a click
KEY = "#ff00fe"           # made transparent; nothing is drawn in it
ARC = "#ffd400"
TRACK = "#202020"
OUTLINE = "#000000"

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010


def ring_state(progress: float, ready: bool, done_until: float, now: float) -> tuple[str, float]:
    """(state, arc extent in degrees, negative = clockwise) for one redraw."""
    if now < done_until:
        return "done", -359.9
    if progress > 0:
        return "filling", -359.9 * min(1.0, progress)
    if ready:
        return "ready", 0.0
    return "hidden", 0.0


class DwellMeter:
    def __init__(self, bus, read_cursor=None, clock=time.monotonic) -> None:
        self.bus = bus
        self.clock = clock
        self._read_cursor = read_cursor
        self.progress = 0.0
        self.ready = False
        self._done_until = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        bus.subscribe(topics.TO_UI, self.on_ui)

    def on_ui(self, msg: dict) -> None:
        """Any thread; only sets fields that the Tk thread reads."""
        kind = msg.get("type")
        if kind == messages.DWELL:
            progress = float(msg.get("progress", 0.0))
            if progress >= 1.0 and self.progress < 1.0:
                self._done_until = self.clock() + DONE_S
            self.progress, self.ready = progress, bool(msg.get("ready", False))
        elif kind == messages.TARGET_SEARCHING:
            self.progress, self.ready = 0.0, False

    def state(self) -> tuple[str, float]:
        return ring_state(self.progress, self.ready, self._done_until, self.clock())

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="dwell-meter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------- Tk thread
    def _run(self) -> None:
        if sys.platform != "win32" and self._read_cursor is None:
            return   # the click-through, see-through window is Windows-only; elsewhere it would be a square
        try:
            import tkinter as tk

            read_cursor = self._read_cursor or _cursor_reader()
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            if sys.platform == "win32":
                root.attributes("-transparentcolor", KEY)
            root.geometry(f"{SIZE}x{SIZE}+-200+-200")
            canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg=KEY, highlightthickness=0, bd=0)
            canvas.pack()
            root.update_idletasks()
            hwnd = _click_through(root)
        except Exception as exc:
            self.bus.publish(topics.TO_UI, messages.make(
                messages.STATUS, level="warn", text=f"No dwell ring on screen ({type(exc).__name__}: {exc})"))
            return

        pad = RING_W / 2 + 2
        box = (pad, pad, SIZE - pad, SIZE - pad)
        drawn = [None, None]
        topmost_at = [0.0]

        def tick():
            if self._stop.is_set():
                root.destroy()
                return
            try:
                state, extent = self.state()
                pos = read_cursor()
                if pos is not None:
                    root.geometry(f"+{int(pos[0]) - SIZE // 2}+{int(pos[1]) - SIZE // 2}")
                key = (state, round(extent))
                if key != drawn[0]:
                    drawn[0] = key
                    canvas.delete("all")
                    if state == "ready":
                        canvas.create_oval(*box, outline=OUTLINE, width=READY_W + 2)
                        canvas.create_oval(*box, outline=ARC, width=READY_W)
                    elif state in ("filling", "done"):
                        canvas.create_oval(*box, outline=OUTLINE, width=RING_W + 2)
                        canvas.create_oval(*box, outline=TRACK, width=RING_W)
                        canvas.create_arc(*box, start=90, extent=extent, style="arc", outline=ARC, width=RING_W)
                now = self.clock()
                if hwnd and now - topmost_at[0] > 1.0:   # stay above a full-screen video
                    topmost_at[0] = now
                    _keep_on_top(hwnd)
            except Exception:
                pass
            root.after(POLL_MS, tick)

        root.after(POLL_MS, tick)
        try:
            root.mainloop()
        except Exception:
            pass


def _cursor_reader():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        def read():
            pt = wintypes.POINT()
            return (pt.x, pt.y) if user32.GetCursorPos(ctypes.byref(pt)) else None
        return read
    try:
        from pynput.mouse import Controller

        mouse = Controller()
        return lambda: mouse.position
    except Exception:
        return lambda: None


def _click_through(root) -> int:
    """Windows: clicks go through the window, it never takes focus or shows in the taskbar. Its hwnd."""
    if sys.platform != "win32":
        return 0
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
    return hwnd


def _keep_on_top(hwnd: int) -> None:
    import ctypes

    ctypes.windll.user32.SetWindowPos(hwnd, ctypes.c_void_p(HWND_TOPMOST), 0, 0, 0, 0,
                                      SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
