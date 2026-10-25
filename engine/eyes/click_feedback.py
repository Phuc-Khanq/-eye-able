"""What a click looks and sounds like: a quick flash where it lands and a short tick. Replaces the ring around the
cursor (Oct 24 night: Khang asked for a click effect instead of the circle).

Route: Khang
Publishes: nothing
Subscribes: topics.OS_CLICK, read only: where it clicks and which button (every click goes through it: the blink,
            the voice "click", look-and-say, dwell, right clicks)
APIs: tkinter, as a Toplevel on the click ring's Tk thread (engine/eyes/dwell_meter.py), so the engine keeps one Tk
      thread; Windows ctypes for click-through, never focused, out of the taskbar and (optionally) out of
      screenshots; winsound plays a WAV made once at startup in the temp folder (a synthesized tick, a few KB)

Flash: a bright dot pops out from the click spot and fades in FLASH_S, like a camera blink: yellow for a left click,
cyan for a right click, with a thin dark edge so it shows on white pages too. It's drawn after the click, so it is
never in UI-Venus's screenshot; with card_hide_from_capture it stays out of screenshots anyway, and with --record
OBS sees it.
Sound: a soft high tick for a left click, a lower one for a right click. click_sound false turns it off;
click_sound_volume is 0 to 1.
"""
from __future__ import annotations

import io
import math
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from engine.core import topics

FLASH_S = 0.28            # the whole flash, pop and fade
SIZE = 72                 # window px
R_START, R_END = 7.0, 26.0    # the dot's radius at the click and when it has faded
ALPHA_START = 0.95
POLL_MS = 16              # about 60 Hz while a flash runs
KEY = "#ff00fe"           # made transparent; nothing is drawn in it
EDGE = "#000000"
CORE = "#ffffff"
COLORS = {"left": "#ffd400", "right": "#22d3ee"}
OFF_SCREEN = "+-5000+-5000"
WDA_EXCLUDEFROMCAPTURE = 0x11

RATE = 44100
TICKS = {"left": (1900.0, 0.010), "right": (1150.0, 0.014)}   # button -> (pitch Hz, decay s)
TICK_S = 0.07


def ease_out(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** 3


class FlashModel:
    """Where the flash is and how it looks right now; no Tk, so tests can drive it."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._flash: tuple[float, float, str, float] | None = None   # x, y, button, start time

    def on_click(self, x: float, y: float, button: str = "left") -> None:
        with self._lock:
            self._flash = (float(x), float(y), button if button in COLORS else "left", self.clock())

    def frame(self) -> tuple[float, float, float, float, str] | None:
        """(x, y, radius, alpha, color) for this redraw, or None when no flash is showing."""
        with self._lock:
            f = self._flash
        if f is None:
            return None
        x, y, button, t0 = f
        p = (self.clock() - t0) / FLASH_S
        if p >= 1.0 or p < 0:
            with self._lock:
                if self._flash is f:
                    self._flash = None
            return None
        radius = R_START + (R_END - R_START) * ease_out(p)
        alpha = ALPHA_START * (1.0 - p) ** 2
        return x, y, radius, alpha, COLORS[button]


def tick_wav(pitch: float, decay: float, volume: float, rate: int = RATE) -> bytes:
    """A short tick as a 16-bit mono WAV: a decaying tone and its octave, plus a 2 ms noise transient."""
    import numpy as np

    n = int(TICK_S * rate)
    t = np.arange(n) / rate
    env = np.exp(-t / decay)
    tone = np.sin(2 * math.pi * pitch * t) + 0.45 * np.sin(2 * math.pi * 2 * pitch * t)
    noise = np.random.default_rng(7).standard_normal(n) * np.exp(-t / 0.002) * 0.35
    s = (tone * env + noise)
    attack = np.minimum(1.0, t / 0.0004)                 # 0.4 ms in: no hard pop at the first sample
    s = s * attack
    s = s / max(1e-9, float(np.abs(s).max())) * 0.7 * min(max(volume, 0.0), 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((s * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class ClickSound:
    """Plays the tick. play(button) never blocks: winsound plays the file asynchronously (Windows only; elsewhere,
    and if anything fails, it stays quiet). `player` replaces winsound in tests."""

    def __init__(self, volume: float = 0.5, player=None, folder: Path | None = None) -> None:
        self.volume = min(max(float(volume), 0.0), 1.0)
        self._player = player
        self._folder = Path(folder) if folder is not None else Path(tempfile.gettempdir())
        self._paths: dict[str, Path] = {}
        self.failed = ""

    def prepare(self) -> None:
        """Write the two WAVs once (call at startup, not on the first click)."""
        if self.volume <= 0:
            return
        try:
            for button, (pitch, decay) in TICKS.items():
                path = self._folder / f"diseyeble_click_{button}_{int(self.volume * 100)}.wav"
                if not path.exists():
                    path.write_bytes(tick_wav(pitch, decay, self.volume))
                self._paths[button] = path
        except Exception as exc:          # no sound is fine; a broken click is not
            self.failed = f"{type(exc).__name__}: {exc}"

    def play(self, button: str = "left") -> None:
        if self.volume <= 0:
            return
        if self._player is not None:
            self._player(button)
            return
        path = self._paths.get(button if button in TICKS else "left")
        if path is None or sys.platform != "win32":
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"


class ClickFeedback:
    def __init__(self, bus, flash: bool = True, sound: bool = True, volume: float = 0.5,
                 hide_from_capture: bool = True, clock=time.monotonic, player=None) -> None:
        self.flash = FlashModel(clock) if flash else None
        self.sound = ClickSound(volume, player=player) if sound else None
        self.hide_from_capture = hide_from_capture
        if self.sound is not None:
            self.sound.prepare()
        bus.subscribe(topics.OS_CLICK, self.on_click)

    def on_click(self, payload: dict) -> None:
        """Any thread: the bus calls this as the click goes out."""
        button = "right" if payload.get("button") == "right" else "left"
        if self.sound is not None:
            self.sound.play(button)
        if self.flash is not None:
            self.flash.on_click(float(payload["x"]), float(payload["y"]), button)

    # ------------------------------------------------------------- Tk thread (the click ring's)
    def attach(self, root) -> None:
        """Build the flash window as a Toplevel of the ring's Tk root; call on that thread."""
        if self.flash is None:
            return
        import tkinter as tk

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=KEY)
        if sys.platform == "win32":
            win.attributes("-transparentcolor", KEY)
        canvas = tk.Canvas(win, width=SIZE, height=SIZE, bg=KEY, highlightthickness=0, bd=0)
        canvas.pack()
        win.geometry(OFF_SCREEN)
        win.update_idletasks()
        hwnd = _setup_window(win, self.hide_from_capture)
        shown = [False]

        def tick():
            try:
                f = self.flash.frame()
                if f is None:
                    if shown[0]:
                        shown[0] = False
                        canvas.delete("all")
                        win.geometry(OFF_SCREEN)
                else:
                    x, y, r, alpha, color = f
                    c = SIZE / 2
                    canvas.delete("all")
                    canvas.create_oval(c - r - 2, c - r - 2, c + r + 2, c + r + 2, fill=EDGE, outline="")
                    canvas.create_oval(c - r, c - r, c + r, c + r, fill=color, outline="")
                    core = max(2.0, r * 0.38)
                    canvas.create_oval(c - core, c - core, c + core, c + core, fill=CORE, outline="")
                    try:
                        win.attributes("-alpha", max(0.05, alpha))
                    except tk.TclError:
                        pass
                    win.geometry(f"+{int(x - c)}+{int(y - c)}")
                    if not shown[0] and hwnd:
                        win.update_idletasks()
                        _keep_on_top(hwnd)
                    shown[0] = True
            except Exception:
                pass
            win.after(POLL_MS, tick)

        win.after(POLL_MS, tick)


def _keep_on_top(hwnd: int) -> None:
    from engine.eyes.dwell_meter import _keep_on_top as on_top

    on_top(hwnd)


def _setup_window(win, hide_from_capture: bool) -> int:
    """Windows: click-through, never focused, out of the taskbar, and (optionally) out of screenshots."""
    if sys.platform != "win32":
        return 0
    import ctypes

    from engine.eyes.dwell_meter import _click_through

    hwnd = _click_through(win)
    if hide_from_capture:
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass
    return hwnd
