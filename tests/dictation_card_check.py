"""Windows: automatic PASS/FAIL checks on the dictation card (engine/eyes/dictation_card.py), on screen.
  hidden from screenshots  a screenshot of the card's area (screen_target.grab(), as UI-Venus takes it) while the
                           card shows has (almost) none of the card's panel colour (the card is drawn solid here,
                           so its colour is exact; what changes behind it doesn't count)
  no focus                 the window in front (GetForegroundWindow) is the same before and after the card shows
  styles                   click-through, never activated, out of the taskbar; display affinity set
  hides                    a dictation_preview with words [] takes it off the screen
tests.card_check plays every state for you to look at; this one measures instead.

Open Notepad and click into it first, then within a few seconds:
    .\\.venv\\Scripts\\python.exe -m tests.dictation_card_check
    .\\.venv\\Scripts\\python.exe -m tests.dictation_card_check --control
--control leaves the card in screenshots, so "hidden from screenshots" must FAIL: proof the check sees the card.
"""
from __future__ import annotations

import argparse
import sys
import time

from engine.config import settings
from engine.core import messages, topics
from engine.core.events import EventBus

STYLES = {"WS_EX_LAYERED": 0x00080000, "WS_EX_TRANSPARENT": 0x00000020, "WS_EX_NOACTIVATE": 0x08000000,
          "WS_EX_TOOLWINDOW": 0x00000080}
WDA_EXCLUDEFROMCAPTURE = 0x11
NEAR = 1              # a pixel is the panel colour when every channel is within this (0-255)
HIDDEN_MAX = 0.01     # hidden: at most 1% of the card's area is the panel colour (the page behind might have it)


def colour_share(img, hex_colour: str) -> float:
    """Share of an RGB PIL image's pixels that are hex_colour."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    data = img.convert("RGB").tobytes()
    hits = sum(1 for i in range(0, len(data), 3)
               if abs(data[i] - r) <= NEAR and abs(data[i + 1] - g) <= NEAR and abs(data[i + 2] - b) <= NEAR)
    return hits / max(1, len(data) // 3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", action="store_true", help="leave the card in screenshots (that check must FAIL)")
    args = p.parse_args(argv)
    if sys.platform != "win32":
        print("Windows only (the card is a Windows window).")
        return 0

    import ctypes
    from ctypes import wintypes

    from engine.eyes import os_input
    from engine.eyes.dictation_card import PANEL, DictationCard
    from engine.eyes.dwell_meter import DwellMeter
    from engine.eyes.screen_target import grab

    os_input.make_dpi_aware()   # as the engine does: Tk, GetWindowRect and mss all in physical pixels
    user32 = ctypes.windll.user32

    def front() -> tuple[int, str]:
        hwnd = user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(200)
        user32.GetWindowTextW(hwnd, buf, 200)
        return hwnd, buf.value

    def rect(hwnd: int) -> tuple[int, int, int, int]:
        r = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        return r.left, r.top, r.right - r.left, r.bottom - r.top

    bus = EventBus()
    bus.subscribe(topics.TO_UI, lambda m: m.get("type") == messages.STATUS
                  and print(f"  status {m['level']}: {m['text']}", flush=True))
    meter = DwellMeter(bus)
    card = DictationCard(bus, hide_from_capture=not args.control, opacity=1.0,   # solid: its colour is exact
                         position=str(getattr(settings, "card_position", "top")))
    found: dict[str, int] = {}

    def attach(root) -> None:   # on the Tk thread: build the card, then note its window handle
        before = set(root.winfo_children())
        card.attach(root)
        win = next(w for w in root.winfo_children() if w not in before)
        found["hwnd"] = user32.GetParent(win.winfo_id()) or win.winfo_id()

    meter.add_window(attach)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}: {name} ({detail})", flush=True)

    fg_start = front()
    print(f"in front at the start: {fg_start[1]!r}", flush=True)
    meter.start()
    try:
        for _ in range(50):
            if "hwnd" in found:
                break
            time.sleep(0.1)
        hwnd = found.get("hwnd")
        if not hwnd:
            print("FAIL: the card's window never appeared")
            return 1
        time.sleep(0.5)
        fg_before = front()
        check("ring and card start without taking focus", fg_before[0] == fg_start[0],
              "the same window is still in front" if fg_before[0] == fg_start[0]
              else f"{fg_before[1]!r} took it (engine/eyes/dwell_meter.py: the ring's window)")
        print(f"in front before the card shows: {fg_before[1]!r}", flush=True)

        bus.publish(topics.TO_UI, messages.make(messages.DICTATION_PREVIEW, words=[
            {"text": w, "conf": 0.95} for w in "see you at nine tomorrow".split()]))
        time.sleep(1.0)
        left, top, width, height = rect(hwnd)
        shown = width > 0 and height > 0 and left > -1000
        check("card shows", shown, f"at ({left}, {top}), {width}x{height}")
        with_card = grab(left, top, width, height) if shown else None
        fg_after = front()
        print(f"in front after:  {fg_after[1]!r}", flush=True)
        check("card shows without taking focus", fg_after[0] == fg_before[0] and fg_after[0] != hwnd,
              "the same window is still in front" if fg_after[0] == fg_before[0] else "the window in front changed")

        style = user32.GetWindowLongW(hwnd, -20)
        missing = [name for name, bit in STYLES.items() if not style & bit]
        check("styles", not missing, "all four set" if not missing else f"missing {', '.join(missing)}")
        affinity = wintypes.DWORD(0)
        user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
        want = 0 if args.control else WDA_EXCLUDEFROMCAPTURE
        check("display affinity", affinity.value == want, f"0x{affinity.value:x}, want 0x{want:x}")

        if with_card is not None:
            share = colour_share(with_card, PANEL)
            check("hidden from screenshots" + (" (--control: must FAIL)" if args.control else ""),
                  share <= HIDDEN_MAX, f"{share:.1%} of the card's area in the screenshot is the card's colour {PANEL}")

        bus.publish(topics.TO_UI, messages.make(messages.DICTATION_PREVIEW, words=[]))
        time.sleep(2.5)   # "Dropped" shows for a moment first (dictation_card.DONE_S)
        gone = rect(hwnd)[0] <= -1000
        check("hides on words []", gone, "moved off the screen" if gone else f"still at {rect(hwnd)[:2]}")
    finally:
        meter.stop()
    ok = all(ok for _, ok, _ in results)
    print(f"{sum(r[1] for r in results)}/{len(results)} passed: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
