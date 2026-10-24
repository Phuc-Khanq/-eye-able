"""Live check, Windows: what the dwell probe sees under the mouse, and the dwell ring, with no camera.

Move the real mouse by hand over a YouTube tile, a Chrome tab, a window's close button, a text box,
blank page space and paragraph text. Each line says whether dwell would count it as clickable, and how:
  hand   the hand cursor (links and buttons in a browser)
  uia    UI Automation, with the element's rectangle (native buttons, taskbar, text boxes)
With --ring the dwell ring follows the mouse too: resting 1 s on something clickable fills it (no
click is made). Check that it never steals focus or clicks, and that the hand cursor still shows
under it (if the ring hid the hand, tier 1 would break).

Run with: python -m tests.clickable_check [--ring] [--no-uia]   (Ctrl+C to stop)
"""
import argparse
import time

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.eyes import os_input
from engine.eyes.clickable import make_probe
from engine.eyes.dwell_click import DwellClicker


def main() -> None:
    p = argparse.ArgumentParser(description="What the dwell probe sees under the mouse")
    p.add_argument("--ring", action="store_true", help="show the dwell ring and fill it on a 1 s rest")
    p.add_argument("--no-uia", action="store_true", help="hand cursor only")
    args = p.parse_args()
    os_input.make_dpi_aware()
    probe, note = make_probe(not args.no_uia)
    if probe is None:
        raise SystemExit("Windows only.")
    if note:
        print(note)
    print("UI Automation:", "on" if probe.ui_at is not None else "off")
    probe.start()
    bus = meter = None
    if args.ring:
        from engine.eyes.dwell_meter import DwellMeter
        bus = EventBus()
        bus.subscribe(topics.TO_UI, lambda m: m.get("type") == messages.STATUS and print(f"[status] {m['text']}"))
        meter = DwellMeter(bus)
        meter.start()
    dwell = DwellClicker()
    last_line, last_print, sent = None, 0.0, (0.0, False)
    t0 = time.monotonic()
    try:
        while True:
            time.sleep(0.033)
            now = time.monotonic()
            got = probe.read_cursor()
            if got is None:
                continue
            x, y, hand = got
            over = probe.over_at(x, y, now)
            at = dwell.update(int((now - t0) * 1000), x, y, over=over, gate=True)
            if bus is not None:
                ready = over is not None and over.clickable and dwell.armed
                state = (1.0 if at is not None else round(dwell.progress, 2), bool(ready))
                if state != sent:
                    bus.publish(topics.TO_UI, messages.make(messages.DWELL, progress=state[0], ready=state[1]))
                    sent = state
            if at is not None:
                print(f"  -> dwell would click at ({at[0]:.0f}, {at[1]:.0f})   (no click made)")
            how = ("unknown" if over is None else "no" if not over.clickable
                   else f"uia {over.rect}" if over.rect else "hand")
            line = f"({x:5d}, {y:5d})  hand cursor: {'yes' if hand else 'no '}  clickable: {how}"
            if line != last_line and now - last_print > 0.2:
                print(line)
                last_line, last_print = line, now
    except KeyboardInterrupt:
        pass
    finally:
        probe.stop()
        if meter is not None:
            meter.stop()


if __name__ == "__main__":
    main()
