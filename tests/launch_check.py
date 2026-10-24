"""Try "open Chrome" / "go to youtube.com" by hand: what launch.match() makes of a phrase, and with --go,
what really opens on this laptop.

    python -m tests.launch_check "open notepad"
    python -m tests.launch_check "open settings" "go to youtube.com" "search for cats" --go

Without --go nothing starts: it only prints the os.launch payload (None = look-and-say keeps it).
With --go the payload goes on a bus to the real OsOutput and its real launcher (os.startfile /
webbrowser), with a fake os_api, so nothing clicks or types.
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from engine.config import settings
from engine.core import messages, topics
from engine.core.events import EventBus
from engine.eyes.os_output import OsOutput
from engine.voice.launch import match


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phrases", nargs="+", help="what the user says, e.g. \"open notepad\"")
    p.add_argument("--go", action="store_true", help="really open it (otherwise only print the payload)")
    args = p.parse_args(argv)

    bus = out = None
    if args.go:
        bus = EventBus()
        bus.subscribe(topics.TO_UI, lambda m: m.get("type") == messages.STATUS
                      and print(f"    status {m['level']}: {m['text']}", flush=True))
        no_clicks = SimpleNamespace()       # OsOutput needs an os_api; launching never uses it
        out = OsOutput(bus, os_api=no_clicks, settings=settings)
    for phrase in args.phrases:
        payload = match(phrase, settings)
        print(f"{phrase!r} -> {payload}" + ("" if payload else "  (not for us: look-and-say)"), flush=True)
        if args.go and payload:
            bus.publish(topics.OS_LAUNCH, payload)
            out.flush(timeout=15)
    if out is not None:
        out.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
