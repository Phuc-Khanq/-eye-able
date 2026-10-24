"""Replays recorded engine messages over ws://127.0.0.1:8765 so the UI works without a camera.
Route: Khoi and Khoa (Khoa leads). Run with: python -m tests.fake_engine

    python -m tests.fake_engine                                   # every tests/recordings/*.jsonl, on loop
    python -m tests.fake_engine tests/recordings/demo_flow.jsonl --speed 2

It uses the real WebSocket relay from engine/core/events.py, so the UI sees exactly what
the engine would send. Messages the UI sends back are printed, so you can check buttons.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay

RECORDINGS = Path(__file__).resolve().parent / "recordings"
UI_MESSAGES = (messages.SET_MODE, messages.CALIBRATE, messages.MENU_SELECT, messages.SETTINGS,
               messages.DICTATION, messages.TARGET, messages.CONSOLE, messages.CONFIRM_ANSWER)
MAX_GAP_S = 3.0   # a long quiet stretch in a recording is shortened to this


def load(path: Path) -> list[tuple[float, dict]]:
    """[(t, msg), ...] from a recording, skipping lines that aren't valid."""
    events = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
            events.append((float(row["t"]), dict(row["msg"])))
        except (ValueError, KeyError, TypeError):
            continue
    return events


async def replay(bus, events: list[tuple[float, dict]], speed: float = 1.0, sleep=asyncio.sleep) -> None:
    """Publish each message on topics.TO_UI with the recorded gaps between them."""
    last = events[0][0] if events else 0.0
    for t, msg in events:
        gap = min(max(t - last, 0.0), MAX_GAP_S) / speed
        if gap:
            await sleep(gap)
        bus.publish(topics.TO_UI, msg)
        last = t


async def run(files: list[Path], host: str, port: int, speed: float, loop: bool = True) -> None:
    bus = EventBus()
    for msg_type in UI_MESSAGES:
        bus.subscribe(topics.from_ui(msg_type), lambda msg: print(f"  UI -> engine: {msg}"))
    relay = WebSocketRelay(bus)
    await relay.start(host, port)
    print(f"Fake engine on ws://{host}:{relay.port}  (Ctrl+C to stop)")
    try:
        while True:
            for path in files:
                print(f"Replaying {path.name}")
                await replay(bus, load(path), speed)
            if not loop:
                break
            await asyncio.sleep(1.0)
    finally:
        await relay.close()


def main() -> None:
    from engine.config import settings

    p = argparse.ArgumentParser(description="Replay recorded engine messages to the UI.")
    p.add_argument("files", nargs="*", type=Path, help="default: every tests/recordings/*.jsonl")
    p.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    p.add_argument("--once", action="store_true", help="play once instead of on loop")
    args = p.parse_args()
    files = args.files or sorted(RECORDINGS.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No recordings in {RECORDINGS}. Make one with: python -m tests.record_events")
    try:
        asyncio.run(run(files, settings.ws_host, settings.ws_port, args.speed, loop=not args.once))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        raise SystemExit(f"Can't serve on port {settings.ws_port}: {exc}. Is the engine already running?")


if __name__ == "__main__":
    main()
