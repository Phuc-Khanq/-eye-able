"""Records real engine messages to tests/recordings/*.jsonl for fake_engine.py.
Records messages only: no camera frames or audio.

Start the engine, then in a second terminal:
    python -m tests.record_events --out tests/recordings/my_session.jsonl
Use the app, then press Ctrl+C. Each line is {"t": seconds since start, "msg": {...}}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from engine.core import messages

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "tests" / "recordings"

# Never written to disk: these carry camera images, screenshots or face points (rule 10).
DROPPED = {messages.DEBUG_FRAME, messages.TARGET_DEBUG, messages.DEBUG_FACE}


def keep(msg: dict) -> bool:
    return isinstance(msg, dict) and isinstance(msg.get("type"), str) and msg["type"] not in DROPPED


def line(t: float, msg: dict) -> str:
    return json.dumps({"t": round(t, 3), "msg": msg}, separators=(",", ":")) + "\n"


async def record(url: str, out: Path, max_messages: int | None = None,
                 seconds: float | None = None) -> int:
    """Write messages from the engine at `url` to `out`. Returns how many were kept."""
    from websockets.asyncio.client import connect

    out.parent.mkdir(parents=True, exist_ok=True)
    kept, t0 = 0, time.monotonic()
    async with connect(url, max_size=None) as ws:
        with out.open("w", encoding="utf-8") as f:
            while max_messages is None or kept < max_messages:
                timeout = None if seconds is None else max(0.0, seconds - (time.monotonic() - t0))
                try:
                    text = await asyncio.wait_for(ws.recv(), timeout)
                except asyncio.TimeoutError:
                    break
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if keep(msg):
                    f.write(line(time.monotonic() - t0, msg))
                    kept += 1
    return kept


def main() -> None:
    from engine.config import settings

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=f"ws://{settings.ws_host}:{settings.ws_port}")
    p.add_argument("--out", type=Path, default=RECORDINGS / f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    args = p.parse_args()
    print(f"Recording {args.url} -> {args.out}  (Ctrl+C to stop)")
    try:
        n = asyncio.run(record(args.url, args.out, seconds=args.seconds))
    except KeyboardInterrupt:
        print(f"Stopped. Saved to {args.out}")
        return
    except OSError as exc:
        raise SystemExit(f"Can't reach the engine at {args.url}: {exc}. Is it running?")
    print(f"Saved {n} messages to {args.out}")


if __name__ == "__main__":
    main()
