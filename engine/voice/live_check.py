"""Talk into the mic and see what Whisper heard, with timings. Nothing is saved.

Route: Khoi and Khoa
Runs the real voice chain: mic -> audio_in (gain + Silero VAD) -> DictationWorker -> transcript.
Uses the Whisper settings from engine/config.py plus config.local.json.

Run from the repo root (Ctrl+C to stop):
    python -m engine.voice.live_check
    python -m engine.voice.live_check --pause 1.0     # end an utterance after 1 s of silence

Each line shows:
    speech    how long you spoke (the utterance audio)
    whisper   time Whisper took (the transcript's "ms")
    after     time from the end of the utterance to the printed text
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from engine.core import messages, topics
from engine.core.events import EventBus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pause", type=float, default=None,
                        help="seconds of silence that end an utterance (1 to 4; default from settings)")
    parser.add_argument("--mic", type=int, default=None, help="mic number from 'python -m engine.voice.mic'")
    parser.add_argument("--seconds", type=float, default=None, help="stop by itself after this long")
    args = parser.parse_args(argv)

    from engine.config import settings
    from engine.voice import mic
    from engine.voice.audio_in import AudioIn
    from engine.voice.dictation import Dictation, DictationWorker, resolve_model

    if args.pause is not None:
        settings.pause_seconds = args.pause
    bus = EventBus()
    sent_at: list[tuple[float, float]] = []      # (time an utterance was published, its seconds)
    lock = threading.Lock()

    def on_utterance(audio) -> None:
        with lock:
            sent_at.append((time.perf_counter(), len(audio) / 16000))

    def on_transcript(t: dict) -> None:
        now = time.perf_counter()
        with lock:
            sent, secs = sent_at.pop(0) if sent_at else (now, 0.0)
        print(f'  "{t["text"]}"\n    speech {secs:4.1f} s | whisper {t["ms"]:6.0f} ms | '
              f"after {1000 * (now - sent):6.0f} ms", flush=True)

    def on_ui(msg: dict) -> None:
        if msg["type"] == messages.LISTENING:
            speaking = msg["speaking"]
            if speaking != on_ui.speaking:
                on_ui.speaking = speaking
                print("  [speaking...]" if speaking else "  [done, transcribing]", flush=True)
        elif msg["type"] == messages.STATUS:
            print(f'  {msg["level"]}: {msg["text"]}', flush=True)
    on_ui.speaking = False

    bus.subscribe(topics.UTTERANCE, on_utterance)          # before the worker: times are honest
    worker = DictationWorker(bus, Dictation(settings))
    bus.subscribe(topics.TRANSCRIPT, on_transcript)
    bus.subscribe(topics.TO_UI, on_ui)

    model = resolve_model(settings.whisper_model)
    print(f"Loading Whisper {model} on {settings.whisper_device} ({settings.whisper_compute_type})...",
          flush=True)
    t0 = time.perf_counter()
    try:
        worker.dictation.warm_up()
    except Exception as exc:
        print(f"Whisper could not load: {exc}")
        return 1
    print(f"Loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    worker.start(warm_up=False)

    try:
        preferred = mic.open_settings(preferred=args.mic, verbose=True) if args.mic is not None else None
        audio_in = AudioIn(bus, settings, mic_settings=preferred)
        audio_in.start()
    except RuntimeError as exc:
        print(exc)
        return 1
    print(f"Listening. Pause {audio_in.pause_seconds:.1f} s ends an utterance. Ctrl+C to stop.\n",
          flush=True)
    t_end = time.monotonic() + args.seconds if args.seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    audio_in.stop()
    worker.stop()
    if audio_in.dropped_chunks:
        print(f"Warning: {audio_in.dropped_chunks} mic chunks dropped (the computer was too busy)")
    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
