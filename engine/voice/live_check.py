"""Talk into the mic and see what Whisper heard, with timings. Nothing is saved.

Route: Khoi and Khoa
Runs the real voice chain: mic -> audio_in (gain + Silero VAD) -> DictationWorker -> transcript.
Uses the Whisper settings from engine/config.py plus config.local.json.

Run from the repo root (Ctrl+C to stop):
    python -m engine.voice.live_check
    python -m engine.voice.live_check --pause 1.0     # end an utterance after 1 s of silence
    python -m engine.voice.live_check --debug         # show level, gain and speech probability
    python -m engine.voice.live_check --seconds 60    # stop by itself after 60 s

Each result shows:
    speech    how long you spoke (the utterance audio)
    whisper   time Whisper took (the transcript's "ms")
    after     time from the end of the utterance to the printed text

It keeps listening until Ctrl+C (or --seconds), and warns if the mic sends no audio
or only silence, so it never just sits there quietly.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from engine.core import messages, topics
from engine.core.events import EventBus

NO_AUDIO_WARN_S = 3.0     # no chunks at all from the mic
SILENT_WARN_S = 8.0       # chunks arrive but the level stays near zero
SILENT_LEVEL = 0.15       # 0-1 meter: about -51 dBFS
DEBUG_EVERY_S = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pause", type=float, default=None,
                        help="seconds of silence that end an utterance (1 to 4; default from settings)")
    parser.add_argument("--mic", type=int, default=None, help="mic number from 'python -m engine.voice.mic'")
    parser.add_argument("--seconds", type=float, default=None, help="stop by itself after this long")
    parser.add_argument("--debug", action="store_true",
                        help="print level, gain and Silero speech probability twice a second")
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

    class ReportingDictation(Dictation):
        """Dictation that also says when Whisper's answer was dropped (the app stays quiet)."""

        def transcribe(self, audio):
            words = super().transcribe(audio)
            if not words:
                with lock:
                    _, secs = sent_at.pop(0) if sent_at else (0.0, 0.0)
                print(f"  (Whisper heard nothing it trusts in {secs:.1f} s of audio: dropped. "
                      "Speak a bit louder or longer.)", flush=True)
            return words

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
    worker = DictationWorker(bus, ReportingDictation(settings))
    bus.subscribe(topics.TRANSCRIPT, on_transcript)
    bus.subscribe(topics.TO_UI, on_ui)

    model = resolve_model(settings.whisper_model)
    print(f"Loading Whisper {model} on {settings.whisper_device} ({settings.whisper_compute_type})...",
          flush=True)
    t0 = time.perf_counter()
    try:
        worker.dictation.load()
        Dictation.transcribe(worker.dictation, __import__("numpy").zeros(16000, "float32"))  # warm up
    except Exception as exc:
        print(f"Whisper could not load: {exc}")
        return 1
    print(f"Loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    worker.start(warm_up=False)

    try:
        chosen = mic.open_settings(preferred=args.mic, verbose=True) if args.mic is not None else None
        audio_in = AudioIn(bus, settings, mic_settings=chosen)
        audio_in.start()
    except Exception as exc:
        print(f"Could not open the mic: {exc}")
        return 1
    print(f"Listening. Speak, then stay quiet {audio_in.pause_seconds:.1f} s. Ctrl+C to stop.\n",
          flush=True)

    t_end = time.monotonic() + args.seconds if args.seconds else None
    started = time.monotonic()
    last_chunks, last_chunk_t = 0, started
    loud_t = started
    warned_no_audio = warned_silent = False
    next_debug = started
    try:
        while t_end is None or time.monotonic() < t_end:
            time.sleep(0.1)
            now = time.monotonic()
            if audio_in.chunks != last_chunks:
                last_chunks, last_chunk_t = audio_in.chunks, now
                warned_no_audio = False
            elif now - last_chunk_t > NO_AUDIO_WARN_S and not warned_no_audio:
                print("  WARNING: the mic sends no audio. Another app may hold it, or it was "
                      "unplugged. Try --mic <number> (list: python -m engine.voice.mic).", flush=True)
                warned_no_audio = True
            if audio_in.level >= SILENT_LEVEL:
                loud_t, warned_silent = now, False
            elif now - loud_t > SILENT_WARN_S and not warned_silent:
                print("  WARNING: the mic hears almost nothing. It is probably MUTED in Windows:\n"
                      "    press the mic-mute key on the keyboard (a microphone icon), or open\n"
                      "    Settings > System > Sound > Input, pick your microphone and unmute it.\n"
                      "    Still silent? Try --mic <number> (list: python -m engine.voice.mic) "
                      "or --debug to watch the level.", flush=True)
                warned_silent = True
            if args.debug and now >= next_debug:
                next_debug = now + DEBUG_EVERY_S
                bar = "#" * int(audio_in.level * 20)
                print(f"  level {audio_in.level:4.2f} {bar:<20} gain {audio_in._gain.gain:4.1f}x "
                      f"speech {audio_in.speech_prob:4.2f} "
                      f"{'SPEAKING' if audio_in.speaking else ''}", flush=True)
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
