"""Starts the engine: threads for camera and vision, audio, Whisper and UI-Venus,
plus the asyncio loop for the WebSocket server. Run with: python -m engine.main

    python -m engine.main                  # the real mic
    python -m engine.main --wav clip.wav   # a recording instead of the mic, played in real time
"""
from __future__ import annotations

import argparse
import asyncio

from engine.config import settings
from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay
from engine.core.mode_manager import ModeManager
from engine.voice.audio_in import ArraySource, AudioIn
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.voice_data import VoiceData


class Engine:
    """Every module on one bus. Tests pass a fake Whisper, an audio array and a VAD."""

    def __init__(self, bus, settings=settings, dictation=None, audio_source=None, vad=None,
                 warm_up: bool = True) -> None:
        self.bus = bus
        self.warm_up = warm_up
        # Start order matters only for subscriptions; nothing is published until start().
        # 1. core: the mode manager turns gaze, gestures and transcripts into actions
        self.mode_manager = ModeManager(bus)
        # 2. eyes: TODO (Khang's route) camera -> face_mesh -> eye_features / head_pose ->
        #    calibration -> smoothing -> os_input, plus head_gestures and screen_target (UI-Venus).
        #    Until then nothing consumes topics.OS_*, so clicks and typing don't reach Windows.
        # 3. voice: audio_in -> dictation -> transcripts for the mode manager
        self.dictation = DictationWorker(bus, dictation or Dictation(settings))
        self.audio_in = AudioIn(bus, settings, source=audio_source, vad=vad)
        self.voice_data = VoiceData(bus, self.dictation.dictation)   # "delete my voice data"

    def start(self) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.MODE, mode=self.mode_manager.mode.value))
        self.mode_manager.start()
        self.dictation.start(warm_up=self.warm_up)   # Whisper loads on its own thread
        self.audio_in.start()                        # no mic -> a status message, not a crash

    def stop(self) -> None:
        self.audio_in.stop()
        self.dictation.stop()
        self.mode_manager.stop()


async def run(wav: str | None = None) -> None:
    bus = EventBus()
    relay = WebSocketRelay(bus)
    await relay.start(settings.ws_host, settings.ws_port)
    bus.subscribe(topics.TO_UI, lambda m: m["type"] == messages.STATUS
                  and print(f"[{m['level']}] {m['text']}"))
    source = ArraySource.from_wav(wav, realtime=True) if wav else None
    engine = Engine(bus, settings, audio_source=source)
    engine.start()
    print(f"Diseyeble engine on ws://{settings.ws_host}:{relay.port}  (Ctrl+C to stop)")
    try:
        await relay.serve_forever()
    finally:
        engine.stop()
        await relay.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Diseyeble engine")
    p.add_argument("--wav", help="use this recording instead of the mic")
    args = p.parse_args()
    try:
        asyncio.run(run(args.wav))
    except KeyboardInterrupt:
        print("Engine stopped.")
    except OSError as exc:
        raise SystemExit(f"Can't serve on port {settings.ws_port}: {exc}. "
                         "Is the engine or tests.fake_engine already running?")


if __name__ == "__main__":
    main()
