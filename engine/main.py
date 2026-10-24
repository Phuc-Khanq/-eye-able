"""Starts the engine: threads for camera and vision, audio, Whisper and UI-Venus,
plus the asyncio loop for the WebSocket server. Run with: python -m engine.main

    python -m engine.main                  # the real mic
    python -m engine.main --wav clip.wav   # a recording instead of the mic, played in real time
"""
from __future__ import annotations

import argparse
import asyncio
import threading

from engine.config import settings
from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay
from engine.core.mode_manager import ModeManager
from engine.voice.audio_in import AudioIn
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.voice_data import VoiceData


class Engine:
    """Every module on one bus. Tests pass a fake Whisper, sounddevice and VAD."""

    def __init__(self, bus, settings=settings, dictation=None, sd=None, mic_settings=None,
                 vad=None, warm_up: bool = True) -> None:
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
        self.audio_in = AudioIn(bus, settings, sd=sd, mic_settings=mic_settings, vad=vad)
        self.voice_data = VoiceData(bus)             # "delete my voice data"

    def start(self, listen: bool = True) -> None:
        """listen=False skips the mic, for --wav and tests (feed audio_in.process instead)."""
        self.bus.publish(topics.TO_UI, messages.make(messages.MODE, mode=self.mode_manager.mode.value))
        self.mode_manager.start()
        self.dictation.start(warm_up=self.warm_up)   # Whisper loads on its own thread
        if listen:
            try:
                self.audio_in.start()
            except Exception as exc:                 # no mic: say so, keep the rest running
                self._status("error", f"No microphone: {exc}")

    def stop(self) -> None:
        self.audio_in.stop()
        self.dictation.stop()
        self.mode_manager.stop()

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))


def read_wav(path: str):
    """(rate, float32 audio) from a WAV file. Only reads; nothing is written."""
    import numpy as np
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return rate, data.astype(np.float32)


def play_wav(audio_in: AudioIn, data, rate: int, stop: threading.Event, realtime: bool = True) -> None:
    """Feed audio to audio_in in real time, as if it came from the mic, then enough
    silence for the pause to end the last sentence."""
    import numpy as np

    tail = np.zeros((int(rate * (audio_in.pause_seconds + 0.5)),) + data.shape[1:], np.float32)
    data = np.concatenate([data, tail])
    step = int(rate * 0.032)
    for i in range(0, len(data), step):
        if stop.wait(0.032 if realtime else 0):
            return
        audio_in.process(data[i:i + step])


async def run(wav: str | None = None) -> None:
    bus = EventBus()
    relay = WebSocketRelay(bus)
    await relay.start(settings.ws_host, settings.ws_port)
    bus.subscribe(topics.TO_UI, lambda m: m["type"] == messages.STATUS
                  and print(f"[{m['level']}] {m['text']}"))
    stop = threading.Event()
    if wav:
        from engine.voice.audio_in import load_vad
        from engine.voice.mic import MicSettings

        rate, data = read_wav(wav)
        channels = 1 if data.ndim == 1 else data.shape[1]
        try:
            vad = load_vad(settings.pause_seconds)
        except Exception as exc:
            raise SystemExit(f"Can't load the speech detector: {exc}")
        engine = Engine(bus, settings, mic_settings=MicSettings(-1, wav, "file", rate, channels), vad=vad)
        engine.start(listen=False)
        threading.Thread(target=play_wav, args=(engine.audio_in, data, rate, stop), daemon=True).start()
    else:
        engine = Engine(bus, settings)
        engine.start()
    print(f"Diseyeble engine on ws://{settings.ws_host}:{relay.port}  (Ctrl+C to stop)")
    try:
        await relay.serve_forever()
    finally:
        stop.set()
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
