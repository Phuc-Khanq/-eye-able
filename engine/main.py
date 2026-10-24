"""Starts the engine: threads for camera and vision, audio, Whisper and UI-Venus,
plus the asyncio loop for the WebSocket server. Run with: python -m engine.main

    python -m engine.main                  # the real mic, camera and cursor
    python -m engine.main --wav clip.wav   # a recording instead of the mic, played in real time
    python -m engine.main --pointer hybrid # gaze jumps, head fine-tunes (needs a gaze calibration)
    python -m engine.main --no-eyes        # voice only: no camera, and nothing clicks or types
    python -m engine.main --stdin          # also take typed lines as if heard (testing without the mic;
                                           # they go straight to voice.transcript, so the mouth gate never sees them)
    python -m engine.main --confirm nod    # say yes with a nod instead of an eyebrow raise (or "both")
    python -m engine.main --no-dwell       # resting the cursor doesn't click (dwell = left click is on by default)
"""
from __future__ import annotations

import argparse
import asyncio
import threading

from engine.config import settings
from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay
from engine.core.mode_manager import CONFIRM_TIMEOUT_S, ModeManager
from engine.voice.audio_in import AudioIn
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.voice_data import VoiceData


class Engine:
    """Every module on one bus. Tests pass a fake Whisper, sounddevice and VAD."""

    def __init__(self, bus, settings=settings, dictation=None, sd=None, mic_settings=None,
                 vad=None, warm_up: bool = True, eyes: bool = False, pointer: str = "head",
                 dwell: bool = True, confirm: str = "brow") -> None:
        self.bus = bus
        self.warm_up = warm_up
        # Start order matters only for subscriptions; nothing is published until start().
        # 1. core: the mode manager turns gaze, gestures and transcripts into actions
        self.mode_manager = ModeManager(bus)
        # 2. eyes (Khang and Hung): camera -> face_mesh -> eye_features / head_pose -> pointer ->
        #    the cursor, plus head_gestures; UI-Venus answers target requests; os_output carries out
        #    topics.OS_* on Windows. Off by default (eyes=False), so tests never touch the camera,
        #    the cursor or the keyboard; run() turns it on.
        self.eyes = self.targets = self.os_output = self.mouth = None
        if eyes:
            from engine.eyes.mouth import MouthTracker
            from engine.eyes.os_output import OsOutput
            from engine.eyes.service import EyesService
            from engine.eyes.target_service import TargetService
            self.eyes = EyesService(bus, settings, pointer=pointer, dwell=dwell, confirm=confirm)
            self.targets = TargetService(bus, settings)
            self.os_output = OsOutput(bus, driver=self.eyes.driver, scroller=self.eyes.scroller)
            # Was the user's mouth moving? For the voice route's "only the user's voice" gate.
            self.mouth = MouthTracker(bus)
        # 3. voice: audio_in -> dictation -> transcripts for the mode manager
        #    (without eyes, mouth is None: no gate, every word counts)
        self.dictation = DictationWorker(bus, dictation or Dictation(settings), mouth=self.mouth,
                                         settings=settings)
        self.audio_in = AudioIn(bus, settings, sd=sd, mic_settings=mic_settings, vad=vad)
        self.voice_data = VoiceData(bus)             # "delete my voice data"

    def start(self, listen: bool = True) -> None:
        """listen=False skips the mic, for --wav and tests (feed audio_in.process instead)."""
        self.bus.publish(topics.TO_UI, messages.make(messages.MODE, mode=self.mode_manager.mode.value))
        self.mode_manager.start()
        if self.eyes:
            self.targets.start()                     # UI-Venus loads on its own worker thread
            self.eyes.start()                        # camera thread and the cursor glide
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
        if self.eyes:
            self.eyes.stop()
            self.targets.stop()
            self.os_output.stop()

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


def log_to_console(bus) -> None:
    """Until the UI overlay exists, print what the engine hears and does, so a test run isn't silent:
    what Whisper heard, mode changes, UI-Venus searches, questions waiting for a yes, the dictation
    preview, gestures, and every click and text typed."""
    def on_ui(m: dict) -> None:
        t = m["type"]
        if t == messages.STATUS:
            print(f"[{m['level']}] {m['text']}")
        elif t == messages.MODE:
            print(f"[mode] {m['mode']}")
        elif t == messages.TARGET_SEARCHING:
            print(f"[ui-venus] looking for '{m['label']}'...")
        elif t == messages.TARGET_FOUND:
            print(f"[ui-venus] found '{m['label']}' at ({m['x']}, {m['y']})")
        elif t == messages.TARGET_NONE:
            print(f"[ui-venus] couldn't find '{m['label']}'")
        elif t == messages.CONFIRM:
            hint = (f"   (raise your eyebrows, or say \"yes\" / \"click it\", within {CONFIRM_TIMEOUT_S:.0f} s; "
                    "\"no\" or a head shake cancels)" if m["state"] == "open" else "")
            print(f"[question] {m['prompt']} -> {m['state']}{hint}")
        elif t == messages.DICTATION_PREVIEW and m["words"]:
            text = " ".join(w["text"] for w in m["words"])
            print(f"[preview] {text}   (raise your eyebrows or say \"yes\" to type it, \"no\" to drop it)")
        elif t == messages.CALIB_RESULT:
            print(f"[calibrate] {'done' if m['ok'] else 'canceled'}")

    bus.subscribe(topics.TO_UI, on_ui)
    bus.subscribe(topics.TRANSCRIPT, lambda tr: print(f"[heard] \"{tr['text']}\""))
    bus.subscribe(topics.GESTURE, lambda g: print(f"[gesture] {g.type}"))
    bus.subscribe(topics.OS_CLICK, lambda p: print(f"[click] ({p['x']}, {p['y']})"))
    bus.subscribe(topics.OS_TYPE, lambda p: print(f"[type] {p['text']!r}"))
    bus.subscribe(topics.OS_KEY, lambda p: print(f"[key] {p['key']}"))
    bus.subscribe(topics.OS_SCROLL, lambda p: print(f"[scroll] {p['direction']}"))


def read_stdin(bus, stop: threading.Event, delay_s: float = 3.0) -> None:
    """--stdin: each typed line counts as heard speech after delay_s, so you can point at the
    target first. Typing goes to whichever window has focus, so test dictation with the mic."""
    import sys

    print(f"Type a command and press Enter (e.g. click search box); you then have {delay_s:.0f} s to point.")
    for line in sys.stdin:
        text = line.strip()
        if stop.is_set():
            return
        if text and not stop.wait(delay_s):
            bus.publish(topics.TRANSCRIPT, {"text": text, "words": [{"text": w, "conf": 1.0} for w in text.split()],
                                            "ms": 0.0})


async def run(wav: str | None = None, eyes: bool = False, pointer: str = "head", dwell: bool = True,
              stdin: bool = False, confirm: str = "brow") -> None:
    bus = EventBus()
    relay = WebSocketRelay(bus)
    await relay.start(settings.ws_host, settings.ws_port)
    log_to_console(bus)
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
        engine = Engine(bus, settings, mic_settings=MicSettings(-1, wav, "file", rate, channels), vad=vad,
                        eyes=eyes, pointer=pointer, dwell=dwell, confirm=confirm)
        engine.start(listen=False)
        threading.Thread(target=play_wav, args=(engine.audio_in, data, rate, stop), daemon=True).start()
    else:
        engine = Engine(bus, settings, eyes=eyes, pointer=pointer, dwell=dwell, confirm=confirm)
        engine.start()
    if stdin:
        threading.Thread(target=read_stdin, args=(bus, stop), daemon=True).start()
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
    p.add_argument("--no-eyes", action="store_true", help="no camera, cursor, clicks or typing (voice only)")
    p.add_argument("--pointer", choices=["head", "head-abs", "hybrid", "gaze"], default="head",
                   help="head = head as a mouse with acceleration (default); head-abs = fixed head angle per "
                        "screen spot; hybrid and gaze need a gaze calibration (tests.gaze_dot)")
    p.add_argument("--dwell", action=argparse.BooleanOptionalAction, default=True,
                   help="resting the cursor dwell_ms (settings, 1 s) left-clicks; on by default, --no-dwell turns it off")
    p.add_argument("--stdin", action="store_true", help="typed lines count as heard speech (testing)")
    p.add_argument("--confirm", choices=["brow", "nod", "both"], default="brow",
                   help="what says yes besides your voice: eyebrow raise (default), nod, or either")
    args = p.parse_args()
    try:
        asyncio.run(run(args.wav, eyes=not args.no_eyes, pointer=args.pointer, dwell=args.dwell,
                        stdin=args.stdin, confirm=args.confirm))
    except KeyboardInterrupt:
        print("Engine stopped.")
    except OSError as exc:
        raise SystemExit(f"Can't serve on port {settings.ws_port}: {exc}. "
                         "Is the engine or tests.fake_engine already running?")


if __name__ == "__main__":
    main()
