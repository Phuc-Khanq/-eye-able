"""Starts the engine: threads for camera and vision, audio, Whisper and UI-Venus,
plus the asyncio loop for the WebSocket server. Run with: python -m engine.main

    python -m engine.main                  # the real mic, camera and cursor
    python -m engine.main --wav clip.wav   # a recording instead of the mic, played in real time
    python -m engine.main --pointer hybrid # gaze jumps, head fine-tunes (needs a gaze calibration)
    python -m engine.main --no-eyes        # voice only: no camera, and nothing clicks or types
    python -m engine.main --stdin          # also take typed lines as if heard (testing without the mic;
                                           # they go straight to voice.transcript, so the mouth gate never sees them)
    python -m engine.main --confirm nod    # say yes with a nod instead of an eyebrow raise (or "both")
    python -m engine.main --dwell          # resting the cursor 1 s also left-clicks (off by default: the left
                                           # click is an eyebrow raise or "click", snapped by UI-Venus)
    python -m engine.main --no-rewrite     # type dictation as heard, without the clean-up model
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading

# Quieter console: MediaPipe's C++ warnings and the model loading bars are noise in a live run (and PowerShell
# shows every stderr line in red). Real errors still print. Set before MediaPipe or transformers load.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

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
                 dwell: bool = False, confirm: str = "brow", rewrite: bool = False, rewriter=None) -> None:
        self.bus = bus
        self.warm_up = warm_up
        # Start order matters only for subscriptions; nothing is published until start().
        # 0. dictation clean-up (Khoi): a small local model (Qwen2.5-1.5B on the GPU) tidies the preview
        #    before it's typed; "yes" waits at most rewrite_wait_s for it. Off by default (rewrite=False),
        #    so tests never load a model; run() turns it on from settings.rewrite_dictation.
        self.rewrite = None
        if rewrite:
            from engine.voice.rewrite import Rewriter, RewriteWorker
            self.rewrite = RewriteWorker(bus, rewriter or Rewriter(settings), settings=settings)
        # 1. core: the mode manager turns gaze, gestures and transcripts into actions
        self.mode_manager = ModeManager(
            bus, settings=settings,
            rewrite_wait_s=float(getattr(settings, "rewrite_wait_s", 1.5)) if rewrite else None)
        # 2. eyes (Khang and Hung): camera -> face_mesh -> eye_features / head_pose -> pointer ->
        #    the cursor, plus head_gestures; UI-Venus answers target requests; os_output carries out
        #    topics.OS_* on Windows. Off by default (eyes=False), so tests never touch the camera,
        #    the cursor or the keyboard; run() turns it on.
        self.eyes = self.targets = self.os_output = self.mouth = self.meter = self.card = None
        if eyes:
            from engine.eyes.mouth import MouthTracker
            from engine.eyes.os_output import OsOutput
            from engine.eyes.service import EyesService
            from engine.eyes.target_service import TargetService
            self.eyes = EyesService(bus, settings, pointer=pointer, dwell=dwell, confirm=confirm)
            self.targets = TargetService(bus, settings, probe=self.eyes.probe)   # snap: on a link already?
            self.os_output = OsOutput(bus, driver=self.eyes.driver, scroller=self.eyes.scroller)
            # Was the user's mouth moving? For the voice route's "only the user's voice" gate.
            self.mouth = MouthTracker(bus)
            self.mode_manager.mouth = self.mouth     # an eyebrow raise while talking is ignored
            card = bool(getattr(settings, "dictation_card", True))
            if getattr(settings, "dwell_meter", True) or card:
                from engine.eyes.dwell_meter import DwellMeter
                self.meter = DwellMeter(bus)         # the dwell ring at the cursor, without the Electron UI
            if card:
                from engine.eyes.dictation_card import DictationCard
                self.card = DictationCard(bus, hide_from_capture=bool(getattr(settings, "card_hide_from_capture", True)),
                                          opacity=float(getattr(settings, "card_opacity", 0.82)),
                                          position=str(getattr(settings, "card_position", "top")))
                self.meter.add_window(self.card.attach)   # what was heard and what will be typed, on screen
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
            if self.meter:
                self.meter.start()                   # its own Tk thread (Windows only)
        self.dictation.start(warm_up=self.warm_up)   # Whisper loads on its own thread
        if self.rewrite:
            self.rewrite.start(warm_up=self.warm_up)  # the clean-up model loads on its own thread
        if listen:
            try:
                self.audio_in.start()
            except Exception as exc:                 # no mic: say so, keep the rest running
                self._status("error", f"No microphone: {exc}")

    def stop(self) -> None:
        self.audio_in.stop()
        self.dictation.stop()
        if self.rewrite:
            self.rewrite.stop()
        self.mode_manager.stop()
        if self.eyes:
            self.eyes.stop()
            self.targets.stop()
            self.os_output.stop()
            if self.meter:
                self.meter.stop()

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))


def cuda_device_count() -> int:
    """NVIDIA GPUs this laptop has (0 if none or no driver). Needs no torch."""
    try:
        import ctranslate2  # comes with faster-whisper; importing it needs no GPU

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def use_cpu_without_gpu(s, cuda_devices: int | None = None) -> bool:
    """Whisper set to CUDA on a laptop with no NVIDIA GPU can't load, so voice would not work.
    Use the CPU (int8) for this run instead and say so. Returns True if it switched.
    The defaults in config.py stay as they are (CONTRIBUTING rule 3)."""
    if s.whisper_device != "cuda":
        return False
    if cuda_devices is None:
        cuda_devices = cuda_device_count()
    if cuda_devices > 0:
        return False
    s.whisper_device, s.whisper_compute_type = "cpu", "int8"
    print("No NVIDIA GPU found: Whisper runs on the CPU (int8) this time. To make that "
          'permanent, put {"whisper_device": "cpu", "whisper_compute_type": "int8"} '
          "in config.local.json.")
    return True


def keep_rewrite_without_gpu(s, rewrite: bool, cuda_devices: int | None = None) -> bool:
    """The dictation clean-up model set to CUDA can't load without an NVIDIA GPU: it would
    download about 3 GB and then fail quietly. On the CPU it takes seconds per sentence, longer
    than rewrite_max_s, so it would only slow Whisper down. So with no GPU it's off for this run
    (the user's own words are shown, as when the clean-up is off). rewrite_device "cpu" in
    config.local.json still forces it on. Returns whether the clean-up stays on."""
    if not rewrite or not str(getattr(s, "rewrite_device", "cuda")).startswith("cuda"):
        return rewrite
    if cuda_devices is None:
        cuda_devices = cuda_device_count()
    if cuda_devices > 0:
        return True
    print("No NVIDIA GPU found: the dictation clean-up is off this time (it needs a GPU to keep up). "
          'To run it on the CPU anyway (slow), put {"rewrite_device": "cpu"} in config.local.json.')
    return False


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
            print("[snap] clicking at the cursor..." if m.get("snap") else f"[ui-venus] looking for '{m['label']}'...")
        elif t == messages.TARGET_FOUND:
            print(f"[snap] click at ({m['x']}, {m['y']})" if m.get("snap")
                  else f"[ui-venus] found '{m['label']}' at ({m['x']}, {m['y']})")
        elif t == messages.TARGET_NONE:
            if m.get("snap") and m.get("timeout"):
                print("[snap] gave up waiting for UI-Venus: no click")
            elif m.get("snap"):
                print("[snap] nothing to snap to: clicking where the cursor is")
            else:
                print(f"[ui-venus] couldn't find '{m['label']}'")
        elif t == messages.CONFIRM:
            hint = (f"   (raise your eyebrows, or say \"yes\" / \"click it\", within {CONFIRM_TIMEOUT_S:.0f} s; "
                    "\"no\" or a head shake cancels)" if m["state"] == "open" else "")
            print(f"[question] {m['prompt']} -> {m['state']}{hint}")
        elif t == messages.DICTATION_PREVIEW and m["words"]:
            text = " ".join(w["text"] for w in m["words"])
            print(f"[preview] {text}   (raise your eyebrows or say \"yes\" to type it, \"no\" to drop it)")
            if m.get("cleaned"):
                print(f"          cleaned up from: {m.get('original', '')!r}   (\"use original\" types your own words)")
        elif t == messages.CALIB_RESULT:
            print(f"[calibrate] {'done' if m['ok'] else 'canceled'}")

    bus.subscribe(topics.TO_UI, on_ui)
    bus.subscribe(topics.TRANSCRIPT, lambda tr: print(f"[heard] \"{tr['text']}\""))
    bus.subscribe(topics.GESTURE, lambda g: print(f"[gesture] {g.type}"))
    bus.subscribe(topics.OS_CLICK, lambda p: print(f"[click] ({p['x']}, {p['y']})"))
    bus.subscribe(topics.OS_TYPE, lambda p: print(f"[type] {p['text']!r}"))
    bus.subscribe(topics.OS_KEY, lambda p: print(f"[key] {p['key']}"))
    bus.subscribe(topics.OS_SCROLL, lambda p: print(f"[scroll] {p['direction']}"))
    bus.subscribe(topics.REWRITE_RESULT, lambda r: print(
        f"[clean-up] {'used' if r['used'] else 'not used (kept your words)'} in {r['ms']:.0f} ms"))


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


async def run(wav: str | None = None, eyes: bool = False, pointer: str = "head", dwell: bool = False,
              stdin: bool = False, confirm: str = "brow", rewrite: bool = False) -> None:
    bus = EventBus()
    relay = WebSocketRelay(bus)
    await relay.start(settings.ws_host, settings.ws_port)
    log_to_console(bus)
    if getattr(settings, "stt_backend", "whisper") == "whisper":
        use_cpu_without_gpu(settings)
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
                        eyes=eyes, pointer=pointer, dwell=dwell, confirm=confirm, rewrite=rewrite)
        engine.start(listen=False)
        threading.Thread(target=play_wav, args=(engine.audio_in, data, rate, stop), daemon=True).start()
    else:
        engine = Engine(bus, settings, eyes=eyes, pointer=pointer, dwell=dwell, confirm=confirm, rewrite=rewrite)
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
    p.add_argument("--dwell", action=argparse.BooleanOptionalAction, default=False,
                   help="resting the cursor dwell_ms (settings, 1 s) also left-clicks; off by default (the left click "
                        "is an eyebrow raise or \"click\", snapped onto the nearest button by UI-Venus)")
    p.add_argument("--blink", action=argparse.BooleanOptionalAction, default=bool(getattr(settings, "blink_click", False)),
                   help="a long, deliberate blink (0.4 to 1.2 s) is also a click, like an eyebrow raise; off by "
                        "default until tests.blink_click_check passes on your face (blink_click in settings)")
    p.add_argument("--stdin", action="store_true", help="typed lines count as heard speech (testing)")
    p.add_argument("--confirm", choices=["brow", "nod", "both"], default="brow",
                   help="what says yes besides your voice: eyebrow raise (default), nod, or either")
    p.add_argument("--rewrite", action=argparse.BooleanOptionalAction,
                   default=bool(getattr(settings, "rewrite_dictation", True)),
                   help="clean up dictated text with a small local model before it's typed (Qwen2.5-1.5B on the "
                        "GPU, downloaded once, about 3 GB); on by default (rewrite_dictation), --no-rewrite turns it off")
    args = p.parse_args()
    for stream in (sys.stdout, sys.stderr):   # a word the Windows console can't show prints as "?", never crashes
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    settings.blink_click = args.blink
    rewrite = keep_rewrite_without_gpu(settings, args.rewrite)
    try:
        asyncio.run(run(args.wav, eyes=not args.no_eyes, pointer=args.pointer, dwell=args.dwell,
                        stdin=args.stdin, confirm=args.confirm, rewrite=rewrite))
    except KeyboardInterrupt:
        print("Engine stopped.")
    except OSError as exc:
        raise SystemExit(f"Can't serve on port {settings.ws_port}: {exc}. "
                         "Is the engine or tests.fake_engine already running?")


if __name__ == "__main__":
    main()
