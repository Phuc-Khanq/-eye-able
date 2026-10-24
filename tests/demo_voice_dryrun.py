"""The demo script with REAL speech recognition: synthetic clips -> audio_in (Silero VAD)
-> Whisper -> mode manager, through engine.main.Engine on the real threaded EventBus.
Only the camera (gaze, gestures) and UI-Venus are faked, injected after each transcript.

Slow (real Whisper), so it isn't a unittest. Run from the repo root after making the clips
(scripts/make_test_speech.ps1):
    python -m tests.demo_voice_dryrun                    # 5 runs, david, normal clips
    python -m tests.demo_voice_dryrun --runs 1 --variant slow --voice zira
    python -m tests.demo_voice_dryrun --pause 3.0
Exit code 0 only if every run did exactly the demo's OS actions.
"""
from __future__ import annotations

import argparse
import dataclasses
import threading
import time

from engine.core import topics
from engine.core.events import EventBus
from engine.core.mode_manager import Mode
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.voice_check import SYNTH, normalize_words
from tests.test_demo_flow import EXPECTED_OS, SCRIPT

TRANSCRIPT_TIMEOUT_S = 20.0   # CPU Whisper on one short utterance


class Watch:
    """Counts utterances and handled transcripts, and keeps every os.* action."""

    def __init__(self, bus) -> None:
        self.lock = threading.Lock()
        self.utterances = self.transcripts = 0
        self.heard: list[str] = []
        self.os: list[tuple[str, dict]] = []
        self.requests: list = []
        self.changed = threading.Condition(self.lock)
        bus.subscribe(topics.UTTERANCE, self._utterance)
        # Subscribed after the mode manager, so this runs once it has handled the transcript.
        bus.subscribe(topics.TRANSCRIPT, self._transcript)
        bus.subscribe(topics.TARGET_REQUEST, self.requests.append)
        for t in (topics.OS_CLICK, topics.OS_TYPE, topics.OS_KEY, topics.OS_SCROLL, topics.OS_UNDO):
            bus.subscribe(t, lambda p, t=t: self._os(t, p))

    def _utterance(self, _audio) -> None:
        with self.changed:
            self.utterances += 1
            self.changed.notify_all()

    def _transcript(self, t: dict) -> None:
        with self.changed:
            self.transcripts += 1
            self.heard.append(t["text"])
            self.changed.notify_all()

    def _os(self, topic: str, payload: dict) -> None:
        with self.lock:
            self.os.append((topic, payload))

    def wait_for_transcripts(self, n: int, timeout: float) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: self.transcripts >= n, timeout)


def os_matches(got: list, expected: list) -> bool:
    """Exact, except typed text is compared like voice_check scores it (Whisper adds case and a period)."""
    if [t for t, _ in got] != [t for t, _ in expected]:
        return False
    for (topic, g), (_, e) in zip(got, expected):
        if topic == topics.OS_TYPE:
            if normalize_words(g["text"]) != normalize_words(e["text"]) or not g["text"].endswith(" "):
                return False
        elif g != e:
            return False
    return True


def run_once(dictation, settings, voice: str, variant: str) -> dict:
    from engine.main import Engine, play_wav
    from engine.voice.audio_in import load_vad
    from engine.voice.mic import MicSettings
    from scipy.io import wavfile

    bus = EventBus()
    engine = Engine(bus, settings, dictation=dictation, warm_up=False,
                    mic_settings=MicSettings(-1, "synthetic clips", "file", 16000, 1),
                    vad=load_vad(settings.pause_seconds))
    watch = Watch(bus)
    engine.start(listen=False)
    t0 = time.perf_counter()
    problems: list[str] = []
    said = 0
    try:
        for kind, value in SCRIPT:
            if kind == "look":
                bus.publish(topics.GAZE, GazePoint(0, *value, 0.9))
            elif kind == "gesture":
                bus.publish(topics.GESTURE, Gesture(0, value))
            elif kind == "ui_venus":
                label = watch.requests[-1].label if watch.requests else value.label
                bus.publish(topics.TARGET_RESULT, dataclasses.replace(value, label=label))
            elif kind == "say":
                said += 1
                rate, data = wavfile.read(SYNTH / f"demo_{said:03d}__{voice}__{variant}.wav")
                before = watch.transcripts
                play_wav(engine.audio_in, data.astype("float32") / 32768.0, rate, threading.Event(),
                         realtime=False)
                with watch.lock:
                    n = watch.utterances
                if n == 0 or not watch.wait_for_transcripts(before + 1, TRANSCRIPT_TIMEOUT_S):
                    problems.append(f"no transcript for {value!r}")
                # if VAD split the clip, let the rest of its transcripts land too
                watch.wait_for_transcripts(n, 5.0)
        mm = engine.mode_manager
        if not (mm.mode == Mode.POINTER and not mm.preview and mm.question is None):
            problems.append(f"ended in {mm.mode.value}, preview={bool(mm.preview)}, "
                            f"question={mm.question is not None}")
    finally:
        engine.stop()
    with watch.lock:
        os_actions = list(watch.os)
    if not os_matches(os_actions, EXPECTED_OS):
        problems.append(f"OS actions {os_actions} != expected {EXPECTED_OS}")
    time.sleep(0.6)                                   # worker threads stop within ~0.25 s
    alive = [t.name for t in threading.enumerate() if t is not threading.main_thread() and t.is_alive()]
    return {"ok": not problems, "problems": problems, "seconds": time.perf_counter() - t0,
            "heard": list(watch.heard), "alive": alive}


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.dictation import Dictation

    p = argparse.ArgumentParser(description="The demo script with real Whisper on synthetic speech.")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--voice", default="david")
    p.add_argument("--variant", default="normal")
    p.add_argument("--pause", type=float, default=None, help="pause_seconds (1 to 4)")
    args = p.parse_args(argv)
    if args.pause is not None:
        settings = dataclasses.replace(settings, pause_seconds=args.pause)

    dictation = Dictation(settings)
    t0 = time.perf_counter()
    dictation.load()
    print(f"Whisper {settings.whisper_model} on {settings.whisper_device} loaded in "
          f"{time.perf_counter() - t0:.1f} s; {args.voice}/{args.variant}, pause {settings.pause_seconds} s")
    passed = 0
    for i in range(1, args.runs + 1):
        r = run_once(dictation, settings, args.voice, args.variant)
        passed += r["ok"]
        print(f"run {i}: {'PASS' if r['ok'] else 'FAIL'} in {r['seconds']:.1f} s; heard {r['heard']}")
        for problem in r["problems"]:
            print(f"    {problem}")
        if r["alive"]:
            print(f"    threads still alive: {r['alive']}")
    print(f"{passed}/{args.runs} runs passed")
    return 0 if passed == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
