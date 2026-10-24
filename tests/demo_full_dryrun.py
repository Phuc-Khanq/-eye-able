"""The whole demo, 5 times in a row in one process, through the real engine pieces: calibrate, search,
open a result, write and send an email, hands in lap. It FINDS where the demo breaks (it fixes nothing):
every step prints PASS or FAIL with what happened, and the exit code is 0 only if every step passed.

Khang's Rig (tests/test_eyes_service.py) is the camera: the real EyesService, ModeManager,
TargetService and OsOutput with a simulated face (calibration, gaze, brow raises). UI-Venus finds every
label next to the cursor, and the dictation clean-up (Qwen) is faked. State carries over from run to run,
like a real rehearsal.

    python -m tests.demo_full_dryrun                     # FakeWhisperModel: hears exactly what was said
    python -m tests.demo_full_dryrun --whisper           # real Whisper on the synthetic clips (slow on CPU)
    python -m tests.demo_full_dryrun --whisper --voice zira --variant slow
    python -m tests.demo_full_dryrun --whisper --model large-v3-turbo   # the demo's model
The clips are tests/speech_phrases.csv's demo rows (scripts/make_test_speech.ps1 makes them).
"""
from __future__ import annotations

import argparse
import math
import time

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel

LAP_S = 60.0                     # hands in lap: this long with nothing happening
FRAME_S = 0.033
QUERY = "funny cat videos"
EMAIL = "hi mom home at seven tonight"
EMAIL_CLEAN = "Hi Mom, home at seven tonight."
CLIPS = {                        # what is said -> its clip id in tests/speech_phrases.csv
    "calibrate": "demo_010", "click search bar": "demo_011", "start dictation": "demo_002",
    QUERY: "demo_012", "send it": "demo_005", "stop dictation": "demo_006",
    "open first result": "demo_013", "click compose": "demo_001", EMAIL: "demo_004",
}


class Demo:
    def __init__(self, whisper: Dictation | None = None, voice: str = "david", variant: str = "normal"):
        from tests.test_dwell_click import ButtonOs
        from tests.test_eyes_service import Rig

        self.r = r = Rig()
        r.os = r.out.os = ButtonOs()
        r.mm.rewrite_wait_s = 1.5              # the clean-up on, as in the app
        self.voice, self.variant = voice, variant
        self.worker = DictationWorker(r.bus, whisper or Dictation(model=FakeWhisperModel()), settings=None,
                                      log=lambda s: None)
        self.whisper = whisper is not None
        self.heard: list[str] = []
        r.bus.subscribe(topics.TRANSCRIPT, lambda t: self.heard.append(t["text"]))
        r.bus.subscribe(topics.REWRITE_REQUEST, self._fake_qwen)
        self.results: list[tuple[int, str, bool, str]] = []
        self.sway = 0.0

    # ------------------------------------------------------------ the user
    def _fake_qwen(self, req: dict) -> None:
        text = req["text"]
        if "mom" in text.lower():
            out = {"id": req["id"], "used": True, "clean": EMAIL_CLEAN}
        else:
            out = {"id": req["id"], "used": False, "clean": ""}   # a search query: nothing to tidy
        self.r.bus.publish(topics.REWRITE_RESULT, out)

    def say(self, text: str) -> None:
        if self.whisper:
            from scipy.io import wavfile

            from engine.voice.voice_check import SYNTH

            rate, data = wavfile.read(SYNTH / f"{CLIPS[text]}__{self.voice}__{self.variant}.wav")
            assert rate == 16000
            self.worker.process(data.astype("float32") / 32768.0)
        else:
            self.worker.process(text)
        self.wait(0.5)                         # Whisper answers after the pause; the face keeps coming

    def wait(self, seconds: float, yaw: float | None = None) -> None:
        """The camera keeps running: frames with the head where it is, swaying a little like a real head."""
        for _ in range(int(seconds / FRAME_S)):
            self.sway += 1
            y = (self.yaw if yaw is None else yaw) + 0.3 * math.sin(self.sway / 25)
            self.r.frames(1, yaw=y, pitch=0.2 * math.sin(self.sway / 40))
            self.r.mm.tick()

    def look(self, yaw: float) -> None:
        """Turn the head (the head pointer moves the cursor there) and settle."""
        steps = 8
        for i in range(1, steps + 1):
            self.r.frames(1, yaw=self.yaw + (yaw - self.yaw) * i / steps)
        self.yaw = yaw
        self.wait(0.3)

    yaw = 0.0

    def raise_brows(self) -> None:
        self.r.frames(15, yaw=self.yaw)
        self.r.frames(12, yaw=self.yaw, brow=0.45)
        self.r.frames(10, yaw=self.yaw)
        self.r.out.flush()

    # ------------------------------------------------------------ checking
    def snapshot(self) -> dict:
        os = self.r.os
        return {"clicks": list(os.clicks), "typed": list(os.typed), "keys": list(os.keys),
                "ui": len(self.r.bus.of(topics.TO_UI)), "heard": len(self.heard)}

    def since(self, snap: dict) -> dict:
        now = self.snapshot()
        ui = self.r.bus.of(topics.TO_UI)[snap["ui"]:]
        return {"clicks": now["clicks"][len(snap["clicks"]):], "typed": now["typed"][len(snap["typed"]):],
                "keys": now["keys"][len(snap["keys"]):], "ui": ui, "heard": self.heard[snap["heard"]:]}

    def check(self, run: int, step: str, ok: bool, detail: str) -> None:
        self.results.append((run, step, ok, detail))
        print(f"  run {run}  {step:<28} {'PASS' if ok else 'FAIL'}  {detail}")

    @staticmethod
    def kinds(ui: list[dict], kind: str) -> list[dict]:
        return [m for m in ui if m["type"] == kind]

    def ask_then_yes(self, d: dict) -> list[str]:
        """The confirm states this step saw, e.g. ["open", "confirmed"]."""
        return [m["state"] for m in self.kinds(d["ui"], messages.CONFIRM)]

    # ------------------------------------------------------------ the demo
    def run(self, n: int) -> None:
        r, mm = self.r, self.r.mm

        # 1. calibrate (the first run also has the one-look calibration of the first face)
        s = self.snapshot()
        self.say("calibrate")
        self.wait(2.0, yaw=0.0)
        self.yaw = 0.0
        d = self.since(s)
        res = self.kinds(d["ui"], messages.CALIB_RESULT)
        self.check(n, "calibrate", bool(res) and res[-1]["ok"] and not d["clicks"],
                   f"heard {d['heard']}, calib_result {res[-1] if res else None}, clicks {d['clicks']}")

        # 2. search: look at the search bar, "click search bar", raise, dictate the query, send it
        s = self.snapshot()
        self.look(-4.0)
        self.say("click search bar")
        r.out.flush()
        self.raise_brows()
        d = self.since(s)
        self.check(n, "search: click search bar", len(d["clicks"]) == 1 and "confirmed" in self.ask_then_yes(d),
                   f"heard {d['heard']}, confirm {self.ask_then_yes(d)}, clicks {d['clicks']}")
        s = self.snapshot()
        self.say("start dictation")
        self.say(QUERY)
        self.say("send it")
        self.raise_brows()
        self.wait(2.0)
        self.say("stop dictation")
        d = self.since(s)
        typed = "".join(d["typed"]).strip().lower().rstrip(".")
        self.check(n, "search: type query + enter",
                   typed == QUERY and d["keys"] == ["enter"] and mm.mode == Mode.POINTER and not d["clicks"],
                   f"heard {d['heard']}, typed {d['typed']}, keys {d['keys']}, clicks {d['clicks']}, "
                   f"mode {mm.mode.value}")

        # 3. open a result
        s = self.snapshot()
        self.look(2.0)
        self.say("open first result")
        r.out.flush()
        self.raise_brows()
        self.wait(1.0)
        d = self.since(s)
        self.check(n, "open first result", len(d["clicks"]) == 1 and "confirmed" in self.ask_then_yes(d),
                   f"heard {d['heard']}, confirm {self.ask_then_yes(d)}, clicks {d['clicks']}")

        # 4. write and send an email
        s = self.snapshot()
        self.look(-6.0)
        self.say("click compose")
        r.out.flush()
        self.raise_brows()
        d = self.since(s)
        self.check(n, "email: click compose", len(d["clicks"]) == 1,
                   f"heard {d['heard']}, confirm {self.ask_then_yes(d)}, clicks {d['clicks']}")
        s = self.snapshot()
        self.say("start dictation")
        self.say(EMAIL)
        previews = self.kinds(self.since(s)["ui"], messages.DICTATION_PREVIEW)
        cleaned = any(p.get("cleaned") for p in previews)
        self.say("send it")
        self.raise_brows()
        self.wait(2.0)
        self.say("stop dictation")
        d = self.since(s)
        self.check(n, "email: dictate, clean, send",
                   d["typed"] == [EMAIL_CLEAN + " "] and d["keys"] == ["enter"] and cleaned
                   and mm.mode == Mode.POINTER and not d["clicks"],
                   f"heard {d['heard']}, cleaned preview {cleaned}, typed {d['typed']}, keys {d['keys']}, "
                   f"clicks {d['clicks']}, mode {mm.mode.value}")

        # 5. hands in lap: a minute of sitting there, looking at the screen
        s = self.snapshot()
        mode = mm.mode
        self.wait(LAP_S)
        r.out.flush()
        d = self.since(s)
        modes = [m["mode"] for m in self.kinds(d["ui"], messages.MODE)]
        statuses = [m["text"] for m in self.kinds(d["ui"], messages.STATUS)]
        self.check(n, "hands in lap 60 s",
                   not d["clicks"] and not d["typed"] and not d["keys"] and not modes and mm.mode == mode
                   and mm.question is None and not mm.preview,
                   f"clicks {d['clicks']}, typed {d['typed']}, modes {modes}, statuses {statuses}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The whole demo, 5 runs in a row (finds failures, fixes nothing).")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--whisper", action="store_true", help="real Whisper on the synthetic clips")
    p.add_argument("--voice", default="david")
    p.add_argument("--variant", default="normal")
    p.add_argument("--model", default=None, help="whisper_model instead of the config's (e.g. large-v3-turbo)")
    args = p.parse_args(argv)

    whisper = None
    if args.whisper:
        import dataclasses

        from engine.config import settings

        if args.model:
            settings = dataclasses.replace(settings, whisper_model=args.model)

        whisper = Dictation(settings)
        t0 = time.perf_counter()
        whisper.load()
        print(f"Whisper {settings.whisper_model} on {settings.whisper_device} loaded in "
              f"{time.perf_counter() - t0:.1f} s; {args.voice}/{args.variant}")
    demo = Demo(whisper, args.voice, args.variant)
    demo.r.frames(60)                         # the first face: the one-look calibration
    for n in range(1, args.runs + 1):
        print(f"run {n}")
        demo.run(n)
    demo.r.out.stop()
    failed = [x for x in demo.results if not x[2]]
    print(f"{len(demo.results) - len(failed)}/{len(demo.results)} steps passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
