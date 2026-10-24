"""Writes tests/recordings/all_messages.jsonl: every message the engine sends the UI, from one short
session through the REAL pieces (Khang's Rig from tests/test_eyes_service.py: the real EyesService,
ModeManager, TargetService and OsOutput with a simulated face; the real AudioIn with a test tone).
Nothing is typed by hand, so the file shows exactly what the UI will get. No camera, mic or GPU.

    python -m tests.make_all_messages             # rewrites the recording
    python -m tests.fake_engine tests/recordings/all_messages.jsonl

For Nhi: replay it with fake_engine and the whole UI can be built without the engine.
Debug messages (debug_*, target_debug) aren't sent by the engine yet, so they aren't in it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.core import messages, topics
from engine.core.types import Gesture
from engine.voice.audio_in import AudioIn
from engine.voice.mic import MicSettings
from tests.test_audio_in import FakeVad, Settings as AudioSettings

OUT = Path(__file__).resolve().parent / "recordings" / "all_messages.jsonl"
GAZE_EVERY = 3          # keep every 3rd gaze message (about 10 a second instead of 30)
WANTED = {
    messages.MODE: {"mode": {"pointer", "dictation", "menu", "paused", "scroll"}},
    messages.GESTURE: {"gesture": {"nod", "brow_raise", "tilt_left", "tilt_right"}},
    messages.CONFIRM: {"state": {"open", "confirmed", "cancelled", "expired"}},
    messages.STATUS: {"level": {"info", "warn", "error"}},
    messages.DICTATION_PREVIEW: {"cleaned": {True, False}},
    messages.LISTENING: {"speaking": {True, False}},
    messages.CALIB_RESULT: {"ok": {True, False}},
    messages.DWELL: {}, messages.SCROLL: {}, messages.GAZE: {}, messages.CALIB_TARGET: {},
    messages.TARGET_SEARCHING: {}, messages.TARGET_FOUND: {}, messages.TARGET_NONE: {},
}


def session() -> list[tuple[float, dict]]:
    """[(t seconds from the start, message), ...] for one short session."""
    from tests.test_eyes_service import Rig, TargetResult

    r = Rig()
    r.mm.rewrite_wait_s = 1.5            # the dictation clean-up on, as in the app
    t0 = r.clock.t
    out: list[tuple[float, dict]] = []
    gazes = [0]

    def record(msg: dict) -> None:
        if msg["type"] == messages.GAZE:
            gazes[0] += 1
            if gazes[0] % GAZE_EVERY:
                return
        out.append((round(r.clock.t - t0, 3), msg))

    r.bus.subscribe(topics.TO_UI, record)
    r.bus.publish(topics.TO_UI, messages.make(messages.MODE, mode=r.mm.mode.value))  # what a new UI gets first

    # 1. the first face: one-look calibration; then a calibration started and cancelled
    r.frames(60)
    r.bus.publish(topics.from_ui(messages.CALIBRATE), {"type": messages.CALIBRATE, "action": "start"})
    r.frames(5)
    r.bus.publish(topics.from_ui(messages.CALIBRATE), {"type": messages.CALIBRATE, "action": "cancel"})
    r.frames(60)

    # 2. look somewhere and rest there: the dwell ring fills and clicks
    r.say("start clicking")                          # dwell on (eyes.dwell)
    r.path(yaws=[1, 2, 3, 4, 5, 6, 6, 6])
    r.frames(45, yaw=6.0)

    # 3. look and say: found, asked, a raise says yes (the demo's confirm)
    r.say("click search bar")
    r.out.flush()
    r.raise_brows()
    r.frames(20)

    # 4. a target that isn't on screen
    submit = r.finder.submit

    def not_found(req):
        f = submit(req)
        res = f.result()
        f2 = type(f)()
        f2.set_result(TargetResult(label=res.label, found=False, x=0, y=0))
        return f2

    r.finder.submit = not_found
    r.say("click the purple button")
    r.out.flush()
    r.frames(20)
    r.finder.submit = submit

    # 5. dictation with the mic: the listening level, the preview, the clean-up, send it
    r.say("start dictation")
    audio = AudioIn(r.bus, AudioSettings(), mic_settings=MicSettings(-1, "synthetic", "file", 16000, 1),
                    vad=FakeVad(min_silence_s=0.5), clock=r.clock)
    rate = 16000
    t = np.arange(int(1.2 * rate)) / rate
    clip = np.concatenate([np.zeros(int(0.3 * rate)), 0.3 * np.sin(2 * np.pi * 300 * t),
                           np.zeros(int(0.8 * rate))]).astype(np.float32)
    for i in range(0, len(clip), 1536):              # mic-sized chunks, the clock moving with them
        part = clip[i:i + 1536]
        r.clock.t += len(part) / rate
        r.t_ms += int(len(part) / rate * 1000)
        audio.process(part, r.clock.t * 1000)
    r.say("hi mom home at seven tonight")              # what Whisper heard in that clip
    rid = r.bus.of(topics.REWRITE_REQUEST)[-1]["id"]
    r.frames(10)
    r.bus.publish(topics.REWRITE_RESULT, {"id": rid, "used": True, "clean": "Hi Mom, home at seven tonight."})
    r.frames(10)
    r.say("send it")
    r.frames(10)
    r.bus.publish(topics.GESTURE, Gesture(t_ms=r.t_ms, type="shake"))   # no: cancelled
    r.frames(10)
    r.say("send it")
    r.frames(int(12.5 / 0.033))                          # nobody answers: expired after 12 s
    r.mm.tick()
    r.say("send it")
    r.raise_brows()                                      # confirmed: typed and sent
    r.frames(10)
    r.say("stop dictation")
    r.frames(10)

    # 6. the menu: move with tilts, pick with a nod or a raise, a shake closes it
    r.say("menu")
    for kind in ("tilt_right", "tilt_left", "nod"):   # what the eyes' gesture detector publishes
        r.bus.publish(topics.GESTURE, Gesture(t_ms=r.t_ms, type=kind))
        r.frames(10)
    r.raise_brows()
    r.bus.publish(topics.GESTURE, Gesture(t_ms=r.t_ms, type="shake"))
    r.frames(10)

    # 7. pause and resume
    r.say("pause")
    r.frames(10)
    r.say("resume")
    r.frames(10)

    # 8. scroll mode: tilt the chin down to scroll, then "stop"
    r.say("scroll")
    for _ in range(20):
        r.frames(1)
        r.eyes.scroller.step(0.033)
    for _ in range(45):
        r.frames(1, pitch=6.0)
        r.eyes.scroller.step(0.033)
    r.say("stop")
    r.frames(10)

    # 9. things going wrong: a click that fails, the face lost and found again
    click = r.os.click

    def broken_click(x, y):
        raise PermissionError("the window is run as administrator")

    r.os.click = broken_click
    r.path(yaws=[-1, -2, -3, -4, -5, -6, -6])
    r.frames(45, yaw=-6.0)                               # dwell click: fails
    r.out.flush()
    r.os.click = click
    for _ in range(90):                                  # 3 s without a face
        r.clock.t += 0.033
        r.t_ms += 33
        r.eyes.process(None, r.clock.t)
    r.frames(20)
    r.out.stop()
    return out


def missing(events: list[tuple[float, dict]]) -> list[str]:
    """What WANTED asks for that the events don't have, e.g. ["confirm state=expired"]."""
    gaps = []
    for kind, fields in WANTED.items():
        msgs = [m for _, m in events if m["type"] == kind]
        if not msgs:
            gaps.append(kind)
        for field, values in fields.items():
            for v in values - {m.get(field) for m in msgs}:
                gaps.append(f"{kind} {field}={v}")
    return gaps


def main() -> int:
    events = session()
    gaps = missing(events)
    if gaps:
        print("missing:", ", ".join(gaps))
        return 1
    OUT.write_text("".join(json.dumps({"t": t, "msg": m}) + "\n" for t, m in events), encoding="utf-8")
    kinds = sorted({m["type"] for _, m in events})
    print(f"wrote {OUT.name}: {len(events)} messages, {len(kinds)} types ({', '.join(kinds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
