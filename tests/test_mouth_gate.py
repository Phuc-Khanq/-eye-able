"""Run with: python -m unittest tests.test_mouth_gate

Only the user's voice counts: DictationWorker keeps a word only if the camera saw the user's
mouth moving while it was said. No camera, no mic, no model: FakeMouth stands in for
engine.eyes.mouth.MouthTracker and FakeWhisperModel says word i from 0.4*i to 0.4*(i+1) s.
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.voice import dictation
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel

S = 10_000.0          # span start (ms) used in most tests
FRAME_MS = 1000 / 30
GATE = SimpleNamespace(mouth_gate=True, mouth_min_moving=0.3, mouth_pad_ms=150.0, mouth_status_every_s=10.0)


def overlaps(t, intervals):
    return any(a <= t <= b for a, b in intervals)


class FakeMouth:
    """Frames at 30 fps wherever the face was seen; a frame is "moving" inside a moving interval.
    Same contract as MouthTracker: None if under half the expected frames had a face,
    otherwise the share of moving frames among the frames with a face."""

    def __init__(self, moving=(), face=((0, 1e9),)):
        self.moving, self.face = list(moving), list(face)
        self.calls = []

    def moving_fraction(self, start_ms, end_ms):
        self.calls.append((start_ms, end_ms))
        n = max(1, int((end_ms - start_ms) // FRAME_MS))
        frames = [start_ms + (i + 0.5) * (end_ms - start_ms) / n for i in range(n)]
        seen = [t for t in frames if overlaps(t, self.face)]
        if len(seen) < 0.5 * n:
            return None
        return sum(overlaps(t, self.moving) for t in seen) / len(seen)


class ConstMouth:
    def __init__(self, f):
        self.f = f

    def moving_fraction(self, start_ms, end_ms):
        return self.f


class BrokenMouth:
    def moving_fraction(self, start_ms, end_ms):
        raise RuntimeError("camera thread died")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(mouth, settings=GATE):
    bus, clock, log = FakeBus(), Clock(), []
    worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()), mouth=mouth, settings=settings,
                             clock=clock, log=log.append)
    return worker, bus, clock, log


def span(start=S, words=4):
    return {"start_ms": start, "end_ms": start + 400 * words}


class TestMouthGate(unittest.TestCase):
    def test_mixed_utterance_keeps_only_the_users_words(self):
        # the user says "hi mom" (0-0.8 s), a teammate's "yes okay" follows (0.8-1.6 s)
        worker, bus, _, log = make(FakeMouth(moving=[(S, S + 800)]))
        worker.process("hi mom yes okay", span())
        [t] = bus.of(topics.TRANSCRIPT)
        self.assertEqual(t["text"], "hi mom")
        self.assertEqual(t["words"], [{"text": "hi", "conf": 0.95}, {"text": "mom", "conf": 0.95}])
        self.assertEqual(len(log), 1)
        self.assertTrue(log[0].startswith("[mouth] kept 2/4: hi("), log[0])
        self.assertIn("dropped: yes(", log[0])

    def test_word_windows_are_the_whisper_times_on_the_span_clock_plus_padding(self):
        mouth = FakeMouth(moving=[(0, 1e9)])
        worker, *_ = make(mouth)
        worker.process("hi mom", span(start=5000.0))
        self.assertEqual(mouth.calls, [(5000 - 150, 5400 + 150), (5400 - 150, 5800 + 150)])

    def test_still_mouth_gives_nothing_and_one_status_per_10_s(self):
        worker, bus, clock, _ = make(FakeMouth(moving=[]))
        for t in (0.0, 2.0, 4.0):                       # 3 utterances in 5 s
            clock.t = t
            worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])
        self.assertEqual(bus.ui(messages.STATUS), [messages.make(
            messages.STATUS, level="info", text="Ignored speech while your mouth wasn't moving")])
        clock.t = 15.0                                   # more than 10 s after the first
        worker.process("okay", span(words=1))
        self.assertEqual(len(bus.ui(messages.STATUS)), 2)

    def test_no_face_changes_nothing(self):
        worker, bus, _, _ = make(FakeMouth(moving=[], face=[]))
        worker.process("hi mom", span(words=2))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "hi mom")

    def test_face_seen_in_under_half_the_frames_means_cant_tell(self):
        # face only in the first 30% of each word's window, mouth still: None -> kept
        worker, bus, _, _ = make(FakeMouth(moving=[], face=[(S - 150, S + 50)]))
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "yes")

    def test_a_broken_tracker_changes_nothing_and_warns_once_a_minute(self):
        worker, bus, clock, log = make(BrokenMouth())
        worker.process("hi mom", span(words=2))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "hi mom")
        warnings = [line for line in log if "tracker failed" in line]
        self.assertEqual(len(warnings), 1)                  # two words, one warning
        clock.t = 30.0
        worker.process("again", span(words=1))
        self.assertEqual(len([line for line in log if "tracker failed" in line]), 1)
        clock.t = 61.0
        worker.process("again", span(words=1))
        self.assertEqual(len([line for line in log if "tracker failed" in line]), 2)

    def test_padding_absorbs_whisper_timing_errors(self):
        # Whisper puts "yes" at 0-0.4 s, but the mouth really moved 150 ms later
        worker, bus, _, _ = make(FakeMouth(moving=[(S + 150, S + 550)]))
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "yes")
        # the mouth only moved from 300 ms after the word ended: someone else said it
        worker, bus, _, _ = make(FakeMouth(moving=[(S + 700, S + 1500)]))
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])

    def test_padding_matters(self):
        late = FakeMouth(moving=[(S + 300, S + 700)])        # 300 ms late: only a quarter of the word
        no_pad = SimpleNamespace(**{**vars(GATE), "mouth_pad_ms": 0.0})
        worker, bus, _, _ = make(late, no_pad)
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])
        worker, bus, _, _ = make(late)                       # with the 150 ms pad: 0.36 -> kept
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "yes")

    def test_the_threshold_itself_is_kept(self):
        for f, kept in ((0.3, True), (0.29, False), (1.0, True), (0.0, False)):
            with self.subTest(f=f):
                worker, bus, _, _ = make(ConstMouth(f))
                worker.process("yes", span(words=1))
                self.assertEqual(bool(bus.of(topics.TRANSCRIPT)), kept)

    def test_threshold_and_pad_come_from_settings(self):
        lenient = SimpleNamespace(**{**vars(GATE), "mouth_min_moving": 0.1})
        worker, bus, _, _ = make(ConstMouth(0.15), lenient)
        worker.process("yes", span(words=1))
        self.assertEqual(bus.of(topics.TRANSCRIPT)[0]["text"], "yes")

    def test_leftover_unsure_words_are_noise(self):
        # the user's only kept word is one Whisper wasn't sure of: drop it like any short unsure result
        worker, bus, _, _ = make(FakeMouth(moving=[(S, S + 400)]))
        worker.process("um? yes okay", span(words=3))
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])

    def test_gate_off_no_tracker_or_no_span_is_the_old_behavior(self):
        off = SimpleNamespace(**{**vars(GATE), "mouth_gate": False})
        for worker, sp in ((make(FakeMouth(moving=[]), off)[0], span()),
                           (make(None)[0], span()),
                           (make(FakeMouth(moving=[]))[0], None)):
            worker.process("hi mom yes okay", sp)
            [t] = worker.bus.of(topics.TRANSCRIPT)
            self.assertEqual(t["text"], "hi mom yes okay")


class TestSpanPairing(unittest.TestCase):
    def test_each_utterance_travels_with_the_span_before_it(self):
        worker, bus, _, _ = make(FakeMouth())
        bus.publish(topics.SPEECH_SPAN, span(1000.0))
        bus.publish(topics.UTTERANCE, "a")
        bus.publish(topics.UTTERANCE, "b")                  # no span published for this one
        bus.publish(topics.SPEECH_SPAN, span(2000.0))
        bus.publish(topics.UTTERANCE, "c")
        items = [worker._queue.get_nowait() for _ in range(3)]
        self.assertEqual(items, [("a", span(1000.0)), ("b", None), ("c", span(2000.0))])

    def test_pairs_survive_the_drop_oldest_queue(self):
        worker, bus, _, _ = make(FakeMouth())
        for i in range(dictation.MAX_QUEUED + 2):
            bus.publish(topics.SPEECH_SPAN, span(1000.0 * i))
            bus.publish(topics.UTTERANCE, f"clip {i}")
        items = [worker._queue.get_nowait() for _ in range(dictation.MAX_QUEUED)]
        self.assertEqual(items, [(f"clip {i}", span(1000.0 * i)) for i in range(2, dictation.MAX_QUEUED + 2)])


class TestTheOriginalBug(unittest.TestCase):
    """Preview open, a teammate says "Yes" nearby: it must not type the preview."""

    def test_teammates_yes_types_nothing_but_the_users_yes_does(self):
        # the user's mouth moves during their own utterances only; the face is always in view
        mouth = FakeMouth()
        moving = mouth.moving
        bus, clock, log = FakeBus(), Clock(), []
        mm = ModeManager(bus, clock=clock)
        worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()), mouth=mouth, settings=GATE,
                                 clock=clock, log=log.append)

        def say(text, user, at_ms):
            words = len(text.split())
            if user:
                moving.append((at_ms, at_ms + 400 * words))
            worker.process(text, {"start_ms": at_ms, "end_ms": at_ms + 400 * words})

        say("start dictation", True, 1_000)
        self.assertEqual(mm.mode, Mode.DICTATION)
        say("hi mom home at seven tonight", True, 5_000)
        self.assertTrue(mm.preview)
        say("yes", False, 10_000)                            # the teammate
        self.assertEqual(bus.of(topics.OS_TYPE), [])
        self.assertTrue(mm.preview)                          # still waiting for the user
        say("yes", True, 14_000)                             # the user
        self.assertEqual([p["text"] for p in bus.of(topics.OS_TYPE)], ["hi mom home at seven tonight "])
        self.assertFalse(mm.preview)


if __name__ == "__main__":
    unittest.main()
