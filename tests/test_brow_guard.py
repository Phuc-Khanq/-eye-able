"""Run with: python -m unittest tests.test_brow_guard

Task 4: an eyebrow raise while the user is talking is ignored (Hung measured about 0.8 accidental
raises a minute while talking; a raise answers questions, and soon right-clicks). FakeMouth from
tests/test_mouth_gate.py stands in for MouthTracker; times are the camera's monotonic ms.
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_mouth_gate import BrokenMouth, FakeMouth

GUARD = SimpleNamespace(brow_ignore_while_talking=True, brow_talk_window_ms=600, brow_talk_min_moving=0.3)
ENGINE_SETTINGS = SimpleNamespace(pause_seconds=1.0, **vars(GUARD))   # what Engine needs
T = 100_000          # the raise happens at camera time T ms


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def make(self, mouth, settings=GUARD):
        self.bus, self.clock, self.log = FakeBus(), Clock(), []
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock, mouth=mouth, settings=settings, log=self.log.append)
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))

    def say(self, text):
        self.worker.process(text)

    def raise_brows(self, t_ms=T):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type="brow_raise"))

    def typed(self):
        return [p["text"] for p in self.bus.of(topics.OS_TYPE)]

    def preview_then_raise(self, mouth, settings=GUARD):
        self.make(mouth, settings)
        self.say("start dictation")
        self.say("hi mom")
        self.raise_brows()


class TestIgnoredWhileTalking(Base):
    def test_a_raise_while_talking_does_nothing(self):
        self.preview_then_raise(FakeMouth(moving=[(T - 2000, T + 500)]))
        self.assertEqual(self.typed(), [])
        self.assertEqual(self.mm._preview_text(), "hi mom")
        self.assertEqual(len(self.log), 1)
        self.assertTrue(self.log[0].startswith("[brow] ignored: you were talking ("), self.log[0])
        self.assertEqual(self.bus.ui(messages.STATUS), [])                 # no UI flash

    def test_it_does_not_answer_a_question_either(self):
        self.make(FakeMouth(moving=[(T - 2000, T + 500)]))
        self.say("click send")
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
        self.raise_brows()
        self.assertEqual(self.bus.of(topics.OS_CLICK), [])
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["state"], "open")

    def test_nor_preconfirm_a_search(self):
        self.make(FakeMouth(moving=[(T - 2000, T + 500)]))
        self.say("click send")
        self.raise_brows()
        self.assertFalse(self.mm.search.preconfirmed)

    def test_raising_while_saying_send_it_does_not_type_early(self):
        # the raise lands while the user is still saying "send it", before Whisper's transcript
        self.make(FakeMouth(moving=[(T - 800, T + 300)]))
        self.say("start dictation")
        self.say("hi mom")
        self.raise_brows()
        self.say("send it")
        self.assertEqual(self.typed(), [])
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["prompt"], 'Insert "hi mom" and send?')


class TestStillWorks(Base):
    def test_a_raise_after_speech_ended_works(self):
        self.preview_then_raise(FakeMouth(moving=[(T - 3000, T - 1000)]))   # stopped talking 1 s ago
        self.assertEqual(self.typed(), ["hi mom "])
        self.assertEqual(self.log, [])

    def test_no_face_means_cant_tell_so_it_works(self):
        self.preview_then_raise(FakeMouth(moving=[(T - 2000, T + 500)], face=[]))
        self.assertEqual(self.typed(), ["hi mom "])

    def test_a_broken_tracker_works_and_warns_once_a_minute(self):
        self.preview_then_raise(BrokenMouth())
        self.assertEqual(self.typed(), ["hi mom "])
        self.say("see you")
        self.clock.t = 30
        self.raise_brows()
        self.assertEqual(self.typed(), ["hi mom ", "see you "])
        self.assertEqual(sum("tracker failed" in line for line in self.log), 1)
        self.say("bye")
        self.clock.t = 61
        self.raise_brows()
        self.assertEqual(sum("tracker failed" in line for line in self.log), 2)

    def test_guard_off_or_no_tracker_is_todays_behavior(self):
        off = SimpleNamespace(**{**vars(GUARD), "brow_ignore_while_talking": False})
        self.preview_then_raise(FakeMouth(moving=[(T - 2000, T + 500)]), off)
        self.assertEqual(self.typed(), ["hi mom "])
        self.preview_then_raise(None)
        self.assertEqual(self.typed(), ["hi mom "])

    def test_only_the_raise_is_guarded(self):
        self.make(FakeMouth(moving=[(T - 2000, T + 500)]))
        self.say("start dictation")
        self.say("hi mom")
        self.bus.publish(topics.GESTURE, Gesture(t_ms=T, type="nod"))     # nod while talking still works
        self.assertEqual(self.typed(), ["hi mom "])
        self.say("see you")
        self.say("yes")                                                    # so does a spoken yes
        self.assertEqual(self.typed(), ["hi mom ", "see you "])

    def test_window_and_threshold_come_from_settings(self):
        # mouth moved only until 200 ms before the raise: inside a 600 ms window, outside a 100 ms one
        moving = [(T - 2000, T - 200)]
        self.preview_then_raise(FakeMouth(moving=moving))
        self.assertEqual(self.typed(), [])
        short = SimpleNamespace(**{**vars(GUARD), "brow_talk_window_ms": 100})
        self.preview_then_raise(FakeMouth(moving=moving), short)
        self.assertEqual(self.typed(), ["hi mom "])


class TestEngineWiring(unittest.TestCase):
    def test_with_eyes_the_mode_manager_gets_the_mouth_tracker(self):
        from unittest import mock

        from engine.core.events import EventBus
        from engine.eyes.mouth import MouthTracker
        from engine.main import Engine
        from engine.voice.mic import MicSettings
        from tests.test_audio_in import FakeVad

        # no camera, UI-Venus or keyboard: only the wiring is under test
        with mock.patch("engine.eyes.service.EyesService"), mock.patch("engine.eyes.target_service.TargetService"), \
             mock.patch("engine.eyes.os_output.OsOutput"):
            e = Engine(EventBus(), ENGINE_SETTINGS, dictation=Dictation(model=FakeWhisperModel()), eyes=True,
                       mic_settings=MicSettings(0, "f", "MME", 16000, 1), vad=FakeVad(), warm_up=False)
        self.assertIsInstance(e.mode_manager.mouth, MouthTracker)
        self.assertIs(e.mode_manager.mouth, e.mouth)
        self.assertIs(e.dictation.mouth, e.mouth)

    def test_without_eyes_there_is_no_guard(self):
        from engine.core.events import EventBus
        from engine.main import Engine
        from engine.voice.mic import MicSettings
        from tests.test_audio_in import FakeVad

        e = Engine(EventBus(), ENGINE_SETTINGS, dictation=Dictation(model=FakeWhisperModel()),
                   mic_settings=MicSettings(0, "f", "MME", 16000, 1), vad=FakeVad(), warm_up=False)
        self.assertIsNone(e.mode_manager.mouth)


if __name__ == "__main__":
    unittest.main()
