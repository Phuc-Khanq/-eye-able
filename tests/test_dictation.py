"""Run with: python -m unittest tests.test_dictation

No model: FakeWhisperModel stands in for faster-whisper.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.voice import dictation
from engine.voice.dictation import Dictation, DictationWorker, resolve_model
from tests.helpers import FakeBus, FakeWhisperModel


class SlowModel(FakeWhisperModel):
    def __init__(self, delay=0.05):
        self.delay = delay
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        return super().transcribe(audio, **kwargs)


class TestDictationWorker(unittest.TestCase):
    def test_utterance_becomes_transcript(self):
        bus = FakeBus()
        DictationWorker(bus, Dictation(model=FakeWhisperModel())).process("hello there")
        [t] = bus.of(topics.TRANSCRIPT)
        self.assertEqual(t["text"], "hello there")
        self.assertEqual([w["text"] for w in t["words"]], ["hello", "there"])
        self.assertGreaterEqual(t["ms"], 0)

    def test_decode_settings(self):
        model = SlowModel(delay=0)
        Dictation(model=model).transcribe("hi")
        kw = model.calls[0]
        self.assertEqual(kw["language"], "en")
        self.assertEqual(kw["beam_size"], dictation.BEAM_SIZE)
        self.assertFalse(kw["condition_on_previous_text"])

    def test_silence_publishes_nothing(self):
        bus = FakeBus()
        DictationWorker(bus, Dictation(model=FakeWhisperModel())).process("")
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])

    def test_unsure_short_result_is_dropped(self):
        bus = FakeBus()
        DictationWorker(bus, Dictation(model=FakeWhisperModel())).process("thanks?")
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])

    def test_runs_on_its_own_thread(self):
        """on_utterance returns at once; the transcript arrives later from the whisper thread."""
        bus = EventBus()
        got = []
        done = threading.Event()

        def on_transcript(t):
            got.append((t["text"], threading.current_thread().name))
            done.set()

        bus.subscribe(topics.TRANSCRIPT, on_transcript)
        worker = DictationWorker(bus, Dictation(model=SlowModel(delay=0.2)))
        worker.start(warm_up=False)
        t0 = time.perf_counter()
        bus.publish(topics.UTTERANCE, "open my email")
        self.assertLess(time.perf_counter() - t0, 0.05)        # audio thread not blocked
        self.assertTrue(done.wait(3.0))
        worker.stop()
        self.assertEqual(got, [("open my email", "whisper")])

    def test_falls_behind_drops_oldest(self):
        bus = FakeBus()
        worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()))
        for i in range(dictation.MAX_QUEUED + 2):
            worker.on_utterance(f"clip {i}")
        left = []
        while not worker._queue.empty():
            left.append(worker._queue.get_nowait())
        self.assertEqual(left, [f"clip {i}" for i in range(2, dictation.MAX_QUEUED + 2)])

    def test_load_failure_is_reported(self):
        class Broken:
            def warm_up(self):
                raise RuntimeError("no CUDA")

        bus = FakeBus()
        worker = DictationWorker(bus, Broken())
        worker._run(warm_up=True)
        [msg] = bus.ui(messages.STATUS)
        self.assertEqual(msg["level"], "error")
        self.assertIn("no CUDA", msg["text"])


class TestResolveModel(unittest.TestCase):
    def test_model_name_stays_a_name(self):
        self.assertEqual(resolve_model("large-v3-turbo"), "large-v3-turbo")
        self.assertEqual(resolve_model("models/does-not-exist"), "models/does-not-exist")

    def test_folder_under_repo_root(self):
        with tempfile.TemporaryDirectory(dir=dictation.REPO_ROOT / "models") as tmp:
            rel = Path(tmp).relative_to(dictation.REPO_ROOT).as_posix()
            self.assertEqual(Path(resolve_model(rel)), Path(tmp))

    def test_load_uses_the_folder(self):
        import sys

        seen = {}

        class FakeWhisper:
            def __init__(self, name, **kw):
                seen["name"] = name

        sys.modules["faster_whisper"] = SimpleNamespace(WhisperModel=FakeWhisper)
        try:
            with tempfile.TemporaryDirectory(dir=dictation.REPO_ROOT / "models") as tmp:
                rel = Path(tmp).relative_to(dictation.REPO_ROOT).as_posix()
                s = SimpleNamespace(whisper_model=rel, whisper_device="cpu",
                                    whisper_compute_type="int8")
                Dictation(settings=s).load()
                self.assertEqual(Path(seen["name"]), Path(tmp))
        finally:
            del sys.modules["faster_whisper"]


class RealWordsModel:
    """Returns faster-whisper's own Word and Segment types, so the attribute names are the real ones."""

    def __init__(self, words, no_speech_prob=0.01):
        self.words = words
        self.no_speech_prob = no_speech_prob

    def transcribe(self, audio, **kwargs):
        from faster_whisper.transcribe import Segment, Word

        words = [Word(start=s, end=e, word=w, probability=p) for s, e, w, p in self.words]
        seg = Segment(id=0, seek=0, start=0.0, end=words[-1].end if words else 0.0, text="", tokens=[],
                      avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=self.no_speech_prob,
                      words=words, temperature=0.0)
        return iter([seg]), SimpleNamespace(language="en")


class TestWordTimes(unittest.TestCase):
    def test_words_get_start_and_end_from_whisper(self):
        d = Dictation(model=RealWordsModel([(0.12345, 0.4, " Hi", 0.9), (0.42, 0.8, " mom,", 0.8)]))
        self.assertEqual(d.transcribe(None), [
            {"text": "Hi", "conf": 0.9, "start": 0.123, "end": 0.4},
            {"text": "mom,", "conf": 0.8, "start": 0.42, "end": 0.8}])

    def test_end_before_start_is_fixed(self):
        [w] = Dictation(model=RealWordsModel([(1.5, 1.2, " yes", 0.9)])).transcribe(None)
        self.assertEqual((w["start"], w["end"]), (1.5, 1.5))

    def test_fake_whisper_words_are_timed_in_order(self):
        words = Dictation(model=FakeWhisperModel()).transcribe("one two three")
        self.assertEqual([(w["start"], w["end"]) for w in words], [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2)])

    def test_transcript_keeps_the_text_and_conf_contract(self):
        bus = FakeBus()
        DictationWorker(bus, Dictation(model=RealWordsModel([(0.0, 0.3, " Send", 0.9),
                                                             (0.3, 0.5, " it.", 0.9)]))).process(None)
        [t] = bus.of(topics.TRANSCRIPT)
        self.assertEqual(t["text"], "Send it.")
        self.assertEqual(t["words"], [{"text": "Send", "conf": 0.9}, {"text": "it.", "conf": 0.9}])

    def test_hallucination_filter_still_works_with_times(self):
        self.assertEqual(Dictation(model=RealWordsModel([(0.0, 0.5, " Thank", 0.2),
                                                         (0.5, 0.9, " you.", 0.25)])).transcribe(None), [])
        self.assertEqual(Dictation(model=RealWordsModel([(0.0, 0.5, " Hi", 0.9)],
                                                        no_speech_prob=0.9)).transcribe(None), [])

    def test_warm_up_still_works(self):
        import numpy as np

        model = SlowModel(delay=0)
        d = Dictation(model=model)
        d.warm_up()                                   # one second of silence through the model
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(Dictation(model=RealWordsModel([])).transcribe(np.zeros(16000, np.float32)), [])


if __name__ == "__main__":
    unittest.main()
