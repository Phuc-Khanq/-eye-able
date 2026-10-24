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


if __name__ == "__main__":
    unittest.main()
