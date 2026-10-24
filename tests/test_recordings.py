"""Run with: python -m unittest tests.test_recordings

The recordings are what Nhi builds the UI from (python -m tests.fake_engine), so they must load, and
all_messages.jsonl must have every Engine -> UI message with exactly the fields messages.FIELDS lists
(the same fields the comments in engine/core/messages.py describe). If this fails after an engine
change, run python -m tests.make_all_messages and update messages.py.
"""
import unittest
from pathlib import Path

from engine.core import messages
from tests import fake_engine, make_all_messages

RECORDINGS = Path(__file__).resolve().parent / "recordings"


class TestRecordings(unittest.TestCase):
    def test_every_recording_loads_in_fake_engine(self):
        paths = sorted(RECORDINGS.glob("*.jsonl"))
        self.assertIn(make_all_messages.OUT, paths)
        for path in paths:
            with self.subTest(path=path.name):
                events = fake_engine.load(path)
                self.assertGreater(len(events), 0)
                self.assertEqual([t for t, _ in events], sorted(t for t, _ in events))   # replays in order

    def test_all_messages_has_everything_the_ui_needs(self):
        events = fake_engine.load(make_all_messages.OUT)
        self.assertEqual(make_all_messages.missing(events), [])

    def test_every_message_has_the_fields_messages_py_lists(self):
        for path in sorted(RECORDINGS.glob("*.jsonl")):
            for _, msg in fake_engine.load(path):
                with self.subTest(path=path.name, msg=msg):
                    self.assertIn(msg["type"], messages.FIELDS)
                    required, optional = messages.FIELDS[msg["type"]]
                    fields = set(msg) - {"type"}
                    self.assertLessEqual(required, fields)
                    self.assertLessEqual(fields, required | optional)

    def test_no_camera_frames_in_recordings(self):
        for path in sorted(RECORDINGS.glob("*.jsonl")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("jpeg", text)
            self.assertNotIn("crop", text)

    def test_the_engine_still_sends_all_of_it(self):
        # the real pieces, run again: the committed file can't silently fall behind the engine
        events = make_all_messages.session()
        self.assertEqual(make_all_messages.missing(events), [])
        for _, msg in events:
            required, optional = messages.FIELDS[msg["type"]]
            self.assertTrue(required <= set(msg) - {"type"} <= required | optional, msg)


if __name__ == "__main__":
    unittest.main()
