"""Run with: python -m unittest tests.test_replay  (record_events.py and fake_engine.py)"""
import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path

from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay
from tests import fake_engine
from tests.fake_engine import load, replay
from tests.helpers import FakeBus
from tests.record_events import keep, line, record


class TestRecordingFiles(unittest.TestCase):
    def test_frames_and_screenshots_are_never_kept(self):
        self.assertFalse(keep(messages.make(messages.DEBUG_FRAME, jpeg="...")))
        self.assertFalse(keep(messages.make(messages.TARGET_DEBUG, crop="...", x=1, y=2, ms=3)))
        self.assertFalse(keep(messages.make(messages.DEBUG_FACE, points=[])))
        self.assertFalse(keep({"no": "type"}))
        self.assertTrue(keep(messages.make(messages.MODE, mode="pointer")))

    def test_load_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(line(0, {"type": "mode", "mode": "pointer"}) + "not json\n{}\n"
                            + line(1.5, {"type": "mode", "mode": "paused"}), encoding="utf-8")
            self.assertEqual(load(path), [(0.0, {"type": "mode", "mode": "pointer"}),
                                          (1.5, {"type": "mode", "mode": "paused"})])

    def test_sample_recording_is_valid(self):
        events = load(fake_engine.RECORDINGS / "demo_flow.jsonl")
        self.assertGreater(len(events), 20)
        types = {m["type"] for _, m in events}
        self.assertTrue({"mode", "gaze", "dictation_preview", "confirm"} <= types)
        self.assertTrue(all(keep(m) for _, m in events))


class TestReplay(unittest.TestCase):
    def test_keeps_gaps_caps_long_ones_and_obeys_speed(self):
        bus, slept = FakeBus(), []

        async def fake_sleep(s):
            slept.append(s)

        events = [(0.0, {"type": "a"}), (0.5, {"type": "b"}), (60.0, {"type": "c"})]
        asyncio.run(replay(bus, events, speed=2.0, sleep=fake_sleep))
        self.assertEqual([m["type"] for m in bus.of(topics.TO_UI)], ["a", "b", "c"])
        self.assertEqual(slept, [0.25, fake_engine.MAX_GAP_S / 2])

    def test_empty_recording(self):
        asyncio.run(replay(FakeBus(), []))


@unittest.skipIf(importlib.util.find_spec("websockets") is None, "websockets not installed")
class TestRecordWhatTheEngineSends(unittest.TestCase):
    def test_round_trip_over_a_real_socket(self):
        sent = [messages.make(messages.MODE, mode="dictation"),
                messages.make(messages.DEBUG_FRAME, jpeg="abc"),      # must not be saved
                messages.make(messages.DICTATION_PREVIEW, words=[{"text": "hi", "conf": 0.9}])]

        async def run(out):
            bus = EventBus()
            relay = WebSocketRelay(bus)
            await relay.start("127.0.0.1", 0)
            try:
                task = asyncio.create_task(record(f"ws://127.0.0.1:{relay.port}", out, max_messages=2))
                while not relay.clients:
                    await asyncio.sleep(0.01)
                for msg in sent:
                    bus.publish(topics.TO_UI, msg)
                return await asyncio.wait_for(task, 3)
            finally:
                await relay.close()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rec.jsonl"
            self.assertEqual(asyncio.run(run(out)), 2)
            self.assertEqual([m for _, m in load(out)], [sent[0], sent[2]])

    def test_engine_closing_the_connection_ends_the_recording_cleanly(self):
        # e.g. fake_engine --once finishing: record() keeps what it got instead of a traceback
        async def run(out):
            bus = EventBus()
            relay = WebSocketRelay(bus)
            await relay.start("127.0.0.1", 0)
            task = asyncio.create_task(record(f"ws://127.0.0.1:{relay.port}", out, seconds=10))
            while not relay.clients:
                await asyncio.sleep(0.01)
            bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="pointer"))
            await asyncio.sleep(0.1)
            await relay.close()
            return await asyncio.wait_for(task, 3)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rec.jsonl"
            self.assertEqual(asyncio.run(run(out)), 1)
            self.assertEqual([m for _, m in load(out)], [messages.make(messages.MODE, mode="pointer")])


if __name__ == "__main__":
    unittest.main()
