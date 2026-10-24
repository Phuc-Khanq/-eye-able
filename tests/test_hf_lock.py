"""Two models never load at the same time (engine/core/hf.py): the UI-Venus loader and the clean-up loader
hold the same lock, so a lazy transformers import on one thread can't be seen half done by the other.

Run with: python -m unittest tests.test_hf_lock
"""
import threading
import time
import unittest

from engine.core import hf
from engine.eyes.screen_target import ScreenTarget
from engine.voice.rewrite import Rewriter


class TestLoadLock(unittest.TestCase):
    def test_the_two_loaders_never_overlap(self):
        inside, overlap = [], []

        def slow(label):
            def load(self, *a, **k):
                inside.append(label)
                if len(inside) > 1:
                    overlap.append(tuple(inside))
                time.sleep(0.15)
                inside.remove(label)
                return 0.0 if label == "venus" else None
            return load

        st, rw = ScreenTarget.__new__(ScreenTarget), Rewriter.__new__(Rewriter)
        st.model = rw._model = rw._tokenizer = None
        real = ScreenTarget._load, Rewriter._load
        ScreenTarget._load, Rewriter._load = slow("venus"), slow("rewrite")
        try:
            threads = [threading.Thread(target=st.load), threading.Thread(target=rw.load)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            ScreenTarget._load, Rewriter._load = real
        self.assertEqual(overlap, [])

    def test_the_lock_is_re_entrant(self):
        with hf.LOAD_LOCK:
            with hf.LOAD_LOCK:
                pass


if __name__ == "__main__":
    unittest.main()
