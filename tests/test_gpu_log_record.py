"""--record (show the dictation card to OBS) and --gpu-log (video memory in use, and the peak).

Run with: python -m unittest tests.test_gpu_log_record
"""
import subprocess
import unittest
from types import SimpleNamespace

from engine.gpu_log import GpuLog, format_line, read_gpu_memory
from engine.main import apply_record


def smi(text):
    def run(cmd, **kw):
        return SimpleNamespace(stdout=text)
    return run


class TestRecordFlag(unittest.TestCase):
    def test_record_shows_the_card_to_recorders(self):
        s = SimpleNamespace(card_hide_from_capture=True)
        self.assertTrue(apply_record(s, True))
        self.assertFalse(s.card_hide_from_capture)

    def test_without_record_nothing_changes(self):
        s = SimpleNamespace(card_hide_from_capture=True)
        self.assertFalse(apply_record(s, False))
        self.assertTrue(s.card_hide_from_capture)


class TestGpuRead(unittest.TestCase):
    def test_reads_used_and_total(self):
        self.assertEqual(read_gpu_memory(smi("5321, 16303\n")), (5321, 16303))

    def test_the_first_gpu_counts(self):
        self.assertEqual(read_gpu_memory(smi("100, 200\n300, 400\n")), (100, 200))

    def test_no_driver_is_none(self):
        def missing(cmd, **kw):
            raise FileNotFoundError("nvidia-smi")
        self.assertIsNone(read_gpu_memory(missing))

    def test_a_failing_command_is_none(self):
        def failing(cmd, **kw):
            raise subprocess.CalledProcessError(9, cmd)
        self.assertIsNone(read_gpu_memory(failing))

    def test_garbage_output_is_none(self):
        self.assertIsNone(read_gpu_memory(smi("N/A, N/A\n")))


class TestGpuLog(unittest.TestCase):
    def test_the_peak_only_goes_up(self):
        readings = iter([(4000, 16000), (9000, 16000), (6000, 16000)])
        log = GpuLog(read=lambda: next(readings), out=lambda *_: None)
        lines = [log.sample() for _ in range(3)]
        self.assertEqual(log.peak, 9000)
        self.assertIn("peak 9000 MB", lines[2])

    def test_warns_over_the_limit(self):
        self.assertIn("WARNING", format_line(14500, 16000, 14500, 14000))
        self.assertNotIn("WARNING", format_line(13900, 16000, 13900, 14000))

    def test_no_gpu_says_so_once_and_stops(self):
        printed = []
        log = GpuLog(every_s=0.01, read=lambda: None, out=printed.append)
        log._loop()
        self.assertEqual(len(printed), 1)
        self.assertIn("no NVIDIA GPU", printed[0])

    def test_stop_returns_the_final_line(self):
        log = GpuLog(read=lambda: (7000, 16000), out=lambda *_: None)
        self.assertIn("7000 / 16000", log.stop())


if __name__ == "__main__":
    unittest.main()
