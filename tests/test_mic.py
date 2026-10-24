"""Run with: python -m unittest tests.test_mic"""
import unittest

from engine.voice.mic import _candidates, open_settings


class FakeSD:
    """Copies Khoi's laptop: the Intel mic array only opens at 48 kHz."""

    HOSTS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"},
             {"name": "Windows WDM-KS"}]
    DEVICES = [
        {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100},
        {"name": "Microphone Array (Intel® Smart ", "hostapi": 0, "max_input_channels": 4, "default_samplerate": 44100},
        {"name": "Speaker (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 44100},
        {"name": "Stereo Mix (Realtek HD Audio Stereo input)", "hostapi": 3, "max_input_channels": 2, "default_samplerate": 48000},
        {"name": "", "hostapi": 2, "max_input_channels": 0, "default_samplerate": 48000},
    ]

    class default:
        device = (1, 3)

    def __init__(self, accepts=lambda dev, rate, ch: dev == 1 and rate == 48000):
        self.accepts = accepts
        self.tried = []

    def query_hostapis(self):
        return self.HOSTS

    def query_devices(self, device=None):
        return self.DEVICES if device is None else self.DEVICES[device]

    def InputStream(self, device, samplerate, channels, dtype):
        self.tried.append((device, samplerate, channels))
        if not self.accepts(device, samplerate, channels):
            raise RuntimeError("The specified format is not supported [MME error 32]")

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Stream()


class TestMic(unittest.TestCase):
    def test_finds_48k_on_the_default_mic(self):
        sd = FakeSD()
        s = open_settings(sd)
        self.assertEqual((s.device, s.rate, s.channels), (1, 48000, 1))
        self.assertEqual(sd.tried[0][0], 1)          # default mic first

    def test_skips_stereo_mix_and_empty_names(self):
        self.assertNotIn(3, _candidates(FakeSD(), None))
        self.assertNotIn(4, _candidates(FakeSD(), None))

    def test_falls_back_to_another_device(self):
        sd = FakeSD(accepts=lambda dev, rate, ch: dev == 0 and ch == 2)
        self.assertEqual(open_settings(sd).device, 0)

    def test_clear_error_when_nothing_opens(self):
        with self.assertRaises(RuntimeError) as ctx:
            open_settings(FakeSD(accepts=lambda *a: False))
        self.assertIn("Privacy", str(ctx.exception))

    def test_to_16k(self):
        try:
            import numpy as np
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("numpy/scipy not installed")
        from engine.voice.mic import to_16k

        out = to_16k(np.ones((48000, 4), dtype=np.float32), 48000)
        self.assertEqual(out.shape, (16000,))
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
