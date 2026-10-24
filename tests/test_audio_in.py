"""Run with: python -m unittest tests.test_audio_in

No mic and no model: a fake sounddevice stream and a loudness-based fake VAD
(same interface as silero_vad.VADIterator). Audio is synthetic silence and tone bursts.
"""
import unittest

import numpy as np

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.voice import audio_in
from engine.voice.audio_in import AudioIn, AutoGain, level_0_1
from engine.voice.mic import MicSettings, Resampler16k, to_16k
from tests.helpers import FakeBus

RATE = 16000


class FakeVad:
    """Loud block = speech. Says "end" after min_silence_samples of quiet, like VADIterator."""

    def __init__(self, min_silence_s=1.0, loud=0.02):
        self.min_silence_samples = int(min_silence_s * RATE)
        self.loud = loud
        self.reset_states()

    def reset_states(self):
        self.current_sample = 0
        self.triggered = False
        self.temp_end = 0

    def __call__(self, block):
        n = len(block)
        self.current_sample += n
        speech = float(np.sqrt(np.mean(block * block))) > self.loud
        if speech and self.temp_end:
            self.temp_end = 0
        if speech and not self.triggered:
            self.triggered = True
            return {"start": self.current_sample - n}
        if not speech and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                end, self.temp_end, self.triggered = self.temp_end, 0, False
                return {"end": end}
        return None


class FakeStream:
    def __init__(self, callback, **kwargs):
        self.callback = callback
        self.kwargs = kwargs
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


class FakeSD:
    def __init__(self):
        self.stream = None

    def InputStream(self, callback, **kwargs):
        self.stream = FakeStream(callback, **kwargs)
        return self.stream


class Settings:
    pause_seconds = 1.0


def silence(seconds, rate=RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def tone(seconds, amp=0.3, rate=RATE, hz=300):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def make(rate=RATE, channels=1, pause=1.0):
    bus = FakeBus()
    sd = FakeSD()
    vad = FakeVad(min_silence_s=pause)
    a = AudioIn(bus, Settings(), sd=sd, mic_settings=MicSettings(0, "fake", "MME", rate, channels),
                vad=vad)
    a.start(run_thread=False)
    return a, bus, sd


def play(a, sd, audio, chunk=1536, channels=1):
    """Feed audio through the sounddevice callback in mic-sized chunks, then process it."""
    for i in range(0, len(audio), chunk):
        part = audio[i:i + chunk]
        if channels > 1:
            part = np.repeat(part[:, None], channels, axis=1)
        else:
            part = part[:, None]
        sd.stream.callback(part, len(part), None, None)
        a.drain()


class TestAudioIn(unittest.TestCase):
    def test_silence_gives_no_utterance(self):
        a, bus, sd = make()
        play(a, sd, silence(5))
        self.assertEqual(bus.of(topics.UTTERANCE), [])
        self.assertTrue(bus.ui(messages.LISTENING))            # the level meter still updates
        self.assertFalse(any(m["speaking"] for m in bus.ui(messages.LISTENING)))

    def test_two_bursts_give_two_utterances(self):
        a, bus, sd = make()
        play(a, sd, np.concatenate([silence(0.5), tone(1.0), silence(1.5),
                                    tone(0.6), silence(1.5)]))
        utts = bus.of(topics.UTTERANCE)
        self.assertEqual(len(utts), 2)
        for u, spoken in zip(utts, (1.0, 0.6)):
            self.assertEqual(u.dtype, np.float32)
            self.assertEqual(u.ndim, 1)
            secs = len(u) / RATE
            # the tone plus a little pre-roll, but not the 1 s of silence VAD waited through
            self.assertGreaterEqual(secs, spoken)
            self.assertLess(secs, spoken + audio_in.PRE_ROLL_S + 0.1)

    def test_short_pause_does_not_split(self):
        a, bus, sd = make(pause=1.0)
        play(a, sd, np.concatenate([tone(0.8), silence(0.5), tone(0.8), silence(1.5)]))
        utts = bus.of(topics.UTTERANCE)
        self.assertEqual(len(utts), 1)
        self.assertGreater(len(utts[0]) / RATE, 2.0)

    def test_blip_is_dropped(self):
        a, bus, sd = make()
        play(a, sd, np.concatenate([silence(0.5), tone(0.1), silence(1.5)]))
        self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_48k_stereo_mic(self):
        a, bus, sd = make(rate=48000, channels=2)
        self.assertEqual(sd.stream.kwargs["samplerate"], 48000)
        self.assertEqual(sd.stream.kwargs["channels"], 2)
        self.assertEqual(sd.stream.kwargs["blocksize"], 1536)
        play(a, sd, np.concatenate([silence(0.5, 48000), tone(1.0, rate=48000),
                                    silence(1.5, 48000)]), channels=2)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)

    def test_long_speech_is_cut_for_whisper(self):
        a, bus, sd = make()
        play(a, sd, tone(audio_in.MAX_UTTERANCE_S + 3), chunk=16000)
        utts = bus.of(topics.UTTERANCE)
        self.assertEqual(len(utts), 1)
        self.assertLessEqual(len(utts[0]) / RATE, audio_in.MAX_UTTERANCE_S + 0.1)

    def test_pause_setting_from_ui(self):
        a, bus, sd = make(pause=1.0)
        bus.publish(topics.from_ui(messages.SETTINGS), {"type": "settings", "pause_seconds": 3})
        self.assertEqual(a.pause_seconds, 3.0)
        self.assertEqual(a._vad.min_silence_samples, 3 * RATE)
        play(a, sd, np.concatenate([tone(0.8), silence(2.0), tone(0.8), silence(3.5)]))
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)    # 2 s is no longer a pause
        bus.publish(topics.from_ui(messages.SETTINGS), {"type": "settings", "pause_seconds": 99})
        self.assertEqual(a.pause_seconds, 4.0)                 # clamped to 1-4

    def test_callback_only_queues(self):
        a, bus, sd = make()
        sd.stream.callback(tone(1.0)[:, None], 16000, None, None)
        self.assertEqual(bus.log, [])                          # nothing happens on the audio thread
        a.drain()
        self.assertTrue(bus.log)

    def test_full_queue_drops_instead_of_blocking(self):
        a, bus, sd = make()
        for _ in range(audio_in.RAW_QUEUE_MAX + 5):
            sd.stream.callback(silence(0.01)[:, None], 160, None, None)
        self.assertEqual(a.dropped_chunks, 5)

    def test_stop_closes_stream(self):
        a, bus, sd = make()
        a.stop()
        self.assertTrue(sd.stream.closed)


class TestAutoGain(unittest.TestCase):
    def run_gain(self, amp, seconds=4.0):
        g = AutoGain()
        audio = np.concatenate([silence(0.5) + 1e-4, tone(seconds, amp=amp)])
        out = [g(audio[i:i + 512]) for i in range(0, len(audio) - 511, 512)]
        return g, np.concatenate(out)

    def test_quiet_voice_is_lifted(self):
        g, out = self.run_gain(amp=0.01)
        self.assertGreater(g.gain, 3.0)
        self.assertLessEqual(g.gain, audio_in.MAX_GAIN)
        self.assertGreater(np.abs(out[-RATE:]).max(), 0.03)

    def test_normal_voice_is_left_alone(self):
        g, _ = self.run_gain(amp=0.1)
        self.assertAlmostEqual(g.gain, 1.0, places=2)

    def test_never_clips(self):
        g, out = self.run_gain(amp=0.02)
        loud = g(tone(0.032, amp=0.9)[:512])
        self.assertLessEqual(float(np.abs(loud).max()), 0.98 + 1e-6)

    def test_does_not_boost_noise_much(self):
        g = AutoGain()
        rng = np.random.default_rng(0)
        for _ in range(200):
            g((rng.standard_normal(512) * 0.005).astype(np.float32))
        self.assertLess(g.gain, 2.5)    # noise * gain stays under NOISE_CEILING

    def test_level_meter(self):
        self.assertEqual(level_0_1(silence(0.03)), 0.0)
        self.assertGreater(level_0_1(tone(0.03, amp=0.5)), 0.8)


class TestResampler(unittest.TestCase):
    def test_stream_matches_whole_clip(self):
        for rate in (48000, 44100):
            x = tone(1.0, rate=rate, hz=440)
            r = Resampler16k(rate)
            y = np.concatenate([r(x[i:i + 1000]) for i in range(0, len(x), 1000)])
            ref = to_16k(x, rate)
            np.testing.assert_allclose(y[100:-100], ref[100:len(y) - 100], atol=1e-6)


class TestRealSilero(unittest.TestCase):
    """The real Silero VAD (bundled with silero-vad, no download): beeps are not speech."""

    def test_silence_and_tones_are_not_speech(self):
        try:
            vad = audio_in.load_vad(1.0)
        except Exception as exc:
            self.skipTest(f"silero-vad not available: {exc}")
        bus = FakeBus()
        sd = FakeSD()
        a = AudioIn(bus, Settings(), sd=sd, mic_settings=MicSettings(0, "fake", "MME", RATE, 1),
                    vad=vad)
        a.start(run_thread=False)
        play(a, sd, np.concatenate([silence(1.0), tone(0.5, hz=1000), silence(1.5),
                                    tone(0.5, hz=440), silence(1.5)]))
        self.assertEqual(bus.of(topics.UTTERANCE), [])


class TestTorchFreeSilero(unittest.TestCase):
    """audio_in runs Silero's ONNX model without torch (4 s faster to start).
    It must give the same probabilities and events as the silero_vad package."""

    def test_matches_silero_package(self):
        try:
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except Exception as exc:
            self.skipTest(f"silero-vad/torch not available: {exc}")
        rng = np.random.default_rng(1)
        # a vowel-like signal: harmonics with a slow wobble, between silences
        t = np.arange(int(1.5 * RATE)) / RATE
        voice = sum(np.sin(2 * np.pi * f * t * (1 + 0.02 * np.sin(6 * t))) / k
                    for k, f in enumerate((140, 280, 420, 700, 1100), start=1))
        x = np.concatenate([silence(0.5), 0.2 * voice, silence(2.0)]).astype(np.float32)
        x += (0.001 * rng.standard_normal(len(x))).astype(np.float32)
        blocks = [x[i:i + 512] for i in range(0, len(x) - 511, 512)]

        mine = audio_in.SileroOnnx()
        ref = load_silero_vad(onnx=True)
        for b in blocks:
            self.assertAlmostEqual(mine(b), ref(torch.from_numpy(b), RATE).item(), places=5)

        ours = audio_in.load_vad(1.0)
        theirs = VADIterator(load_silero_vad(onnx=True), threshold=audio_in.VAD_THRESHOLD,
                             sampling_rate=RATE, min_silence_duration_ms=1000,
                             speech_pad_ms=audio_in.SPEECH_PAD_MS)
        self.assertEqual([ours(b) for b in blocks],
                         [theirs(torch.from_numpy(b)) for b in blocks])


class TestEventBus(unittest.TestCase):
    def test_publish_reaches_every_subscriber(self):
        bus = EventBus()
        got = []
        bus.subscribe("t", lambda p: got.append(("a", p)))
        bus.subscribe("t", lambda p: got.append(("b", p)))
        bus.publish("t", 1)
        bus.publish("other", 2)
        self.assertEqual(got, [("a", 1), ("b", 1)])

    def test_failing_handler_does_not_stop_others(self):
        bus = EventBus()
        got = []

        def bad(_):
            raise ValueError("boom")

        bus.subscribe("t", bad)
        bus.subscribe("t", got.append)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bus.publish("t", 5)
        self.assertEqual(got, [5])

    def test_threads(self):
        import threading

        bus = EventBus()
        got = []
        lock = threading.Lock()

        def add(p):
            with lock:
                got.append(p)

        bus.subscribe("t", add)
        threads = [threading.Thread(target=lambda: [bus.publish("t", i) for i in range(500)])
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(got), 2000)


if __name__ == "__main__":
    unittest.main()
