"""Run with: python -m unittest tests.test_assemblyai  (no network, no key needed)"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from engine.voice import assemblyai_stt as stt
from engine.voice.dictation import Dictation


def tone(seconds: float, amp: float = 0.3):
    t = np.linspace(0, seconds, int(seconds * stt.SAMPLE_RATE), dtype=np.float32)
    return amp * np.sin(2 * np.pi * 440 * t)


class TrimTests(unittest.TestCase):
    def test_trims_silence_but_keeps_pad(self):
        audio = np.concatenate([np.zeros(16000), tone(0.5), np.zeros(16000)]).astype(np.float32)
        out = stt.trim_silence(audio)
        pad = int(stt.TRIM_PAD * stt.SAMPLE_RATE)
        self.assertLess(len(out), len(audio) / 2)
        self.assertAlmostEqual(len(out), 8000 + 2 * pad, delta=200)

    def test_all_silence_is_empty(self):
        self.assertEqual(len(stt.trim_silence(np.zeros(16000, dtype=np.float32))), 0)

    def test_wav_is_16khz_mono_16bit(self):
        import io
        import wave

        with wave.open(io.BytesIO(stt.to_wav_bytes(tone(0.5)))) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))


class FlowTests(unittest.TestCase):
    def make(self, results):
        client = stt.AssemblyAI(api_key="k")
        calls = []

        def fake(method, path, body=None, content_type="application/json"):
            calls.append((method, path, body))
            if path == "/v2/upload":
                return {"upload_url": "https://cdn/x"}
            if method == "POST":
                return {"id": "t1"}
            return results.pop(0)

        client._request = fake
        return client, calls

    def test_returns_words_with_confidence(self):
        client, calls = self.make([
            {"status": "processing"},
            {"status": "completed", "words": [{"text": "Hello,", "confidence": 0.98765},
                                              {"text": "world.", "confidence": 0.9}]},
        ])
        with mock.patch.object(stt.time, "sleep"):
            words = client.transcribe(tone(1.0), keyterms=["Khoa"])
        self.assertEqual(words, [{"text": "Hello,", "conf": 0.988}, {"text": "world.", "conf": 0.9}])
        import json
        job = json.loads(calls[1][2])
        self.assertEqual(job["keyterms_prompt"], ["Khoa"])
        self.assertEqual(job["language_code"], "en")

    def test_short_clip_is_not_sent(self):
        client, calls = self.make([])
        self.assertEqual(client.transcribe(tone(0.1)), [])
        self.assertEqual(calls, [])

    def test_error_status_raises(self):
        client, _ = self.make([{"status": "error", "error": "bad audio"}])
        with self.assertRaises(stt.AssemblyAIError):
            client.transcribe(tone(1.0))


class DictationTests(unittest.TestCase):
    def test_cloud_backend_is_used_and_noise_filtered(self):
        d = Dictation(settings=SimpleNamespace(stt_backend="assemblyai"))
        d._cloud = mock.Mock()
        d._cloud.transcribe.return_value = [{"text": "Thank you.", "conf": 0.1}]
        self.assertEqual(d.transcribe(tone(1.0)), [])
        d._cloud.transcribe.return_value = [{"text": "Send", "conf": 0.9}]
        self.assertEqual(d.transcribe(tone(1.0)), [{"text": "Send", "conf": 0.9}])


if __name__ == "__main__":
    unittest.main()
