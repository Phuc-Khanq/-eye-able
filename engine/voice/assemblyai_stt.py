"""AssemblyAI speech-to-text, as an opt-in alternative to local Whisper.

Route: Khoi and Khoa (Khoi leads)
Used by: dictation.Dictation when settings.stt_backend == "assemblyai"
APIs: POST /v2/upload, POST /v2/transcript, GET /v2/transcript/{id} (plain HTTPS, no SDK)

Privacy: this sends each finished utterance to AssemblyAI. That is why it is off by
default (CONTRIBUTING rule 10). The key comes from the ASSEMBLYAI_API_KEY environment
variable or the git-ignored .env file at the repo root, never from code.

Cost: AssemblyAI bills per second of audio, not per token, so we only send what is needed:
one VAD-cut utterance at a time, 16 kHz mono 16-bit, silence trimmed from both ends,
clips too short to hold a word skipped, and no paid add-ons requested.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

BASE_URL = "https://api.assemblyai.com"
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.25        # shorter than this is a click or breath, not a word
TRIM_THRESHOLD = 0.01     # amplitude below this at the clip edges counts as silence
TRIM_PAD = 0.15           # seconds of lead-in/out kept so word edges are not clipped
POLL_SECONDS = 0.25
TIMEOUT_SECONDS = 20


class AssemblyAIError(RuntimeError):
    pass


def load_api_key() -> str:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ASSEMBLYAI_API_KEY":
                return value.strip().strip("\"'")
    raise AssemblyAIError("No AssemblyAI key. Set ASSEMBLYAI_API_KEY or put it in .env")


def trim_silence(audio):
    """Drop quiet audio from both ends (keeps a small pad). Returns float32."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    loud = np.flatnonzero(np.abs(audio) > TRIM_THRESHOLD)
    if loud.size == 0:
        return audio[:0]
    pad = int(TRIM_PAD * SAMPLE_RATE)
    return audio[max(0, loud[0] - pad): loud[-1] + pad + 1]


def to_wav_bytes(audio) -> bytes:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class AssemblyAI:
    def __init__(self, api_key: str | None = None, speech_models: list[str] | None = None,
                 base_url: str = BASE_URL) -> None:
        self.api_key = api_key
        self.speech_models = speech_models or ["universal-3-5-pro", "universal-2"]
        self.base_url = base_url

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str = "application/json") -> dict:
        if not self.api_key:
            self.api_key = load_api_key()
        req = urllib.request.Request(self.base_url + path, data=body, method=method, headers={
            "authorization": self.api_key, "content-type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise AssemblyAIError(f"AssemblyAI {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AssemblyAIError(f"AssemblyAI unreachable: {exc.reason}") from exc

    def transcribe(self, audio, keyterms: list[str] | None = None) -> list[dict]:
        """Return [{"text", "conf"}, ...] for 16 kHz mono float32 audio."""
        audio = trim_silence(audio)
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return []
        upload = self._request("POST", "/v2/upload", to_wav_bytes(audio), "application/octet-stream")
        job = {"audio_url": upload["upload_url"], "language_code": "en",
               "speech_models": self.speech_models}
        if keyterms:
            job["keyterms_prompt"] = keyterms[:200]
        tid = self._request("POST", "/v2/transcript", json.dumps(job).encode("utf-8"))["id"]

        deadline = time.monotonic() + TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            result = self._request("GET", f"/v2/transcript/{tid}")
            if result["status"] == "completed":
                return [{"text": w["text"], "conf": round(float(w["confidence"]), 3)}
                        for w in result.get("words") or [] if w["text"].strip()]
            if result["status"] == "error":
                raise AssemblyAIError(result.get("error", "transcription failed"))
            time.sleep(POLL_SECONDS)
        raise AssemblyAIError("AssemblyAI timed out")
