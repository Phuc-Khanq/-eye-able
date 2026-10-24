"""Transcribes each utterance with per-word confidence.

Route: Khoi and Khoa (Khoi leads)
Publishes: topics.TRANSCRIPT {"text", "words": [{"text", "conf"}], "ms"}
Subscribes: topics.UTTERANCE (float32 audio from audio_in)
APIs: faster-whisper WhisperModel("large-v3-turbo", device="cuda", compute_type="float16"),
      transcribe(word_timestamps=True, hotwords=...)

The mode manager decides whether a transcript is a command or dictation, and only
dictation becomes a dictation_preview, so "send it" never flashes up as typed text.

Whisper runs in its own worker thread (CONTRIBUTING rule 9): on_utterance only
puts audio on a queue, so audio_in's thread is never blocked.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from engine.core import topics

REPO_ROOT = Path(__file__).resolve().parents[2]

BEAM_SIZE = 1             # greedy: fastest; short utterances barely gain from beam search
NO_SPEECH_SKIP = 0.6      # drop a segment Whisper thinks is silence
HALLUCINATION_CONF = 0.35  # a 1-3 word result this unsure is usually noise ("Thank you.")
MAX_QUEUED = 3            # if Whisper falls behind, drop the oldest utterance


def resolve_model(name: str) -> str:
    """A folder under the repo root (e.g. "models/whisper-elder-ct2") becomes its full path;
    anything else ("large-v3-turbo", "small") is a model name faster-whisper downloads."""
    path = Path(name)
    if not path.is_absolute() and (REPO_ROOT / path).is_dir():
        return str(REPO_ROOT / path)
    return name


class Dictation:
    """Wraps faster-whisper. Pass `model` in tests to skip loading Whisper."""

    def __init__(self, settings=None, model=None, hotwords: str | None = None) -> None:
        self.settings = settings
        self.hotwords = hotwords  # e.g. names the user says often: "Khoa Nhi Gmail"
        self._model = model

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy: import only when used (rule 6)

            s = self.settings
            if s is None:
                from engine.config import settings as s
            self._model = WhisperModel(resolve_model(s.whisper_model), device=s.whisper_device,
                                       compute_type=s.whisper_compute_type)

    def warm_up(self) -> None:
        """First CUDA call is slow; run one second of silence at startup."""
        import numpy as np

        self.load()
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio) -> list[dict]:
        """Return [{"text": str, "conf": float}, ...] for the preview bubble."""
        self.load()
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=BEAM_SIZE,
            word_timestamps=True,
            vad_filter=False,                # audio_in already cut the utterance with Silero VAD
            condition_on_previous_text=False,
            hotwords=self.hotwords,
        )
        words: list[dict] = []
        for seg in segments:                 # segments is lazy: this loop runs the model
            if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_SKIP:
                continue
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append({"text": text, "conf": round(float(w.probability), 3)})
        if words and len(words) <= 3 and max(w["conf"] for w in words) < HALLUCINATION_CONF:
            return []
        return words


def words_to_text(words: list[dict]) -> str:
    # Whisper keeps punctuation on the word ("hello,"), so joining with spaces is enough.
    return " ".join(w["text"] for w in words)


class DictationWorker:
    """Listens for utterances on the bus and transcribes them on its own thread."""

    def __init__(self, bus, dictation: Dictation) -> None:
        self.bus = bus
        self.dictation = dictation
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUED)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        bus.subscribe(topics.UTTERANCE, self.on_utterance)

    def start(self, warm_up: bool = True) -> None:
        self._thread = threading.Thread(target=self._run, args=(warm_up,),
                                        name="whisper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def on_utterance(self, audio) -> None:
        """Called on audio_in's thread: queue and return immediately."""
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            try:
                self._queue.get_nowait()       # drop the oldest
            except queue.Empty:
                pass
            self._queue.put_nowait(audio)

    def _run(self, warm_up: bool) -> None:
        if warm_up:
            try:
                self.dictation.warm_up()
            except Exception as exc:
                self._error(f"Whisper could not load: {exc}")
                return
        while not self._stop.is_set():
            try:
                audio = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.process(audio)

    def process(self, audio) -> None:
        """Transcribe one utterance and publish it. Tests call this directly."""
        t0 = time.perf_counter()
        try:
            words = self.dictation.transcribe(audio)
        except Exception as exc:               # one bad clip must not kill the demo
            self._error(f"Transcription failed: {exc}")
            return
        if not words:
            return
        self.bus.publish(topics.TRANSCRIPT, {
            "text": words_to_text(words),
            "words": words,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    def _error(self, text: str) -> None:
        from engine.core import messages

        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level="error", text=text))
