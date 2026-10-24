"""Transcribes each utterance with per-word confidence.

Route: Khoi and Khoa (Khoi leads)
Publishes: topics.TRANSCRIPT {"text", "words": [{"text", "conf"}], "ms"}
Subscribes: topics.UTTERANCE (float32 audio from audio_in) and topics.SPEECH_SPAN (when it was
            spoken, for the mouth gate: only the user's voice counts; see DictationWorker)
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
            name = resolve_model(s.whisper_model)
            kwargs = dict(device=s.whisper_device, compute_type=s.whisper_compute_type)
            try:
                # already downloaded: load offline (skips a ~2 s online check, nothing leaves the laptop)
                self._model = WhisperModel(name, local_files_only=True, **kwargs)
            except Exception:
                self._model = WhisperModel(name, **kwargs)   # first run: download it once

    def warm_up(self) -> None:
        """First CUDA call is slow; run one second of silence at startup."""
        import numpy as np

        self.load()
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio) -> list[dict]:
        """Return [{"text": str, "conf": float, "start": s, "end": s}, ...].

        start / end are seconds from the start of `audio` (Whisper's word timestamps). The
        DictationWorker uses them to line words up with the camera, then drops them: the
        transcript and the preview bubble only get "text" and "conf".
        """
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
                    start = float(getattr(w, "start", None) or 0.0)
                    end = max(start, float(getattr(w, "end", None) or 0.0))
                    words.append({"text": text, "conf": round(float(w.probability), 3),
                                  "start": round(start, 3), "end": round(end, 3)})
        if words and len(words) <= 3 and max(w["conf"] for w in words) < HALLUCINATION_CONF:
            return []
        return words


def words_to_text(words: list[dict]) -> str:
    # Whisper keeps punctuation on the word ("hello,"), so joining with spaces is enough.
    return " ".join(w["text"] for w in words)


def public_words(words: list[dict]) -> list[dict]:
    """Words as the transcript and the UI get them: text and confidence, no timing."""
    return [{"text": w["text"], "conf": w["conf"]} for w in words]


class DictationWorker:
    """Listens for utterances on the bus and transcribes them on its own thread.

    Only the user's voice counts (the mouth gate). With teammates talking near the mic, Whisper
    hears them too. audio_in publishes topics.SPEECH_SPAN (when the clip was spoken) right before
    each UTTERANCE; with `mouth` (engine.eyes.mouth.MouthTracker) the utterance is kept only if
    the camera saw the user's mouth moving while it was said (mouth_gate_mode "utterance"):
        f = mouth.moving_fraction(span start + first word start - pad, span start + last word end + pad)
        keep every word if f is None or f >= mouth_min_moving, else drop them all
    Per utterance, not per word: a word with little jaw movement ("search" in "Click search bar")
    scored low on its own and was dropped, so UI-Venus clicked "Bar". mouth_gate_mode "word" is
    the old gate (the same test on each word's own window), kept as a fallback.
    MouthTracker's contract (confirmed with Khang): it returns None when the face was seen in
    under half of the window's expected frames ("can't tell", so we keep the word); otherwise
    the share of moving frames among the frames where the face was seen. It is thread-safe
    (it locks and copies its history), so calling it from this Whisper thread while the
    camera thread adds frames is fine. If it raises, the word is kept (fail open: a broken
    camera must not mute the user). No mouth, mouth_gate off, or an utterance with no span:
    every word is kept, as before.
    """

    def __init__(self, bus, dictation: Dictation, mouth=None, settings=None,
                 clock=time.monotonic, log=print) -> None:
        if settings is None:
            from engine.config import settings
        self.bus = bus
        self.dictation = dictation
        self.mouth = mouth
        self.gate = bool(getattr(settings, "mouth_gate", True))
        self.gate_mode = str(getattr(settings, "mouth_gate_mode", "utterance"))
        self.min_moving = float(getattr(settings, "mouth_min_moving", 0.3))
        self.pad_ms = float(getattr(settings, "mouth_pad_ms", 150.0))
        self.status_every_s = float(getattr(settings, "mouth_status_every_s", 10.0))
        self._clock = clock
        self._log = log                       # console lines for tuning; never audio
        self._span: dict | None = None        # the SPEECH_SPAN for the next UTTERANCE
        self._span_lock = threading.Lock()
        self._last_ignored = float("-inf")    # clock time of the last "Ignored speech" status
        self._last_warning = float("-inf")    # ... and of the last "tracker failed" warning
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUED)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        bus.subscribe(topics.SPEECH_SPAN, self.on_speech_span)
        bus.subscribe(topics.UTTERANCE, self.on_utterance)

    def start(self, warm_up: bool = True) -> None:
        self._thread = threading.Thread(target=self._run, args=(warm_up,),
                                        name="whisper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def on_speech_span(self, span: dict) -> None:
        """Called on audio_in's thread right before the UTTERANCE it belongs to."""
        with self._span_lock:
            self._span = span

    def on_utterance(self, audio) -> None:
        """Called on audio_in's thread: queue the audio with its span and return immediately."""
        with self._span_lock:
            item, self._span = (audio, self._span), None
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()       # drop the oldest (its span goes with it)
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def _run(self, warm_up: bool) -> None:
        if warm_up:
            try:
                self.dictation.warm_up()
            except Exception as exc:
                self._error(f"Whisper could not load: {exc}")
                return
        while not self._stop.is_set():
            try:
                audio, span = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.process(audio, span)

    def process(self, audio, span: dict | None = None) -> None:
        """Transcribe one utterance and publish it. Tests call this directly."""
        t0 = time.perf_counter()
        try:
            words = self.dictation.transcribe(audio)
        except Exception as exc:               # one bad clip must not kill the demo
            self._error(f"Transcription failed: {exc}")
            return
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if words and span is not None and self.mouth is not None and self.gate:
            words = self._mouth_gate(words, span)
            if not words:
                self._ignored()
                return
        if not words:
            return
        self.bus.publish(topics.TRANSCRIPT, {
            "text": words_to_text(words),
            "words": public_words(words),
            "ms": ms,
        })

    # ----------------------------------------------------------- mouth gate
    def _fraction(self, start_ms: float, end_ms: float) -> float | None:
        """mouth.moving_fraction, or None (keep) if the tracker fails."""
        try:
            return self.mouth.moving_fraction(start_ms, end_ms)
        except Exception as exc:
            now = self._clock()
            if now - self._last_warning >= 60:
                self._last_warning = now
                self._log(f"[mouth] the mouth tracker failed ({exc}); keeping the words")
            return None

    def _mouth_gate(self, words: list[dict], span: dict) -> list[dict]:
        """The words said while the user's mouth moved (see the class docstring)."""
        start = float(span["start_ms"])
        per_word = [(w, self._fraction(start + w["start"] * 1000 - self.pad_ms,
                                       start + w["end"] * 1000 + self.pad_ms)) for w in words]

        def show(pairs):
            return " ".join(f"{w['text']}({'-' if f is None else f'{f:.2f}'})" for w, f in pairs)

        if self.gate_mode != "word":
            f = self._fraction(start + words[0]["start"] * 1000 - self.pad_ms,
                               start + words[-1]["end"] * 1000 + self.pad_ms)
            keep = f is None or f >= self.min_moving
            number = "-" if f is None else f"{f:.2f} {'>=' if keep else '<'} {self.min_moving:.2f}"
            self._log(f"[mouth] utterance {number}: {'kept' if keep else 'dropped'} "
                      f"{len(words)}/{len(words)}: {show(per_word)}")
            return list(words) if keep else []

        kept = [(w, f) for w, f in per_word if f is None or f >= self.min_moving]
        dropped = [(w, f) for w, f in per_word if not (f is None or f >= self.min_moving)]
        line = f"[mouth] kept {len(kept)}/{len(words)}: {show(kept)}"
        self._log(line + (f"  dropped: {show(dropped)}" if dropped else ""))
        kept_words = [w for w, _ in kept]
        if (dropped and kept_words and len(kept_words) <= 3
                and max(w["conf"] for w in kept_words) < HALLUCINATION_CONF):
            return []                           # what's left is a few unsure words: noise
        return kept_words

    def _ignored(self) -> None:
        now = self._clock()
        if now - self._last_ignored >= self.status_every_s:
            self._last_ignored = now
            from engine.core import messages

            self.bus.publish(topics.TO_UI, messages.make(
                messages.STATUS, level="info", text="Ignored speech while your mouth wasn't moving"))

    def _error(self, text: str) -> None:
        from engine.core import messages

        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level="error", text=text))
