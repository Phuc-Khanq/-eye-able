"""16 kHz mono mic stream, automatic gain for quiet voices, speech start and end.

Route: Khoi and Khoa (Khoa leads)
Publishes: topics.UTTERANCE (float32, 16 kHz mono) to dictation; messages.LISTENING to the UI
Subscribes: topics.from_ui(messages.SETTINGS) (pause_seconds)
APIs: sounddevice InputStream (via mic.py), silero-vad (512-sample frames at 16 kHz)

The mic callback only queues blocks; VAD and cutting run on this module's own thread
(CONTRIBUTING rule 9). The source and the VAD are injectable, so tests and laptops
without a working mic feed a WAV array instead:

    python -m engine.voice.audio_in clip.wav     # print the utterances found in a file

Audio stays in memory. Nothing here writes to disk (rule 10).
"""

from __future__ import annotations

import math
import queue
import threading
from collections import deque

import numpy as np

from engine.core import messages, topics
from engine.voice.mic import TARGET_RATE, to_16k

FRAME = 512                     # Silero needs exactly 512 samples at 16 kHz (32 ms)
FRAME_S = FRAME / TARGET_RATE
START_PROB = 0.5                # speech starts at this VAD probability...
STOP_PROB = 0.35                # ...and still counts as speech down to this one
PRE_ROLL_S = 0.3                # keep this much before the start, so the first sound isn't cut
TAIL_S = 0.25                   # and this much after the last speech frame
MIN_SPEECH_S = 0.25             # shorter bursts are coughs and clicks
MAX_UTTERANCE_S = 28.0          # Whisper's window is 30 s
MIN_PAUSE_S, MAX_PAUSE_S = 1.0, 4.0
TARGET_RMS = 0.08               # about -22 dBFS: where Whisper is happy
MAX_AUTO_GAIN = 10.0            # +20 dB at most, so room noise isn't blown up
LEVEL_EVERY_S = 0.1             # listening messages at 10 Hz
MAX_QUEUED_BLOCKS = 200         # about 6 s; if the thread falls behind, drop the oldest


# ----------------------------------------------------------------------- VAD
class SileroVad:
    """Speech probability per 512-sample frame. Needs torch and silero-vad (demo laptop)."""

    def __init__(self) -> None:
        import torch                              # heavy: import only when used (rule 6)
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self._model = load_silero_vad()

    def prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(frame), TARGET_RATE))

    def reset(self) -> None:
        self._model.reset_states()


class EnergyVad:
    """Loudness-only fallback for laptops without torch, and for tests."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def prob(self, frame: np.ndarray) -> float:
        return 1.0 if rms(frame) >= self.threshold else 0.0

    def reset(self) -> None:
        pass


def load_vad(warn=print):
    try:
        return SileroVad()
    except (ImportError, OSError) as exc:     # no torch, or a broken CUDA / DLL install
        warn(f"Silero VAD unavailable ({exc}); using a simple loudness check instead.")
        return EnergyVad()


# ------------------------------------------------------------------- sources
class MicSource:
    """The real mic, in whatever format Windows accepts (mic.open_settings)."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self.rate = TARGET_RATE

    def open(self, on_block):
        import sounddevice as sd

        from engine.voice.mic import open_settings

        s = open_settings(sd, self.device, verbose=True)   # RuntimeError if nothing opens
        self.rate = s.rate

        def callback(indata, frames, time_info, status) -> None:
            on_block(indata.mean(axis=1))                  # mono, and a copy of PortAudio's buffer

        stream = sd.InputStream(device=s.device, samplerate=s.rate, channels=s.channels,
                                dtype="float32", blocksize=int(s.rate * FRAME_S), callback=callback)
        stream.start()
        return stream                                       # has .close()


class ArraySource:
    """Audio already in memory (a test signal or a WAV file) instead of the mic."""

    def __init__(self, audio, rate: int = TARGET_RATE, realtime: bool = False) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        self.audio = audio.mean(axis=1) if audio.ndim == 2 else audio
        self.rate = rate
        self.realtime = realtime
        self._stop = threading.Event()

    @classmethod
    def from_wav(cls, path, realtime: bool = False) -> "ArraySource":
        from scipy.io import wavfile

        rate, data = wavfile.read(path)
        if data.dtype.kind in "iu":
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        return cls(data, rate, realtime)

    def blocks(self):
        step = max(1, int(self.rate * FRAME_S))
        for i in range(0, len(self.audio), step):
            yield self.audio[i:i + step]

    def open(self, on_block):
        def run() -> None:
            for block in self.blocks():
                if self._stop.wait(FRAME_S if self.realtime else 0):
                    return
                on_block(block)

        threading.Thread(target=run, name="array-source", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------- helpers
def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0


def auto_gain(audio: np.ndarray) -> np.ndarray:
    """Bring a quiet utterance up toward TARGET_RMS without clipping. Loud ones are left alone."""
    level = rms(audio)
    if level < 1e-6:
        return audio
    peak = float(np.max(np.abs(audio)))
    gain = min(TARGET_RMS / level, MAX_AUTO_GAIN, 0.95 / peak)
    return (audio * gain).astype(np.float32) if gain > 1.0 else audio


def level_0_1(audio: np.ndarray) -> float:
    """-60 dBFS -> 0, 0 dBFS -> 1, for the UI's level bar."""
    r = rms(audio)
    return 0.0 if r <= 1e-6 else min(1.0, max(0.0, (20 * math.log10(r) + 60) / 60))


# ------------------------------------------------------------------- AudioIn
class AudioIn:
    def __init__(self, bus, settings=None, source=None, vad=None) -> None:
        if settings is None:
            from engine.config import settings
        self.bus = bus
        self.source = source
        self.vad = vad
        self.mic_gain = float(getattr(settings, "mic_gain", 1.0))
        self.pause_seconds = _clamp_pause(getattr(settings, "pause_seconds", 1.5))
        self.rate = getattr(source, "rate", TARGET_RATE)
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
        self._stop = threading.Event()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._reset_stream()
        bus.subscribe(topics.from_ui(messages.SETTINGS), self.on_settings)

    def _reset_stream(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)   # 16 kHz samples not yet in a frame
        self._buf: deque[np.ndarray] = deque()           # source-rate audio, mono
        self._buf_start = 0                              # source sample index of _buf[0][0]
        self._n_in = 0                                   # source samples seen
        self._t = 0.0                                    # stream time at the end of the last frame
        self._speaking = False
        self._start_s = self._last_speech_s = self._speech_s = 0.0
        self._next_level_s = 0.0

    # ----------------------------------------------------------------- control
    def start(self) -> bool:
        """Open the mic (or the injected source) and start the audio thread.
        Returns False, with a status message for the UI, if no mic works."""
        if self.vad is None:
            self.vad = load_vad(lambda text: self._status("warn", text))
        if self.source is None:
            self.source = MicSource()
        try:
            self._stream = self.source.open(self._on_block)
        except Exception as exc:
            self._status("error", f"No microphone: {exc}")
            return False
        self.rate = self.source.rate
        self._thread = threading.Thread(target=self._run, name="audio-in", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.close()

    def on_settings(self, msg: dict) -> None:
        if "pause_seconds" in msg:
            self.pause_seconds = _clamp_pause(msg["pause_seconds"])

    def _on_block(self, block) -> None:
        """The mic callback: queue and return right away."""
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(block)

    def _run(self) -> None:
        failed = False
        while not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.feed(block)
            except Exception as exc:              # one bad block must not stop listening
                if not failed:
                    self._status("error", f"Audio processing failed: {exc}")
                failed = True

    # -------------------------------------------------------------- processing
    def feed(self, block) -> None:
        """Process one block of source-rate audio. Tests call this directly."""
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 2:
            block = block.mean(axis=1)
        if self.mic_gain != 1.0:
            block = np.clip(block * self.mic_gain, -1.0, 1.0)
        self._buf.append(block)
        self._n_in += len(block)
        self._pending = np.concatenate([self._pending, to_16k(block, self.rate)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            self._frame(np.ascontiguousarray(frame))

    def flush(self) -> None:
        """End of a file: finish an utterance that is still open."""
        if self._speaking:
            self._finish()

    def _frame(self, frame: np.ndarray) -> None:
        start, self._t = self._t, self._t + FRAME_S
        prob = self.vad.prob(frame)
        was_speaking = self._speaking
        if not self._speaking:
            if prob >= START_PROB:
                self._speaking = True
                self._start_s = max(0.0, start - PRE_ROLL_S)
                self._last_speech_s, self._speech_s = self._t, FRAME_S
            else:
                self._trim(self._t - PRE_ROLL_S)
        else:
            if prob >= STOP_PROB:
                self._last_speech_s = self._t
                self._speech_s += FRAME_S
            if (self._t - self._last_speech_s >= self.pause_seconds
                    or self._t - self._start_s >= MAX_UTTERANCE_S):
                self._finish()
        if self._speaking != was_speaking or self._t >= self._next_level_s:
            self._next_level_s = self._t + LEVEL_EVERY_S
            self.bus.publish(topics.TO_UI, messages.make(
                messages.LISTENING, level=round(level_0_1(frame), 2), speaking=self._speaking))

    def _finish(self) -> None:
        self._speaking = False
        self.vad.reset()
        end_s = min(self._last_speech_s + TAIL_S, self._t)
        if self._speech_s >= MIN_SPEECH_S:
            audio = self._slice(self._start_s, end_s)
            self.bus.publish(topics.UTTERANCE, auto_gain(to_16k(audio, self.rate)))
        self._trim(self._t - PRE_ROLL_S)

    def _slice(self, start_s: float, end_s: float) -> np.ndarray:
        audio = np.concatenate(list(self._buf)) if self._buf else np.zeros(0, np.float32)
        a = max(0, int(start_s * self.rate) - self._buf_start)
        b = max(a, int(end_s * self.rate) - self._buf_start)
        return audio[a:b]

    def _trim(self, keep_from_s: float) -> None:
        """Drop source audio older than keep_from_s (whole blocks only)."""
        keep_from = int(max(0.0, keep_from_s) * self.rate)
        while self._buf and self._buf_start + len(self._buf[0]) <= keep_from:
            self._buf_start += len(self._buf.popleft())

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))


def _clamp_pause(value) -> float:
    try:
        return min(MAX_PAUSE_S, max(MIN_PAUSE_S, float(value)))
    except (TypeError, ValueError):
        return 1.5


if __name__ == "__main__":
    import sys

    from engine.core.events import EventBus

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m engine.voice.audio_in clip.wav")
    bus = EventBus()
    bus.subscribe(topics.UTTERANCE, lambda a: print(f"utterance: {len(a) / TARGET_RATE:.2f} s, "
                                                   f"rms {rms(a):.3f}"))
    bus.subscribe(topics.TO_UI, lambda m: m["type"] == messages.STATUS and print(m["text"]))
    src = ArraySource.from_wav(sys.argv[1])
    audio_in = AudioIn(bus, source=src, vad=load_vad())
    for b in src.blocks():
        audio_in.feed(b)
    audio_in.flush()
