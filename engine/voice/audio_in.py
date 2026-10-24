"""16 kHz mono mic stream, automatic gain for quiet voices, speech start and end.

Route: Khoi and Khoa (Khoa leads)
Publishes: listening, finished utterances (float32 audio) to dictation
Subscribes: settings (pause_seconds)
APIs: sounddevice InputStream (512-sample blocks), silero-vad VADIterator

How audio flows:
  sounddevice callback   copies each chunk onto a queue and returns (never blocks)
  "audio-in" thread      mic.Resampler16k -> 512-sample blocks -> gain -> Silero VAD
                         -> on speech end: topics.UTTERANCE (16 kHz mono float32)

An utterance ends after settings.pause_seconds of silence (the user picks 1 to 4 s),
or at MAX_UTTERANCE_S, because Whisper only hears 30 s at a time.
Audio stays in memory; nothing is written to disk (CONTRIBUTING rule 10).
"""

from __future__ import annotations

import queue
import threading
from collections import deque

from engine.core import messages, topics
from engine.voice import mic

BLOCK = 512                 # samples at 16 kHz = 32 ms, the size Silero VAD wants
RATE = mic.TARGET_RATE
VAD_THRESHOLD = 0.5
SPEECH_PAD_MS = 150         # keep a little audio around speech so first/last sounds aren't cut
PRE_ROLL_S = 0.3            # audio kept from before VAD noticed speech
MIN_SPEECH_S = 0.25         # shorter blips (a cough, a click) are dropped
MAX_UTTERANCE_S = 28.0
LEVEL_EVERY = 3             # send a listening level every 3 blocks (about 10 per second)
RAW_QUEUE_MAX = 200         # about 6 s of mic chunks; if we fall that far behind, drop audio

# Gentle automatic gain: lift quiet voices toward TARGET_RMS, never turn anyone down.
TARGET_RMS = 0.05           # about -26 dBFS, a normal speaking level
MAX_GAIN = 8.0              # +18 dB at most
NOISE_CEILING = 0.01        # never lift background noise above this
GAIN_SMOOTHING = 0.05       # how fast the gain follows the voice (per loud block)
LOUD_OVER_NOISE = 3.0       # a block this much above the noise floor counts as voice


class AutoGain:
    """Slow automatic gain. Tracks the noise floor and the level of loud (voice) blocks."""

    def __init__(self) -> None:
        self.noise = 0.002       # running noise floor (RMS)
        self.voice = TARGET_RMS  # running voice level (RMS); starts neutral = gain 1
        self.gain = 1.0

    def __call__(self, block):
        import numpy as np

        rms = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        # noise floor falls fast and rises slowly, so it settles on the quiet parts
        self.noise += (rms - self.noise) * (0.2 if rms < self.noise else 0.005)
        self.noise = max(self.noise, 1e-5)
        if rms > LOUD_OVER_NOISE * self.noise and rms > 1e-4:
            self.voice += (rms - self.voice) * GAIN_SMOOTHING
        wanted = min(MAX_GAIN, TARGET_RMS / max(self.voice, 1e-6),
                     NOISE_CEILING / self.noise)
        wanted = max(1.0, wanted)
        self.gain += (wanted - self.gain) * 0.1
        out = block * self.gain
        peak = float(np.abs(out).max()) if len(out) else 0.0
        if peak > 0.98:          # limiter: a sudden loud sound must not clip
            out = out * (0.98 / peak)
        return out.astype(np.float32)


def level_0_1(block) -> float:
    """RMS as a 0-1 meter: -60 dBFS (or quieter) is 0, 0 dBFS is 1."""
    import numpy as np

    rms = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
    if rms <= 1e-6:
        return 0.0
    return round(min(1.0, max(0.0, 1.0 + 20.0 * np.log10(rms) / 60.0)), 3)


def load_vad(pause_seconds: float):
    """Silero VAD (ONNX, small and bundled with the package; no download)."""
    from silero_vad import VADIterator, load_silero_vad  # heavy: import only when used

    model = load_silero_vad(onnx=True)
    return VADIterator(model, threshold=VAD_THRESHOLD, sampling_rate=RATE,
                       min_silence_duration_ms=int(pause_seconds * 1000),
                       speech_pad_ms=SPEECH_PAD_MS)


class AudioIn:
    """Mic -> utterances on the bus.

    Tests pass `sd` (a fake sounddevice), `mic_settings` and `vad` to skip the real mic
    and model, then call the stream's callback and drain() themselves.
    """

    def __init__(self, bus, settings=None, sd=None, mic_settings=None, vad=None) -> None:
        if settings is None:
            from engine.config import settings
        self.bus = bus
        self.settings = settings
        self.pause_seconds = float(settings.pause_seconds)
        self._sd = sd
        self.mic_settings = mic_settings
        self._vad = vad
        self._raw: queue.Queue = queue.Queue(maxsize=RAW_QUEUE_MAX)
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.dropped_chunks = 0
        self._reset_stream_state()
        bus.subscribe(topics.from_ui(messages.SETTINGS), self.on_settings)

    def _reset_stream_state(self) -> None:
        import numpy as np

        self._resampler = None
        self._pending = np.zeros(0, dtype=np.float32)  # 16 kHz samples not yet a full block
        self._gain = AutoGain()
        self._blocks = 0                               # blocks processed (sample clock / BLOCK)
        self._pre = deque(maxlen=max(1, round(PRE_ROLL_S * RATE / BLOCK)))
        self._speech: list = []                        # blocks of the current utterance
        self._speech_start = 0                         # sample index of _speech[0]
        self._voice_start = 0                          # sample index where VAD heard speech
        self._vad_offset = 0                           # our sample index when the VAD clock was 0
        self.speaking = False

    # ------------------------------------------------------------------ control
    def start(self, run_thread: bool = True) -> None:
        """Open the mic and start the processing thread. Raises RuntimeError if no mic opens."""
        if self._sd is None:
            import sounddevice as sd  # heavy: import only when used

            self._sd = sd
        if self.mic_settings is None:
            self.mic_settings = mic.open_settings(self._sd, verbose=True)
        if self._vad is None:
            self._vad = load_vad(self.pause_seconds)
        self._resampler = mic.Resampler16k(self.mic_settings.rate)
        s = self.mic_settings
        native_block = max(BLOCK, round(BLOCK * s.rate / RATE))
        self._stream = self._sd.InputStream(device=s.device, samplerate=s.rate,
                                            channels=s.channels, dtype="float32",
                                            blocksize=native_block, callback=self._callback)
        self._stream.start()
        self._stop.clear()
        if run_thread:
            self._thread = threading.Thread(target=self._run, name="audio-in", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def on_settings(self, msg: dict) -> None:
        value = msg.get("pause_seconds")
        if value is None:
            return
        self.pause_seconds = min(4.0, max(1.0, float(value)))
        if self._vad is not None:
            # VADIterator reads min_silence_samples on every block, so this applies right away
            self._vad.min_silence_samples = int(self.pause_seconds * RATE)

    # --------------------------------------------------------- sounddevice side
    def _callback(self, indata, frames, time_info, status) -> None:
        """Runs on PortAudio's thread: copy and return, nothing else."""
        try:
            self._raw.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_chunks += 1

    # ------------------------------------------------------------ worker side
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._raw.get(timeout=0.2)
            except queue.Empty:
                continue
            self.process(chunk)

    def drain(self) -> None:
        """Process every queued chunk now (tests, or a thread-less start)."""
        while True:
            try:
                chunk = self._raw.get_nowait()
            except queue.Empty:
                return
            self.process(chunk)

    def process(self, chunk) -> None:
        """Turn one native-rate mic chunk into blocks and run them through gain and VAD."""
        import numpy as np

        if self._resampler is None:
            self._resampler = mic.Resampler16k(self.mic_settings.rate if self.mic_settings else RATE)
        audio = self._resampler(chunk)
        if len(audio):
            self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= BLOCK:
            block, self._pending = self._pending[:BLOCK], self._pending[BLOCK:]
            self._block(block)

    def _block(self, block) -> None:
        block = self._gain(block)
        start_sample = self._blocks * BLOCK
        self._blocks += 1
        event = self._vad(block)

        if self.speaking:
            self._speech.append(block)
        else:
            self._pre.append((start_sample, block))

        if event and "start" in event and not self.speaking:
            self.speaking = True
            self._voice_start = int(event["start"]) + self._vad_offset
            first = self._pre[0][0] if self._pre else start_sample
            self._speech_start = first
            self._speech = [b for _, b in self._pre]
            self._pre.clear()
        elif event and "end" in event and self.speaking:
            self._finish(end_sample=int(event["end"]) + self._vad_offset)
        elif self.speaking and len(self._speech) * BLOCK >= MAX_UTTERANCE_S * RATE:
            self._finish(end_sample=None)
            if hasattr(self._vad, "reset_states"):
                self._vad.reset_states()
                self._vad_offset = self._blocks * BLOCK

        if self._blocks % LEVEL_EVERY == 0:
            self.bus.publish(topics.TO_UI, messages.make(
                messages.LISTENING, level=level_0_1(block), speaking=self.speaking))

    def _finish(self, end_sample: int | None) -> None:
        import numpy as np

        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
        end = self._speech_start + len(audio)
        if end_sample is not None and self._speech_start < end_sample < end:
            audio = audio[:end_sample - self._speech_start]  # drop the silence VAD waited through
            end = end_sample
        voiced = end - max(self._voice_start, self._speech_start)
        self._speech = []
        self.speaking = False
        self.bus.publish(topics.TO_UI, messages.make(messages.LISTENING, level=0.0, speaking=False))
        if voiced >= MIN_SPEECH_S * RATE:
            self.bus.publish(topics.UTTERANCE, audio.astype(np.float32))
