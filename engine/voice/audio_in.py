"""16 kHz mono mic stream, automatic gain for quiet voices, speech start and end.

Route: Khoi and Khoa (Khoa leads)
Publishes: listening, finished utterances (float32 audio) to dictation, and right before
           each one topics.SPEECH_SPAN: when that audio was spoken
Subscribes: settings (pause_seconds)
APIs: sounddevice InputStream (512-sample blocks), silero-vad VADIterator

How audio flows:
  sounddevice callback   stamps each chunk with its arrival time, queues it, returns (never blocks)
  "audio-in" thread      mic.Resampler16k -> 512-sample blocks -> gain -> Silero VAD
                         -> on speech end: topics.SPEECH_SPAN, then topics.UTTERANCE (16 kHz mono float32)

When was it said? Each 16 kHz sample's time comes from the arrival stamp of its own mic chunk
(time.monotonic() ms, the camera's clock too), not from "now" when the thread gets to it, so a
backlog in the queue or a dropped chunk doesn't shift the times. The voice route's mouth gate
lines Whisper's word times up with the camera with it.

An utterance ends after settings.pause_seconds of silence (the user picks 1 to 4 s),
or at MAX_UTTERANCE_S, because Whisper only hears 30 s at a time.
Audio stays in memory; nothing is written to disk (CONTRIBUTING rule 10).
"""

from __future__ import annotations

import bisect
import queue
import threading
import time
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
LIVE_VOICE_RMS = 0.015      # a block this loud (after gain) counts as voice for the live preview, so a pause isn't re-transcribed
LIVE_NEW_VOICE_S = 0.25     # a new partial needs this much new voice since the last one
MOUTH_STILL = 0.05          # mouth-moving share (over the last pause_seconds) at or under this = the user stopped talking
MOUTH_EXTRA_S = 0.5         # a clip must be this much longer than pause_seconds before the mouth can end it
MOUTH_CHECK_EVERY = 6       # ask the mouth tracker every 6 blocks (about 5 times a second)
MOUTH_TAIL_S = 0.3          # keep this much audio after the mouth stopped
LEVEL_EVERY = 3             # send a listening level every 3 blocks (about 10 per second)
RAW_QUEUE_MAX = 200         # about 6 s of mic chunks; if we fall that far behind, drop audio
TIME_HISTORY_S = 35.0       # sample times kept for SPEECH_SPAN (longer than any utterance plus pre-roll)

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


def _silero_onnx_path():
    """The ONNX model file inside the silero-vad package, found WITHOUT importing the package
    (its __init__ imports torch, which alone takes about 4 s to load)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("silero-vad is not installed (pip install -r requirements.txt)")
    path = Path(list(spec.submodule_search_locations)[0]) / "data" / "silero_vad.onnx"
    if not path.exists():
        raise RuntimeError(f"Silero VAD model not found at {path}")
    return path


class SileroOnnx:
    """Silero VAD speech probability for one 512-sample block, with numpy + onnxruntime only.
    Same model file and same input handling as silero_vad's OnnxWrapper (64-sample context)."""

    CONTEXT = 64

    def __init__(self) -> None:
        import numpy as np
        import onnxruntime  # comes with faster-whisper

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(_silero_onnx_path()), sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
        self._sr = np.array(RATE, dtype=np.int64)
        self.last = 0.0                  # last speech probability (live_check --debug)
        self.reset_states()

    def reset_states(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, block, sr: int = RATE) -> float:
        import numpy as np

        x = np.concatenate([self._context, np.asarray(block, dtype=np.float32).reshape(1, -1)], axis=1)
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self.CONTEXT:]
        self.last = float(out.reshape(-1)[0])
        return self.last


class VadIterator:
    """Speech start / end events, the same rules as silero_vad.VADIterator:
    start when probability >= threshold; end after min_silence_samples below threshold - 0.15.
    Returns {"start": sample}, {"end": sample} or None for each 512-sample block."""

    def __init__(self, model, threshold: float = VAD_THRESHOLD, min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30) -> None:
        self.model = model
        self.threshold = threshold
        self.min_silence_samples = RATE * min_silence_duration_ms // 1000
        self.speech_pad_samples = RATE * speech_pad_ms // 1000
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, block):
        n = len(block)
        self.current_sample += n
        prob = self.model(block, RATE)
        if prob >= self.threshold and self.temp_end:
            self.temp_end = 0
        if prob >= self.threshold and not self.triggered:
            self.triggered = True
            return {"start": max(0, self.current_sample - self.speech_pad_samples - n)}
        if prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            end = self.temp_end + self.speech_pad_samples - n
            self.temp_end = 0
            self.triggered = False
            return {"end": end}
        return None


def load_vad(pause_seconds: float):
    """Silero VAD (the ONNX model bundled with silero-vad; no download, no torch)."""
    return VadIterator(SileroOnnx(), threshold=VAD_THRESHOLD,
                       min_silence_duration_ms=int(pause_seconds * 1000),
                       speech_pad_ms=SPEECH_PAD_MS)


class AudioIn:
    """Mic -> utterances on the bus.

    Tests pass `sd` (a fake sounddevice), `mic_settings` and `vad` to skip the real mic
    and model, then call the stream's callback and drain() themselves.
    """

    def __init__(self, bus, settings=None, sd=None, mic_settings=None, vad=None,
                 clock=time.monotonic) -> None:
        if settings is None:
            from engine.config import settings
        self.bus = bus
        self.settings = settings
        self.pause_seconds = float(settings.pause_seconds)
        self._sd = sd
        self.mic_settings = mic_settings
        self._vad = vad
        self._clock = clock             # seconds; tests pass a fake one
        self._raw: queue.Queue = queue.Queue(maxsize=RAW_QUEUE_MAX)
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.dropped_chunks = 0
        self.chunks = 0                 # mic chunks received (live_check warns if this stalls)
        self.level = 0.0                # last block's 0-1 level, after gain
        # live preview: the audio so far goes out as topics.PARTIAL_AUDIO every live_every_s while speaking
        self.live = bool(getattr(settings, "live_preview", False))
        self.live_every_s = max(0.3, float(getattr(settings, "live_every_s", 1.0)))
        self.live_min_s = float(getattr(settings, "live_min_s", 0.8))
        # End a clip when the user's MOUTH has been still for pause_seconds, not only when the VAD hears silence.
        # In a room where other people talk, the VAD never hears a pause: clips ran to the 28 s cap, the user
        # waited that long, and the mouth gate then judged all 28 s at once and dropped "hi Dizzy". Needs `mouth`
        # (the MouthTracker, set by the engine); no mouth or no face seen: the VAD alone decides, as before.
        self.mouth = None
        self.mouth_endpoint = bool(getattr(settings, "mouth_endpoint", False))
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
        self._times: deque = deque()                   # (16 kHz samples produced after a chunk, its time ms)
        self._partial_at = 0                           # samples of _speech when the last partial audio went out
        self._voiced = 0                               # voiced samples in this utterance so far (live preview)
        self._voiced_at = 0                            # ... when the last partial audio went out
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
        """Runs on PortAudio's thread: stamp, copy and return, nothing else."""
        self.chunks += 1
        try:
            self._raw.put_nowait((self._clock() * 1000.0, indata.copy()))
        except queue.Full:
            self.dropped_chunks += 1

    # ------------------------------------------------------------ worker side
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                t_ms, chunk = self._raw.get(timeout=0.2)
            except queue.Empty:
                continue
            self.process(chunk, t_ms)

    def drain(self) -> None:
        """Process every queued chunk now (tests, or a thread-less start)."""
        while True:
            try:
                t_ms, chunk = self._raw.get_nowait()
            except queue.Empty:
                return
            self.process(chunk, t_ms)

    def process(self, chunk, t_ms: float | None = None) -> None:
        """Turn one native-rate mic chunk into blocks and run them through gain and VAD.

        t_ms: when the chunk arrived (time.monotonic() ms, the moment its last sample was
        captured). None (tests, a WAV file) means now.
        """
        import numpy as np

        if t_ms is None:
            t_ms = self._clock() * 1000.0
        if self._resampler is None:
            self._resampler = mic.Resampler16k(self.mic_settings.rate if self.mic_settings else RATE)
        audio = self._resampler(chunk)
        if len(audio):
            self._pending = np.concatenate([self._pending, audio])
            produced = self._blocks * BLOCK + len(self._pending)   # 16 kHz samples so far
            # the resampler's newest output is a few ms older than the chunk's last sample
            self._times.append((produced, t_ms - self._resampler.delay_ms))
            while self._times and self._times[0][0] < produced - TIME_HISTORY_S * RATE:
                self._times.popleft()
        while len(self._pending) >= BLOCK:
            block, self._pending = self._pending[:BLOCK], self._pending[BLOCK:]
            self._block(block)

    def _sample_time_ms(self, sample: int, edge: str) -> float:
        """time.monotonic() ms at 16 kHz sample index `sample`.

        edge "start": the moment that sample began, timed from its own chunk.
        edge "end": the end of sample - 1, timed from the chunk that sample - 1 is in.
        The two only differ across a dropped chunk, where the real clock jumps.
        """
        if not self._times:
            return self._clock() * 1000.0
        ends = [n for n, _ in self._times]
        i = bisect.bisect_right(ends, sample) if edge == "start" else bisect.bisect_left(ends, sample)
        produced, t_ms = self._times[min(i, len(self._times) - 1)]
        return t_ms - (produced - sample) * 1000.0 / RATE

    @property
    def speech_prob(self) -> float:
        """Silero's last speech probability (0-1), or -1 with a fake VAD."""
        model = getattr(self._vad, "model", None)
        return getattr(model, "last", -1.0)

    def _block(self, block) -> None:
        import numpy as np

        block = self._gain(block)
        self.level = level_0_1(block)
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
            self._partial_at = self._voiced = self._voiced_at = 0
            self._pre.clear()
        elif event and "end" in event and self.speaking:
            self._finish(end_sample=int(event["end"]) + self._vad_offset)
        elif self.speaking and len(self._speech) * BLOCK >= MAX_UTTERANCE_S * RATE:
            self._finish(end_sample=None)
            self._restart_vad()
        elif self.speaking and self._mouth_stopped():
            self._finish(end_sample=self._blocks * BLOCK - int(max(0.0, self.pause_seconds - MOUTH_TAIL_S) * RATE))
            self._restart_vad()

        if self.live and self.speaking:
            if float(np.sqrt(np.mean(block * block))) >= LIVE_VOICE_RMS:
                self._voiced += BLOCK
            self._partial()

        if self._blocks % LEVEL_EVERY == 0:
            self.bus.publish(topics.TO_UI, messages.make(
                messages.LISTENING, level=self.level, speaking=self.speaking))

    def _restart_vad(self) -> None:
        """After a clip ended without the VAD's say-so, so the VAD can start the next one."""
        if hasattr(self._vad, "reset_states"):
            self._vad.reset_states()
            self._vad_offset = self._blocks * BLOCK

    def _mouth_stopped(self) -> bool:
        """True when the camera saw the user's mouth still for the last pause_seconds (see mouth_endpoint)."""
        if self.mouth is None or not self.mouth_endpoint or self._blocks % MOUTH_CHECK_EVERY:
            return False
        if len(self._speech) * BLOCK < (self.pause_seconds + MOUTH_EXTRA_S) * RATE:
            return False
        try:
            end_ms = self._sample_time_ms(self._blocks * BLOCK, "end")
            f = self.mouth.moving_fraction(end_ms - self.pause_seconds * 1000.0, end_ms)
        except Exception:
            return False                # a broken tracker must never cut the user off
        return f is not None and f <= MOUTH_STILL

    def _partial(self) -> None:
        """The audio of this utterance so far, for the live preview (the dictation worker transcribes it
        when it is idle and drops it if the finished utterance gets there first)."""
        import numpy as np

        n = len(self._speech) * BLOCK
        if (self._voiced < self.live_min_s * RATE or n - self._partial_at < self.live_every_s * RATE
                or self._voiced - self._voiced_at < LIVE_NEW_VOICE_S * RATE):
            return                      # too little voice, too soon, or only a pause since the last one
        self._partial_at, self._voiced_at = n, self._voiced
        audio = np.concatenate(self._speech).astype(np.float32)
        self.bus.publish(topics.PARTIAL_AUDIO, {
            "audio": audio,
            "start_ms": round(self._sample_time_ms(self._speech_start, "start"), 1),
            "end_ms": round(self._sample_time_ms(self._speech_start + len(audio), "end"), 1)})

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
            # same thread, right before the audio: dictation pairs the two in order
            self.bus.publish(topics.SPEECH_SPAN, {
                "start_ms": round(self._sample_time_ms(self._speech_start, "start"), 1),
                "end_ms": round(self._sample_time_ms(end, "end"), 1)})
            self.bus.publish(topics.UTTERANCE, audio.astype(np.float32))
