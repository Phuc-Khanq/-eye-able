"""Find a mic format Windows accepts, and turn its audio into 16 kHz mono.

Route: Khoi and Khoa
Used by: whisper_bench.py (record) and audio_in.py (open_settings, Resampler16k).

Why: laptop mic arrays (Intel Smart Sound, Realtek) often refuse 44.1 kHz or 2 channels
and fail with "format is not supported" (MME error 32) or DSERR_BADFORMAT. We try each
input device with several rates and channel counts until one opens, then resample.

    python -m engine.voice.mic      # list which mic formats actually open
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_RATE = 16_000
RATES = (48_000, 44_100, 16_000, 32_000, 96_000)
HOST_ORDER = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS")
SKIP_NAMES = ("speaker", "stereo mix", "output")   # inputs that record system sound, not you


@dataclass(frozen=True)
class MicSettings:
    device: int
    name: str
    host: str
    rate: int
    channels: int

    def __str__(self) -> str:
        return f"#{self.device} '{self.name}' ({self.host}) at {self.rate} Hz, {self.channels} ch"


def _candidates(sd, preferred: int | None) -> list[int]:
    """Input device numbers to try, best first."""
    devices = sd.query_devices()
    hosts = [h["name"] for h in sd.query_hostapis()]
    if preferred is not None:
        return [preferred]

    def host_rank(i: int) -> int:
        host = hosts[devices[i]["hostapi"]]
        return HOST_ORDER.index(host) if host in HOST_ORDER else len(HOST_ORDER)

    inputs = [i for i, d in enumerate(devices)
              if d["max_input_channels"] > 0
              and not any(s in d["name"].lower() for s in SKIP_NAMES)
              and d["name"].strip(" ()")]
    order = sorted(inputs, key=host_rank)
    try:
        default = sd.default.device[0]
    except Exception:
        default = -1
    if default in order:
        order.remove(default)
        order.insert(0, default)
    return order


def _formats(device_info) -> list[tuple[int, int]]:
    rates = [int(device_info["default_samplerate"])] + [r for r in RATES]
    rates = list(dict.fromkeys(rates))                  # dedupe, keep order
    max_ch = int(device_info["max_input_channels"])
    channels = list(dict.fromkeys([1, 2, max_ch]))
    return [(r, c) for r in rates for c in channels if c <= max_ch]


def open_settings(sd=None, preferred: int | None = None, verbose: bool = False) -> MicSettings:
    """Return the first (device, rate, channels) that really opens.

    Opens a stream for a moment to test it, because check_input_settings sometimes
    says yes and the driver still refuses.
    """
    if sd is None:
        import sounddevice as sd

    hosts = [h["name"] for h in sd.query_hostapis()]
    errors: list[str] = []
    for device in _candidates(sd, preferred):
        info = sd.query_devices(device)
        for rate, channels in _formats(info):
            try:
                with sd.InputStream(device=device, samplerate=rate, channels=channels,
                                    dtype="float32"):
                    pass
            except Exception as exc:
                errors.append(f"#{device} {rate} Hz {channels} ch: {exc}")
                continue
            found = MicSettings(device, info["name"], hosts[info["hostapi"]], rate, channels)
            if verbose:
                print(f"Using mic {found}")
            return found
    raise RuntimeError(
        "No mic format worked. Check Windows Settings > Privacy & security > Microphone "
        "(let desktop apps use it) and that no other app holds the mic.\n"
        + "\n".join(errors[-6:])
    )


def to_16k(audio, rate: int):
    """Mix to mono and resample to 16 kHz float32, with a proper low-pass filter."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != TARGET_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(TARGET_RATE, rate)
        audio = resample_poly(audio, TARGET_RATE // g, rate // g)
    return audio.astype(np.float32)


class Resampler16k:
    """to_16k for a live stream: feed chunks of any size, get 16 kHz mono back.

    Resampling each small chunk on its own clicks at the edges, so we keep a short
    overlap from the previous chunk and only return output the filter fully covered.
    The output runs about PAD_MS behind the input.
    """

    PAD_MS = 4

    def __init__(self, rate: int) -> None:
        import numpy as np
        from math import ceil, gcd

        g = gcd(TARGET_RATE, rate)
        self.up, self.down = TARGET_RATE // g, rate // g
        min_pad = max(64, rate * self.PAD_MS // 1000)
        self.pad = self.down * ceil(min_pad / self.down)   # whole output samples
        self._buf = np.zeros(self.pad, dtype=np.float32)  # silence as the first history

    def __call__(self, chunk):
        import numpy as np

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 2:
            chunk = chunk.mean(axis=1)
        if self.up == self.down:
            return chunk
        from scipy.signal import resample_poly

        buf = np.concatenate([self._buf, chunk])
        usable = (len(buf) // self.down) * self.down
        if usable < 2 * self.pad + self.down:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        y = resample_poly(buf[:usable], self.up, self.down)
        a = self.pad * self.up // self.down
        b = (usable - self.pad) * self.up // self.down
        self._buf = buf[usable - 2 * self.pad:]
        return y[a:b].astype(np.float32)


def record(seconds: float, preferred: int | None = None):
    """Record a clip into memory (never to disk). Returns 16 kHz mono float32."""
    import sounddevice as sd

    s = open_settings(sd, preferred, verbose=True)
    audio = sd.rec(int(seconds * s.rate), samplerate=s.rate, channels=s.channels,
                   dtype="float32", device=s.device)
    sd.wait()
    return to_16k(audio, s.rate)


if __name__ == "__main__":
    import sounddevice as sd

    hosts = [h["name"] for h in sd.query_hostapis()]
    for device in _candidates(sd, None):
        info = sd.query_devices(device)
        ok = []
        for rate, channels in _formats(info):
            try:
                with sd.InputStream(device=device, samplerate=rate, channels=channels,
                                    dtype="float32"):
                    ok.append(f"{rate}/{channels}ch")
            except Exception:
                pass
        print(f"#{device:>2} {info['name'][:40]:<40} {hosts[info['hostapi']]:<20} "
              f"{', '.join(ok) or 'nothing opens'}")
