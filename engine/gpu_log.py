"""GPU memory log: how much video memory the models use, with everything loaded.

Route: core (Khang)
Publishes: a console line, and a status warning on the bus when memory runs high
APIs: nvidia-smi (comes with the NVIDIA driver; no torch, no new package)

    python -m engine.main --gpu-log      # prints used / total MB every gpu_log_every_s seconds, plus the peak

Whisper large-v3-turbo, UI-Venus 2B and the Qwen clean-up model share one 16 GB card. The demo needs the peak to
stay under gpu_warn_mb (14000), so there is room for the browser and the screen recorder.
"""
from __future__ import annotations

import subprocess
import threading

QUERY = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]


def read_gpu_memory(run=subprocess.run) -> tuple[int, int] | None:
    """(used MB, total MB) of the first NVIDIA GPU, or None when there is no driver or nvidia-smi fails."""
    try:
        out = run(QUERY, capture_output=True, text=True, timeout=5, check=True).stdout
        used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def format_line(used: int, total: int, peak: int, warn_mb: int) -> str:
    line = f"[gpu] {used} / {total} MB used (peak {peak} MB)"
    return line + f"  WARNING: over {warn_mb} MB, close other GPU programs" if used > warn_mb else line


class GpuLog:
    def __init__(self, every_s: float = 10.0, warn_mb: int = 14000, read=read_gpu_memory, out=print) -> None:
        self.every_s, self.warn_mb, self.read, self.out = every_s, warn_mb, read, out
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> str | None:
        mem = self.read()
        if mem is None:
            return None
        used, total = mem
        self.peak = max(self.peak, used)
        return format_line(used, total, self.peak, self.warn_mb)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-log", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        if self.sample() is None:
            self.out("[gpu] no NVIDIA GPU or nvidia-smi found; the memory log is off")
            return
        while not self._stop.wait(self.every_s):
            line = self.sample()
            if line:
                self.out(line)

    def stop(self) -> str | None:
        """Stops the log and returns the final line (the peak) for the console."""
        self._stop.set()
        return self.sample()
