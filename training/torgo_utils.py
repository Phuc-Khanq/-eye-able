"""Small pure-Python helpers for the TORGO training pipeline (no torch / datasets needed).

Route: voice (Khoi). Tested in tests/test_training.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# TORGO speaker codes: F01, M05 = dysarthric; FC01, MC04 = control (typical speech).
SPEAKER_RE = re.compile(r"(?<![A-Z0-9])([FM]C?\d{2})(?=[_\-/\\.])")

# Held out from training. Results on these speakers decide whether we ship.
# One female + one male dysarthric speaker, one female + one male control speaker.
DEFAULT_TEST_SPEAKERS = ("F04", "M03", "FC03", "MC04")

MIN_SECONDS = 0.3
MAX_SECONDS = 30.0   # Whisper's window

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def speaker_of(path: str) -> str | None:
    """'.../FC01_1_arrayMic_0066.wav' -> 'FC01'."""
    m = SPEAKER_RE.search(path or "")
    return m.group(1) if m else None


def is_dysarthric_speaker(speaker: str) -> bool:
    return len(speaker) == 3          # F01 / M05; controls have a C: FC01 / MC04


def _number_word(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    return str(n)


def normalize(text: str) -> str:
    """For scoring only: lowercase, no punctuation, digits as words ("7" == "seven")."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = [_number_word(int(w)) if w.isdigit() else w.strip("'") for w in text.split()]
    return " ".join(w for w in " ".join(words).split() if w)


def is_usable_transcript(text: str) -> bool:
    """Drop prompts that aren't real speech targets: '[say Ah-P-Eee repeatedly]', 'xxx', paths."""
    t = (text or "").strip()
    if not t or "[" in t or "]" in t or "/" in t or "\\" in t:
        return False
    if t.lower() in {"xxx", "xx", "x"} or t.lower().endswith((".jpg", ".png")):
        return False
    return bool(normalize(t))


def training_target(text: str) -> str:
    """The text we teach Whisper to write. Sentence case plus a final period, like Whisper's
    own output, so the adapted model keeps writing tidy text in dictation."""
    t = " ".join(text.strip().split())
    if t.isupper() or t.islower():
        t = t.lower()
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".?!":
        t += "."
    return t


# ----------------------------------------------------------------------- scoring
def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, words in ref) on normalized text."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def wer(pairs) -> float:
    """Word error rate over (ref, hyp) pairs. 0.25 = 25 %."""
    errors = words = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        errors, words = errors + e, words + n
    return errors / words if words else 0.0


@dataclass
class Gate:
    min_gain: float = 0.02         # dysarthric WER must drop by at least 2 points...
    max_healthy_loss: float = 0.01  # ...and typical speech may get at most 1 point worse

    def decide(self, base_dys: float, new_dys: float, base_ok: float, new_ok: float) -> tuple[bool, str]:
        if base_dys - new_dys < self.min_gain:
            return False, (f"not enough gain on dysarthric speech: {base_dys:.1%} -> {new_dys:.1%} "
                           f"(needs {self.min_gain:.0%} better)")
        if new_ok - base_ok > self.max_healthy_loss:
            return False, (f"typical speech got worse: {base_ok:.1%} -> {new_ok:.1%} "
                           f"(allowed {self.max_healthy_loss:.0%})")
        return True, (f"dysarthric {base_dys:.1%} -> {new_dys:.1%}, "
                      f"typical {base_ok:.1%} -> {new_ok:.1%}")
