"""Cleans up dictated text before it's typed, like Claude's dictation: fillers, repeats and false
starts go, punctuation and capitals get fixed, and the meaning stays.

Route: Khoi and Khoa (Khoi leads)
Publishes: topics.REWRITE_RESULT {"id", "clean", "used", "ms"} (RewriteWorker)
Subscribes: topics.REWRITE_REQUEST {"id", "text"} (from the mode manager)
APIs: transformers AutoModelForCausalLM / AutoTokenizer, Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0),
      greedy decoding. Runs on this laptop; nothing leaves it (CONTRIBUTING rule 10).

Safety first: a small model sometimes answers, summarizes or drops a fact. check_rewrite()
compares the rewrite with the original and rejects it if a number or name went missing, too much
of the content changed, or the length jumped. A rejected rewrite means the user simply sees and
types their own words. The user always sees the text before it's typed.
"""

from __future__ import annotations

import re
from collections import Counter

from engine.voice.voice_check import normalize_words

# ------------------------------------------------------------------ the guard
MAX_LENGTH_RATIO = 1.3      # the rewrite may be at most this much longer than the original...
MIN_LENGTH_RATIO = 0.4      # ...and at least this long (in normalized words)
MIN_CONTENT_KEPT = 0.6      # share of the original's distinct content words the rewrite must keep
MAX_CONTENT_NEW = 0.25      # share of the rewrite's distinct content words that may be new

# Number words after normalize_words ("7" -> "seven", "21" -> "twenty one", "4729" -> four seven two nine).
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "half", "quarter",
}

# Words that carry no content of their own, so dropping or adding them doesn't change the meaning:
# articles, pronouns, auxiliaries and their contractions, prepositions, conjunctions, determiners,
# question words and speech fillers. Built in on purpose (no download); English only.
STOPWORDS = set("""
a an the
i me my mine myself we us our ours you your yours he him his she her hers it its they them their theirs
this that these those there here who whom whose which what when where why how
i'm i'll i've i'd you're you'll you've you'd he's she's it's we're we'll we've they're they'll they've
that's there's here's what's who's where's how's let's
am is are was were be been being have has had having do does did doing done
will would shall should can could may might must ought
isn't aren't wasn't weren't don't doesn't didn't won't wouldn't can't couldn't shouldn't haven't hasn't hadn't
to of in on at for with from by about as into onto over under after before up down out off through
between against during without within upon around near via
and or but so because if than then though although while whether nor yet also too
not no yes all any some each every both either neither other another such only own same just very
more most much many few less least
um uh umm uhh er erm hmm mm oh ah like well okay ok yeah yep right actually basically literally really
mean know kind sort
""".split())

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "nor", "neither"}  # plus any "...n't"

_NOT_NAMES = {"i", "i'm", "i'll", "i've", "i'd"}


def _negations(words: list[str]) -> int:
    """How many negations: "not", "never", "don't" ... each counts once ("do not" == "don't")."""
    return sum(1 for w in words if w in NEGATIONS or w.endswith("n't"))
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def _content(words: list[str]) -> set[str]:
    return {w for w in words if w not in STOPWORDS and w not in NUMBER_WORDS}


def _swapped(orig_w: list[str], clean_w: list[str]) -> list[tuple[str, str]]:
    """Content words replaced in place by a word the original never had ("photos" -> "videos").
    Clean-up removes and fixes; it has no reason to swap one thing for another."""
    from difflib import SequenceMatcher

    a = [w for w in orig_w if w not in STOPWORDS and w not in NUMBER_WORDS]
    b = [w for w in clean_w if w not in STOPWORDS and w not in NUMBER_WORDS]
    seen = set(a)
    swaps = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "replace":
            new = [w for w in b[j1:j2] if w not in seen]
            if new:
                swaps.append((" ".join(a[i1:i2]), " ".join(new)))
    return swaps


def _names(text: str) -> set[str]:
    """Capitalized words that aren't the first word of a sentence: probably names ("call Khoa")."""
    names, start = set(), True
    for token in text.replace("’", "'").split():
        word = re.sub(r"^[^\w']+|[^\w']+$", "", token)
        if word and not start and word[0].isupper() and word.lower() not in _NOT_NAMES:
            names.add(re.sub(r"'s$", "", word.lower()))
        start = bool(_SENTENCE_END.search(token))
    return names


def check_rewrite(original: str, clean: str) -> str | None:
    """Why the rewrite must not be used, or None if it's safe to show instead of the original.

    Errs on the safe side: a self-correction like "meet at 5, no wait, at 6" -> "meet at 6" is
    rejected because the 5 (and the "no") went missing. Better to show the user's own words than
    to lose a fact.
    """
    if not clean or not clean.strip():
        return "empty"
    orig_w, clean_w = normalize_words(original), normalize_words(clean)
    if not orig_w:
        return "nothing to clean"
    if not clean_w:
        return "empty"
    ratio = len(clean_w) / len(orig_w)
    if ratio > MAX_LENGTH_RATIO:
        return f"too long ({ratio:.2f}x)"
    if ratio < MIN_LENGTH_RATIO:
        return f"too short ({ratio:.2f}x)"
    orig_n = Counter(w for w in orig_w if w in NUMBER_WORDS)
    clean_n = Counter(w for w in clean_w if w in NUMBER_WORDS)
    missing = orig_n - clean_n
    if missing:
        return "number missing: " + " ".join(sorted(missing.elements()))
    if _negations(orig_w) != _negations(clean_w):    # "I'm not coming" -> "I'm coming" flips the meaning
        return f"negation changed ({_negations(orig_w)} -> {_negations(clean_w)})"
    clean_set = {re.sub(r"'s$", "", w) for w in clean_w}
    lost = sorted(n for n in _names(original) if n not in clean_set)
    if lost:
        return "name missing: " + " ".join(lost)
    swaps = _swapped(orig_w, clean_w)
    if swaps:
        return "word swapped: " + ", ".join(f"{a} -> {b}" for a, b in swaps)
    orig_c, clean_c = _content(orig_w), _content(clean_w)
    if orig_c:
        kept = len(orig_c & clean_c) / len(orig_c)
        if kept < MIN_CONTENT_KEPT:
            return f"content changed (kept {kept:.0%})"
    if clean_c:
        new = len(clean_c - orig_c) / len(clean_c)
        if new > MAX_CONTENT_NEW:
            return f"content added ({new:.0%} new)"
    return None


def accept_rewrite(original: str, clean: str) -> bool:
    """True if the rewrite keeps every number and name and roughly the same content."""
    return check_rewrite(original, clean) is None
