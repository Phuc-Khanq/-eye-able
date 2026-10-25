"""Stock vs fine-tuned Whisper on the short commands: exactly the same audio, decoded the way the app does.

    .\\scripts\\make_command_speech.ps1                                    # the clips, once (again after new phrases)
    python -m engine.voice.command_bench --models large-v3-turbo models/whisper-elder-ct2
    python -m engine.voice.command_bench --models tiny.en --device cpu --compute-type int8 --limit 10
    python -m engine.voice.command_bench --models large-v3-turbo models/whisper-elder-ct2 --mic 1   # + you
    python -m engine.voice.command_bench --models large-v3-turbo models/whisper-elder-ct2 --mic 1 --mic-only --takes 2
    python -m engine.voice.command_bench --mic 1 --mic-only --hotwords "pause yes no"   # does a word hint help you?

Phrases: tests/short_commands.csv (yes, no, nope, write it, don't write it, pause, take a break, click, right click,
go back, escape, scroll, stop, send it, new tab, copy). Mode "answer" scores a phrase the way the app hears it while
something waits for an answer: "Know." is a no there (commands.answer). Clips: data/voice/commands, every English Windows voice x normal / slow / quiet. The models run one at a
time (each is freed before the next loads) over the same clips. Each clip goes through the app's own path, like
voice_check: audio_in (Silero VAD) -> Dictation.transcribe (the app's Whisper settings) -> commands.parse.
Synthetic voices, not elderly speakers: it compares the models, it doesn't say how well an older user is heard.

Two scores per clip:
  exact    the words, lowercased, without punctuation or filler words ("um", "please"), are the phrase:
           "Yes." is right, "Yeah" is wrong, "Send it." is right, "sent it" is wrong
  command  the app would do what the phrase does: commands.parse gives the phrase's own command ("Yeah" is a
           yes; "stop" counts when it would leave scroll mode). Phrases the app has no command for ("new tab",
           "copy") fall back to exact. The headline is this score.
A clip Whisper splits into two utterances is wrong: the app would act on each part.

--mic N also takes you saying each phrase (shuffled, --takes times) into memory; the same audio goes to every
model, and it's never saved. It's one speaker, so it's printed as its own line, a sanity check, and its
transcripts are printed only. Writes data/voice/commands/report.md (git-ignored; the synthetic clips' table and
wrong transcripts) and the numbers only to checkpoints/command_bench_<model>.json.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from engine.voice import commands
from engine.voice.voice_check import command_name, normalize_words

REPO = Path(__file__).resolve().parents[2]
PHRASES_CSV = REPO / "tests" / "short_commands.csv"
CLIPS = REPO / "data" / "voice" / "commands"
VARIANTS = ("normal", "slow", "quiet")
SCROLL_EXIT = "scroll_exit"
TAKES = 3                 # --mic: takes per phrase
TAKE_S = 3.0              # --mic: seconds recorded per take


def load_phrases(path: Path = PHRASES_CSV) -> list[dict]:
    """[{"id", "text", "mode"}] in file order. mode is where the phrase is said ("stop" in scroll mode)."""
    with path.open(encoding="utf-8", newline="") as f:
        return [{"id": r["id"], "text": r["text"], "mode": r.get("mode") or "pointer"} for r in csv.DictReader(f)]


def heard_command(heard: str, mode: str) -> str:
    """What the app does with this text in the phrase's mode: a command name, target:<Label>, scroll_exit,
    none or empty. The phrase itself gives what it should do."""
    from engine.core.mode_manager import SCROLL_EXIT as EXIT

    if mode == "scroll" and commands.normalize(heard) in EXIT:
        return SCROLL_EXIT
    if mode == "answer" and heard.strip() and commands.answer(heard):   # something waits for a yes or no:
        return commands.answer(heard)                                    # "Know." is a no (the mode manager)
    return (command_name(commands.parse(heard, allow_targets=mode in ("pointer", "answer"))) if heard.strip()
            else "empty")


def exact_words(text: str) -> list[str]:
    """voice_check's normalize_words without the filler words: "Um, yes please." -> ["yes"]."""
    return [w for w in normalize_words(text) if w not in commands.FILLER_WORDS]


def score(text: str, mode: str, heard: list[str]) -> dict:
    """One clip: heard is every transcript Whisper gave for it (usually one)."""
    said = " ".join(heard).strip()
    if not said:
        return {"text": False, "command": False, "empty": True, "became": "empty"}
    exact = exact_words(said) == exact_words(text)
    want = heard_command(text, mode)
    got = [heard_command(h, mode) for h in heard if h.strip()]
    command = exact if want == "none" else got == [want]   # no app command ("new tab"): the words must be right
    return {"text": exact, "command": command, "empty": False, "became": " | ".join(got)}


def clips(phrases: list[dict], manifest: Path = CLIPS / "manifest.csv") -> list[dict]:
    """One item per clip whose WAV exists: {"phrase", "mode", "voice", "variant", "file"}. Ordered variant,
    voice, phrase, so --limit 10 still covers most phrases."""
    by_id = {p["id"]: p for p in phrases}
    order = [p["id"] for p in phrases]
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["id"] in by_id and (CLIPS / r["file"]).exists()]
    rows.sort(key=lambda r: (VARIANTS.index(r["variant"]) if r["variant"] in VARIANTS else len(VARIANTS),
                             r["voice"], order.index(r["id"])))
    return [{"phrase": by_id[r["id"]]["text"], "mode": by_id[r["id"]]["mode"], "voice": r["voice"],
             "variant": r["variant"], "file": CLIPS / r["file"]} for r in rows]


def hear(audio, rate: int, dictation, settings, name: str = "clip") -> list[str]:
    """Every transcript the app gets from this audio: audio_in (Silero VAD) cuts it into utterances, as from
    the mic, and Dictation transcribes each one (voice_check.check_clip, for audio in memory)."""
    import threading

    from engine.core import topics
    from engine.core.events import EventBus
    from engine.main import play_wav
    from engine.voice.audio_in import AudioIn, load_vad
    from engine.voice.dictation import words_to_text
    from engine.voice.mic import MicSettings

    bus = EventBus()
    utterances: list = []
    bus.subscribe(topics.UTTERANCE, utterances.append)
    audio_in = AudioIn(bus, settings, mic_settings=MicSettings(-1, name, "file", rate, 1),
                       vad=load_vad(settings.pause_seconds))
    play_wav(audio_in, audio, rate, threading.Event(), realtime=False)
    texts = []
    for utt in utterances:
        words = dictation.transcribe(utt)
        if words:
            texts.append(words_to_text(words))
    return texts


def read_clip(path: Path):
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    return data.astype("float32") / 32768.0, rate


def run(items: list[dict], transcribe, log=print) -> list[dict]:
    """Score every item with one model. transcribe(item) -> [texts]. Returns one score per item, plus heard."""
    out = []
    for i, item in enumerate(items, 1):
        heard = transcribe(item)
        out.append({**score(item["phrase"], item["mode"], heard), "heard": " | ".join(heard)})
        if i % 20 == 0:
            log(f"    ... {i}/{len(items)} clips")
    return out


def label(model: str) -> str:
    """"stock large-v3-turbo" for a model name, "fine-tuned whisper-elder-ct2" for a folder (an export)."""
    from engine.voice.dictation import resolve_model

    return f"{'fine-tuned' if Path(resolve_model(model)).is_dir() else 'stock'} {Path(model).name}"


def pct(right: int, total: int) -> str:
    return f"{right / total:.0%}" if total else "-"


def headline(items: list[dict], results: dict[str, list[dict]], what: str = "Short commands") -> str:
    """"Short commands, 16 phrases x 6 clips: stock A 71% (68/96) vs fine-tuned B 88% (84/96), same audio"."""
    n_phrases = len({it["phrase"] for it in items})
    per = len(items) // n_phrases if n_phrases and len(items) % n_phrases == 0 else 0
    size = f"{n_phrases} phrases x {per} clips" if per else f"{n_phrases} phrases, {len(items)} clips"
    parts = [f"{name} {pct(sum(r['command'] for r in rs), len(rs))} ({sum(r['command'] for r in rs)}/{len(rs)})"
             for name, rs in results.items()]
    return f"{what}, {size}: {' vs '.join(parts)}, same audio"


def table(items: list[dict], results: dict[str, list[dict]]) -> list[str]:
    """Markdown: per phrase and variant, then per variant and in total, exact and command right / clips."""
    names = list(results)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[(it["phrase"], it["variant"])].append(i)
        groups[("all", it["variant"])].append(i)
        groups[("TOTAL", "all")].append(i)
    phrase_order = list(dict.fromkeys(it["phrase"] for it in items)) + ["all", "TOTAL"]
    variant_order = list(VARIANTS) + sorted({it["variant"] for it in items} - set(VARIANTS)) + ["all"]
    head = "| phrase | variant | clips | " + " | ".join(f"{n} exact | {n} command" for n in names) + " |"
    lines = [head, "|" + " --- |" * (3 + 2 * len(names))]
    for key in sorted(groups, key=lambda k: (phrase_order.index(k[0]), variant_order.index(k[1]))):
        idx = groups[key]
        cells = []
        for n in names:
            rs = [results[n][i] for i in idx]
            cells += [f"{sum(r['text'] for r in rs)}/{len(rs)}", f"{sum(r['command'] for r in rs)}/{len(rs)}"]
        phrase, variant = key
        bold = "**" if phrase == "TOTAL" else ""
        lines.append(f"| {bold}{phrase}{bold} | {variant} | {len(idx)} | " + " | ".join(cells) + " |")
    return lines


def wrong(items: list[dict], results: dict[str, list[dict]]) -> list[str]:
    """Every clip some model got wrong (command score): what each model heard."""
    lines = []
    for i, it in enumerate(items):
        rs = {n: results[n][i] for n in results}
        if all(r["command"] for r in rs.values()):
            continue
        heard = "; ".join(f"{n}: {r['heard'] or '(nothing)'!r} {'ok' if r['command'] else 'WRONG -> ' + r['became']}"
                          for n, r in rs.items())
        lines.append(f"- {it['phrase']!r} ({it['voice']}/{it['variant']}): {heard}")
    return lines or ["- none"]


def record_takes(phrases: list[dict], device: int, takes: int = TAKES, seconds: float = TAKE_S,
                 seed: int | None = None) -> list[dict]:
    """You say each phrase, in random order: one item per take, the audio in memory only (rule 10)."""
    import sounddevice as sd

    from engine.voice.mic import open_settings, to_16k

    mic = open_settings(sd, device, verbose=True)
    order = [p for p in phrases for _ in range(takes)]
    random.Random(seed).shuffle(order)
    items = []
    print(f"Say each phrase when it shows ({seconds:.0f} s each, {len(order)} takes). Nothing is saved.")
    for i, p in enumerate(order, 1):
        input(f"  [{i}/{len(order)}] press Enter, then say: {p['text']!r} ")
        audio = sd.rec(int(seconds * mic.rate), samplerate=mic.rate, channels=mic.channels, dtype="float32",
                       device=mic.device)
        sd.wait()
        items.append({"phrase": p["text"], "mode": p["mode"], "voice": "you", "variant": "mic",
                      "audio": to_16k(audio, mic.rate)})
    return items


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.dictation import Dictation

    p = argparse.ArgumentParser(description="Stock vs fine-tuned Whisper on the short commands, same audio.")
    p.add_argument("--models", nargs="+", default=["large-v3-turbo"],
                   help="Whisper names or CTranslate2 folders, e.g. large-v3-turbo models/whisper-elder-ct2")
    p.add_argument("--device", default=None, help="cuda or cpu (default: the config's)")
    p.add_argument("--compute-type", default=None, help="float16 on the GPU, int8 on the CPU")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (a quick smoke test)")
    p.add_argument("--mic", type=int, default=None, help="also record live takes from this mic (in memory only)")
    p.add_argument("--takes", type=int, default=TAKES, help="--mic: takes per phrase")
    p.add_argument("--mic-only", action="store_true", help="only your own takes (needs --mic N), no synthetic clips")
    p.add_argument("--hotwords", default=None,
                   help='lean Whisper toward these words, e.g. "pause yes no" (settings.whisper_hotwords)')
    p.add_argument("--report", type=Path, default=CLIPS / "report.md")
    args = p.parse_args(argv)
    changes = {}
    if args.hotwords is not None:
        changes["whisper_hotwords"] = args.hotwords
    if args.device:
        changes["whisper_device"] = args.device
    if args.compute_type:
        changes["whisper_compute_type"] = args.compute_type
    settings = dataclasses.replace(settings, **changes)

    phrases = load_phrases()
    manifest = CLIPS / "manifest.csv"
    if args.mic_only:
        if args.mic is None:
            print("--mic-only needs --mic N (your microphone's number)")
            return 1
        items = []
    else:
        if not manifest.exists():
            print(f"No {manifest.relative_to(REPO)}: run .\\scripts\\make_command_speech.ps1 first (or use --mic N --mic-only)")
            return 1
        items = clips(phrases, manifest)
        missing = sorted({ph["text"] for ph in phrases} - {it["phrase"] for it in items})
        if missing:
            print(f"no clips for {missing}: run .\\scripts\\make_command_speech.ps1 (it makes only the missing ones)")
            return 1
    if args.limit:
        items = items[:args.limit]
    takes = record_takes(phrases, args.mic, args.takes) if args.mic is not None else []

    results: dict[str, list[dict]] = {}
    live: dict[str, list[dict]] = {}
    for model in args.models:
        s = dataclasses.replace(settings, whisper_model=model)
        name = label(model)
        dictation = Dictation(s)
        t0 = time.perf_counter()
        dictation.load()
        print(f"{name} on {s.whisper_device} ({s.whisper_compute_type}) loaded in {time.perf_counter() - t0:.1f} s")
        t0 = time.perf_counter()
        results[name] = run(items, lambda it: hear(*read_clip(it["file"]), dictation, s, it["file"].name))
        took = time.perf_counter() - t0
        print(f"  {len(items)} clips in {took:.0f} s ({took / max(1, len(items)):.1f} s a clip)")
        if takes:
            live[name] = run(takes, lambda it: hear(it["audio"], 16000, dictation, s, "mic take"))
        if items:                                      # numbers for the synthetic clips (not for --mic-only)
            out = REPO / "checkpoints" / f"command_bench_{Path(model).name}.json"
            out.parent.mkdir(exist_ok=True)
            rs = results[name]
            out.write_text(json.dumps({"model": model, "device": s.whisper_device,
                                       "compute_type": s.whisper_compute_type, "hotwords": s.whisper_hotwords,
                                       "clips": len(rs), "command_right": sum(r["command"] for r in rs),
                                       "exact_right": sum(r["text"] for r in rs),
                                       "empty": sum(r["empty"] for r in rs)}, indent=2), encoding="utf-8")
        del dictation                                  # free this model before the next one loads
        gc.collect()

    lines: list[str] = []
    if items:
        top = headline(items, results)
        exact = "Exact text: " + " vs ".join(f"{n} {pct(sum(r['text'] for r in rs), len(rs))} "
                                              f"({sum(r['text'] for r in rs)}/{len(rs)})" for n, rs in results.items())
        rows, misses = table(items, results), wrong(items, results)
        print("\n" + "\n".join(rows) + "\n\nWrong (what each model heard):\n" + "\n".join(misses))
        print(f"\n{top}\n{exact}")
        caveat = ("Synthetic Windows TTS voices (normal, slow, quiet), not real elderly speakers; "
                  "the fine-tuning data was TORGO.")
        lines += [f"# {top}", "", exact, "", f"Whisper on {settings.whisper_device} ({settings.whisper_compute_type}); "
                  f"generated {time.strftime('%Y-%m-%d %H:%M')}. {caveat}", "",
                  "## Per phrase and variant (right / clips)", "", *rows, "", "## Wrong clips", "", *misses, ""]
    if live:
        mine = headline(takes, live, "Your voice (one speaker; a sanity check)")
        mine_rows, mine_wrong = table(takes, live), wrong(takes, live)
        print("\n" + "\n".join(mine_rows) + "\n" + mine + "\n" + "\n".join(mine_wrong))
        if args.hotwords is not None:
            print(f"(hotwords: {args.hotwords!r})")
        lines += [f"## {mine}", "", *mine_rows, "", *mine_wrong, ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {args.report.relative_to(REPO) if args.report.is_relative_to(REPO) else args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
