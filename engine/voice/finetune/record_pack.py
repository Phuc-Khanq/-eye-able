"""Record your own voice pack: read short phrases aloud, one at a time.

    python -m engine.voice.finetune.record_pack            # start or continue
    python -m engine.voice.finetune.record_pack --list     # show the phrases
    python -m engine.voice.finetune.record_pack --mic 2    # use mic number 2
    python -m engine.voice.finetune.record_pack --script engine/voice/finetune/voice_script.txt
                                                           # read your own script, with how to read each part

PRIVACY: this is an explicit, consented recording session, the ONE place where audio is
written to disk. The app itself never saves microphone audio. Recordings go to
data/voice/personal/ on this laptop only (git ignores data/), as 16 kHz WAV files plus
transcripts.csv (file,text). They are used to fine-tune and test Whisper on your voice
(prepare.py splits them 80/20 by sentence). Nothing is uploaded. Delete them any time:
say "delete my voice data" in the app, or run  python -m engine.voice.voice_data

For each phrase: press Enter, read it, press Enter again. Also: r = redo the last one,
s = skip, q = quit. You can stop and continue later; finished phrases are skipped.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import time
import wave
from pathlib import Path

from engine.voice.finetune import common

# About 150 things people say to a computer, plus everyday sentences.
COMMANDS = [
    "open my email", "scroll down", "scroll up", "send it", "send message", "scratch that",
    "undo that", "new line", "next line", "start dictation", "stop dictation", "done typing",
    "pause", "resume", "wake up", "open menu", "cancel", "never mind", "yes", "no", "confirm",
    "calibrate", "click send", "click compose", "click reply", "click forward", "click delete",
    "click search", "click settings", "click back", "click next", "click play", "click pause",
    "click close", "open the first result", "open Chrome", "open YouTube", "open Gmail",
    "open the calendar", "open my documents", "select all", "copy that", "paste it",
    "delete that word", "go back", "go forward", "refresh the page", "zoom in", "zoom out",
    "volume up", "volume down", "mute", "unmute", "close this window", "switch window",
    "minimize", "maximize", "take a screenshot", "search for the weather", "read it to me",
    "call my daughter", "call my son", "text my wife", "text my husband", "show my photos",
    "play some music", "stop the music", "turn on captions", "make the text bigger",
    "what time is it", "set a reminder", "set an alarm for seven", "open the news",
    "check the weather", "start a video call", "end the call", "answer the call",
    "add an attachment", "save this file", "print this page", "go to the top",
    "go to the bottom", "page down", "page up", "click the blue button", "open settings",
    "delete my voice data", "help me", "show the menu", "type my address",
]
SENTENCES = [
    "Hi mom, I'll be home at seven tonight.",
    "Thank you so much for the lovely flowers.",
    "Can we meet for coffee on Tuesday morning?",
    "I have a doctor's appointment at two thirty.",
    "Please remember to pick up my medicine.",
    "The weather is nice today, let's go for a walk.",
    "I'm running a little late, start without me.",
    "Happy birthday, I hope you have a wonderful day.",
    "Could you send me the photos from the weekend?",
    "My password reset email never arrived.",
    "I would like to order a large pizza with mushrooms.",
    "The grandchildren are visiting this Sunday.",
    "Don't forget we have dinner with the neighbors.",
    "I finished the book you gave me, it was great.",
    "What time does the pharmacy close today?",
    "Please call me back when you get this message.",
    "I need to pay the electricity bill this week.",
    "The bus was late again this morning.",
    "Let's watch a movie together tonight.",
    "I made soup, come over if you are hungry.",
    "My knee feels much better after the exercises.",
    "Can you help me set up the new phone?",
    "I will see you at church on Sunday.",
    "The garden needs watering before it gets too hot.",
    "Please turn the heating up a little.",
    "I can't find my glasses anywhere.",
    "We should plan a trip to the beach this summer.",
    "Tell the kids I said hello and I love them.",
    "The meeting has been moved to Thursday afternoon.",
    "I am tired today, so I will rest a bit.",
    "Could you read the letter from the bank for me?",
    "My sister is coming to stay for a few days.",
    "The new neighbors seem very friendly.",
    "I'd like a cup of tea with a little milk.",
    "Remember to lock the door when you leave.",
    "The game starts at eight on channel four.",
    "I feel a little dizzy, I'm going to sit down.",
    "Please write down the phone number for me.",
    "We are out of bread and eggs.",
    "The nurse will come by on Friday at ten.",
    "Thank you for driving me to the hospital.",
    "Let me know when you arrive safely.",
    "The library books are due next Monday.",
    "I love the picture you drew for me.",
    "Can you turn the television down, please?",
    "The cat has not eaten her breakfast.",
    "I want to send a card to my old friend.",
    "My hands are cold, could you bring a blanket?",
    "We had a lovely time at the party.",
    "The doctor said my blood pressure is fine.",
    "Please check if the stove is turned off.",
    "I'll call you after lunch.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "Good morning, how did you sleep?",
    "Good night, sleep well.",
    "See you tomorrow.",
    "I'm sorry, could you say that again?",
    "That sounds like a good idea.",
    "I don't understand, please explain it slowly.",
]
PHRASES = COMMANDS + SENTENCES


def load_script(path: Path) -> list[tuple[str, str]]:
    """[(text, how to read it)] from a script file. A line "## Title | how to read this part" sets the hint for
    the lines under it; "#" lines and blank lines are ignored; every other line is one thing to read. Each text
    must differ from every other (a finished text is skipped), so a second pass of a phrase needs different
    punctuation, e.g. "Take a break!"."""
    items: list[tuple[str, str]] = []
    hint, seen = "", set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("##")):
            continue
        if line.startswith("##"):
            title, _, how = line.lstrip("#").partition("|")
            hint = f"{title.strip()}: {how.strip()}" if how.strip() else title.strip()
            continue
        if line in seen:
            raise ValueError(f"{path}: '{line}' appears twice (a finished text is skipped; change its punctuation)")
        seen.add(line)
        items.append((line, hint))
    return items


MIN_PEAK = 0.02          # quieter than this: probably not heard; offer a redo


def personal_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "personal"


def done_texts(table: Path) -> set[str]:
    if not table.exists():
        return set()
    with open(table, encoding="utf-8", newline="") as f:
        return {r["text"] for r in csv.DictReader(f)}


def append_row(table: Path, file_name: str, text: str) -> None:
    new = not table.exists()
    with open(table, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text"])
        if new:
            w.writeheader()
        w.writerow({"file": file_name, "text": text})


def remove_row(table: Path, file_name: str) -> None:
    with open(table, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] != file_name]
    with open(table, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text"])
        w.writeheader()
        w.writerows(rows)


def write_wav(path: Path, audio) -> None:
    """16 kHz mono 16-bit WAV with the standard library (no extra packages)."""
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(common.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def record_until_enter(sd, s):
    """Record from the mic until the user presses Enter. Returns 16 kHz mono float32."""
    import numpy as np

    from engine.voice.mic import to_16k

    chunks: queue.Queue = queue.Queue()
    stream = sd.InputStream(device=s.device, samplerate=s.rate, channels=s.channels,
                            dtype="float32", callback=lambda d, f, t, st: chunks.put(d.copy()))
    with stream:
        input("    ...recording. Press Enter when you are done. ")
        time.sleep(0.2)                  # keep the last syllable
    parts = []
    while not chunks.empty():
        parts.append(chunks.get())
    audio = np.concatenate(parts) if parts else np.zeros((0, s.channels), np.float32)
    return to_16k(audio, s.rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--mic", type=int, default=None, help="mic number from 'python -m engine.voice.mic'")
    parser.add_argument("--list", action="store_true", help="print the phrases and exit")
    parser.add_argument("--script", type=Path, default=None,
                        help="read the phrases in this file (with a hint on how to read each part) instead of "
                             "the built-in 150")
    args = parser.parse_args(argv)
    items = load_script(args.script) if args.script else [(p, "") for p in PHRASES]
    if args.list:
        last_hint = None
        for i, (p, h) in enumerate(items, 1):
            if h != last_hint:
                print(f"\n  [{h}]" if h else "")
                last_hint = h
            print(f"{i:3d}. {p}")
        return 0

    folder = personal_dir(args.data_dir)
    table = folder / "transcripts.csv"
    finished = done_texts(table)
    todo = [(p, h) for p, h in items if p not in finished]
    print(__doc__.split("For each phrase")[0].strip().split("\n\n", 1)[1])
    print(f"\n{len(items) - len(todo)} of {len(items)} phrases are already recorded.")
    if not todo:
        print("All done. Thank you!")
        return 0
    try:
        if input("Type yes to record your voice to this laptop: ").strip().lower() != "yes":
            print("Nothing recorded.")
            return 1
    except EOFError:
        print("Nothing recorded (no keyboard input).")
        return 1

    import numpy as np
    import sounddevice as sd

    from engine.voice import mic

    s = mic.open_settings(sd, args.mic, verbose=True)
    folder.mkdir(parents=True, exist_ok=True)
    last: str | None = None
    i = 0
    while i < len(todo):
        text, hint = todo[i]
        n = len(items) - len(todo) + i + 1
        print(f"\n[{n}/{len(items)}]  Say:  {text}")
        if hint:
            print(f"    How:  {hint}")
        try:
            key = input("    Enter = start, s = skip, r = redo the last one, q = quit: ").strip().lower()
        except EOFError:
            key = "q"
        if key == "q":
            break
        if key == "s":
            i += 1
            continue
        if key == "r" and last:
            remove_row(table, last)
            (folder / last).unlink(missing_ok=True)
            i = max(0, i - 1)
            last = None
            continue
        audio = record_until_enter(sd, s)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if len(audio) < 0.3 * common.SAMPLE_RATE or peak < MIN_PEAK:
            print(f"    That was very quiet or short (peak {peak:.3f}). Let's try again.")
            continue
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{n:03d}.wav"
        write_wav(folder / name, audio)
        append_row(table, name, text)
        print(f"    Saved ({len(audio) / common.SAMPLE_RATE:.1f} s).")
        last = name
        i += 1
    print(f"\n{len(done_texts(table))} phrases recorded in {folder}. "
          "Next: python -m engine.voice.finetune.prepare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
