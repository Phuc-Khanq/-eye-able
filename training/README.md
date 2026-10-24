# Adapting Whisper to dysarthric speech

Some users (older adults after a stroke, people with ALS, Parkinson's or cerebral palsy)
have **dysarthria**: slurred, slow or quiet speech that stock Whisper often gets wrong.
This folder fine-tunes Whisper large-v3-turbo on the TORGO dataset and only switches the
engine to the new model if it measurably helps.

Everything here runs on the RTX 5080 demo laptop. Nothing in `training/` runs inside the engine.

## One command

```powershell
.\scripts\train_whisper.ps1 -Check      # any laptop, CPU ok: every step on a tiny fake model, ~1 min
.\scripts\train_whisper.ps1 -Quick      # RTX 5080: 300 steps + 200 test clips
.\scripts\train_whisper.ps1             # RTX 5080: full run
```

Run `-Check` first: it catches library-version problems in a minute instead of after hours.

Or step by step, after the script has installed the packages once (it pins your CUDA
PyTorch so pip can't replace it with the CPU-only build):

```powershell
.\.venv\Scripts\python.exe -m training.prepare_torgo       # download + split by speaker -> data/torgo
.\.venv\Scripts\python.exe -m training.finetune_whisper    # LoRA training -> models/whisper-turbo-torgo-ct2
.\.venv\Scripts\python.exe -m training.evaluate            # stock vs adapted -> SHIP / DONT SHIP
```

The script installs `requirements-train.txt` with `requirements.lock.txt` as a constraint,
so pip can't change any package the demo already uses, then checks the GPU still works.

## Use it in the app

Only if `evaluate` says **SHIP**, add this to `config.local.json` on that laptop:

```json
{"whisper_model": "models/whisper-turbo-torgo-ct2"}
```

Remove the line to go back to stock Whisper. The default in `engine/config.py` does not change.

## How it avoids fooling us

- **Split by speaker.** Four TORGO speakers (F04, M03, FC03, MC04) are never trained on.
  A model that only memorized the training speakers would fail on them.
- **Unseen sentences.** TORGO speakers read the same prompts, so `evaluate` also reports
  WER on test sentences no training speaker read.
- **Typical speech is checked too.** It ships only if dysarthric WER drops by at least
  2 points **and** typical speech gets at most 1 point worse.
- **Same decoding as the app.** `evaluate` transcribes through `engine/voice/dictation.py`.

## Limits to know

- TORGO is 8 dysarthric speakers (cerebral palsy or ALS) reading words and short sentences.
  It is not a dataset of older adults, and it has no long free speech. Expect smaller gains
  on a new person than on the test table, and check dictation punctuation by hand.
- The biggest wins for one user usually come from that user's own recordings
  (`personal_commands.py`, Next tier), plus the longer pause setting and mic gain.

## Data terms

TORGO is free for academic, non-profit use; cite Rudzicz, Namasivayam and Wolff (2012),
"The TORGO database of acoustic and articulatory speech from speakers with dysarthria".
**Don't commit or upload the data or the trained model.** `data/` and `models/` are git-ignored.

A larger option for later: the Speech Accessibility Project (UIUC) shares data under a
data use agreement; approval takes weeks.

## What `-Check` does

`python -m training.smoke_test` builds a tiny random Whisper and fake clips, then runs
every real step (split, LoRA train, merge, export, load through `dictation.py`, score)
in about a minute on CPU. No downloads, nothing saved outside a temp folder.
