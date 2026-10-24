# Run the Whisper training on the RTX 5080 laptop

## What it does (short)

1. **Download** the TORGO dysarthric-speech dataset and split it by speaker. Four speakers
   (F04, M03, FC03, MC04) are held out and never trained on.
2. **Train** a small LoRA adapter on top of Whisper large-v3-turbo (2000 steps by default),
   so it hears slurred or slow speech better. Then merge it and convert it for faster-whisper.
3. **Evaluate** stock Whisper against the adapted one on the held-out speakers, using the
   same decoding the app uses. It prints **SHIP** or **DONT SHIP**.

Nothing in the running app changes unless you opt in at the end.

## Steps

From the repo root in PowerShell, with the demo venv already installed (`scripts\install.ps1`):

```powershell
git pull
.\scripts\train_whisper.ps1 -Check    # 1 min: tests every step on a tiny fake model
.\scripts\train_whisper.ps1 -Quick    # ~300 steps + 200 test clips: a trial run
.\scripts\train_whisper.ps1           # full run
```

Run them in that order. If `-Check` fails, stop and send the error: it is a library problem.
`-Quick` shows whether the whole chain works and roughly how long the full run will take.

The script installs only `datasets` and `peft` (everything else, including your CUDA
PyTorch, is pinned), checks the GPU, then runs prepare, train, evaluate. It stops on the
first failure. Re-running skips the download if `data\torgo` already exists.

## Outputs

- `data\torgo`: dataset (git-ignored)
- `models\whisper-turbo-torgo-ct2`: adapted model (git-ignored)
- The last lines printed: SHIP / DONT SHIP with before and after error rates

## If it says SHIP

Add this to `config.local.json` on that laptop:

```json
{"whisper_model": "models/whisper-turbo-torgo-ct2"}
```

Delete the line to go back to stock Whisper. If it says DONT SHIP, change nothing.

## Rules

- Keep the laptop plugged in and awake; the full run takes hours.
- Don't commit or upload `data\` or `models\`: the TORGO terms forbid it.
- If you run out of GPU memory, rerun training by hand with a smaller batch:
  `.\.venv\Scripts\python.exe -m training.finetune_whisper --batch 4 --grad-accum 4`
