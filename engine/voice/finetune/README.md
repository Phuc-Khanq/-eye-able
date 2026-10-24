# Fine-tuning Whisper for impaired and elderly speech

This folder trains a version of Whisper that understands dysarthric (impaired) speech
better, without forgetting normal voices. Training runs on a **free Kaggle notebook**
(one NVIDIA T4 GPU). Your laptop is only used to test that the code runs.

What you get at the end: `whisper-elder-ct2.zip`, a model the app loads instead of the
stock `large-v3-turbo`, plus `RESULTS.md` with the scores.

> Read [DATA.md](DATA.md) first. The model is trained on TORGO, which is for academic,
> non-profit use only: **never publish or upload the model, the adapter or the data.**

---

## 1. What gets measured (the target)

Word error rate (**WER**, lower is better) on test speakers the model never heard during
training (F04 and M01 from TORGO):

| Group | Target vs the stock model |
| --- | --- |
| Dysarthric, **unseen-text sentences** (PRIMARY) | at least 20% lower WER |
| Dysarthric, seen-text sentences | at least 20% lower WER |
| Dysarthric, single words | at least 10% lower WER |
| LibriSpeech test-clean (typical speech) | no more than 0.5 WER points worse |
| Warm latency for a 5 s clip | no more than 15% slower |

"Unseen text" matters because TORGO speakers read the same prompts. A model can look great
just by memorizing sentences it was trained on, so the honest number is the one on
sentences no training speaker read.

---

## 2. One-time Kaggle setup (about 10 minutes)

1. Make an account at **kaggle.com**.
2. **Verify your phone:** click your profile picture (top right) → **Settings** → **Phone
   verification**. Without it, Kaggle gives you no GPU and no internet.
3. Create a notebook: **Create** (left menu) → **New Notebook**.
4. In the notebook's right-hand panel (**Settings**; click the small arrow if it's hidden):
   - **Accelerator:** **GPU T4 x2** (we use one of them). Not "None", not TPU.
   - **Internet:** **On**. It needs to download the code, the data and the Whisper model.
   - Language: Python. Environment: leave the default ("Always use latest environment").
5. Rename it at the top, for example `diseyeble-whisper-finetune`.

---

## 3. The two notebook cells

Delete any example cells, then add these two code cells.

**Cell 1: get the code and the training packages**

```python
!mkdir -p /kaggle/temp
!rm -rf /kaggle/temp/diseyeble
!git clone --depth 1 --branch voice https://github.com/Phuc-Khanq/diseyeble.git /kaggle/temp/diseyeble
!pip install -q -r /kaggle/temp/diseyeble/requirements-train.txt
!python -c "import sys, torch; print(sys.version); print('torch', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))"
```

**Cell 2: run everything (about 10 hours, nobody needs to watch)**

```python
%cd /kaggle/temp/diseyeble
!python -m engine.voice.finetune.run_all --platform kaggle --hours 10 --out /kaggle/working
```

Why `/kaggle/temp`: Kaggle saves everything in `/kaggle/working` as downloadable output.
The code, the data and the big checkpoints live in `/kaggle/temp` (scratch space), so
**TORGO audio never ends up in the output**. Only small result files go to `/kaggle/working`.

---

## 4. Start the run: Save & Run All

Don't just press "Run": that stops when you close the browser.

1. Click **Save Version** (top right).
2. Choose **Save & Run All (Commit)**, then **Save**.
3. Kaggle now runs the notebook on its own computer, up to 12 hours. You can close the
   browser and turn your laptop off.
4. It uses about 10 of your ~30 free GPU hours per week. Check them under Settings → Quotas.

To watch progress: open the notebook → **Versions** (or "View Active Events" at the bottom
left) → click the running version → **Logs**.

What `run_all` does, in order:

1. downloads TORGO (dysarthric speakers only) and LibriSpeech dev/test (`get_data`)
2. builds the clip lists and splits (`prepare`)
3. scores the stock model on the test set (the baseline)
4. trains each setting in `search.json`, picking the best on the **dev** set only.
   It saves the adapter every 30 minutes, measures its speed in the first 50 steps and
   shortens a run that wouldn't fit in the time left.
5. converts the best one for the app and scores it on the test set once
6. writes the results to `/kaggle/working`

It never asks a question, so it can't get stuck waiting for you. If a step fails, the
error is in the log and in `RESULTS.md`, and everything that finished is still saved.

---

## 5. Download the results

When the version shows **Succeeded** (or finishes with an error):

1. Open the notebook version → **Output** tab.
2. Click **Download** (download all). You get:
   - `RESULTS.md`: the table "stock vs fine-tuned vs target" and every run's dev numbers
   - `results.json`: the same numbers for scripts
   - `whisper-elder-ct2.zip`: the model for the app (about 1.6 GB)
   - `best_adapter/`: the small LoRA adapter (to resume or re-export)
   - `run_all.log`: what happened, minute by minute
   - `resume_adapter/`: only if a run was cut short
3. Put them in the repo's `downloads/` folder (git ignores it), for example
   `downloads/2026-10-25/`.

There's no audio and there are no transcripts in the output.

---

## 6. Use the model in the app

1. Unzip `whisper-elder-ct2.zip` into the repo's `models/` folder, so you have
   `models/whisper-elder-ct2/model.bin`.
2. Add this to `config.local.json` in the repo root (keep any settings already there):

   ```json
   {"whisper_model": "models/whisper-elder-ct2"}
   ```
3. Run the app or `python -m engine.voice.live_check`. With no config line, the app uses
   the stock `large-v3-turbo` as before. The defaults in `engine/config.py` never change.

**Teammates don't need to train again.** Give them the zip privately (USB stick, or a
private team share) and they do steps 1–2. Never upload it publicly (TORGO license).
On a CPU-only laptop the fine-tuned model is slow (it's the large model). Use it on a GPU.

---

## 7. Continue from a previous run (resume)

To use a second session, for example to try more settings or finish a run cut short:

1. Open the notebook → **Edit**.
2. Right panel → **Add Input** → **Your Work** → pick this notebook's previous version.
   It appears under `/kaggle/input/<notebook-name>/`. Look at the right panel for the exact path.
3. Change cell 2 to:

   ```python
   %cd /kaggle/temp/diseyeble
   !python -m engine.voice.finetune.run_all --platform kaggle --hours 10 --out /kaggle/working \
       --resume-from /kaggle/input/diseyeble-whisper-finetune
   ```
4. Save & Run All again.

It reuses the baseline from the earlier output, skips finished settings, continues a run
that was cut short, and keeps the best adapter so far.

To try different settings, edit `search.json` on the `voice` branch, push, and resume.
Each entry can change `lora_r`, `lora_alpha`, `lr`, `typical_mix`, `max_steps`, `eval_every`,
etc. (see `TrainConfig` in `train.py`).

---

## 8. Test on your laptop first (the tiny smoke test, a few minutes, CPU)

This runs the whole pipeline with Whisper tiny, 40 LibriSpeech clips and 5 training steps.
The numbers mean nothing; it only proves the code works end to end.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m engine.voice.finetune.get_data --tiny      # 347 MB, asks first
.\.venv\Scripts\python.exe -m engine.voice.finetune.prepare --tiny
.\.venv\Scripts\python.exe -m engine.voice.finetune.train --tiny
.\.venv\Scripts\python.exe -m engine.voice.finetune.export --tiny
.\.venv\Scripts\python.exe -m engine.voice.finetune.evaluate --tiny
# or all at once, never asking anything:
.\.venv\Scripts\python.exe -m engine.voice.finetune.run_all --tiny --out downloads\tiny-test
```

---

## 9. The scripts

Each one runs with `python -m engine.voice.finetune.<name>`, has `--help`, and takes
`--data-dir` (default `<repo>/data/voice`).

| Script | What it does |
| --- | --- |
| `get_data` | downloads TORGO and LibriSpeech; shows sizes and free disk space, then asks (`--yes` skips) |
| `prepare` | builds `data/voice/manifests/all.csv` (path, text, speaker, group, split) and prints clips and minutes per set |
| `evaluate` | WER per group and per test speaker, with the app's own decoding (`Dictation.transcribe`), plus 5 s latency; saves JSON in `checkpoints/` |
| `train` | LoRA fine-tune of large-v3-turbo; dev checks, best checkpoint, early stop, `--resume` |
| `export` | merges the LoRA and converts to CTranslate2 float16 → `models/whisper-elder-ct2/`, then evaluates |
| `run_all` | everything above, unattended, for Kaggle |
| `record_pack` | records your own voice pack (about 150 phrases), only after you agree |

Settings tried by `run_all` are in `search.json`. Speakers per split are in `split.json`.

---

## 10. If something goes wrong

| Problem | What to do |
| --- | --- |
| "No GPU" or a CUDA error in cell 1 | Accelerator must be **GPU T4 x2**, and your phone must be verified. |
| Download errors in `get_data` | Internet must be **On**. Then just run again: it resumes. |
| `faster-whisper failed here ... using the transformers backend` | Fine: scoring falls back to transformers with the same greedy decoding, and `RESULTS.md` says so. |
| `Out of GPU memory: micro-batch is now 2` | Fine: it halves the batch by itself and keeps going. |
| The run stopped at 12 h | Use step 7 (resume) with `--hours` a bit lower. |
| `RESULTS.md` says "not met" | That's a real result too. Try other settings in `search.json` and resume. Never tune on the test set. |
