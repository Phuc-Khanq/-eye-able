# Datasets for the Whisper fine-tune

**The rule:** the data, and any model or LoRA adapter trained on TORGO, are for academic,
non-profit use only. They are **never published or uploaded**: not to GitHub, not to the
Hugging Face Hub, not as a public Kaggle dataset or model. Share a trained model inside the
team only, privately (for example a USB stick). Git ignores `data/`, `checkpoints/`,
`models/` and `downloads/`, so they can't be committed by accident.

`python -m engine.voice.finetune.get_data` downloads everything below except the personal
pack. It is meant to run on Kaggle, where the data stays in `/kaggle/temp` and never goes
into the notebook's output.

| Dataset | What we use | Size kept | License | Where |
| --- | --- | --- | --- | --- |
| TORGO | dysarthric speakers F01 F03 F04 M01 M02 M03 M04 M05: array-mic WAVs + prompts | ~2 GB | academic, non-profit use | `data/voice/torgo/` |
| LibriSpeech | dev-clean and test-clean (typical speech), FLAC + transcripts | ~0.7 GB | CC BY 4.0 | `data/voice/librispeech/` |
| Personal voice pack | your own recordings from `record_pack` | small | yours | `data/voice/personal/` |

## TORGO

Speakers with dysarthria (speech affected by cerebral palsy or ALS) reading single words,
sentences and short phrases. We use only the eight dysarthric speakers, not the controls.

- Source: University of Toronto, https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
  (archives `F.tar.bz2` and `M.tar.bz2`).
- License: free for academic, non-profit use. No redistribution.
- Kept: `wav_arrayMic` WAVs and `prompts/` texts. Where an utterance has no array-mic file,
  its `wav_headMic` file is used instead. The archives are deleted after extraction.
- Cleaned out: prompts that aren't words to transcribe (bracketed instructions such as
  `[say Ah for 5 seconds]`, picture descriptions `input/images/*.jpg`, `xxx`, empty files).
  A word with a hint, `lead [as in I will lead you]`, keeps just the word.
- Citation:

  > Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of acoustic and
  > articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*,
  > 46(4), 523–541. https://doi.org/10.1007/s10579-011-9145-0

## LibriSpeech

Audiobook recordings of typical adult speech. We use it to make sure the fine-tuned model
doesn't get worse on normal voices.

- Source: OpenSLR resource 12, https://www.openslr.org/12 (`dev-clean.tar.gz`, `test-clean.tar.gz`).
  We don't download `train-clean` (too big).
- License: CC BY 4.0 (attribution required).
- Citation:

  > Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: an ASR corpus
  > based on public domain audio books. *Proc. ICASSP 2015*, 5206–5210.

## Personal voice pack (optional, never on Kaggle)

`python -m engine.voice.finetune.record_pack` records you reading about 150 everyday
computer phrases and sentences, after you type "yes" to agree. This consented session is the
only place the project writes audio to disk. The files stay in `data/voice/personal/`
(WAV + `transcripts.csv` with `file,text`). Delete them any time: say "delete my voice data"
in the app, or run `python -m engine.voice.voice_data`.

## Splits (who is used for what)

`split.json` holds speaker IDs only (no audio, no transcripts).

- **TORGO, by speaker.** No speaker is in two splits.
  - test: **F04** (female, very mild) and **M01** (male, severe)
  - dev: **M05** (male, moderate to severe)
  - train: F01, F03, M02, M03, M04
- **TORGO repeats prompts across speakers**, so test results are split by text, judged
  against the training text:
  - *unseen-text sentences*: 3+ words, never in training (the **primary** number)
  - *seen-text sentences*: 3+ words, also read by a training speaker
  - *single words*: 1–2 words
- **LibriSpeech:** test-clean = test. dev-clean speakers are split in half (alternating
  within each gender): one half is the typical-speech mix during training (at most 80 clips
  per speaker), the other half is dev (300 clips).
- **Personal pack:** 80/20 by sentence (the same sentence is never in both).
- Dev is for choosing settings. Test is only run on the stock model and on a finished
  candidate, never to tune.

Clips shorter than 0.3 s or longer than 30 s (Whisper's limit) are left out.

## Other files

- `english_spelling.json`: British-to-American spellings used by Whisper's text normalizer
  when scoring WER. A copy of `normalizer.json` from the openai/whisper models on Hugging
  Face (MIT license, © OpenAI). It's kept in the repo so every evaluation scores the same way.
