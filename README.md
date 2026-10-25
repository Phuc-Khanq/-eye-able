# [eye]able

Control your whole computer hands-free, with nothing but a webcam and a microphone. Your head moves the cursor, a long blink clicks, and your voice runs commands and types. UI-Venus 1.5 (released February 2026) turns a spoken name, like "click Send", into an exact click.

Built at Gator Hack V (UF AI Days 2026). Everything runs on the laptop: camera frames, audio and screenshots stay in memory and are never saved.

Two ways to try it:

- **The website, no install:** [phuc-khanq.github.io/-eye-able](https://phuc-khanq.github.io/-eye-able/). The landing page explains every control; **Live demo** runs a simulated Windows desktop that you drive with your head, blinks and voice, right in the browser.
- **The app on your own PC:** the installer below. It moves the real Windows pointer.

## Install on your own PC

**Needs:** Windows 10 or 11, a webcam, a microphone, internet for the first install, and 4 to 22 GB of disk (depends on your GPU, below). No Python or other setup needed.

1. Download `EYEable-<version>.zip` from [Releases](../../releases) (or the green **Code -> Download ZIP** button) and unzip it anywhere, for example `Documents\EYEable`.
2. Double-click **`Install EYEable.bat`**. If Windows says "Windows protected your PC", click **More info -> Run anyway** (the script isn't signed). It installs its own Python, the packages and the models into that folder, picks what your PC can run, and adds an **EYEable** shortcut to the Desktop and Start menu (file names spell it EYEable, since Windows paths and brackets don't mix well). Takes 10 to 40 minutes, mostly downloads.
3. Start [eye]able from the **EYEable** shortcut. Face the screen and look at its center for a second. Say "hi Dizzy" to dictate, "bye" to stop dictating, "quit" to close it.

What you get depends on your graphics card (the installer picks for you; `scripts\setup.ps1 -Profile cpu` forces one):

| Your PC | Profile | What runs | Disk |
| --- | --- | --- | --- |
| NVIDIA with 16 GB (RTX 4080, 5080, 4090, 5090, 4060 Ti 16 GB) | `full` | Everything, as in our demo: head pointer, blink click, voice commands, dictation with live preview and clean-up, look-and-say | about 22 GB |
| NVIDIA with 12 GB | `gpu12` | All of it except the dictation clean-up (your own words are typed) | about 14 GB |
| NVIDIA with 8 GB | `gpu8` | All of it except look-and-say; a smaller clean-up model | about 14 GB |
| No NVIDIA card (most laptops) | `cpu` | Head pointer, blink click, voice commands and dictation (Whisper small on the CPU: slower, less accurate) | about 4 GB |

If it doesn't start: run `scripts\preflight.ps1` (right-click -> Run with PowerShell). It checks the camera isn't busy, the GPU has room, the microphone works and the files are complete, and says what to fix. Camera or microphone blocked: Windows Settings -> Privacy & security -> Camera / Microphone -> turn on "Let desktop apps access". To uninstall, delete the folder and the shortcuts; the models are in `%USERPROFILE%\.cache\huggingface\hub`.

Privacy: camera frames, audio and screenshots stay in memory on your PC and are never saved or sent anywhere. The only network use is downloading the models once.

Not included: our Whisper model fine-tuned on dysarthric speech (TORGO) is for academic, non-profit use and is never published, so downloads use the standard Whisper model. macOS and Linux aren't supported yet (the overlay and the clicking use Windows APIs).

## Routes

| Route | Main engineer | Helpers | Code |
| --- | --- | --- | --- |
| Eyes, cursor and UI-Venus | Khang | Hung | `engine/eyes/` |
| Voice, dictation and control | Khoi | Khoa | `engine/voice/`, `engine/core/` |
| Website, UI and live demo | Nhi | Hung, Khoa | `eyeable-frontend/eyeable-frontend/` |

Shared contracts live in `engine/core/types.py` and `engine/core/messages.py`. Tell the team before changing either one.

## Developer setup (Windows, demo laptop)

Needs Python 3.11, Git, and an NVIDIA GPU for the full profile (RTX 50-series needs the CUDA 12.8 PyTorch build, which the script installs). Node.js 20 or newer only for the website's JavaScript tests.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1            # demo laptop; add -CpuOnly on a laptop without an NVIDIA GPU
```

The script creates `.venv`, installs PyTorch and `requirements.txt`, and downloads the MediaPipe face model to `models/`. UI-Venus and Whisper download to the Hugging Face cache the first time they run.

## Run

| What | How |
| --- | --- |
| The app (moves the real pointer) | `EYEable.bat`, or `.\scripts\run_demo.ps1` |
| The website with the real engine | `EYEable Website.bat`, then **Live demo -> Allow camera** (it starts the engine) |
| The engine on its own | `.\.venv\Scripts\python.exe -m engine.main` (`--help` lists the options) |
| Website work without a camera | `.\.venv\Scripts\python.exe -m tests.fake_engine` |
| Checks before a demo | `.\scripts\preflight.ps1` |

The pointer mode is `head` (the head moves the pointer like a mouse). `--pointer head-eyes` adds the eye pull. `--pointer head-push` is a paused experiment, kept as an option. The website is plain HTML, CSS and JavaScript in `eyeable-frontend/eyeable-frontend/`, with no build step. Every push to `main` that touches it republishes the hosted site (`.github/workflows/pages.yml`).

## Working together

**Read `CONTRIBUTING.md` before your first commit.** The demo runs on Khang's laptop, and those rules keep your code working there.

- `main` is the release: the installer zip, the hosted website and the demo laptop all come from it. Work on a route branch (`eyes`, `voice`, `ui`) and merge into `main` through a pull request.
- Never commit model files, recordings, calibration data or keys. `.gitignore` covers these.
- Keep modules talking through the event bus only, so any one can be swapped for a stub.

## Third-party models

- UI-Venus 1.5 (Ant Group): Hugging Face lists Apache 2.0; the GitHub page says research and educational use only.
- MediaPipe Face Landmarker (Google): Apache 2.0.
- Whisper weights (OpenAI) via faster-whisper: MIT.
- TORGO dysarthric speech database (University of Toronto), used only in `engine/voice/finetune/` to adapt Whisper: free for academic, non-profit use; cite Rudzicz et al. (2012). The data and adapted model are never committed. See `engine/voice/finetune/DATA.md`.

## License

Our code is MIT licensed. See `LICENSE`.
