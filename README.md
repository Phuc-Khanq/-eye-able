# EYEable

Control your whole computer hands-free, with nothing but a webcam and a microphone. Your head moves the cursor, a long blink clicks, and your voice runs commands and types. UI-Venus 1.5 (released February 2026) turns a spoken name, like "click Send", into an exact click.

Built at Gator Hack V (UF AI Days 2026). Everything runs on the laptop: camera frames, audio and screenshots stay in memory and are never saved.

## Install on your own PC

**Needs:** Windows 10 or 11, a webcam, a microphone, internet for the first install, and 4 to 22 GB of disk (depends on your GPU, below). No Python or other setup needed.

1. Download `EYEable-<version>.zip` from [Releases](../../releases) (or the green **Code -> Download ZIP** button) and unzip it anywhere, for example `Documents\EYEable`.
2. Double-click **`Install EYEable.bat`**. If Windows says "Windows protected your PC", click **More info -> Run anyway** (the script isn't signed). It installs its own Python, the packages and the models into that folder, picks what your PC can run, and adds a **EYEable** shortcut to the Desktop and Start menu. Takes 10 to 40 minutes, mostly downloads.
3. Start **EYEable** from the shortcut. Face the screen and look at its center for a second. Say "hi Dizzy" to dictate, "bye" to stop dictating, "quit" to close it.

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
| UI, UX and front end | Nhi | Hung, Khoa | `ui/` |

Shared contracts live in `engine/core/types.py` and `engine/core/messages.py`. Tell the team before changing either one.

## Developer setup (Windows, demo laptop)

Needs Python 3.11, Node.js 20 or newer, Git and an NVIDIA GPU (RTX 50-series needs the CUDA 12.8 PyTorch build, which the script installs).

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1
```

The script creates `.venv`, installs PyTorch and `requirements.txt`, and downloads the MediaPipe face model to `models/`. UI-Venus and Whisper download to the Hugging Face cache the first time they run.

### Front end (one time, Nhi)

From the repo root:

```powershell
npm create @quick-start/electron@latest ui -- --template react-ts
cd ui
npm install
```

Then shape `ui/src/` like this:

```
ui/src/main/                 windows, tray, hotkey, starts the engine
ui/src/preload/              safe bridge between Electron and React
ui/src/renderer/cursor/      cursor layer: ring states, bubble, radial menu, calibration
ui/src/renderer/console/     test console: webcam, bars, UI-Venus panel, log, settings
ui/src/renderer/shared/      WebSocket client, message types, style tokens
```

## Run

```powershell
.\scripts\run_demo.ps1
```

Or run the parts on their own:

```powershell
.\.venv\Scripts\python.exe -m engine.main      # engine
.\.venv\Scripts\python.exe -m tests.fake_engine  # UI work without a camera
npm --prefix ui run dev                          # front end
```

## Working together

**Read `CONTRIBUTING.md` before your first commit.** The demo runs on Khang's laptop, and those rules keep your code working there.

- One branch per route: `eyes`, `voice`, `ui`. Merge into `main` often through pull requests.
- Never commit model files, recordings, calibration data or keys. `.gitignore` covers these.
- Keep modules talking through the event bus only, so any one can be swapped for a stub.

## Third-party models

- UI-Venus 1.5 (Ant Group): Hugging Face lists Apache 2.0; the GitHub page says research and educational use only.
- MediaPipe Face Landmarker (Google): Apache 2.0.
- Whisper weights (OpenAI) via faster-whisper: MIT.
- TORGO dysarthric speech database (University of Toronto), used only in `engine/voice/finetune/` to adapt Whisper: free for academic, non-profit use; cite Rudzicz et al. (2012). The data and adapted model are never committed. See `engine/voice/finetune/DATA.md`.

## License

Our code is MIT licensed. See `LICENSE`.
