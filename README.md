# Diseyeble

Control your whole computer hands-free, with nothing but a webcam. Eye tracking moves the cursor, head gestures click, and dictation types. UI-Venus 1.5 (released February 2026) turns a rough look plus a spoken name, like "click Send", into an exact click.

Built at Gator Hack V (UF AI Days 2026). Everything runs on the laptop: camera frames, audio and screenshots stay in memory and are never saved.

## Routes

| Route | Main engineer | Helpers | Code |
| --- | --- | --- | --- |
| Eyes, cursor and UI-Venus | Khang | Hung | `engine/eyes/` |
| Voice, dictation and control | Khoi | Khoa | `engine/voice/`, `engine/core/` |
| UI, UX and front end | Nhi | Hung, Khoa | `ui/` |

Shared contracts live in `engine/core/types.py` and `engine/core/messages.py`. Tell the team before changing either one.

## Setup (Windows, demo laptop)

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
- TORGO dysarthric speech database (University of Toronto), used only in `training/` to adapt Whisper: free for academic, non-profit use; cite Rudzicz et al. (2012). The data and adapted model are never committed. See `training/README.md`.

## License

Our code is MIT licensed. See `LICENSE`.
