# Rules for every commit

Diseyeble runs on Khang's laptop: Windows, Python 3.11, an RTX 5080 with 16 GB, and Node 20 or newer. Code that doesn't run there doesn't count.

## Environment

1. **Python 3.11 in the repo's `.venv`,** set up with `scripts\install.ps1`. Don't use features from Python 3.12 or newer.
2. **Add a package everywhere or nowhere.** A new Python package goes into `requirements.txt` with an exact version (`name==1.2.3`), and a new npm package goes into `ui/package.json`, in the same commit as the code that needs it. Never rely on something installed only on your own laptop.
3. **Defaults are Khang's laptop.** Defaults in `engine/config.py` must work on the demo laptop (GPU on, Whisper large-v3-turbo, UI-Venus 2B). If your laptop differs, override it in `config.local.json` (git ignores it), never by changing a default.

## Code

4. **No hard-coded paths or machine details.** Build paths with `pathlib` from the repo root; no `C:\Users\...`, no drive letters. Camera index, ports, devices and model names come from `config.py`.
5. **Open every text file with `encoding="utf-8"`.** Windows doesn't default to UTF-8.
6. **Import heavy libraries inside the function that uses them** (torch, transformers, faster-whisper), so the engine still starts on a laptop without CUDA.
7. **Routes only talk through the contracts.** Use `engine/core/types.py`, `engine/core/messages.py` and the event bus. Never import another route's internals, and don't edit another route's folder without asking its owner.
8. **Contract changes need a heads-up.** A change to `types.py` or `messages.py` gets a message in the team chat and a review from each affected route. If messages change, the UI's copy changes in the same pull request.
9. **Never block the camera or audio thread.** Whisper and UI-Venus run in their own worker threads.
10. **Privacy is not optional.** Camera frames, audio and screenshots stay in memory and are never written to disk. Nothing leaves the laptop at runtime except the first-run model downloads, and no API keys go in the code.

## Git

11. **Work on your route branch** (`eyes`, `voice`, `ui`) and merge into `main` only through a pull request. Never commit straight to `main`, and never force-push it.
12. **Pull `main` into your branch before you push,** and fix any conflicts on your branch.
13. **Small, labeled commits.** Start each message with the route: `eyes:`, `voice:`, `ui:` or `core:`.
14. **Never commit** models, recordings, calibration data, `config.local.json`, `.env`, `.venv`, `node_modules`, the project description, or any file over 50 MB. Stage your own folder (`git add engine/eyes`), not `git add .`.

## Before you open a pull request

15. **Prove it starts.** On the latest `main` plus your changes:
    - `python -m compileall -q engine tests` shows no errors
    - `python -m engine.main` starts without a traceback
    - `npm --prefix ui run build` succeeds, if you touched `ui/`

## Khang's merge check

16. **The demo laptop has the final say.** After a merge, Khang pulls `main` and reruns `scripts\install.ps1` if `requirements.txt` changed. Then he runs `scripts\run_demo.ps1`. If it breaks, the merge is undone with `git revert` and fixed on the route branch, so `main` never stays broken.

## A normal work cycle

```powershell
git switch eyes
git pull origin main
# ...code...
git add engine/eyes
git commit -m "eyes: skip blink frames in eye_features"
git push origin eyes
# then open a pull request from eyes into main on GitHub
```
