## What this changes

Route: eyes / voice / ui / core

## Checklist (see CONTRIBUTING.md)

- [ ] I pulled the latest `main` into my branch first
- [ ] Only my route's folder changed, or I told the owner of anything else I touched
- [ ] If `core/types.py` or `core/messages.py` changed: I told the team, and the UI's copy matches
- [ ] New packages are in `requirements.txt` or `ui/package.json` with exact versions
- [ ] No hard-coded paths, drive letters or machine-specific settings (those go in `config.local.json`)
- [ ] Heavy libraries (torch, transformers, faster-whisper) are imported inside functions
- [ ] Nothing is saved to disk from the camera, mic or screen
- [ ] No models, recordings, keys, `config.local.json` or files over 50 MB
- [ ] `python -m compileall -q engine tests` has no errors
- [ ] `python -m engine.main` starts without a traceback
- [ ] `npm --prefix ui run build` succeeds (if `ui/` changed)
