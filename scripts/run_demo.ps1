# Starts the engine in its own window, then the front end.
# Later, Electron will start the engine itself (ui/src/main).
$ErrorActionPreference = "Stop"
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "engine.main"
npm --prefix ui run dev
