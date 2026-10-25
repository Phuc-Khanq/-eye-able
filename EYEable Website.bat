@echo off
rem Opens the [eye]able website (landing page and the live demo) in your browser. On the live demo page, click
rem "Allow camera" to start the engine: the page then shows your webcam and the simulated desktop obeys your head,
rem blinks and voice. Close this window (or press Ctrl+C) to stop.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run "Install EYEable.bat" first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "eyeable-frontend\eyeable-frontend\serve_demo.py" %*
if errorlevel 1 pause
