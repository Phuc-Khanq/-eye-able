@echo off
rem Starts Diseyeble. Say "quit" (or press Ctrl+C in this window) to stop.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_demo.ps1" %*
if errorlevel 1 pause
