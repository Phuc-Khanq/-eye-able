# Starts the Diseyeble engine for the demo. Run from anywhere:  .\scripts\run_demo.ps1
#   .\scripts\run_demo.ps1                 normal run (head pointer, long-blink click, voice, dictation)
#   .\scripts\run_demo.ps1 -Record         recording the demo video: the dictation card shows up in OBS
#   .\scripts\run_demo.ps1 -GpuLog         also print the GPU memory in use, and the peak when you stop
#   .\scripts\run_demo.ps1 -- --no-blink   anything after -- goes to engine.main as it is
# The output is shown here and also saved to calibration\incoming\run.log (git ignores that folder).
# Stop with Ctrl+C. There is no separate front end: the engine draws its own feedback (the ring at the cursor
# and the dictation card).
param(
    [switch]$Record,
    [switch]$GpuLog,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$EngineArgs
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "No .venv here. Run  .\scripts\install.ps1  first."
}

# PyTorch's DLLs must come before any other copy on PATH, or Whisper/UI-Venus fail to load on Windows.
$torchLib = Join-Path $root ".venv\Lib\site-packages\torch\lib"
if (Test-Path $torchLib) { $env:PATH = "$torchLib;$env:PATH" }

$flags = @()
if ($Record) { $flags += "--record" }
if ($GpuLog) { $flags += "--gpu-log" }
foreach ($a in $EngineArgs) { if ($a -ne "--") { $flags += $a } }

if (-not (Test-Path "config.local.json")) {
    Write-Host "Note: no config.local.json, so the defaults in engine\config.py are used (stock Whisper)." -ForegroundColor Yellow
}
New-Item -ItemType Directory -Force "calibration\incoming" | Out-Null
$log = "calibration\incoming\run.log"

$cmd = "`"$py`" -m engine.main " + ($flags -join " ") + " 2>&1"
Write-Host "Starting: python -m engine.main $($flags -join ' ')" -ForegroundColor Cyan
# cmd /c so PowerShell does not paint every stderr line red
cmd /c $cmd | Tee-Object -FilePath $log
