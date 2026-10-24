# Fine-tune Whisper on TORGO (dysarthric speech), then decide if it ships.
# Run from the repo root:
#   .\scripts\train_whisper.ps1 -Check   # any laptop, CPU is fine: test every step on a tiny fake model (~1 min)
#   .\scripts\train_whisper.ps1 -Quick   # RTX 5080: short real run (300 steps, 200 test clips)
#   .\scripts\train_whisper.ps1          # RTX 5080: full run
param([switch]$Check, [switch]$Quick)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Step($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: $what (exit code $LASTEXITCODE)" }
}

# Pin the installed PyTorch too: pip must never swap a CUDA build for the CPU-only one from PyPI.
$torch = (& $py -c "import torch; print(torch.__version__)").Trim()
if ($LASTEXITCODE -ne 0) { throw "PyTorch isn't installed. Run .\scripts\install.ps1 first." }
$pin = Join-Path $env:TEMP "diseyeble-torch-pin.txt"
"torch==$torch" | Set-Content -Encoding ascii $pin
Step "Install training packages (demo packages and PyTorch $torch stay as they are)" {
    & $py -m pip install -r requirements-train.txt -c requirements.lock.txt -c $pin
}

if ($Check) {
    Step "Smoke test: every step on a tiny fake model" { & $py -m training.smoke_test }
    exit 0
}

Step "Check the GPU works" {
    & $py -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)"
}
if (-not (Test-Path "data\torgo\dataset_dict.json")) {
    Step "Download TORGO and split by speaker" { & $py -m training.prepare_torgo }
}
if ($Quick) {
    Step "Train (quick: 300 steps)" { & $py -m training.finetune_whisper --max-steps 300 --eval-every 100 }
    Step "Evaluate (200 test clips)" { & $py -m training.evaluate --limit 200 }
} else {
    Step "Train" { & $py -m training.finetune_whisper }
    Step "Evaluate" { & $py -m training.evaluate }
}
