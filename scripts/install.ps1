# One-command setup. Run from the repo root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install.ps1
# Safe to rerun, for example after requirements change.
$ErrorActionPreference = "Stop"

function Step($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: $what (exit code $LASTEXITCODE)" }
}

Step "Check Python 3.11" { py -3.11 --version }
Step "Create .venv" { py -3.11 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
Step "Upgrade pip" { & $py -m pip install --upgrade pip }

# PyTorch built for CUDA 12.8 (RTX 50-series GPUs need it; also runs on CPU-only laptops)
Step "Install PyTorch (large download)" { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 }

# Exact versions from the demo laptop once they exist, otherwise the loose list
if (Test-Path "requirements.lock.txt") {
    Step "Install packages from requirements.lock.txt" { & $py -m pip install -r requirements.lock.txt }
} else {
    Step "Install packages from requirements.txt" { & $py -m pip install -r requirements.txt }
}

Write-Host "`n== Download the MediaPipe face model" -ForegroundColor Cyan
New-Item -ItemType Directory -Force models | Out-Null
if (-not (Test-Path "models\face_landmarker.task")) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
}

# Front end, once Nhi has created ui/
if (Test-Path "ui\package.json") { Step "Install front-end packages" { npm --prefix ui install } }

Step "Check imports" { & $py -c "import cv2, mediapipe, numpy, sklearn, mss, pynput, transformers, faster_whisper, sounddevice, rapidfuzz, librosa, websockets; print('All imports OK')" }
Step "Check GPU" { & $py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok, torch.cuda.get_device_name(0) if ok else '(CPU only)')" }
Write-Host "`nSetup done." -ForegroundColor Green
