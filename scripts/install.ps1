# One-command setup. Run from the repo root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install.ps1            # demo laptop: PyTorch for the RTX 5080 (2.8 GB)
#   .\scripts\install.ps1 -CpuOnly   # coding laptops: small CPU-only PyTorch
# Safe to rerun, for example after requirements change.
param([switch]$CpuOnly)
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

# PyTorch: CUDA 12.8 build for the demo laptop (RTX 50-series needs it), small CPU build otherwise
if ($CpuOnly) {
    Step "Install PyTorch, CPU only" { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu }
    if (-not (Test-Path "config.local.json")) {
        '{"whisper_device": "cpu", "whisper_model": "small", "whisper_compute_type": "int8", "use_ui_venus": false}' |
            Set-Content -Encoding ascii config.local.json
        Write-Host "Created config.local.json for a CPU-only laptop"
    }
} else {
    Step "Install PyTorch for CUDA 12.8 (large download)" { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 }
}

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
