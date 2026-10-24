# One-command setup for the demo laptop. Run from the repo root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install.ps1
$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

# PyTorch built for CUDA 12.8 (RTX 50-series GPUs need it)
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt

# MediaPipe face model
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"

# Front end, once Nhi has created ui/
if (Test-Path "ui\package.json") { npm --prefix ui install }

& $py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok, torch.cuda.get_device_name(0) if ok else '')"
Write-Host "Setup done."
