# Installs EYEable on this Windows PC. Double-click "Install EYEable.bat" in the folder, or run:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   ... -Profile cpu        force a profile (auto, full, gpu12, gpu8, cpu); auto picks from the GPU
#   ... -SkipModels         don't download the models now (the app downloads them on its first start)
#   ... -NoShortcuts        no Desktop / Start menu shortcut
# Needs Windows 10 or 11, internet for the downloads, a webcam and a microphone. No Python needed: it
# installs its own (with uv), only inside this folder and uv's cache. Safe to run again (it repairs).
# It changes nothing outside this folder except: uv in %USERPROFILE%\.local\bin, the models in the
# Hugging Face cache (%USERPROFILE%\.cache\huggingface), and the two shortcuts.
param(
    [ValidateSet("auto", "full", "gpu12", "gpu8", "cpu")] [string]$Profile = "auto",
    [switch]$SkipModels,
    [switch]$NoShortcuts
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"      # Invoke-WebRequest is 10x faster without the progress bar
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($what) { Write-Host "`n== $what" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "Failed: $what (exit code $LASTEXITCODE). Run the installer again; if it fails twice, send this window's text to the team." } }

Write-Host "EYEable setup in $root" -ForegroundColor Green
if ($env:OS -ne "Windows_NT") { throw "EYEable runs on Windows 10 or 11 only." }

# ---------------------------------------------------------------- 1. the GPU decides the profile
Step "Checking this PC"
$vram = 0; $gpuName = ""; $driver = ""
$smi = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try { $smi = (nvidia-smi --query-gpu=memory.total,name,driver_version --format=csv,noheader,nounits) 2>$null } catch { $smi = $null }
}
if ($smi) {
    $parts = ($smi | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }
    $vram = [int]$parts[0]; $gpuName = $parts[1]; $driver = $parts[2]
    Write-Host "NVIDIA GPU: $gpuName, $vram MB, driver $driver"
    if ([double]($driver.Split(".")[0]) -lt 570) {
        Write-Host "WARNING: driver $driver is older than 570, which CUDA 12.8 needs. Update it from nvidia.com/drivers, or the GPU parts fall back to the CPU." -ForegroundColor Yellow
    }
} else {
    Write-Host "No NVIDIA GPU found: EYEable will run on the CPU (fewer features)."
}
$useGpu = ($Profile -ne "cpu") -and ($vram -ge 7500)
$needGb = if (-not $useGpu) { 4 } elseif ($vram -ge 15000) { 22 } else { 14 }
$drive = Get-PSDrive -Name (Split-Path -Qualifier $root).TrimEnd(":")
$freeGb = [math]::Round($drive.Free / 1GB, 1)
$homeDrive = Get-PSDrive -Name $env:USERPROFILE.Substring(0, 1)
Write-Host "Free space: $freeGb GB here, $([math]::Round($homeDrive.Free / 1GB, 1)) GB on $($env:USERPROFILE.Substring(0,2)) (models go to the Hugging Face cache there); about $needGb GB needed in all"
if (($drive.Free + $(if ($homeDrive.Name -ne $drive.Name) { $homeDrive.Free } else { 0 })) / 1GB -lt $needGb) {
    throw "Not enough disk space: about $needGb GB needed."
}

# ---------------------------------------------------------------- 2. uv and Python 3.11
Step "Getting uv (it installs Python and the packages)"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uv)) {
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        Check "install uv"
    }
}
& $uv --version; Check "uv"

Step "Python 3.11 and the .venv folder"
$py = Join-Path $root ".venv\Scripts\python.exe"
$ok = $false
if (Test-Path $py) {
    try { $v = (& $py -c "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null; $ok = ($v -eq "3.11") } catch { $ok = $false }
}
if (-not $ok) {
    $sys311 = $null
    try { $sys311 = (py -3.11 -c "import sys; print(sys.executable)") 2>$null } catch { }
    if ($sys311 -and (Test-Path $sys311)) {
        Write-Host "Using the installed Python 3.11: $sys311"
        & $uv venv --python $sys311 .venv; Check "create .venv"
    } else {
        & $uv python install 3.11; Check "install Python 3.11"
        & $uv venv --python 3.11 .venv; Check "create .venv"
    }
} else { Write-Host ".venv with Python 3.11 is already here" }

# ---------------------------------------------------------------- 3. packages
if ($useGpu) {
    Step "PyTorch for NVIDIA GPUs (CUDA 12.8, about 2.8 GB download)"
    & $uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128; Check "install PyTorch (CUDA)"
} else {
    Step "PyTorch, CPU only (small)"
    & $uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cpu; Check "install PyTorch (CPU)"
}
Step "EYEable's packages (exact versions from the demo PC)"
& $uv pip install --python $py -r requirements.lock.txt; Check "install packages"

Step "MediaPipe face model"
New-Item -ItemType Directory -Force models | Out-Null
if (-not (Test-Path "models\face_landmarker.task")) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
}
Write-Host "models\face_landmarker.task ready"

Step "Checking the packages load"
& $py -c "import cv2, mediapipe, numpy, faster_whisper, sounddevice, rapidfuzz, pynput, mss, websockets, tkinter; print('All imports OK')"; Check "imports"

# ---------------------------------------------------------------- 4. settings and models
Step "Choosing what this PC runs"
$profArgs = @("-m", "engine.setup_profile", "--write", "--vram", "$vram")
if ($Profile -ne "auto") { $profArgs += @("--profile", $Profile) }
& $py @profArgs; Check "choose profile"

if (-not $SkipModels) {
    Step "Downloading the models (once; can take 10 to 30 minutes on the full profile)"
    $torchLib = Join-Path $root ".venv\Lib\site-packages\torch\lib"
    if (Test-Path $torchLib) { $env:PATH = "$torchLib;$env:PATH" }
    & $py -m engine.setup_profile --download; Check "download models"
}

# ---------------------------------------------------------------- 5. shortcuts
if (-not $NoShortcuts) {
    Step "Shortcuts"
    $shell = New-Object -ComObject WScript.Shell
    foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
        $lnk = $shell.CreateShortcut((Join-Path $dir "EYEable.lnk"))
        $lnk.TargetPath = Join-Path $root "EYEable.bat"
        $lnk.WorkingDirectory = $root
        $lnk.Description = "Hands-free computer control with your head, eyes and voice"
        $lnk.IconLocation = "$env:SystemRoot\System32\shell32.dll,22"
        $lnk.Save()
        Write-Host "Shortcut: $(Join-Path $dir 'EYEable.lnk')"
    }
}

Step "Final check"
& (Join-Path $PSScriptRoot "preflight.ps1") -NoTests
Write-Host "`nEYEable is installed. Start it with the EYEable shortcut (or EYEable.bat in this folder)." -ForegroundColor Green
Write-Host "First start: face the screen and look at its center for a second; say `"hi Dizzy`" to dictate, `"quit`" to stop."
