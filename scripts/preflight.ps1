# Checks the laptop is ready to run [eye]able, in about a minute, before every demo or test:
#   .\scripts\preflight.ps1            checks, then runs the unit tests with the GPU hidden
#   .\scripts\preflight.ps1 -NoTests   checks only
# It changes nothing. Each line says OK, WARN (read it) or FAIL (fix it before running the app).
param([switch]$NoTests)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$script:fails = 0
function Ok($t)   { Write-Host "  OK    $t" -ForegroundColor Green }
function Warn($t) { Write-Host "  WARN  $t" -ForegroundColor Yellow }
function Fail($t) { Write-Host "  FAIL  $t" -ForegroundColor Red; $script:fails++ }

Write-Host "1. The code" -ForegroundColor Cyan
if ((Test-Path ".git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $branch = (git rev-parse --abbrev-ref HEAD) 2>$null
    $last = (git log -1 --format="%h %s") 2>$null
    Ok "branch $branch, last commit: $last"
    $dirty = (git status --porcelain) 2>$null
    if ($dirty) { Warn "uncommitted changes:`n$(($dirty | Select-Object -First 8) -join "`n")" } else { Ok "everything is committed" }
} else { Ok "downloaded copy (not a git clone)" }
# Features that must be in the files (a missing one means a robocopy folder wasn't applied):
$needs = @(
    @("engine\config.py", "mouth_endpoint",   "a still mouth ends a clip (eyes46)"),
    @("engine\config.py", "live_preview",     "live dictation preview (eyes43)"),
    @("engine\core\topics.py", "PARTIAL_AUDIO", "live preview audio (eyes43)"),
    @("engine\core\topics.py", "QUIT",        '"quit" command (eyes38)'),
    @("engine\config.py", "refine_dictation", "whole-passage refine (eyes39)"),
    @("engine\eyes\nose.py", "class NoseTracker", "head-nose pointer (eyes42)"),
    @("engine\main.py", "--no-venus",         "--no-venus (eyes44)"),
    @("engine\voice\dictation.py", "[whisper]", "timing lines (eyes45)"),
    @("engine\setup_profile.py", "def pick",  "installer profiles (eyes47)")
)
foreach ($n in $needs) {
    if ((Test-Path $n[0]) -and (Select-String -Path $n[0] -SimpleMatch -Pattern $n[1] -Quiet)) { Ok $n[2] }
    else { Fail "$($n[2]): not found in $($n[0]); this copy is out of date (download it again, or apply the missing engine-eyesNN folder)" }
}

Write-Host "2. Settings and models" -ForegroundColor Cyan
if (Test-Path "config.local.json") {
    $raw = Get-Content "config.local.json" -Raw
    Ok "config.local.json: $($raw.Trim())"
    try {
        $cfg = $raw | ConvertFrom-Json
        if ($cfg.whisper_model -and ($cfg.whisper_model -like "*\*" -or $cfg.whisper_model -like "*/*")) {
            if (Test-Path $cfg.whisper_model) { Ok "Whisper model folder exists" } else { Fail "whisper_model folder not found: $($cfg.whisper_model)" }
        }
    } catch { Fail "config.local.json is not valid JSON (a byte-order mark or a typo): $($_.Exception.Message)" }
} else { Warn "no config.local.json: the stock Whisper model is used" }
if (Test-Path ".venv\Scripts\python.exe") { Ok ".venv found" } else { Fail "no .venv: double-click Install EYEable.bat (developers: .\scripts\install.ps1)" }

Write-Host "3. Camera, microphone and GPU" -ForegroundColor Cyan
$py = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match "engine\.main|pointer_feel|camera_probe|gaze|head_|blink|mouth_|look_and_say|click" }
if ($py) {
    Fail "other Python programs may hold the camera or GPU. Stop them (Stop-Process -Id <id>):"
    $py | ForEach-Object { Write-Host "          $($_.ProcessId)  $($_.CommandLine)" }
} else { Ok "no other [eye]able Python program is running" }
$trainer = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match "finetune" }
if ($trainer) { Fail "training is running and uses the GPU: wait for it to finish" }
# What the settings will put on the GPU (MB, measured on the demo PC): Whisper 2500, UI-Venus 6000, clean-up 5000 (3B) / 3500 (1.5B)
$cfg = $null
if (Test-Path "config.local.json") { try { $cfg = Get-Content "config.local.json" -Raw | ConvertFrom-Json } catch { } }
if (-not $cfg) { $cfg = [pscustomobject]@{} }
$need = 0
if (-not $cfg.whisper_device -or $cfg.whisper_device -eq "cuda") { $need += 2500 }
if ($cfg.use_ui_venus -ne $false) { $need += 6000 }
if ($cfg.rewrite_dictation -ne $false) { if ("$($cfg.rewrite_model)" -like "*1.5B*") { $need += 3500 } else { $need += 5000 } }
$smi = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try { $smi = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits) 2>$null } catch { }
}
if ($need -eq 0) { Ok "these settings run everything on the CPU" }
elseif ($smi) {
    $p = ($smi | Select-Object -First 1).Split(",") | ForEach-Object { [int]($_.Trim()) }
    $free = $p[1] - $p[0]
    $line = "GPU: $($p[0]) of $($p[1]) MB used, $($p[2])% busy, $free MB free; these settings need about $need MB"
    if ($free -ge $need + 1000) { Ok $line }
    elseif ($free -ge $need) { Warn "$line. Tight: close Chrome and other GPU programs" }
    else { Fail "$line. Close the programs using the GPU (nvidia-smi lists them), or run with --no-venus" }
} else { Warn "no NVIDIA GPU found, but the settings ask for one: Whisper falls back to the CPU (slower) and look-and-say and the clean-up are off" }
if (Test-Path ".venv\Scripts\python.exe") {
    $mic = & ".venv\Scripts\python.exe" -c "import sounddevice as sd; d=sd.query_devices(kind='input'); print(d['name'])" 2>$null
    if ($mic) { Ok "default microphone: $mic" } else { Fail "no microphone found" }
}

if (-not $NoTests -and (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "4. Unit tests (GPU hidden)" -ForegroundColor Cyan
    $env:CUDA_VISIBLE_DEVICES = "-1"
    $out = & ".venv\Scripts\python.exe" -m unittest discover -s tests -t . 2>&1
    $env:CUDA_VISIBLE_DEVICES = $null
    $tail = ($out | Select-Object -Last 3) -join " "
    if ($tail -match "OK") { Ok $tail } else { Fail "tests did not pass: $tail" }
}

Write-Host ""
if ($script:fails) { Write-Host "$script:fails problem(s): fix the FAIL lines, then run this again." -ForegroundColor Red; exit 1 }
Write-Host "Ready: .\scripts\run_demo.ps1" -ForegroundColor Green
