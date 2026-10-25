# Checks the laptop is ready to run Diseyeble, in about a minute, before every demo or test:
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
$branch = (git rev-parse --abbrev-ref HEAD) 2>$null
$last = (git log -1 --format="%h %s") 2>$null
Ok "branch $branch, last commit: $last"
$dirty = (git status --porcelain) 2>$null
if ($dirty) { Warn "uncommitted changes:`n$(($dirty | Select-Object -First 8) -join "`n")" } else { Ok "everything is committed" }
# Features that must be in the files (a missing one means a robocopy folder wasn't applied):
$needs = @(
    @("engine\config.py", "mouth_endpoint",   "eyes46: a still mouth ends a clip"),
    @("engine\config.py", "live_preview",     "eyes43: live dictation preview"),
    @("engine\core\topics.py", "PARTIAL_AUDIO", "eyes43: live preview audio topic"),
    @("engine\core\topics.py", "QUIT",        "eyes38: quit"),
    @("engine\config.py", "refine_dictation", "eyes39: whole-passage refine"),
    @("engine\eyes\nose.py", "class NoseTracker", "eyes42: head-nose pointer"),
    @("engine\main.py", "--no-venus",         "eyes44: --no-venus"),
    @("engine\voice\dictation.py", "[whisper]", "eyes45: timing lines")
)
foreach ($n in $needs) {
    if ((Test-Path $n[0]) -and (Select-String -Path $n[0] -SimpleMatch -Pattern $n[1] -Quiet)) { Ok $n[2] }
    else { Fail "$($n[2]): not found in $($n[0]); apply its engine-eyesNN folder (robocopy ... . /E)" }
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
if (Test-Path ".venv\Scripts\python.exe") { Ok ".venv found" } else { Fail "no .venv: run .\scripts\install.ps1" }

Write-Host "3. Camera, microphone and GPU" -ForegroundColor Cyan
$py = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match "engine\.main|pointer_feel|camera_probe|gaze|head_|blink|mouth_|look_and_say|click" }
if ($py) {
    Fail "other Python programs may hold the camera or GPU. Stop them (Stop-Process -Id <id>):"
    $py | ForEach-Object { Write-Host "          $($_.ProcessId)  $($_.CommandLine)" }
} else { Ok "no other Diseyeble Python program is running" }
$trainer = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match "finetune" }
if ($trainer) { Fail "training is running and uses the GPU: wait for it to finish" }
$smi = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits) 2>$null
if ($smi) {
    $p = $smi.Split(",") | ForEach-Object { [int]($_.Trim()) }
    $free = $p[1] - $p[0]
    if ($free -ge 11000) { Ok "GPU: $($p[0]) of $($p[1]) MB used, $($p[2])% busy, $free MB free" }
    elseif ($free -ge 8000) { Warn "GPU: only $free MB free ($($p[0]) used, $($p[2])% busy); the full app peaks near 15000 MB: close Chrome/other GPU programs or use --no-venus" }
    else { Fail "GPU: only $free MB free ($($p[0]) used, $($p[2])% busy): close the programs using it (nvidia-smi shows them)" }
} else { Warn "nvidia-smi not found: can't check the GPU" }
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
