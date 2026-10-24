# Shows that the Whisper fine-tune (engine.voice.finetune.run_all) is alive, even when it
# prints nothing for a few minutes (unpacking TORGO, downloading a model, a dev check).
#
# Run it in a SECOND PowerShell window, from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_training.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_training.ps1 -Out downloads\tiny-test
#
# Read-only: it never changes, pauses or stops the training.
# Ctrl+C in THIS window only stops the watcher. Never press Ctrl+C in the training window.
param(
    [string]$Out = "downloads\5080-run",
    [int]$Every = 30
)
$ErrorActionPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
$log  = Join-Path $root "$Out\run_all.log"
$data = Join-Path $root "data\voice"
$ckpt = Join-Path $root "checkpoints"
$hf   = Join-Path $env:USERPROFILE ".cache\huggingface\hub"

function Get-SizeGB([string]$path) {
    if (-not (Test-Path $path)) { return 0.0 }
    $sum = (Get-ChildItem -LiteralPath $path -Recurse -File -Force | Measure-Object Length -Sum).Sum
    if (-not $sum) { return 0.0 }
    return [math]::Round($sum / 1GB, 2)
}

function Get-TrainingProcessIds {
    # The run_all process plus everything it started (data-loader workers, the model converter).
    $all  = @(Get-CimInstance Win32_Process)
    $main = @($all | Where-Object { $_.CommandLine -like "*engine.voice.finetune*" -and
                                    $_.Name -like "python*" })
    if ($main.Count -eq 0) { return @() }
    $ids = New-Object System.Collections.Generic.List[int]
    foreach ($p in $main) { $ids.Add([int]$p.ProcessId) }
    $grew = $true
    while ($grew) {
        $grew = $false
        foreach ($p in $all) {
            if ($ids.Contains([int]$p.ParentProcessId) -and -not $ids.Contains([int]$p.ProcessId)) {
                $ids.Add([int]$p.ProcessId); $grew = $true
            }
        }
    }
    return $ids.ToArray()
}

Write-Host "Watching the Whisper training every $Every s. Ctrl+C here stops only this watcher." -ForegroundColor Cyan
Write-Host "Log: $log`n"

$prevCpu  = $null
$prevData = Get-SizeGB $data
$prevHf   = Get-SizeGB $hf
$prevCk   = Get-SizeGB $ckpt
$quiet    = 0

while ($true) {
    $now = Get-Date -Format "HH:mm:ss"
    $ids = Get-TrainingProcessIds

    if ($ids.Count -eq 0) {
        Write-Host "[$now] Training is NOT running." -ForegroundColor Yellow
        if (Test-Path $log) { Write-Host "         last log line: $(Get-Content $log -Tail 1)" }
        Write-Host ""
        $prevCpu = $null
        Start-Sleep -Seconds $Every
        continue
    }

    $cpu = (Get-Process -Id $ids | Measure-Object CPU -Sum).Sum
    $cpuDelta = if ($null -ne $prevCpu) { [math]::Round($cpu - $prevCpu) } else { 0 }
    $prevCpu = $cpu

    $gpuText = "GPU ?"
    $gpuUtil = 0
    $smi = $null
    try { $smi = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null } catch { }
    if ($smi) {
        $g = ($smi | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }
        $gpuUtil = [int]$g[0]
        $gpuText = "GPU $($g[0])%, $($g[1]) / $($g[2]) MB"
    }

    $dataNow = Get-SizeGB $data
    $hfNow   = Get-SizeGB $hf
    $ckNow   = Get-SizeGB $ckpt
    $dD = [math]::Round($dataNow - $prevData, 2)
    $dH = [math]::Round($hfNow - $prevHf, 2)
    $dC = [math]::Round($ckNow - $prevCk, 2)
    $prevData = $dataNow; $prevHf = $hfNow; $prevCk = $ckNow

    $stage = "(no log yet)"; $last = ""; $age = -1
    if (Test-Path $log) {
        $lines = Get-Content $log
        $st = $lines | Where-Object { $_ -match "===|--- config|Run .*: " } | Select-Object -Last 1
        if ($st) { $stage = $st }
        $last = $lines | Select-Object -Last 1
        $age = [int]((Get-Date) - (Get-Item $log).LastWriteTime).TotalSeconds
    }

    $busy = ($cpuDelta -gt 2) -or ($gpuUtil -gt 10) -or ($dD -ne 0) -or ($dH -ne 0) -or ($dC -ne 0)
    if ($busy) { $quiet = 0 } else { $quiet++ }

    if ($busy) {
        Write-Host "[$now] WORKING" -ForegroundColor Green -NoNewline
    } elseif ($quiet -lt 4) {
        Write-Host "[$now] quiet" -ForegroundColor DarkYellow -NoNewline
    } else {
        Write-Host "[$now] NO ACTIVITY for $($quiet * $Every) s: tell Claude before stopping anything" -ForegroundColor Red -NoNewline
    }
    Write-Host "  | CPU +$cpuDelta s in the last $Every s | $gpuText"
    Write-Host "         data\voice $dataNow GB ($('{0:+0.00;-0.00;0}' -f $dD)) | model cache $hfNow GB ($('{0:+0.00;-0.00;0}' -f $dH)) | checkpoints $ckNow GB ($('{0:+0.00;-0.00;0}' -f $dC))"
    Write-Host "         stage: $stage"
    if ($last -and $last -ne $stage) { Write-Host "         last line ($age s ago): $last" }
    Write-Host ""
    Start-Sleep -Seconds $Every
}
