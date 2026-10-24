# Make synthetic speech clips for testing the voice chain without a mic.
# Uses Windows' built-in, offline text-to-speech (System.Speech). Synthetic voices only,
# never a real person, and the clips go only under data\voice\synth\ (git-ignored).
#
# Run from the repo root in Windows PowerShell 5.1:
#   .\scripts\make_test_speech.ps1            # every English voice x normal / slow / quiet
#   .\scripts\make_test_speech.ps1 -Force     # remake clips that already exist
# Then: .\.venv\Scripts\python.exe -m tests.add_noise   (adds the noisy copies)
#
# Reads tests\speech_phrases.csv (id, text, category, expected, mode).
# Writes data\voice\synth\<id>__<voice>__<variant>.wav (16 kHz, mono, 16-bit) and manifest.csv.
param([switch]$Force)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$root = Split-Path -Parent $PSScriptRoot
$phrases = Import-Csv -Path (Join-Path $root "tests\speech_phrases.csv") -Encoding UTF8
$outDir = Join-Path $root "data\voice\synth"
New-Item -ItemType Directory -Force $outDir | Out-Null

# name -> (rate -10..10, volume 0..100). Slow is closer to dysarthric pacing; quiet is a soft voice.
$variants = [ordered]@{ normal = @(0, 100); slow = @(-4, 100); quiet = @(0, 25) }

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like "en-*" } |
    ForEach-Object { $_.VoiceInfo.Name }
if (-not $voices) { throw "No English text-to-speech voice is installed (Settings > Time & language > Speech)." }
Write-Host ("Voices: " + ($voices -join ", "))
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$rows = New-Object System.Collections.Generic.List[object]
$made = 0
foreach ($voice in $voices) {
    $synth.SelectVoice($voice)
    $short = ($voice -replace "^Microsoft ", "" -replace " Desktop$", "" -replace "\W", "").ToLower()
    foreach ($v in $variants.Keys) {
        $synth.Rate = $variants[$v][0]
        $synth.Volume = $variants[$v][1]
        foreach ($p in $phrases) {
            $file = "{0}__{1}__{2}.wav" -f $p.id, $short, $v
            $path = Join-Path $outDir $file
            if ($Force -or -not (Test-Path $path)) {
                $synth.SetOutputToWaveFile($path, $format)
                $synth.Speak($p.text)
                $synth.SetOutputToNull()
                $made++
            }
            $rows.Add([pscustomobject]@{ id = $p.id; text = $p.text; category = $p.category;
                expected = $p.expected; mode = $p.mode; voice = $short; variant = $v;
                rate = $variants[$v][0]; volume = $variants[$v][1]; file = $file })
        }
        Write-Host ("  {0} {1}: {2} clips" -f $short, $v, $phrases.Count)
    }
}
$synth.Dispose()

# UTF-8 without a BOM, so Python reads it with a plain encoding="utf-8"
$csv = $rows | ConvertTo-Csv -NoTypeInformation
[System.IO.File]::WriteAllLines((Join-Path $outDir "manifest.csv"), $csv, (New-Object System.Text.UTF8Encoding($false)))
$size = (Get-ChildItem $outDir -Filter *.wav | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Made {0} new clips; manifest has {1} rows; {2:N1} MB of WAVs in {3}" -f $made, $rows.Count, $size, $outDir)
