# Builds the zip people download: dist\EYEable-<version>.zip, made by git from the committed files only
# (no models, no .venv, no config.local.json, no TORGO data, no private team notes: .gitignore keeps them out).
#   .\scripts\make_release.ps1 -Version 1.0
# Then on GitHub: Releases -> Draft a new release -> tag v1.0 -> attach the zip -> Publish.
# (GitHub also attaches its own "Source code (zip)" of the tag, which works the same way.)
param([Parameter(Mandatory = $true)] [string]$Version)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (git status --porcelain) { throw "Commit or stash your changes first: the zip is made from the last commit." }
New-Item -ItemType Directory -Force dist | Out-Null
$out = "dist\EYEable-$Version.zip"
git archive --format=zip --prefix="EYEable/" -o $out HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
$mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
Write-Host "Wrote $out ($mb MB) from $(git log -1 --format='%h %s')" -ForegroundColor Green
Write-Host "People unzip it and double-click 'Install EYEable.bat'."
