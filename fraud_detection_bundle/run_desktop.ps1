# MythosBank Fraud Detection - desktop launcher (Windows PowerShell).
#
# Sets up a virtualenv, installs the package + web deps + pywebview,
# then runs desktop_app.py - which boots uvicorn locally and opens a
# native desktop window pointing at it.
#
# Windows note: pywebview uses Edge WebView2 by default. On Windows 10
# 1803+ and all Windows 11, this is preinstalled with Edge. If it's
# missing, install the runtime from https://aka.ms/webview2 .

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-OK   { param($m) Write-Host $m -ForegroundColor Green }
function Write-Step { param($m) Write-Host $m -ForegroundColor Cyan }
function Write-Warn { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Bad  { param($m) Write-Host $m -ForegroundColor Red }

$py = $null
foreach ($c in @("python", "python3", "py")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
        try {
            $v = [int](& $c -c "import sys;print(sys.version_info[0]*100+sys.version_info[1])")
            if ($v -ge 310) { $py = $c; break }
        } catch {}
    }
}
if (-not $py) {
    Write-Bad "Need Python 3.10+. Install from https://www.python.org/downloads/"
    exit 1
}
Write-Host "-> MythosBank Forensic Integrity Console (desktop)" -ForegroundColor White
Write-OK ("  Python:  " + (& $py --version))

$venv = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Step "  Creating virtualenv at .venv\ ..."
    & $py -m venv $venv
}
. (Join-Path $venv "Scripts\Activate.ps1")
python -m pip install --upgrade pip --quiet

$needed = $false
try { python -c "import fastapi, uvicorn, anthropic, PIL, numpy, webview" 2>$null } catch { $needed = $true }
if ($LASTEXITCODE -ne 0) { $needed = $true }
if ($needed) {
    Write-Step "  Installing dependencies (~45 s first time) ..."
    pip install --quiet -e .
    pip install --quiet -r web/requirements.txt
    pip install --quiet pywebview
    foreach ($extra in @("opencv-python-headless", "pypdfium2", "scikit-image")) {
        pip install --quiet $extra
        if ($LASTEXITCODE -ne 0) { Write-Warn "  (skipped $extra)" }
    }
}
Write-OK "  Deps:    ready"

$env:PYTHONPATH = $PSScriptRoot
Write-OK "  Launching desktop window ..."
Write-Host ""
& python desktop_app.py
