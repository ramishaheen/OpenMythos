# MythosBank Fraud Detection - build a native .exe for Windows and copy
# it onto the user's Desktop.
#
# Usage:  .\build_and_export.ps1
#
# Output: ~\Desktop\MythosBankFraudDetection.exe  (double-click)
#
# Prereq: Edge WebView2 runtime (built into Win 10 1803+ and Win 11; if
#         missing, install from https://aka.ms/webview2 ).

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

$desktop = [System.Environment]::GetFolderPath("Desktop")

Write-Host "-> MythosBank Fraud Detection - desktop build" -ForegroundColor White
Write-OK ("  Platform: Windows ($([System.Environment]::OSVersion.Platform))")
Write-OK ("  Python:   " + (& $py --version))
Write-OK ("  Desktop:  $desktop")

# --- venv ---
$venv = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Step "  Creating virtualenv at .venv\ ..."
    & $py -m venv $venv
}
. (Join-Path $venv "Scripts\Activate.ps1")
python -m pip install --upgrade pip --quiet

# --- deps ---
$need = $false
try { python -c "import fastapi, uvicorn, anthropic, PIL, numpy, webview, PyInstaller" 2>$null } catch { $need = $true }
if ($LASTEXITCODE -ne 0) { $need = $true }
if ($need) {
    Write-Step "  Installing build dependencies (~1 min first time) ..."
    pip install --quiet -e .
    pip install --quiet -r web/requirements.txt
    pip install --quiet pywebview pyinstaller
    foreach ($extra in @("opencv-python-headless", "pypdfium2", "scikit-image")) {
        pip install --quiet $extra
        if ($LASTEXITCODE -ne 0) { Write-Warn "  (skipped $extra)" }
    }
}
Write-OK "  Deps:     ready"

# --- clean prior build ---
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist"  }

# --- build ---
Write-Step "  Running PyInstaller (~30-60s) ..."
$env:PYTHONPATH = $PSScriptRoot
$logFile = "$env:TEMP\mb_pyi_build.log"
& pyinstaller --noconfirm --log-level WARN desktop_app.spec *>$logFile
if ($LASTEXITCODE -ne 0) {
    Write-Bad "PyInstaller failed. Last 30 log lines:"
    Get-Content $logFile -Tail 30
    exit 2
}
Write-OK "  Build:    OK"

# --- copy to Desktop ---
$src = Join-Path $PSScriptRoot "dist\MythosBankFraudDetection.exe"
if (-not (Test-Path $src)) {
    Write-Bad "Build succeeded but $src not found. dist contents:"
    Get-ChildItem dist
    exit 3
}
$target = Join-Path $desktop "MythosBankFraudDetection.exe"
Copy-Item -Force $src $target
$size = (Get-Item $target).Length
$sizeMB = [math]::Round($size / 1MB, 1)
Write-OK "  Exported: $target"
Write-OK "  Size:     $sizeMB MB"
Write-Host ""
Write-Host "Double-click MythosBankFraudDetection.exe on your Desktop to launch." -ForegroundColor White
Write-Host "Close any running browser-mode instance first." -ForegroundColor Yellow
