# MythosBank Fraud Detection - local one-shot runner (Windows PowerShell).
#
# Usage (from the fraud_detection_bundle directory):
#   .\run_local.ps1
#
# Stop: Ctrl+C in this terminal.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-OK    { param($m) Write-Host $m -ForegroundColor Green }
function Write-Step  { param($m) Write-Host $m -ForegroundColor Cyan  }
function Write-Warn  { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Bad   { param($m) Write-Host $m -ForegroundColor Red   }

# ---- Python ----
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
    Write-Bad "Python 3.10 or newer is required. Install from https://www.python.org/downloads/"
    exit 1
}
Write-Host "-> MythosBank Forensic Integrity Console" -ForegroundColor White -BackgroundColor Black
Write-OK ("  Python:  " + (& $py --version))

# ---- venv ----
$venv = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Step "  Creating virtualenv at .venv\ ..."
    & $py -m venv $venv
}
$activate = Join-Path $venv "Scripts\Activate.ps1"
. $activate
python -m pip install --upgrade pip --quiet

# ---- deps ----
$needInstall = $false
try { python -c "import fastapi, uvicorn, anthropic, PIL, numpy" 2>$null } catch { $needInstall = $true }
if ($LASTEXITCODE -ne 0) { $needInstall = $true }
if ($needInstall) {
    Write-Step "  Installing dependencies (~30 s first time) ..."
    pip install --quiet -e .
    pip install --quiet -r web/requirements.txt
    foreach ($extra in @("opencv-python-headless", "pypdfium2", "scikit-image")) {
        pip install --quiet $extra
        if ($LASTEXITCODE -ne 0) { Write-Warn "  (skipped $extra)" }
    }
}
Write-OK "  Deps:    ready"

# ---- port ----
$port = if ($env:PORT) { [int]$env:PORT } else { 8000 }
function Test-PortBusy([int]$p) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $p)
        $c.Close()
        return $true
    } catch { return $false }
}
$attempts = 0
while ((Test-PortBusy $port) -and ($attempts -lt 9)) { $port++; $attempts++ }
$url = "http://127.0.0.1:$port"
Write-OK "  URL:     $url"

# ---- open browser ----
Start-Job -ScriptBlock { Start-Sleep -Seconds 2; Start-Process $using:url } | Out-Null

# ---- boot ----
$env:PYTHONPATH = $PSScriptRoot
Write-OK "  Booting uvicorn - Ctrl+C to stop."
Write-Host ""
& uvicorn web.app:app --host 127.0.0.1 --port $port --workers 1
