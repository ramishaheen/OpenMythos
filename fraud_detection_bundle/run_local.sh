#!/usr/bin/env bash
# MythosBank Fraud Detection — local one-shot runner.
#
# Usage:  ./run_local.sh
#
# What it does on your machine:
#   1. Picks the newest available Python >= 3.10 on $PATH.
#   2. Creates / reuses a virtualenv at .venv/.
#   3. Installs the package + web deps if anything is missing.
#   4. Picks port 8000 (or 8001..8009 if busy).
#   5. Boots uvicorn at 127.0.0.1:<port>.
#   6. Opens that URL in your default browser.
#
# Environment knobs:
#   PORT=8000          override the port
#   FRAUD_AUTH_TOKEN=  enable bearer-auth (off by default for local use)
#   ANTHROPIC_API_KEY  enable Claude vision corroboration
#
# Stop:  Ctrl+C in this terminal.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "\033[0;32m%s\033[0m\n" "$*"; }
blue()   { printf "\033[0;34m%s\033[0m\n" "$*"; }
yellow() { printf "\033[0;33m%s\033[0m\n" "$*"; }
red()    { printf "\033[0;31m%s\033[0m\n" "$*" >&2; }

# ----------------------------------------------------------------- python
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null || echo 0)
    if [ "${v:-0}" -ge 310 ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  red "Python 3.10 or newer is required. Install from https://www.python.org/downloads/"
  exit 1
fi
bold "→ MythosBank Forensic Integrity Console"
green "  Python:  $($PY --version)"

# ----------------------------------------------------------------- venv
VENV=".venv"
if [ ! -d "$VENV" ]; then
  blue "  Creating virtualenv at $VENV/ …"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

# ----------------------------------------------------------------- deps
need_install=0
python -c 'import fastapi, uvicorn, anthropic, PIL, numpy' 2>/dev/null || need_install=1
if [ "$need_install" -eq 1 ]; then
  blue "  Installing dependencies (≈ 30 s the first time) …"
  pip install --quiet -e .
  pip install --quiet -r web/requirements.txt
  # Optional analyzers (video / PDF / OCR / SSIM). Best-effort: don't fail
  # the launcher if any single wheel doesn't have a pre-built binary on
  # this platform.
  pip install --quiet opencv-python-headless 2>/dev/null || yellow "  (skipped opencv — video analysis disabled)"
  pip install --quiet pypdfium2          2>/dev/null || yellow "  (skipped pypdfium2 — PDF rendering disabled)"
  pip install --quiet scikit-image       2>/dev/null || yellow "  (skipped scikit-image — SSIM in signatures disabled)"
fi
green "  Deps:    ready"

# ----------------------------------------------------------------- port
PORT="${PORT:-8000}"
is_busy() {
  local p=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$p" >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -tln | awk '{print $4}' | grep -qE "[:.]$p\$"
  else
    python -c "import socket,sys;s=socket.socket();s.settimeout(0.3);
try:s.connect(('127.0.0.1',$p));sys.exit(0)
except:sys.exit(1)" 2>/dev/null
  fi
}
attempt=0
while is_busy "$PORT" && [ "$attempt" -lt 9 ]; do
  PORT=$((PORT + 1))
  attempt=$((attempt + 1))
done
URL="http://127.0.0.1:$PORT"
green "  URL:     $URL"

# ----------------------------------------------------------------- open browser (best-effort)
(
  sleep 2
  case "$(uname -s)" in
    Darwin)              open "$URL" 2>/dev/null || true ;;
    Linux)               xdg-open "$URL" 2>/dev/null || true ;;
    MINGW*|CYGWIN*|MSYS*) start "$URL" 2>/dev/null || true ;;
  esac
) &

# ----------------------------------------------------------------- boot
green "  Booting uvicorn — Ctrl+C to stop."
echo
export PYTHONPATH="$PWD"
exec uvicorn web.app:app --host 127.0.0.1 --port "$PORT" --workers 1
