#!/usr/bin/env bash
# MythosBank Fraud Detection — desktop launcher (macOS / Linux / WSL).
#
# Sets up a virtualenv, installs the package + web deps + pywebview,
# then runs desktop_app.py — which boots uvicorn locally and opens a
# native desktop window pointing at it.
#
# Linux note: pywebview needs the system WebKitGTK runtime. If the app
# refuses to launch with a "No module named gi" error, install:
#   sudo apt install python3-gi gir1.2-webkit2-4.0
# (Debian/Ubuntu) or the equivalent on your distro.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "\033[0;32m%s\033[0m\n" "$*"; }
blue()   { printf "\033[0;34m%s\033[0m\n" "$*"; }
yellow() { printf "\033[0;33m%s\033[0m\n" "$*"; }
red()    { printf "\033[0;31m%s\033[0m\n" "$*" >&2; }

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null || echo 0)
    if [ "${v:-0}" -ge 310 ]; then PY="$c"; break; fi
  fi
done
[ -z "$PY" ] && { red "Need Python 3.10+. Install from https://www.python.org/"; exit 1; }
bold "→ MythosBank Forensic Integrity Console (desktop)"
green "  Python:  $($PY --version)"

VENV=".venv"
if [ ! -d "$VENV" ]; then
  blue "  Creating virtualenv at $VENV/ …"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

needed=0
python -c 'import fastapi, uvicorn, anthropic, PIL, numpy, webview' 2>/dev/null || needed=1
if [ "$needed" -eq 1 ]; then
  blue "  Installing dependencies (~45 s the first time) …"
  pip install --quiet -e .
  pip install --quiet -r web/requirements.txt
  pip install --quiet pywebview
  pip install --quiet opencv-python-headless 2>/dev/null || yellow "  (skipped opencv)"
  pip install --quiet pypdfium2          2>/dev/null || yellow "  (skipped pypdfium2)"
  pip install --quiet scikit-image       2>/dev/null || yellow "  (skipped scikit-image)"
fi
green "  Deps:    ready"

green "  Launching desktop window …"
echo
export PYTHONPATH="$PWD"
exec python desktop_app.py
