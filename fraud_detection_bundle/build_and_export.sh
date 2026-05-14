#!/usr/bin/env bash
# MythosBank Fraud Detection — build a native binary for THIS machine
# and copy it onto the user's Desktop folder.
#
# Usage (macOS / Linux / WSL):
#   ./build_and_export.sh
#
# Output:
#   macOS  → ~/Desktop/MythosBankFraudDetection.app          (double-click)
#   Linux  → ~/Desktop/MythosBankFraudDetection              (executable)
#   WSL    → ~/Desktop/MythosBankFraudDetection (Linux ELF)
#
# What it does:
#   1. Reuses .venv/ if present, creates one otherwise.
#   2. Installs the package + web deps + pywebview + pyinstaller.
#   3. Runs `pyinstaller desktop_app.spec`.
#   4. Copies the output onto your Desktop and prints the launch command.
#
# Notes:
#   * Cross-compilation is NOT possible. This script builds for the OS
#     it's running on. To target a different OS, run the script there.
#   * Linux: the resulting ELF expects WebKitGTK at runtime. If launching
#     it errors out on `gi.repository`, install:
#         sudo apt install gir1.2-webkit2-4.0
#   * macOS: WebKit is built into the system. No extra runtime needed.

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

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) PRETTY="macOS";   DESKTOP="$HOME/Desktop" ;;
  Linux)  PRETTY="Linux";   DESKTOP="$HOME/Desktop" ;;
  *)      PRETTY="$OS";     DESKTOP="$HOME/Desktop" ;;
esac

bold "→ MythosBank Fraud Detection — desktop build"
green "  Platform: $PRETTY ($ARCH)"
green "  Python:   $($PY --version)"

# --- venv ---
VENV=".venv"
if [ ! -d "$VENV" ]; then
  blue "  Creating virtualenv at $VENV/ …"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

# --- deps ---
need=0
python -c 'import fastapi, uvicorn, anthropic, PIL, numpy, webview, PyInstaller' 2>/dev/null || need=1
if [ "$need" -eq 1 ]; then
  blue "  Installing build dependencies (~1 min the first time) …"
  pip install --quiet -e .
  pip install --quiet -r web/requirements.txt
  pip install --quiet pywebview pyinstaller
  pip install --quiet opencv-python-headless 2>/dev/null || yellow "  (skipped opencv)"
  pip install --quiet pypdfium2          2>/dev/null || yellow "  (skipped pypdfium2)"
  pip install --quiet scikit-image       2>/dev/null || yellow "  (skipped scikit-image)"
fi
green "  Deps:     ready"

# --- clean previous build ---
if [ -d build ]; then rm -rf build; fi
if [ -d dist ];  then rm -rf dist;  fi

# --- build ---
blue "  Running PyInstaller (≈ 30–60 s) …"
PYTHONPATH="$PWD" pyinstaller --noconfirm --log-level WARN desktop_app.spec >/tmp/mb_pyi_build.log 2>&1 \
  || { red "PyInstaller failed. Last 30 log lines:"; tail -30 /tmp/mb_pyi_build.log; exit 2; }
green "  Build:    OK"

# --- locate the artifact + copy to Desktop ---
mkdir -p "$DESKTOP"

if [ -d "dist/MythosBankFraudDetection.app" ]; then
  # macOS .app bundle.
  rm -rf "$DESKTOP/MythosBankFraudDetection.app"
  cp -R "dist/MythosBankFraudDetection.app" "$DESKTOP/"
  TARGET="$DESKTOP/MythosBankFraudDetection.app"
  # Strip macOS quarantine attribute so Gatekeeper doesn't refuse
  # the unsigned .app the first time the user double-clicks.
  if command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$TARGET" 2>/dev/null || true
  fi
  green "  Exported: $TARGET"
  echo
  bold  "  Double-click MythosBankFraudDetection.app on your Desktop to launch."
  yellow "  (macOS only) If Gatekeeper blocks it, right-click the .app and pick"
  echo   "  'Open' instead of double-clicking the first time."
elif [ -f "dist/MythosBankFraudDetection" ]; then
  # Linux ELF binary.
  cp "dist/MythosBankFraudDetection" "$DESKTOP/MythosBankFraudDetection"
  chmod +x "$DESKTOP/MythosBankFraudDetection"
  TARGET="$DESKTOP/MythosBankFraudDetection"
  green "  Exported: $TARGET"
  echo
  bold  "  Launch with:"
  echo  "    $TARGET"
  yellow "  (Linux only) If launch errors with a 'gi.repository' message, run:"
  echo   "    sudo apt install gir1.2-webkit2-4.0"
else
  red "Build succeeded but no expected output found in dist/. Contents:"
  ls -la dist/ || true
  exit 3
fi

SIZE=$(du -sh "$TARGET" 2>/dev/null | awk '{print $1}')
green "  Size:     ${SIZE:-?}"
echo
bold "All set. Close any running browser-mode instance before launching the desktop app."
