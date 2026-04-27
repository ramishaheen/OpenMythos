#!/usr/bin/env bash
# postCreateCommand — runs once when the Codespace is built.
set -euo pipefail

echo "→ Setting up MythosBank Fraud Detection venv …"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip --quiet
pip install --quiet -e .
pip install --quiet -r web/requirements.txt
# Optional analyzers — best-effort, don't fail the build.
pip install --quiet opencv-python-headless || true
pip install --quiet pypdfium2              || true
pip install --quiet scikit-image           || true

echo "✓ Setup complete. Run ./.devcontainer/start.sh to launch the service."
