#!/usr/bin/env bash
# postAttachCommand — runs each time you open / re-open the Codespace.
set -euo pipefail

cd "$(dirname "$0")/.."

# Make sure setup ran (idempotent — Codespaces sometimes re-uses a container
# without running postCreateCommand).
if [ ! -d .venv ]; then
  bash .devcontainer/setup.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Don't double-launch if uvicorn is already running.
if curl -sf -m 1 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "✓ Service already running on port 8000."
  exit 0
fi

echo "→ Booting MythosBank Fraud Detection on port 8000 …"
echo "  Once you see 'Uvicorn running', open the auto-forwarded URL"
echo "  shown by Codespaces in the Ports tab (https://*.app.github.dev)."
echo

# Run in foreground so the integrated terminal shows the logs.
exec uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 1
