# Codespaces config

When this branch is opened in GitHub Codespaces, the container:

1. Builds a Python 3.12 environment.
2. Runs `setup.sh` — creates `.venv/`, installs the package + web deps + the
   optional analyzers (opencv-python-headless, pypdfium2, scikit-image).
3. Runs `start.sh` on attach — boots `uvicorn web.app:app` on port 8000.
4. Codespaces auto-forwards port 8000 to a public HTTPS URL (visible in the
   "Ports" tab) and opens the browser tab automatically.

The forwarded URL looks like `https://<codespace>-8000.app.github.dev`.

## To launch

From the repo on github.com:

  Code (green button) → Codespaces → Create codespace on
  claude/document-fraud-detection-Xixkd

…then wait ~30 s for setup to finish. The browser tab opens itself. The
auto-forwarded port has `visibility: public` so you can also share the
URL with anyone (note: anyone with the URL can use the service — combine
with `FRAUD_AUTH_TOKEN` if that's a concern).
