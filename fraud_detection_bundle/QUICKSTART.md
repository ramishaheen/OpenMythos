# Quickstart — run it on your machine

One command on macOS / Linux / WSL:

```bash
git clone <this-repo> mythos-fraud
cd mythos-fraud/fraud_detection_bundle
./run_local.sh
```

Windows PowerShell:

```powershell
git clone <this-repo> mythos-fraud
cd mythos-fraud\fraud_detection_bundle
.\run_local.ps1
```

The launcher will:

1. Pick the newest available Python ≥ 3.10.
2. Create a `.venv/` (re-uses it on later runs).
3. Install the package, the web deps, and the optional analyzers
   (`opencv-python-headless`, `pypdfium2`, `scikit-image`). Any optional
   wheel that won't build on your platform is skipped with a warning;
   the rest of the system still runs.
4. Pick `http://127.0.0.1:8000` (or the next free port up to 8009).
5. Start uvicorn and open the URL in your default browser.

Stop the server with **Ctrl + C** in the terminal where you launched it.

## Optional environment variables

| Variable | Effect |
| --- | --- |
| `PORT` | Override the default 8000. |
| `ANTHROPIC_API_KEY` | Enables Claude vision corroboration. Without it the deterministic offline pipeline runs every detector locally. |
| `FRAUD_AUTH_TOKEN` | Adds bearer-token auth to `/api/analyze`. The browser SPA does not send the header, so leave this unset for local use. |

## Verifying the install

```bash
.venv/bin/pytest -q              # 18 tests, ~1.5 s
.venv/bin/python -m fraud_detection.calibration   # rebuilds calibration_report.json
```

## What you should see

After `./run_local.sh`:

```
→ MythosBank Forensic Integrity Console
  Python:  Python 3.12.3
  Deps:    ready
  URL:     http://127.0.0.1:8000
  Booting uvicorn — Ctrl+C to stop.
```

Then your browser opens to a dark, animated console: a drag-drop zone on
the left, a placeholder shield on the right, and a status pill in the
top-right showing whether the agent is online (Claude API) or in
deterministic offline mode.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Python 3.10 or newer is required` | Install from <https://www.python.org/downloads/>. |
| `pip install` complains about `externally-managed-environment` | The launcher creates a venv to avoid this; if it still hits it, delete `.venv/` and re-run. |
| Port 8000 is busy | The launcher will try 8001..8009. Or `PORT=8765 ./run_local.sh`. |
| `opencv-python-headless` skipped | Video analysis disabled; everything else still works. |
| Browser doesn't auto-open | Open `http://127.0.0.1:8000` manually. |

## Skipping the launcher (manual setup)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r web/requirements.txt
pip install opencv-python-headless pypdfium2 scikit-image  # optional
PYTHONPATH=. uvicorn web.app:app --host 127.0.0.1 --port 8000
```
