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

## Run as a desktop application

Wraps the same UI in a native OS window using your system's webview
(Edge WebView2 on Windows, WebKit on macOS, WebKitGTK on Linux). No
browser tab, no port to type — looks and feels like a desktop app.

```bash
./run_desktop.sh        # macOS / Linux / WSL
./run_desktop.ps1       # Windows PowerShell
```

The launcher:

1. Creates `.venv/` if missing.
2. Installs everything `run_local.sh` does PLUS `pywebview`.
3. Starts uvicorn on a random free loopback port.
4. Opens a 1280 × 820 native window pointing at it.
5. Shuts uvicorn down cleanly when you close the window.

### Build a redistributable binary (PyInstaller)

```bash
source .venv/bin/activate
pip install pyinstaller
pyinstaller desktop_app.spec
# → dist/MythosBankFraudDetection (Linux/macOS) or .exe (Windows)
```

The spec excludes the heaviest optional analyzers (opencv, skimage,
pypdfium2) to keep the binary lean. Re-enable them by moving the entry
from `excludes=` to `hiddenimports=` in `desktop_app.spec`.

### Linux note

pywebview needs **WebKitGTK** at runtime. If the desktop app fails to
launch with a `gi.repository.Gtk` import error:

```bash
# Debian / Ubuntu
sudo apt install python3-gi gir1.2-webkit2-4.0
# Fedora
sudo dnf install python3-gobject webkit2gtk3
# Arch
sudo pacman -S python-gobject webkit2gtk
```

### Windows note

Edge WebView2 ships with Edge on Windows 10 1803+ and Windows 11. If
the window won't open, install the runtime from <https://aka.ms/webview2>.

## Skipping the launcher (manual setup)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r web/requirements.txt
pip install opencv-python-headless pypdfium2 scikit-image  # optional
PYTHONPATH=. uvicorn web.app:app --host 127.0.0.1 --port 8000
```
