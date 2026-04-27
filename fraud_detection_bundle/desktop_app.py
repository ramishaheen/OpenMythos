"""MythosBank Fraud Detection — desktop wrapper.

Runs the FastAPI service in a background thread and opens a native
desktop window pointing at it. The window uses the OS's system webview
(Edge WebView2 on Windows, WebKit on macOS, WebKitGTK on Linux) — no
Electron, no separate browser process.

Usage:
    python desktop_app.py
or
    ./run_desktop.sh        # macOS / Linux / WSL
    ./run_desktop.ps1       # Windows PowerShell

Environment knobs:
    MYTHOS_PORT             override the loopback port (default: random free)
    MYTHOS_NO_AUTOCLOSE     keep the uvicorn server alive after the window
                            closes (default: shut down with the window)
    ANTHROPIC_API_KEY,      passed straight through to the agent
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY        — choose any one to enable LLM reasoning;
                              omit them all to run in deterministic
                              offline mode.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Force the FastAPI app to bind to a fresh per-process port — no clashes
# with the dev server.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout_s: float = 8.0) -> bool:
    """Poll /api/health until the server answers (or timeout)."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ConnectionError):
            pass
        time.sleep(0.15)
    return False


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run `pip install -r web/requirements.txt`.",
            file=sys.stderr,
        )
        return 1
    try:
        import webview  # type: ignore
    except ImportError:
        print(
            "pywebview is not installed. Run `pip install pywebview` or "
            "`pip install -e .[desktop]`.",
            file=sys.stderr,
        )
        return 1

    # Import the FastAPI app lazily so missing deps fail with a clean message
    # before importing pywebview's GUI bindings.
    from web.app import app

    port = int(os.environ.get("MYTHOS_PORT", "0") or 0) or _find_free_port()

    # Subclass uvicorn.Server so the main thread (where pywebview lives)
    # owns the signal handlers — pywebview needs SIGINT for clean shutdown.
    class _NoSignalsServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            return

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level=os.environ.get("FRAUD_LOG_LEVEL", "warning").lower(),
        access_log=False,
        workers=1,
        lifespan="on",
    )
    server = _NoSignalsServer(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    if not _wait_for_server(port, timeout_s=10):
        print(
            f"Server failed to start on 127.0.0.1:{port}. "
            "Check that port is available and try again.",
            file=sys.stderr,
        )
        server.should_exit = True
        return 2

    url = f"http://127.0.0.1:{port}"
    print(f"MythosBank Forensic Console ready at {url}")
    print("Close the window or press Ctrl+C to quit.")

    # Create the native window. The min size keeps the two-panel layout
    # readable; background colour matches the CSS so there's no flash on
    # first paint.
    webview.create_window(
        title="MythosBank · Forensic Integrity Console",
        url=url,
        width=1280,
        height=820,
        min_size=(900, 600),
        background_color="#08090D",
        text_select=True,
        confirm_close=False,
    )

    try:
        # webview.start blocks until the window is closed.
        webview.start(debug=bool(os.environ.get("MYTHOS_DEBUG")))
    finally:
        if not os.environ.get("MYTHOS_NO_AUTOCLOSE"):
            server.should_exit = True
            thread.join(timeout=3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
