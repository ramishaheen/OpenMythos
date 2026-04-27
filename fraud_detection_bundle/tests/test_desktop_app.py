"""Sanity tests for the desktop wrapper.

We can't actually open the pywebview window in CI (no DISPLAY / no
WebKitGTK), so we validate the helpers and that the module imports
cleanly when ``webview`` and ``uvicorn`` are unavailable.
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_desktop_module():
    if "desktop_app" in sys.modules:
        del sys.modules["desktop_app"]
    return importlib.import_module("desktop_app")


def test_find_free_port_returns_usable_port() -> None:
    mod = _import_desktop_module()
    port = mod._find_free_port()
    assert 1024 < port < 65536
    # The port should still be free immediately after — bind once to confirm.
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_wait_for_server_times_out_when_nothing_running() -> None:
    mod = _import_desktop_module()
    free = mod._find_free_port()
    # Nothing is listening on `free` — the helper should give up cleanly.
    assert mod._wait_for_server(free, timeout_s=0.6) is False


def test_main_aborts_with_clear_message_when_uvicorn_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If uvicorn is not installed, main() must exit non-zero with a
    one-line install hint instead of crashing."""
    mod = _import_desktop_module()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("uvicorn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rc = mod.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "uvicorn is not installed" in err


def test_main_aborts_with_clear_message_when_webview_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _import_desktop_module()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "webview":
            raise ImportError("pywebview not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rc = mod.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "pywebview is not installed" in err
