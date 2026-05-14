# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — bundle the desktop app into a single executable.
#
# Build:
#   pip install pyinstaller pywebview
#   pyinstaller desktop_app.spec
#
# Output:
#   Linux:   dist/MythosBankFraudDetection
#   macOS:   dist/MythosBankFraudDetection.app  (double-clickable .app bundle)
#   Windows: dist/MythosBankFraudDetection.exe
#
# This spec uses PyInstaller's collect_submodules / collect_data_files
# helpers to be robust against FastAPI / uvicorn / pydantic / anthropic /
# openai missing-import gotchas. If you add a new provider or a new web
# dependency, edit the lists at the top of `Analysis`.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve()
IS_MACOS = sys.platform == "darwin"

# --- robust submodule collection -------------------------------------------
# These libraries have lazy / dynamic imports PyInstaller's static analysis
# routinely misses. Collecting all submodules avoids "ModuleNotFoundError"
# at runtime inside the frozen binary.
hidden = []
for pkg in (
    "fraud_detection",
    "web",
    "uvicorn",
    "starlette",
    "fastapi",
    "pydantic",
    "pydantic_core",
    "anyio",
    "anthropic",
    "openai",
    "webview",
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        pass

# Explicit imports that are sometimes only referenced by string.
hidden += [
    "fraud_detection.detectors.base",
    "fraud_detection.detectors.ela",
    "fraud_detection.detectors.jpeg_qtable",
    "fraud_detection.detectors.benford",
    "fraud_detection.detectors.cfa",
    "fraud_detection.detectors.copy_move",
    "fraud_detection.detectors.lighting",
    "fraud_detection.detectors.noise_residue",
    "fraud_detection.providers.anthropic_provider",
    "fraud_detection.providers.openai_provider",
    "fraud_detection.providers.deepseek_provider",
    "fraud_detection.providers.offline_provider",
    "fraud_detection.providers.factory",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.loops.auto",
    "uvicorn.loops.uvloop",
    "uvicorn.loops.asyncio",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.logging",
    "anyio._backends._asyncio",
    "multipart",          # python-multipart for FastAPI file uploads
    "email.feedparser",
    "email.parser",
    # Pillow may JIT-load these at runtime.
    "PIL",
    "PIL._imagingmath",
    "PIL._imagingft",
    "PIL.JpegImagePlugin",
    "PIL.PngImagePlugin",
    "PIL.WebPImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.BmpImagePlugin",
]

# --- bundled data files ----------------------------------------------------
datas = [
    (str(ROOT / "web" / "static"), "web/static"),
    (
        str(ROOT / "fraud_detection" / "calibration" / "calibration_report.json"),
        "fraud_detection/calibration",
    ),
]
# Some libraries ship JSON / cert / model files alongside their .py code.
for pkg in ("anthropic", "openai", "anyio"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass


a = Analysis(
    [str(ROOT / "desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hidden)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep the binary lean — drop heavy optional analyzers. Re-enable
        # by removing the relevant entry from `excludes` here and adding
        # it to the `hidden` list above.
        "cv2",
        "skimage",
        "pypdfium2",
        "pytesseract",
        "matplotlib",
        "scipy",
        "torch",
        "tensorflow",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MythosBankFraudDetection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # native window only — no terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# macOS: also wrap into a double-clickable .app bundle.
if IS_MACOS:
    app = BUNDLE(
        exe,
        name="MythosBankFraudDetection.app",
        icon=None,
        bundle_identifier="com.mythosbank.fraud-detection",
        info_plist={
            "CFBundleDisplayName": "MythosBank Fraud Detection",
            "CFBundleShortVersionString": "0.3.0",
            "CFBundleVersion": "0.3.0",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.business",
            "NSHumanReadableCopyright": "Forensic Integrity Console",
        },
    )
