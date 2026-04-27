# PyInstaller spec — bundle the desktop app into a single executable.
#
# Build:
#   pip install pyinstaller
#   pyinstaller desktop_app.spec
#
# Output:
#   dist/MythosBankFraudDetection (Linux/macOS) or .exe (Windows)
#
# The output bundles the FastAPI server, the SPA static assets, all
# detectors, and pywebview. It does NOT bundle the optional opencv /
# pdfium / scikit-image / openai wheels by default — add them to
# `hiddenimports` below if you need video or PDF analysis or non-Anthropic
# providers in the binary.

# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()
IS_MACOS = sys.platform == "darwin"

a = Analysis(
    [str(ROOT / "desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "web" / "static"), "web/static"),
        (str(ROOT / "fraud_detection" / "calibration" / "calibration_report.json"),
         "fraud_detection/calibration"),
    ],
    hiddenimports=[
        "fraud_detection",
        "fraud_detection.agent",
        "fraud_detection.cli",
        "fraud_detection.image_forensics",
        "fraud_detection.signature_verifier",
        "fraud_detection.video_analyzer",
        "fraud_detection.document_analyzer",
        "fraud_detection.detectors",
        "fraud_detection.detectors.base",
        "fraud_detection.detectors.ela",
        "fraud_detection.detectors.jpeg_qtable",
        "fraud_detection.detectors.benford",
        "fraud_detection.detectors.cfa",
        "fraud_detection.detectors.copy_move",
        "fraud_detection.detectors.lighting",
        "fraud_detection.detectors.noise_residue",
        "fraud_detection.providers",
        "fraud_detection.providers.base",
        "fraud_detection.providers.factory",
        "fraud_detection.providers.anthropic_provider",
        "fraud_detection.providers.openai_provider",
        "fraud_detection.providers.deepseek_provider",
        "fraud_detection.providers.offline_provider",
        "web",
        "web.app",
        "anthropic",
        "openai",
        "fastapi",
        "uvicorn",
        "uvicorn.lifespan.on",
        "uvicorn.protocols.http.auto",
        "uvicorn.loops.auto",
        "starlette",
        "pydantic",
        "webview",
        "PIL",
        "PIL._imagingmath",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep the binary lean — drop heavy optional analyzers. Re-enable
        # in `hiddenimports` if you need them.
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
