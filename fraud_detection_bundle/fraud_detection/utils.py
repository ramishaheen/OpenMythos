"""Shared helpers — image encoding, file typing, score normalization."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MediaKind = Literal["image", "document", "video", "signature", "unknown"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DOCUMENT_EXTS = {".pdf"}


@dataclass(frozen=True)
class FileMeta:
    path: Path
    mime: str
    size_bytes: int
    sha256: str


def file_meta(path: str | Path) -> FileMeta:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return FileMeta(p, mime, p.stat().st_size, h.hexdigest())


def detect_kind(path: str | Path, hint: MediaKind | None = None) -> MediaKind:
    if hint and hint != "unknown":
        return hint
    suf = Path(path).suffix.lower()
    if suf in IMAGE_EXTS:
        return "image"
    if suf in VIDEO_EXTS:
        return "video"
    if suf in DOCUMENT_EXTS:
        return "document"
    return "unknown"


def encode_image_b64(path: str | Path, max_side: int = 1568) -> tuple[str, str]:
    """Return (base64, media_type) suitable for Claude vision.

    Downscales the longer edge to ``max_side`` to control token cost while
    keeping forensic detail readable.
    """
    from PIL import Image  # local import to keep top-level light

    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        from io import BytesIO

        buf = BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def risk_label(score: float) -> str:
    """Map a 0-1 fraud score to a human-readable risk band.

    Bands are aligned with the calibrated thresholds in
    ``fraud_detection.image_forensics.DEFAULT_THRESHOLDS``. They are
    deliberately conservative on the synthetic baseline; production
    deployments should re-calibrate against labelled data.
    """
    if score >= 0.4434:
        return "high"
    if score >= 0.2412:
        return "medium"
    if score >= 0.1912:
        return "low"
    return "minimal"
