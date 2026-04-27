"""Smoke tests using synthetic images — no real samples required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fraud_detection.image_forensics import ImageForensics
from fraud_detection.signature_verifier import SignatureVerifier
from fraud_detection.utils import detect_kind, risk_label


@pytest.fixture
def clean_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    base = rng.normal(loc=128, scale=16, size=(256, 256, 3)).clip(0, 255).astype("uint8")
    p = tmp_path / "clean.jpg"
    Image.fromarray(base).save(p, format="JPEG", quality=95)
    return p


@pytest.fixture
def tampered_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    base = rng.normal(loc=128, scale=16, size=(256, 256, 3)).clip(0, 255).astype("uint8")
    # Splice a sharp, high-contrast block in the middle (different statistics).
    base[80:160, 80:160] = rng.integers(0, 255, size=(80, 80, 3), dtype="uint8")
    p = tmp_path / "tampered.jpg"
    # Re-save twice with different qualities to leave an ELA fingerprint.
    Image.fromarray(base).save(p, format="JPEG", quality=70)
    with Image.open(p) as im:
        im.save(p, format="JPEG", quality=92)
    return p


def test_image_forensics_runs(clean_image: Path) -> None:
    fx = ImageForensics()
    r = fx.analyze(clean_image)
    assert 0.0 <= r.overall_score <= 1.0
    assert r.risk in ("minimal", "low", "medium", "high")


def test_tampered_scores_higher_than_clean(
    clean_image: Path, tampered_image: Path
) -> None:
    fx = ImageForensics()
    clean = fx.analyze(clean_image)
    tampered = fx.analyze(tampered_image)
    # Not a strict guarantee in adversarial settings, but holds on this synthetic case.
    assert tampered.overall_score >= clean.overall_score


def test_detect_kind() -> None:
    assert detect_kind("foo.jpg") == "image"
    assert detect_kind("foo.pdf") == "document"
    assert detect_kind("foo.mp4") == "video"
    assert detect_kind("foo.bin") == "unknown"
    assert detect_kind("foo.bin", "signature") == "signature"


def test_risk_labels() -> None:
    assert risk_label(0.0) == "minimal"
    assert risk_label(0.3) == "low"
    assert risk_label(0.5) == "medium"
    assert risk_label(0.8) == "high"


def test_signature_intrinsic(tmp_path: Path) -> None:
    # Tiny synthetic "signature": a few black strokes on white.
    arr = np.full((128, 256), 255, dtype="uint8")
    arr[40:50, 40:200] = 0
    arr[60:70, 60:180] = 0
    arr[80:90, 80:160] = 0
    p = tmp_path / "sig.png"
    Image.fromarray(arr).save(p)
    sv = SignatureVerifier()
    v = sv.analyze_single(p)
    assert 0.0 <= v.forgery_score <= 1.0
    assert v.features.contour_count >= 1
