"""Smoke tests for individual detectors. Each must:
   - run on a synthetic JPEG without raising
   - return score in [0, 1]
   - return confidence in [0, 1]
   - report a non-empty algorithm version string
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fraud_detection.detectors import (
    BenfordDCTDetector,
    CFAInconsistencyDetector,
    ErrorLevelAnalysisDetector,
    JPEGQuantizationDetector,
    LightingConsistencyDetector,
    NoiseResidueDetector,
    PerceptualCopyMoveDetector,
)
from fraud_detection.detectors.base import safe_run


@pytest.fixture
def jpeg_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = rng.normal(128, 30, size=(192, 192, 3)).clip(0, 255).astype("uint8")
    p = tmp_path / "noise.jpg"
    Image.fromarray(arr).save(p, "JPEG", quality=90)
    return p


@pytest.fixture
def png_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = rng.normal(128, 30, size=(192, 192, 3)).clip(0, 255).astype("uint8")
    p = tmp_path / "noise.png"
    Image.fromarray(arr).save(p, "PNG")
    return p


@pytest.mark.parametrize(
    "factory",
    [
        ErrorLevelAnalysisDetector,
        JPEGQuantizationDetector,
        BenfordDCTDetector,
        CFAInconsistencyDetector,
        PerceptualCopyMoveDetector,
        NoiseResidueDetector,
        LightingConsistencyDetector,
    ],
)
def test_detector_runs(factory, jpeg_image: Path) -> None:
    detector = factory()
    result = safe_run(detector, jpeg_image)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.name
    assert result.version
    assert result.error is None


def test_jpeg_qtable_skips_png(png_image: Path) -> None:
    """Q-table detector should bail gracefully on non-JPEG input."""
    result = safe_run(JPEGQuantizationDetector(), png_image)
    assert result.confidence == 0.0
    assert result.error is None


def test_safe_run_swallows_exceptions(tmp_path: Path) -> None:
    """A detector that crashes should produce a structured zero-evidence result."""

    class Boom:
        name = "boom"
        version = "0.0"

        def run(self, _path):  # noqa: ARG002
            raise RuntimeError("intentional")

    result = safe_run(Boom(), tmp_path / "missing.jpg")
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.error and "intentional" in result.error
