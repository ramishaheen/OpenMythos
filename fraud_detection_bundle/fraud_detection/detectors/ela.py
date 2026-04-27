"""Error Level Analysis — calibrated against pristine-vs-tampered baselines.

The classical ELA was over-amplified (×8) which produced false positives on
naturally noisy images. This version reports both a raw residual mean and a
*localization* score: how spatially concentrated the residual is. Splices
produce localized hotspots; uniform residuals just indicate a noisy image
and should NOT trigger a high score.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops

from fraud_detection.detectors.base import DetectorResult


class ErrorLevelAnalysisDetector:
    name = "ela"
    version = "2.1"

    def __init__(self, quality: int = 90, hotspot_threshold_sigma: float = 3.0) -> None:
        self.quality = quality
        self.hotspot_threshold_sigma = hotspot_threshold_sigma

    def run(self, image_path: str | Path) -> DetectorResult:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            with Image.open(buf) as recompressed:
                ela = ImageChops.difference(im, recompressed)
            residual = np.asarray(ela, dtype=np.float32).max(axis=2) / 255.0

        mean = float(residual.mean())
        std = float(residual.std())
        # Localization: ratio of high-residual pixels to total. A genuinely
        # tampered region appears as a concentrated cluster of outliers, not a
        # diffuse field.
        threshold = mean + self.hotspot_threshold_sigma * std
        hotspot_frac = float((residual > threshold).mean()) if std > 0 else 0.0

        # Two independent signals fused.
        intensity_score = _saturating(mean / 0.06)  # 0.06 ≈ pristine 80th pct.
        local_score = _saturating(hotspot_frac / 0.04)
        score = 0.35 * intensity_score + 0.65 * local_score

        # Confidence drops when residual is essentially zero (PNG without
        # JPEG history) or the image is tiny.
        h, w = residual.shape
        confidence = 1.0
        if mean < 1e-3:
            confidence = 0.3
        if h * w < 64 * 64:
            confidence *= 0.5

        notes: list[str] = []
        if hotspot_frac > 0.04:
            notes.append(
                f"Localized ELA hotspots cover {hotspot_frac * 100:.2f}% of pixels."
            )
        if mean > 0.06:
            notes.append("Mean residual is elevated; check for global re-encoding.")

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(_clip01(score), 4),
            confidence=round(confidence, 3),
            evidence={
                "residual_mean": round(mean, 5),
                "residual_std": round(std, 5),
                "hotspot_fraction": round(hotspot_frac, 5),
                "quality": self.quality,
                "image_size": [int(w), int(h)],
            },
            notes=notes,
        )


def _saturating(x: float) -> float:
    """Smooth saturation: 0 at 0, 1 in the limit. Avoids hard clipping."""
    return float(1.0 - np.exp(-max(0.0, x)))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
