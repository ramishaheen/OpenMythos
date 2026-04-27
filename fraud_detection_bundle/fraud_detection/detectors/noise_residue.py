"""PRNU-style noise-residue consistency detector.

Photo Response Non-Uniformity (Lukáš, Fridrich, Goljan 2006) is the
canonical sensor-noise camera-fingerprint. A full PRNU pipeline requires
a per-camera reference fingerprint estimated from many images. Without
that reference we can still run *internal* PRNU consistency: extract a
denoised residual from the image, then check whether the residual's
statistics are *uniform* across the frame. Spliced regions exhibit
sharply different residual statistics (different sensor or post-processing
chain).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraud_detection.detectors.base import DetectorResult


class NoiseResidueDetector:
    name = "prnu_consistency"
    version = "1.1"

    def __init__(self, block: int = 64) -> None:
        self.block = block

    def run(self, image_path: str | Path) -> DetectorResult:
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)

        h, w = gray.shape
        if h < self.block * 2 or w < self.block * 2:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.0,
                notes=["Image too small for block analysis."],
            )

        residual = gray - _box_blur(gray, k=3)

        # Per-block residual statistics.
        b = self.block
        bh = h // b
        bw = w // b
        cropped = residual[: bh * b, : bw * b].reshape(bh, b, bw, b).swapaxes(1, 2)
        means = cropped.mean(axis=(2, 3))
        stds = cropped.std(axis=(2, 3))
        kurts = _kurtosis_block(cropped)

        rel_std = float(stds.std() / (stds.mean() + 1e-6))
        rel_kurt = float(kurts.std() / (np.abs(kurts).mean() + 1e-6))

        # Detect any single block whose stats are far from the global mean.
        z_std = np.abs(stds - stds.mean()) / (stds.std() + 1e-6)
        z_kurt = np.abs(kurts - kurts.mean()) / (kurts.std() + 1e-6)
        outlier_blocks = int(((z_std > 3.0) | (z_kurt > 3.0)).sum())
        outlier_ratio = outlier_blocks / max(1, bh * bw)

        score = float(
            min(1.0, 0.5 * min(1.0, rel_std / 0.5) + 0.5 * min(1.0, outlier_ratio / 0.05))
        )
        confidence = 0.75

        notes = []
        if outlier_ratio > 0.05:
            notes.append(
                f"{outlier_blocks} blocks have noise statistics >3σ from the "
                "global distribution; possible splice."
            )

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(score, 4),
            confidence=confidence,
            evidence={
                "block_size": self.block,
                "blocks": int(bh * bw),
                "block_std_relative_dispersion": round(rel_std, 4),
                "outlier_block_ratio": round(outlier_ratio, 4),
                "block_kurtosis_dispersion": round(rel_kurt, 4),
            },
            notes=notes,
        )


def _box_blur(arr: np.ndarray, k: int) -> np.ndarray:
    """Cheap separable box blur. ``k`` must be odd. Returns same shape as ``arr``.

    Implemented with prepended-zero cumulative sums so the row/column
    differences yield exactly the input shape — the previous version had an
    off-by-one and produced (H-1, W-1) outputs which crashed downstream.
    """
    if k % 2 != 1 or k < 1:
        raise ValueError(f"box-blur kernel must be a positive odd integer, got {k}")
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge").astype(np.float32)
    cs = np.zeros((padded.shape[0] + 1, padded.shape[1]), dtype=np.float32)
    cs[1:] = padded.cumsum(axis=0)
    blurred_y = (cs[k:] - cs[:-k]) / float(k)
    cs2 = np.zeros((blurred_y.shape[0], blurred_y.shape[1] + 1), dtype=np.float32)
    cs2[:, 1:] = blurred_y.cumsum(axis=1)
    blurred = (cs2[:, k:] - cs2[:, :-k]) / float(k)
    return blurred


def _kurtosis_block(blocks: np.ndarray) -> np.ndarray:
    flat = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    m = flat.mean(axis=-1, keepdims=True)
    s = flat.std(axis=-1, keepdims=True) + 1e-6
    z = (flat - m) / s
    return (z**4).mean(axis=-1) - 3.0
