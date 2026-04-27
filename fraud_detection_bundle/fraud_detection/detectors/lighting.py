"""Lighting-direction consistency detector.

Splices typically combine regions illuminated from different directions.
The inverse-lighting-direction estimator (Johnson & Farid 2005) recovers
the dominant lighting vector per region from intensity gradients along
object boundaries. Mis-aligned vectors across regions imply tampering.

This module implements a coarse, gradient-histogram-based proxy: it
splits the image into NxN blocks, computes the dominant gradient
direction per block, and reports the angular dispersion. Pristine
photographs cluster tightly; spliced ones disperse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraud_detection.detectors.base import DetectorResult


class LightingConsistencyDetector:
    name = "lighting_consistency"
    version = "1.0"

    def __init__(self, blocks: int = 6) -> None:
        self.blocks = blocks

    def run(self, image_path: str | Path) -> DetectorResult:
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)

        h, w = gray.shape
        if h < 64 or w < 64:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.0,
                notes=["Image too small."],
            )

        gx, gy = np.gradient(gray)
        magnitude = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx)

        bh = h // self.blocks
        bw = w // self.blocks
        if bh < 8 or bw < 8:
            return DetectorResult(
                name=self.name, version=self.version, score=0.0, confidence=0.0
            )

        dominant_angles = []
        for by in range(self.blocks):
            for bx in range(self.blocks):
                m = magnitude[by * bh : (by + 1) * bh, bx * bw : (bx + 1) * bw]
                a = angle[by * bh : (by + 1) * bh, bx * bw : (bx + 1) * bw]
                # Restrict to high-magnitude pixels (edges) — those carry the
                # lighting signature.
                edge_thresh = float(np.percentile(m, 90))
                mask = m > edge_thresh
                if mask.sum() < 32:
                    continue
                # Circular mean.
                cs = np.cos(a[mask]).mean()
                ss = np.sin(a[mask]).mean()
                dominant_angles.append(np.arctan2(ss, cs))

        if len(dominant_angles) < 4:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.3,
                notes=["Too few high-gradient blocks for reliable estimation."],
            )

        angles = np.asarray(dominant_angles, dtype=np.float32)
        # Circular variance — 0 = aligned, 1 = uniformly dispersed.
        cs = np.cos(angles).mean()
        ss = np.sin(angles).mean()
        circ_var = float(1.0 - np.hypot(cs, ss))

        # Pristine photos have a partial CV (~0.4-0.7); splices push above
        # ~0.85. Calibrate accordingly.
        score = float(min(1.0, max(0.0, (circ_var - 0.7) / 0.25)))
        confidence = 0.6  # this is a weak detector, weight it modestly

        notes = []
        if circ_var > 0.85:
            notes.append(
                f"Block-wise dominant gradient directions are highly dispersed "
                f"(circular variance {circ_var:.2f})."
            )

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(score, 4),
            confidence=confidence,
            evidence={
                "blocks_analysed": len(dominant_angles),
                "circular_variance": round(circ_var, 4),
            },
            notes=notes,
        )
