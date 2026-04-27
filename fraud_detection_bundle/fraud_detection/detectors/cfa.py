"""CFA (Color Filter Array) demosaicing inconsistency detector.

Pristine images from a Bayer-CFA sensor show a characteristic high-frequency
pattern in the green channel (rows alternate between G-rich and R/B-rich).
Edited regions break this pattern. Popescu & Farid (2005) introduced the
canonical EM-based CFA detector; this module implements the simpler
variance-ratio variant that's robust on resampled/cropped inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraud_detection.detectors.base import DetectorResult


class CFAInconsistencyDetector:
    name = "cfa_inconsistency"
    version = "1.0"

    def run(self, image_path: str | Path) -> DetectorResult:
        with Image.open(image_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)

        h, w, _ = arr.shape
        if h < 64 or w < 64:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.0,
                notes=["Image too small for CFA analysis."],
            )

        green = arr[..., 1]
        # Bayer pattern: green at (even, even) and (odd, odd).
        diag = np.zeros_like(green, dtype=bool)
        diag[0::2, 0::2] = True
        diag[1::2, 1::2] = True
        anti = ~diag

        # Per-pixel residual against a 3x3 mean (cheap demosaic predictor).
        from numpy.lib.stride_tricks import sliding_window_view

        if green.shape[0] < 3 or green.shape[1] < 3:
            return DetectorResult(
                name=self.name, version=self.version, score=0.0, confidence=0.0
            )
        windows = sliding_window_view(green, (3, 3))
        local_mean = windows.mean(axis=(-1, -2))
        residual = np.zeros_like(green)
        residual[1:-1, 1:-1] = green[1:-1, 1:-1] - local_mean

        # Compare residual variance on diagonal vs anti-diagonal positions in
        # blocks. If CFA is intact the two should differ; if not (edited),
        # they equalise.
        bs = 32
        h_b = (h // bs) * bs
        w_b = (w // bs) * bs
        if h_b < bs * 2 or w_b < bs * 2:
            return DetectorResult(
                name=self.name, version=self.version, score=0.0, confidence=0.0
            )
        residual = residual[:h_b, :w_b]
        d = diag[:h_b, :w_b]
        per_block_ratios = []
        for y in range(0, h_b, bs):
            for x in range(0, w_b, bs):
                rb = residual[y : y + bs, x : x + bs]
                db = d[y : y + bs, x : x + bs]
                v_diag = float(rb[db].var()) if db.any() else 0.0
                v_anti = float(rb[~db].var()) if (~db).any() else 0.0
                if v_diag + v_anti < 1e-3:
                    continue
                ratio = v_diag / (v_anti + 1e-6)
                per_block_ratios.append(ratio)

        if not per_block_ratios:
            return DetectorResult(
                name=self.name, version=self.version, score=0.0, confidence=0.0
            )
        ratios = np.asarray(per_block_ratios, dtype=np.float32)
        # In a pristine CFA-camera image, ratios are tightly clustered
        # (low std). Tampering shows up as block-to-block variance.
        rel_std = float(ratios.std() / (ratios.mean() + 1e-6))
        # Fraction of blocks with ratio close to 1.0 (CFA pattern destroyed).
        flat_frac = float(((ratios > 0.85) & (ratios < 1.18)).mean())
        score = 0.5 * min(1.0, rel_std / 0.6) + 0.5 * flat_frac
        confidence = 0.7

        notes = []
        if flat_frac > 0.5:
            notes.append(
                f"{flat_frac * 100:.0f}% of blocks show no CFA residual signature; "
                "possible re-rendered or non-camera origin."
            )

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(float(min(1.0, max(0.0, score))), 4),
            confidence=confidence,
            evidence={
                "block_count": len(per_block_ratios),
                "ratio_std_over_mean": round(rel_std, 4),
                "blocks_without_cfa_pattern": round(flat_frac, 4),
            },
            notes=notes,
        )
