"""JPEG quantization-table fingerprinting + double-JPEG detection.

Forensic background:
  - Stock cameras/phones use a small set of well-documented quantization
    tables (Hass-Lichtenauer, Farid 2006). Software re-encodes (Photoshop,
    GIMP, web pipelines) typically use IJG-derived tables that differ from
    camera tables.
  - When a JPEG is decoded, edited, and re-saved, its DCT coefficients show
    *double-quantization* artifacts: periodic gaps/peaks in the histogram
    of low-frequency coefficients (Lukáš & Fridrich 2003, Pevný & Fridrich
    2008). This is one of the most cited primary-evidence detectors in
    image forensics.

This module estimates both signals using only Pillow + NumPy. For best
fidelity in production, add libjpeg-turbo bindings (`jpegio`) to read raw
DCT coefficients without redecoding.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from fraud_detection.detectors.base import DetectorResult

# Standard IJG quality-50 luminance table — landmark for "edited in software".
IJG_Q50_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


class JPEGQuantizationDetector:
    name = "jpeg_qtable"
    version = "1.2"

    def run(self, image_path: str | Path) -> DetectorResult:
        path = Path(image_path)
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.0,
                evidence={"input_format": path.suffix},
                notes=["Not a JPEG — detector skipped."],
            )

        with Image.open(path) as im:
            qtables = getattr(im, "quantization", None) or {}
            im_rgb = im.convert("YCbCr")
        if not qtables:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.3,
                confidence=0.5,
                evidence={"qtables_present": False},
                notes=[
                    "JPEG had no readable quantization tables — unusual for "
                    "in-camera output."
                ],
            )

        luma = np.asarray(qtables.get(0, []), dtype=np.float32)
        if luma.size != 64:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.0,
                evidence={"qtable_size": int(luma.size)},
                notes=["Luma quantization table missing or malformed."],
            )
        luma = luma.reshape(8, 8)

        # 1. Distance to standard IJG table (signal of software re-save).
        # Scale by quality estimate.
        q_estimate = _estimate_quality(luma)
        ref = _scale_table(IJG_Q50_LUMA, q_estimate)
        ijg_distance = float(np.abs(luma - ref).mean() / max(ref.mean(), 1.0))
        ijg_match = ijg_distance < 0.05  # very close → software-derived

        # 2. Double-JPEG via DCT-coefficient histogram analysis.
        dq_score = _double_quantization_score(np.asarray(im_rgb)[..., 0])

        # IJG match alone is *not* sufficient evidence — most software pipelines
        # use IJG tables, including phone-side post-processing. We treat it as
        # *informational* and only contribute to the score when paired with
        # double-quantization evidence (Pevný & Fridrich 2008).
        software_score = 1.0 if ijg_match else 0.0
        score = dq_score + 0.10 * (software_score * dq_score)
        confidence = 0.85 if luma.size == 64 else 0.4
        # Lower confidence when no double-quant signal — the detector's primary
        # signal is missing.
        if dq_score < 0.05:
            confidence *= 0.5

        notes: list[str] = []
        notes.append(f"Estimated JPEG quality factor: {q_estimate:.0f}")
        if ijg_match:
            notes.append(
                "Luma quantization table matches IJG library (informational; "
                "most software pipelines use this table)."
            )
        else:
            notes.append("Quantization table is non-standard (camera-like).")
        if dq_score > 0.4:
            notes.append(
                "DCT coefficient histogram shows double-quantization peaks — "
                "evidence of re-saved JPEG."
            )

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(float(min(1.0, score)), 4),
            confidence=round(float(confidence), 3),
            evidence={
                "estimated_quality": round(float(q_estimate), 2),
                "ijg_distance": round(ijg_distance, 5),
                "ijg_match": bool(ijg_match),
                "double_quantization_score": round(float(dq_score), 4),
                "qtable_luma_first_row": [int(v) for v in luma[0].tolist()],
            },
            notes=notes,
        )


# --------------------------------------------------------------------- helpers
def _estimate_quality(qtable: np.ndarray) -> float:
    """Reverse-engineer JPEG quality factor from the luma quantization table.

    Uses the standard IJG formula (Wallace 1991): scale = 50/quality if
    quality >= 50 else 5000/quality.
    """
    avg = float(qtable.mean())
    ref_avg = float(IJG_Q50_LUMA.mean())
    if avg <= 0:
        return 50.0
    # Solve avg = ref_avg * scale/100, where scale = max(1, min(100, ...))
    scale = avg / ref_avg * 100.0
    if scale < 100.0:
        q = 100.0 - scale / 2.0
    else:
        q = 5000.0 / scale
    return float(np.clip(q, 1.0, 100.0))


def _scale_table(base: np.ndarray, quality: float) -> np.ndarray:
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2 * quality
    out = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(out, 1, 255)


def _double_quantization_score(luma_channel: np.ndarray) -> float:
    """Approximate double-JPEG signature via 8x8 block DCT histogram peakiness.

    Methodology follows the spirit of Pevný & Fridrich 2008 — tampered JPEGs
    show periodic spikes in DCT coefficient histograms because two
    quantizations leave a remainder pattern. We measure histogram
    periodicity at the (1,2) AC coefficient.
    """
    h, w = luma_channel.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 16 or w8 < 16:
        return 0.0
    img = luma_channel[:h8, :w8].astype(np.float32) - 128.0
    blocks = img.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)
    coeffs = _dct2(blocks)  # (Bh, Bw, 8, 8)
    # AC coefficient at (1, 2) — sensitive to double quantization.
    ac = coeffs[..., 1, 2].ravel()
    # Histogram around 0 with unit spacing.
    hist, _ = np.histogram(ac, bins=np.arange(-32, 33))
    if hist.sum() == 0:
        return 0.0
    pdf = hist.astype(np.float32) / hist.sum()
    # Power at periodicity ~ q (we don't know q; check periods 2..8).
    power = []
    for period in range(2, 9):
        comb = pdf[::period]
        if comb.size > 1:
            power.append(comb.mean() / (pdf.mean() + 1e-9))
    if not power:
        return 0.0
    peak = float(max(power))
    return float(min(1.0, max(0.0, (peak - 1.0) / 1.5)))


def _dct2(blocks: np.ndarray) -> np.ndarray:
    """2-D DCT-II over the last two axes, separable, no scipy dependency."""
    n = blocks.shape[-1]
    k = np.arange(n)
    factor = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)).astype(
        np.float32
    )
    norm = np.full(n, np.sqrt(2.0 / n), dtype=np.float32)
    norm[0] = np.sqrt(1.0 / n)
    m = norm[:, None] * factor
    out = m @ blocks @ m.T
    return out
