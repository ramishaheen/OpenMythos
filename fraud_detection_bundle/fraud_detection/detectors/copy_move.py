"""Perceptual-hash-based copy-move detector — replaces the crude block hash.

Uses a 64-bit DCT perceptual hash (pHash, Zauner 2010) per overlapping block
plus a small Hamming-radius search for matches. Resists JPEG re-compression
and small geometric changes much better than the previous mean-sign hash,
and yields far fewer false positives on noisy images.

For full court-grade analysis, swap this for SIFT/SURF keypoint matching
(opencv-contrib) or a CNN-based copy-move detector. The agent's tool
contract is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraud_detection.detectors.base import DetectorResult
from fraud_detection.detectors.jpeg_qtable import _dct2


class PerceptualCopyMoveDetector:
    name = "copy_move_phash"
    version = "1.0"

    def __init__(
        self,
        block_size: int = 32,
        stride: int = 16,
        hamming_radius: int = 4,
        min_separation: int = 48,
    ) -> None:
        self.block_size = block_size
        self.stride = stride
        self.hamming_radius = hamming_radius
        self.min_separation = min_separation

    def run(self, image_path: str | Path) -> DetectorResult:
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)

        h, w = gray.shape
        # Downscale very large images for tractable runtime.
        max_side = 768
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            with Image.open(image_path) as im:
                gray = np.asarray(
                    im.convert("L").resize((new_w, new_h), Image.LANCZOS),
                    dtype=np.float32,
                )
            h, w = gray.shape

        b = self.block_size
        s = self.stride
        if h < b * 2 or w < b * 2:
            return DetectorResult(
                name=self.name, version=self.version, score=0.0, confidence=0.0
            )

        positions = []
        hashes = []
        for y in range(0, h - b, s):
            for x in range(0, w - b, s):
                block = gray[y : y + b, x : x + b]
                # 8x8 reduced via mean-pooling, then DCT, then hash.
                small = block.reshape(8, b // 8, 8, b // 8).mean(axis=(1, 3))
                dct = _dct2(small[None, None]).squeeze()
                low = dct[:8, :8].copy()
                low[0, 0] = 0.0
                med = float(np.median(low))
                bits = (low > med).astype(np.uint8).ravel()
                # Pack 64 bits into uint64.
                packed = np.uint64(0)
                for bi in bits:
                    packed = (packed << np.uint64(1)) | np.uint64(int(bi))
                positions.append((y, x))
                hashes.append(int(packed))

        hashes_np = np.asarray(hashes, dtype=np.uint64)
        positions_np = np.asarray(positions, dtype=np.int32)
        # Bucket by high-order bits to avoid O(N^2). Each 16-bit prefix
        # bucket gathers candidates; do exact Hamming inside each bucket.
        prefix = (hashes_np >> np.uint64(48)).astype(np.int64)
        order = np.argsort(prefix)
        hashes_sorted = hashes_np[order]
        positions_sorted = positions_np[order]
        prefix_sorted = prefix[order]

        match_pairs = 0
        offset_histogram: dict[tuple[int, int], int] = {}
        i = 0
        n = len(hashes_sorted)
        while i < n:
            j = i + 1
            while j < n and prefix_sorted[j] == prefix_sorted[i]:
                j += 1
            for a in range(i, j):
                for c in range(a + 1, j):
                    xor = int(hashes_sorted[a]) ^ int(hashes_sorted[c])
                    ham = bin(xor).count("1")
                    if ham > self.hamming_radius:
                        continue
                    pa = positions_sorted[a]
                    pc = positions_sorted[c]
                    if abs(int(pa[0]) - int(pc[0])) + abs(int(pa[1]) - int(pc[1])) < self.min_separation:
                        continue
                    match_pairs += 1
                    dy = int(pc[0]) - int(pa[0])
                    dx = int(pc[1]) - int(pa[1])
                    # Quantize offset to 8-px buckets.
                    key = (dy // 8, dx // 8)
                    offset_histogram[key] = offset_histogram.get(key, 0) + 1
            i = j

        # Real copy-move shows a *consistent shift vector*: many pairs share
        # the same offset. A few scattered matches are false positives.
        if match_pairs == 0:
            return DetectorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                confidence=0.9,
                evidence={"matched_pairs": 0},
                notes=[],
            )
        peak_offset, peak_count = max(offset_histogram.items(), key=lambda kv: kv[1])
        peak_share = peak_count / match_pairs
        # Score: needs both a non-trivial number of matches AND a dominant
        # offset. Both gates suppress textural false positives.
        density = match_pairs / max(1, len(hashes_np))
        score = float(
            min(1.0, max(0.0, (peak_share - 0.20) * 2.0)) * min(1.0, density / 0.05)
        )
        confidence = 0.85

        notes = []
        if peak_count >= 5 and peak_share > 0.3:
            notes.append(
                f"{peak_count} of {match_pairs} matched block pairs share "
                f"offset (~{peak_offset[0] * 8},{peak_offset[1] * 8}) — "
                "consistent with copy-move tampering."
            )

        return DetectorResult(
            name=self.name,
            version=self.version,
            score=round(score, 4),
            confidence=confidence,
            evidence={
                "matched_pairs": match_pairs,
                "block_count": int(len(hashes_np)),
                "dominant_offset": list(peak_offset),
                "dominant_offset_share": round(peak_share, 4),
            },
            notes=notes,
        )
