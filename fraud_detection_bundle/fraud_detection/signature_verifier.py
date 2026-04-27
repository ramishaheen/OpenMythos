"""Signature verification — geometric features + ORB + optional SSIM.

Inspired by sainipankaj15/Signature-Forgery-Detection and the broader
GitHub `signature-detection` topic. The classical pipeline:

  1. Preprocess: grayscale -> Otsu threshold -> denoise -> bbox crop.
  2. Geometric features: aspect ratio, ink density, centroid offset,
     stroke-thickness estimate, skeleton length.
  3. Reference matching: ORB keypoints + Hamming distance on descriptors,
     plus optional SSIM (scikit-image) on size-normalised crops.
  4. Verdict: distance fused into a 0-1 forgery score.

Claude (in agent.py) gets the raw features and renders the final call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from fraud_detection.utils import clamp01, risk_label


@dataclass
class SignatureFeatures:
    aspect_ratio: float
    ink_density: float
    centroid_offset: tuple[float, float]
    stroke_thickness: float
    contour_count: int
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SignatureVerdict:
    questioned: str
    reference: str | None
    features: SignatureFeatures
    reference_features: SignatureFeatures | None
    feature_distance: float | None
    orb_distance: float | None
    ssim_score: float | None
    forgery_score: float
    risk: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = self.features.to_dict()
        d["reference_features"] = (
            self.reference_features.to_dict() if self.reference_features else None
        )
        return d


class SignatureVerifier:
    def __init__(self, target_size: tuple[int, int] = (256, 128)) -> None:
        self.target_size = target_size

    # ------------------------------------------------------- preprocess
    def preprocess(self, path: str | Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("L")
            arr = np.asarray(im, dtype=np.uint8)

        # Otsu threshold (hand-rolled to avoid skimage dep here).
        thresh = _otsu(arr)
        # `<=` so that when Otsu picks t=0 on a sharply bimodal image
        # (pure black ink on pure white paper) we still capture the ink.
        binary = (arr <= thresh).astype(np.uint8)  # 1 = ink

        ys, xs = np.where(binary > 0)
        if ys.size == 0:
            return binary, (0, 0, binary.shape[1] - 1, binary.shape[0] - 1)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return binary, bbox

    # -------------------------------------------------------- features
    def extract_features(self, path: str | Path) -> SignatureFeatures:
        binary, bbox = self.preprocess(path)
        x0, y0, x1, y1 = bbox
        crop = binary[y0 : y1 + 1, x0 : x1 + 1]
        h, w = max(1, crop.shape[0]), max(1, crop.shape[1])
        ink = float(crop.sum())
        density = ink / float(h * w)

        if ink > 0:
            ys, xs = np.where(crop > 0)
            cy = float(ys.mean()) / h
            cx = float(xs.mean()) / w
        else:
            cy = cx = 0.5

        # Stroke thickness ~= ink_pixels / skeleton_length.
        skeleton = _skeletonize(crop)
        skel_len = max(1, int(skeleton.sum()))
        thickness = ink / skel_len

        return SignatureFeatures(
            aspect_ratio=round(w / h, 4),
            ink_density=round(density, 4),
            centroid_offset=(round(cx - 0.5, 4), round(cy - 0.5, 4)),
            stroke_thickness=round(thickness, 4),
            contour_count=int(_count_components(crop)),
            bbox=bbox,
        )

    # --------------------------------------------------------- compare
    def compare(
        self, questioned: str | Path, reference: str | Path
    ) -> SignatureVerdict:
        q_feat = self.extract_features(questioned)
        r_feat = self.extract_features(reference)

        dist = _feature_distance(q_feat, r_feat)
        orb_dist = _orb_distance(questioned, reference)
        ssim = _ssim_score(questioned, reference, self.target_size)

        # Fuse what we have.
        components: list[float] = [dist]
        if orb_dist is not None:
            components.append(orb_dist)
        if ssim is not None:
            components.append(1.0 - ssim)
        forgery_score = clamp01(float(np.mean(components)))

        notes: list[str] = []
        if dist > 0.4:
            notes.append("Geometric features diverge from the reference.")
        if orb_dist is not None and orb_dist > 0.5:
            notes.append("ORB descriptor distance is high.")
        if ssim is not None and ssim < 0.55:
            notes.append("Pixel-level structural similarity is low.")
        if not notes:
            notes.append("Classical metrics are within tolerance of the reference.")

        return SignatureVerdict(
            questioned=str(questioned),
            reference=str(reference),
            features=q_feat,
            reference_features=r_feat,
            feature_distance=round(dist, 4),
            orb_distance=round(orb_dist, 4) if orb_dist is not None else None,
            ssim_score=round(ssim, 4) if ssim is not None else None,
            forgery_score=round(forgery_score, 4),
            risk=risk_label(forgery_score),
            notes=notes,
        )

    # --------------------------------------------------------- single
    def analyze_single(self, path: str | Path) -> SignatureVerdict:
        """Score a signature without a reference — flags only intrinsic anomalies."""
        feats = self.extract_features(path)
        anomalies = 0
        notes: list[str] = []
        if feats.ink_density < 0.02:
            anomalies += 1
            notes.append("Unusually sparse ink — possible erasure or scan artefact.")
        if feats.ink_density > 0.45:
            anomalies += 1
            notes.append("Excessive ink density — possible overdrawing.")
        if abs(feats.centroid_offset[0]) > 0.25 or abs(feats.centroid_offset[1]) > 0.25:
            anomalies += 1
            notes.append("Centroid is heavily skewed within the bounding box.")
        if feats.contour_count > 80:
            anomalies += 1
            notes.append("Excessive disconnected components — possible patching.")

        score = clamp01(anomalies / 4.0)
        return SignatureVerdict(
            questioned=str(path),
            reference=None,
            features=feats,
            reference_features=None,
            feature_distance=None,
            orb_distance=None,
            ssim_score=None,
            forgery_score=round(score, 4),
            risk=risk_label(score),
            notes=notes or ["No intrinsic anomalies detected."],
        )


# ---------------------------------------------------------------- helpers
def _otsu(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 255))
    total = arr.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    threshold = 127
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t * hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def _skeletonize(binary: np.ndarray, max_iter: int = 25) -> np.ndarray:
    """Zhang-Suen-ish skeleton — good enough for stroke estimation."""
    img = binary.astype(np.uint8).copy()
    prev = np.zeros_like(img)
    for _ in range(max_iter):
        if np.array_equal(img, prev):
            break
        prev = img.copy()
        for sub in (0, 1):
            removals = []
            h, w = img.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] == 0:
                        continue
                    p = img[y - 1 : y + 2, x - 1 : x + 2].copy()
                    p[1, 1] = 0
                    n = int(p.sum())
                    if n < 2 or n > 6:
                        continue
                    seq = [
                        img[y - 1, x],
                        img[y - 1, x + 1],
                        img[y, x + 1],
                        img[y + 1, x + 1],
                        img[y + 1, x],
                        img[y + 1, x - 1],
                        img[y, x - 1],
                        img[y - 1, x - 1],
                    ]
                    transitions = sum(
                        1 for i in range(8) if seq[i] == 0 and seq[(i + 1) % 8] == 1
                    )
                    if transitions != 1:
                        continue
                    if sub == 0:
                        if seq[0] * seq[2] * seq[4] != 0:
                            continue
                        if seq[2] * seq[4] * seq[6] != 0:
                            continue
                    else:
                        if seq[0] * seq[2] * seq[6] != 0:
                            continue
                        if seq[0] * seq[4] * seq[6] != 0:
                            continue
                    removals.append((y, x))
            for y, x in removals:
                img[y, x] = 0
    return img


def _count_components(binary: np.ndarray) -> int:
    if binary.size == 0 or not binary.any():
        return 0
    visited = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    count = 0
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or binary[cy, cx] == 0:
                    continue
                visited[cy, cx] = True
                stack.extend(
                    [
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ]
                )
    return count


def _feature_distance(a: SignatureFeatures, b: SignatureFeatures) -> float:
    diffs = [
        abs(a.aspect_ratio - b.aspect_ratio) / max(a.aspect_ratio, b.aspect_ratio, 1e-6),
        abs(a.ink_density - b.ink_density) / max(a.ink_density, b.ink_density, 1e-6),
        abs(a.stroke_thickness - b.stroke_thickness)
        / max(a.stroke_thickness, b.stroke_thickness, 1e-6),
        abs(a.centroid_offset[0] - b.centroid_offset[0]) * 2.0,
        abs(a.centroid_offset[1] - b.centroid_offset[1]) * 2.0,
        min(1.0, abs(a.contour_count - b.contour_count) / max(1, b.contour_count)),
    ]
    return clamp01(float(np.mean(diffs)))


def _orb_distance(a: str | Path, b: str | Path) -> float | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    img_a = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return None
    orb = cv2.ORB_create(nfeatures=500)
    _, des_a = orb.detectAndCompute(img_a, None)
    _, des_b = orb.detectAndCompute(img_b, None)
    if des_a is None or des_b is None or len(des_a) == 0 or len(des_b) == 0:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    if not matches:
        return 1.0
    matches = sorted(matches, key=lambda m: m.distance)[: min(50, len(matches))]
    avg = float(np.mean([m.distance for m in matches]))
    return clamp01(avg / 64.0)  # Hamming over 32 bytes => max 256


def _ssim_score(
    a: str | Path, b: str | Path, size: tuple[int, int]
) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore
    except ImportError:
        return None
    with Image.open(a) as ia, Image.open(b) as ib:
        ia = ia.convert("L").resize(size, Image.LANCZOS)
        ib = ib.convert("L").resize(size, Image.LANCZOS)
        return float(ssim(np.asarray(ia), np.asarray(ib)))
