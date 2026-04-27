"""Image-forensics primitives — ELA, EXIF, copy-move, noise.

The techniques here mirror the open-source IFAKE pipeline:
  - Error Level Analysis (ELA): re-encode at fixed quality and diff
  - EXIF / metadata sanity checks
  - Block-based copy-move detection (DCT-feature hashing)
  - High-frequency noise residue scoring

All routines run with Pillow + NumPy only — no GPU, no model weights.
The agent treats these scores as evidence and lets Claude weigh them.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, ImageChops

from fraud_detection.utils import clamp01, risk_label


@dataclass
class ImageForensicsResult:
    path: str
    ela_score: float
    ela_hotspots: list[tuple[int, int, int, int]]
    copy_move_score: float
    copy_move_pairs: int
    noise_score: float
    metadata: dict[str, Any]
    metadata_flags: list[str]
    overall_score: float
    risk: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageForensics:
    """Stateless image-forensics analyzer."""

    def __init__(self, ela_quality: int = 90, block_size: int = 16) -> None:
        self.ela_quality = ela_quality
        self.block_size = block_size

    # ------------------------------------------------------------------ ELA
    def error_level_analysis(
        self, path: str | Path
    ) -> tuple[float, list[tuple[int, int, int, int]], np.ndarray]:
        """Compute ELA score, hotspot bounding boxes, and the ELA image."""
        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.ela_quality)
            buf.seek(0)
            with Image.open(buf) as recompressed:
                ela = ImageChops.difference(im, recompressed)
            arr = np.asarray(ela, dtype=np.float32)

        # Per-pixel max-channel residual, normalised to [0,1].
        residual = arr.max(axis=2) / 255.0
        score = float(residual.mean() * 8.0)  # IFAKE-style amplification
        score = clamp01(score)

        # Hotspots: connected high-residual blocks.
        thresh = max(0.15, residual.mean() + 2.0 * residual.std())
        mask = residual > thresh
        hotspots = _bbox_clusters(mask, min_area=64, max_boxes=8)
        return score, hotspots, residual

    # ---------------------------------------------------------------- EXIF
    def metadata_inspection(
        self, path: str | Path
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract EXIF and flag suspicious patterns."""
        meta: dict[str, Any] = {}
        flags: list[str] = []
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    meta[tag] = _coerce(value)
                meta["_format"] = im.format
                meta["_mode"] = im.mode
                meta["_size"] = list(im.size)
        except Exception as e:  # noqa: BLE001
            flags.append(f"metadata_read_error: {e!s}")
            return meta, flags

        software = str(meta.get("Software", "")).lower()
        for needle in ("photoshop", "gimp", "lightroom", "affinity", "snapseed"):
            if needle in software:
                flags.append(f"editor_signature:{needle}")
                break

        if not exif_has_camera(meta):
            flags.append("missing_camera_make_model")

        if "DateTime" in meta and "DateTimeOriginal" in meta:
            if meta["DateTime"] != meta["DateTimeOriginal"]:
                flags.append("datetime_mismatch")

        if meta.get("_format") == "JPEG" and "JPEGThumbnail" not in meta and not exif:
            flags.append("jpeg_without_exif")

        return meta, flags

    # --------------------------------------------------------- copy-move
    def copy_move_detection(self, path: str | Path) -> tuple[float, int]:
        """Block-DCT hashing — counts matching block pairs.

        Not a state-of-the-art detector, but flags obvious clone-stamps.
        """
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            scale = min(1.0, 512 / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32)

        b = self.block_size
        H, W = arr.shape
        if H < b * 2 or W < b * 2:
            return 0.0, 0

        # Coarse 8-bit DCT signature per overlapping block (stride b//2).
        stride = max(1, b // 2)
        sigs: dict[bytes, list[tuple[int, int]]] = {}
        pairs = 0
        for y in range(0, H - b, stride):
            for x in range(0, W - b, stride):
                block = arr[y : y + b, x : x + b]
                sig = _block_signature(block)
                bucket = sigs.setdefault(sig, [])
                for py, px in bucket:
                    if abs(py - y) + abs(px - x) > b * 2:  # ignore neighbours
                        pairs += 1
                bucket.append((y, x))

        denom = max(1, ((H // stride) * (W // stride)) // 16)
        score = clamp01(pairs / denom)
        return score, pairs

    # ----------------------------------------------------------- noise
    def noise_residue_score(self, path: str | Path) -> float:
        """High-pass residual variance — splices often disrupt sensor noise."""
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
        # 3x3 Laplacian
        k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        from numpy.lib.stride_tricks import sliding_window_view

        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        windows = sliding_window_view(arr, (3, 3))
        lap = (windows * k).sum(axis=(-1, -2))
        # Block-wise variance heterogeneity: legit photos look homogeneous.
        bs = 32
        H, W = lap.shape
        bh, bw = H // bs, W // bs
        if bh < 2 or bw < 2:
            return 0.0
        cropped = lap[: bh * bs, : bw * bs].reshape(bh, bs, bw, bs)
        block_var = cropped.var(axis=(1, 3))
        # Coefficient of variation across blocks.
        mean = float(block_var.mean()) + 1e-6
        cov = float(block_var.std() / mean)
        return clamp01((cov - 0.6) / 1.5)

    # ---------------------------------------------------------- compose
    def analyze(self, path: str | Path) -> ImageForensicsResult:
        ela, hotspots, _ = self.error_level_analysis(path)
        meta, meta_flags = self.metadata_inspection(path)
        cm_score, cm_pairs = self.copy_move_detection(path)
        noise = self.noise_residue_score(path)

        # Weighted blend — tuned conservatively. Claude can override in agent.
        overall = clamp01(
            0.45 * ela
            + 0.25 * cm_score
            + 0.20 * noise
            + 0.10 * min(1.0, len(meta_flags) / 3.0)
        )
        notes = []
        if ela > 0.4:
            notes.append("Elevated ELA residuals suggest local re-encoding.")
        if cm_score > 0.3:
            notes.append(f"Detected {cm_pairs} matching block pairs (copy-move).")
        if noise > 0.4:
            notes.append("Block-wise noise heterogeneity is high.")
        for f in meta_flags:
            notes.append(f"Metadata flag: {f}")

        return ImageForensicsResult(
            path=str(path),
            ela_score=round(ela, 4),
            ela_hotspots=hotspots,
            copy_move_score=round(cm_score, 4),
            copy_move_pairs=cm_pairs,
            noise_score=round(noise, 4),
            metadata=meta,
            metadata_flags=meta_flags,
            overall_score=round(overall, 4),
            risk=risk_label(overall),
            notes=notes,
        )


# ---------------------------------------------------------------- helpers
def _coerce(v: Any) -> Any:
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return repr(v)
    if isinstance(v, (tuple, list)):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce(val) for k, val in v.items()}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def exif_has_camera(meta: dict[str, Any]) -> bool:
    return bool(meta.get("Make")) and bool(meta.get("Model"))


def _block_signature(block: np.ndarray) -> bytes:
    """8-bit quantised DCT-ish signature — robust to mild JPEG noise."""
    b = block - block.mean()
    # Cheap separable transform: row + column means of sign pattern.
    rows = (b.mean(axis=1) > 0).astype(np.uint8)
    cols = (b.mean(axis=0) > 0).astype(np.uint8)
    return rows.tobytes() + cols.tobytes()


def _bbox_clusters(
    mask: np.ndarray, min_area: int, max_boxes: int
) -> list[tuple[int, int, int, int]]:
    """Coarse connected-region extractor (no scipy dependency)."""
    if not mask.any():
        return []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    # 8-connected flood-fill via iterative DFS.
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                ys.append(cy)
                xs.append(cx)
                stack.extend(
                    [
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                        (cy - 1, cx - 1),
                        (cy - 1, cx + 1),
                        (cy + 1, cx - 1),
                        (cy + 1, cx + 1),
                    ]
                )
            if len(ys) >= min_area:
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
    boxes.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    return boxes[:max_boxes]
