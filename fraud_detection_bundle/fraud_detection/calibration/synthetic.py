"""Synthetic dataset generation for calibration.

Pristine class simulates a camera-origin JPEG:
    multi-scale Perlin-like noise → Bayer mosaic → bilinear demosaic
    → save at one JPEG quality.
Tampered class:
    same camera-origin baseline → splice a region from another camera
    image → save at q1, reload, save at q2 (double-JPEG).
Copy-move class:
    camera-origin baseline → clone a 64x64 patch elsewhere in the image.
Recompressed class (negative control):
    camera-origin baseline → save at q1 → reload → save at q2 (no splice).

This is close enough to real data for *baseline* calibration; production
should re-run on labeled domain data (CASIA, CoMoFoD, custom KYC corpus).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _multiscale_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    """Pseudo-Perlin texture: sum of low-freq Gaussian noise upsampled."""
    out = np.zeros((size, size), dtype=np.float32)
    octaves = [4, 8, 16, 32, 64]
    weights = [0.5, 0.3, 0.15, 0.08, 0.04]
    for o, w in zip(octaves, weights):
        coarse = rng.normal(0, 1, size=(o, o)).astype(np.float32)
        # Bilinear-ish upsample via PIL.
        im = Image.fromarray(((coarse - coarse.min()) / max(1e-6, np.ptp(coarse)) * 255).astype(np.uint8))
        up = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        out += w * (up - 0.5)
    return out


def _camera_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Synthesize a 'camera' RGB image: noise → Bayer → demosaic.

    The Bayer→demosaic pass leaves the characteristic CFA artefact that the
    cfa_inconsistency detector keys on.
    """
    # Three independent channels for chrominance variation.
    chans = []
    for _ in range(3):
        n = _multiscale_noise(rng, size)
        n = (n - n.min()) / max(1e-6, float(np.ptp(n)))
        chans.append(n)
    rgb = np.stack(chans, axis=-1)  # 0..1
    # Mild gamma + brightness.
    rgb = np.clip(rgb ** 0.9 * 0.9 + 0.05, 0, 1) * 255.0
    rgb = rgb.astype(np.uint8)

    # Bayer mosaic (RGGB).
    h, w, _ = rgb.shape
    bayer = np.zeros((h, w), dtype=np.uint8)
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G1
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G2
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B

    # Bilinear demosaic — coarse but enough to leave the CFA fingerprint.
    out = np.zeros_like(rgb, dtype=np.float32)
    out[..., 0] = _interp_channel(bayer, "r")
    out[..., 1] = _interp_channel(bayer, "g")
    out[..., 2] = _interp_channel(bayer, "b")
    out += rng.normal(0, 1.2, size=out.shape).astype(np.float32)  # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _interp_channel(bayer: np.ndarray, ch: str) -> np.ndarray:
    h, w = bayer.shape
    mask = np.zeros((h, w), dtype=bool)
    if ch == "r":
        mask[0::2, 0::2] = True
    elif ch == "b":
        mask[1::2, 1::2] = True
    else:  # g
        mask[0::2, 1::2] = True
        mask[1::2, 0::2] = True
    chan = np.where(mask, bayer.astype(np.float32), 0.0)
    cnt = mask.astype(np.float32)
    # Box-blur 3x3 to fill missing samples.
    pad_c = np.pad(chan, 1, mode="edge")
    pad_n = np.pad(cnt, 1, mode="edge")
    s = (
        pad_c[0:-2, 0:-2] + pad_c[0:-2, 1:-1] + pad_c[0:-2, 2:]
        + pad_c[1:-1, 0:-2] + pad_c[1:-1, 1:-1] + pad_c[1:-1, 2:]
        + pad_c[2:, 0:-2] + pad_c[2:, 1:-1] + pad_c[2:, 2:]
    )
    n = (
        pad_n[0:-2, 0:-2] + pad_n[0:-2, 1:-1] + pad_n[0:-2, 2:]
        + pad_n[1:-1, 0:-2] + pad_n[1:-1, 1:-1] + pad_n[1:-1, 2:]
        + pad_n[2:, 0:-2] + pad_n[2:, 1:-1] + pad_n[2:, 2:]
    )
    return np.where(mask, bayer.astype(np.float32), s / np.maximum(n, 1.0)).astype(np.float32)


def _save_jpeg(arr: np.ndarray, dst: Path, quality: int) -> Path:
    Image.fromarray(arr).save(dst, format="JPEG", quality=int(quality))
    return dst


def _double_jpeg(arr: np.ndarray, dst: Path, q1: int, q2: int) -> Path:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(q1))
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        im.save(dst, format="JPEG", quality=int(q2))
    return dst


def make_pristine(rng: np.random.Generator, dst: Path) -> Path:
    arr = _camera_image(rng)
    return _save_jpeg(arr, dst, quality=int(rng.integers(85, 96)))


def make_tampered(rng: np.random.Generator, dst: Path) -> Path:
    """Splice + double-JPEG (the canonical forensic adversary)."""
    arr = _camera_image(rng)
    other = _camera_image(rng)
    h, w, _ = arr.shape
    bh, bw = int(rng.integers(50, 110)), int(rng.integers(50, 110))
    y, x = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    arr[y : y + bh, x : x + bw] = other[y : y + bh, x : x + bw]
    return _double_jpeg(arr, dst, q1=int(rng.integers(60, 78)), q2=int(rng.integers(86, 95)))


def make_copy_move(rng: np.random.Generator, dst: Path) -> Path:
    arr = _camera_image(rng)
    h, w, _ = arr.shape
    bh, bw = 64, 64
    y1, x1 = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    y2, x2 = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    while abs(y1 - y2) + abs(x1 - x2) < 96:
        y2, x2 = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    arr[y2 : y2 + bh, x2 : x2 + bw] = arr[y1 : y1 + bh, x1 : x1 + bw]
    return _save_jpeg(arr, dst, quality=int(rng.integers(85, 95)))


def make_recompressed(rng: np.random.Generator, dst: Path) -> Path:
    """Negative control: re-saved with no edits."""
    arr = _camera_image(rng)
    return _double_jpeg(arr, dst, q1=int(rng.integers(78, 86)), q2=int(rng.integers(86, 95)))


def build_dataset(
    out_dir: Path,
    n_per_class: int = 25,
    seed: int = 0,
) -> list[tuple[Path, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples: list[tuple[Path, str]] = []
    for i in range(n_per_class):
        samples.append((make_pristine(rng, out_dir / f"clean_{i:03d}.jpg"), "pristine"))
    for i in range(n_per_class):
        samples.append((make_tampered(rng, out_dir / f"splice_{i:03d}.jpg"), "tampered"))
    for i in range(n_per_class):
        samples.append((make_copy_move(rng, out_dir / f"cmove_{i:03d}.jpg"), "tampered"))
    for i in range(n_per_class):
        samples.append((make_recompressed(rng, out_dir / f"recomp_{i:03d}.jpg"), "pristine"))
    return samples
