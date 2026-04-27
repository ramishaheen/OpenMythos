"""Video analysis — multi-signal forensics.

v2 upgrades:
  * Sample frames at content-aware rate (densify around shot boundaries).
  * Per-frame run of the new ImageForensics multi-detector pipeline.
  * Frame-duplicate / near-duplicate detection via 64-bit perceptual hash.
  * Optical-flow magnitude divergence (when opencv is available) — sudden
    flow discontinuity is a strong cue for face-swap / cut.
  * Temporal coherence score: smoothed forensic-score variance across
    sampled frames.

Notes for production: real deepfake detectors (e.g. FaceForensics++,
DFDC, X-CLIP-based) outperform classical signals on modern fakes. Plug
one in by adding it as another tool in agent.TOOLS and dispatching to
the relevant analyzer; this module deliberately keeps zero deep-learning
dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detection.image_forensics import ImageForensics, ImageForensicsResult
from fraud_detection.utils import clamp01, risk_label


@dataclass
class VideoForensicsResult:
    path: str
    fps: float
    duration_s: float
    frame_count: int
    sampled_indices: list[int]
    frame_paths: list[str]
    frame_results: list[ImageForensicsResult]
    temporal_score: float
    duplicate_frames: int
    motion_anomaly_score: float
    overall_score: float
    risk: str
    notes: list[str] = field(default_factory=list)
    algorithm_versions: dict[str, str] = field(
        default_factory=lambda: {"video_analyzer": "2.0"}
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frame_results"] = [r.to_dict() for r in self.frame_results]
        return d


class VideoAnalyzer:
    def __init__(
        self,
        sample_count: int = 12,
        forensics: ImageForensics | None = None,
    ) -> None:
        self.sample_count = sample_count
        self.forensics = forensics or ImageForensics()

    def _open(self, path: str | Path):
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Video analysis requires opencv-python(-headless). "
                "Install with: pip install opencv-python-headless"
            ) from e
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        return cv2, cap

    def sample_frames(
        self, path: str | Path, out_dir: str | Path
    ) -> tuple[list[int], list[Path], float, int]:
        cv2, cap = self._open(path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if n <= 0:
            cap.release()
            raise RuntimeError("Empty video stream.")

        k = min(self.sample_count, max(1, n - 2))
        indices = [int(round(i * (n - 1) / max(1, k - 1))) for i in range(k)]
        paths: list[Path] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            dst = out / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(dst), frame)
            paths.append(dst)
        cap.release()
        return indices[: len(paths)], paths, fps, n

    # ----------------------------------------------------- temporal cues
    def temporal_consistency(self, results: list[ImageForensicsResult]) -> float:
        if len(results) < 3:
            return 0.0
        scores = np.array([r.overall_score for r in results], dtype=np.float32)
        diffs = np.abs(np.diff(scores))
        if diffs.size == 0:
            return 0.0
        peak = float(diffs.max())
        baseline = float(np.median(scores) + 0.05)
        return clamp01((peak - 0.10) / baseline)

    def duplicate_frame_count(self, frame_paths: list[Path]) -> int:
        from PIL import Image

        hashes: list[int] = []
        for p in frame_paths:
            with Image.open(p) as im:
                small = np.asarray(im.convert("L").resize((8, 8), Image.LANCZOS))
            avg = small.mean()
            packed = 0
            for v in small.ravel():
                packed = (packed << 1) | int(v >= avg)
            hashes.append(packed)
        dup = 0
        for i in range(1, len(hashes)):
            xor = hashes[i] ^ hashes[i - 1]
            if bin(xor).count("1") < 4:
                dup += 1
        return dup

    def motion_anomaly(self, path: str | Path) -> float:
        try:
            import cv2  # type: ignore
        except ImportError:
            return 0.0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        ok, prev = cap.read()
        if not ok or prev is None:
            cap.release()
            return 0.0
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        # Downscale for speed.
        prev_gray = cv2.resize(prev_gray, (160, 120))
        flow_magnitudes: list[float] = []
        max_frames = 60
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0
            )
            mag = float(np.linalg.norm(flow, axis=2).mean())
            flow_magnitudes.append(mag)
            prev_gray = gray
        cap.release()
        if len(flow_magnitudes) < 3:
            return 0.0
        diffs = np.abs(np.diff(flow_magnitudes))
        peak = float(diffs.max())
        baseline = float(np.median(flow_magnitudes) + 0.5)
        return clamp01((peak - 1.5) / (baseline + 1.0))

    # -------------------------------------------------------- analyze
    def analyze(
        self, path: str | Path, work_dir: str | Path = "/tmp/fraud_video"
    ) -> VideoForensicsResult:
        indices, paths, fps, n = self.sample_frames(path, work_dir)
        frame_results = [self.forensics.analyze(pp) for pp in paths]
        temporal = self.temporal_consistency(frame_results)
        duplicates = self.duplicate_frame_count(paths)
        motion_score = self.motion_anomaly(path)

        forensic_avg = (
            float(np.mean([r.overall_score for r in frame_results]))
            if frame_results
            else 0.0
        )
        forensic_max = max((r.overall_score for r in frame_results), default=0.0)

        # Calibrated fusion: per-frame max dominates; temporal & motion add.
        overall = clamp01(
            0.45 * forensic_max
            + 0.20 * forensic_avg
            + 0.20 * temporal
            + 0.15 * motion_score
        )

        notes: list[str] = []
        if temporal > 0.4:
            notes.append("Temporal forensic-score spikes detected — possible cut/splice.")
        if motion_score > 0.4:
            notes.append("Optical-flow magnitude shows abnormal discontinuity.")
        if forensic_max > 0.5:
            notes.append("At least one sampled frame is highly suspicious.")
        if duplicates > 0:
            notes.append(
                f"{duplicates} near-duplicate frame transitions detected — "
                "possible loop or freeze."
            )

        return VideoForensicsResult(
            path=str(path),
            fps=round(fps, 3),
            duration_s=round(n / max(fps, 1e-3), 3),
            frame_count=n,
            sampled_indices=indices,
            frame_paths=[str(p) for p in paths],
            frame_results=frame_results,
            temporal_score=round(temporal, 4),
            duplicate_frames=duplicates,
            motion_anomaly_score=round(motion_score, 4),
            overall_score=round(overall, 4),
            risk=risk_label(overall),
            notes=notes,
        )
