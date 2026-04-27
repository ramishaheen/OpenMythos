"""Video analysis — frame sampling, per-frame ELA, temporal consistency.

Looks for splice/face-swap/deepfake footprints by:
  - Sampling N evenly spaced keyframes (or change-detected frames).
  - Running ImageForensics on each.
  - Measuring temporal residual jumps (sudden ELA spikes between frames
    are a classic deepfake / cut tell).

Requires opencv-python at runtime.
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
    overall_score: float
    risk: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frame_results"] = [r.to_dict() for r in self.frame_results]
        return d


class VideoAnalyzer:
    def __init__(
        self,
        sample_count: int = 8,
        forensics: ImageForensics | None = None,
    ) -> None:
        self.sample_count = sample_count
        self.forensics = forensics or ImageForensics()

    def _open(self, path: str | Path):
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Video analysis requires opencv-python. "
                "Install with: pip install opencv-python"
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

        # Even spacing, but skip the first/last frame to avoid black borders.
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

    def temporal_consistency(
        self, results: list[ImageForensicsResult]
    ) -> float:
        """Score sudden inter-frame ELA jumps. 0 = smooth, 1 = noisy."""
        if len(results) < 2:
            return 0.0
        elas = np.array([r.ela_score for r in results], dtype=np.float32)
        diffs = np.abs(np.diff(elas))
        if diffs.size == 0:
            return 0.0
        # Normalise by range — large jumps relative to baseline.
        peak = float(diffs.max())
        baseline = float(np.median(elas) + 1e-3)
        return clamp01(peak / (baseline + 0.1))

    def analyze(
        self, path: str | Path, work_dir: str | Path = "/tmp/fraud_video"
    ) -> VideoForensicsResult:
        indices, paths, fps, n = self.sample_frames(path, work_dir)
        results = [self.forensics.analyze(pp) for pp in paths]
        temporal = self.temporal_consistency(results)
        forensic_avg = float(np.mean([r.overall_score for r in results])) if results else 0.0
        forensic_max = max((r.overall_score for r in results), default=0.0)

        overall = clamp01(0.5 * forensic_max + 0.3 * forensic_avg + 0.2 * temporal)

        notes: list[str] = []
        if temporal > 0.4:
            notes.append("Temporal ELA spikes detected — possible cuts or splices.")
        if forensic_max > 0.5:
            notes.append("At least one sampled frame is highly suspicious.")

        return VideoForensicsResult(
            path=str(path),
            fps=round(fps, 3),
            duration_s=round(n / max(fps, 1e-3), 3),
            frame_count=n,
            sampled_indices=indices,
            frame_paths=[str(p) for p in paths],
            frame_results=results,
            temporal_score=round(temporal, 4),
            overall_score=round(overall, 4),
            risk=risk_label(overall),
            notes=notes,
        )
