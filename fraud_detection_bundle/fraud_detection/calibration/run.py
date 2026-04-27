"""Calibrate fusion weights and risk thresholds against synthetic data.

Method:
  1. Build the synthetic dataset (pristine vs. tampered).
  2. Run every detector on every sample → matrix X (N x D), labels y.
  3. Per-detector AUC = how well that single signal separates classes.
  4. Per-detector weight ∝ max(0, AUC - 0.5) — chance-level detectors get 0.
  5. Fused score on training set → choose risk thresholds at empirical
     quantiles (low=80th of pristine, medium=Youden-optimal, high=95th
     of tampered).

This is a *baseline* calibration; production deployments should re-run
against labeled domain data and keep the resulting JSON in version control
as part of chain-of-custody.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fraud_detection.calibration.synthetic import build_dataset
from fraud_detection.detectors import (
    BenfordDCTDetector,
    CFAInconsistencyDetector,
    ErrorLevelAnalysisDetector,
    JPEGQuantizationDetector,
    LightingConsistencyDetector,
    NoiseResidueDetector,
    PerceptualCopyMoveDetector,
)
from fraud_detection.detectors.base import safe_run

DETECTOR_FACTORIES = [
    ErrorLevelAnalysisDetector,
    JPEGQuantizationDetector,
    BenfordDCTDetector,
    CFAInconsistencyDetector,
    PerceptualCopyMoveDetector,
    NoiseResidueDetector,
    LightingConsistencyDetector,
]

# Literature-informed priors. Blended with synthetic-AUC weights so the
# production weights don't collapse onto a single detector that happens to
# dominate on synthetic data.
LITERATURE_PRIOR: dict[str, float] = {
    "ela": 0.18,
    "jpeg_qtable": 0.16,
    "benford_dct": 0.10,
    "cfa_inconsistency": 0.18,
    "copy_move_phash": 0.16,
    "prnu_consistency": 0.17,
    "lighting_consistency": 0.05,
}
PRIOR_BLEND = 0.6  # 60% prior, 40% synthetic AUC
MIN_WEIGHT = 0.03  # never zero-weight a detector — it may still help on real data


@dataclass
class CalibrationReport:
    n_samples: int
    detectors: list[str]
    auc: dict[str, float]
    weights: dict[str, float]
    thresholds: dict[str, float]
    score_stats: dict[str, dict[str, float]]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U-statistic AUC. labels: 1 = positive (tampered)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Cheap O(N log N) AUC via rank.
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort().astype(np.float64) + 1.0
    rank_pos = ranks[: pos.size].sum()
    auc = (rank_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def calibrate(
    n_per_class: int = 25, seed: int = 0
) -> tuple[CalibrationReport, list[Path]]:
    notes: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="fraud_calib_"))
    samples = build_dataset(work, n_per_class=n_per_class, seed=seed)

    detectors = [factory() for factory in DETECTOR_FACTORIES]
    names = [d.name for d in detectors]
    n = len(samples)
    X = np.zeros((n, len(detectors)), dtype=np.float64)
    y = np.zeros(n, dtype=np.int32)

    for i, (path, label) in enumerate(samples):
        y[i] = 1 if label == "tampered" else 0
        for j, det in enumerate(detectors):
            res = safe_run(det, path)
            X[i, j] = res.score * (res.confidence if res.confidence > 0 else 1.0)

    auc = {name: _auc(X[:, j], y) for j, name in enumerate(names)}

    # Convert AUC to weights. Map [0.5, 1.0] → [0, 1], floor at 0.
    raw = np.array([max(0.0, auc[n] - 0.5) * 2.0 for n in names], dtype=np.float64)
    if raw.sum() == 0:
        notes.append("No detector beat chance on synthetic data — using prior only.")
        synthetic_weights = {n: LITERATURE_PRIOR[n] for n in names}
    else:
        norm = raw / raw.sum()
        synthetic_weights = {n: float(w) for n, w in zip(names, norm)}

    # Blend prior + synthetic AUC, then floor at MIN_WEIGHT and renormalise.
    blended = {
        n: PRIOR_BLEND * LITERATURE_PRIOR[n]
        + (1.0 - PRIOR_BLEND) * synthetic_weights[n]
        for n in names
    }
    blended = {n: max(MIN_WEIGHT, w) for n, w in blended.items()}
    total = sum(blended.values())
    weights = {n: round(w / total, 4) for n, w in blended.items()}
    notes.append(
        f"Blended weights: {int(PRIOR_BLEND * 100)}% literature prior + "
        f"{int((1 - PRIOR_BLEND) * 100)}% synthetic AUC, floor {MIN_WEIGHT}."
    )

    # Fused score on training set with these weights.
    w_vec = np.array([weights[n] for n in names], dtype=np.float64)
    fused = X @ w_vec

    # Threshold selection. Use class-conditional quantiles.
    pos_scores = fused[y == 1]
    neg_scores = fused[y == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        thresholds = {"low": 0.18, "medium": 0.40, "high": 0.62}
    else:
        # Youden's J at the medium threshold.
        all_thr = np.linspace(0, 1, 200)
        tpr = np.array([(pos_scores >= t).mean() for t in all_thr])
        fpr = np.array([(neg_scores >= t).mean() for t in all_thr])
        j = tpr - fpr
        medium = float(all_thr[int(np.argmax(j))])
        # Bands: low at 80th percentile of pristine; high at 95th of pristine
        # OR 50th of tampered (whichever is higher) — keeps high stringent.
        low = float(np.percentile(neg_scores, 80))
        high = float(max(np.percentile(neg_scores, 95), np.percentile(pos_scores, 50)))
        # Enforce ordering.
        low = min(low, medium - 0.05) if medium - 0.05 > 0 else low
        high = max(high, medium + 0.10)
        thresholds = {
            "low": round(low, 4),
            "medium": round(medium, 4),
            "high": round(high, 4),
        }

    score_stats = {
        "pristine": {
            "mean": round(float(neg_scores.mean()), 4) if neg_scores.size else 0.0,
            "p95": round(float(np.percentile(neg_scores, 95)), 4) if neg_scores.size else 0.0,
        },
        "tampered": {
            "mean": round(float(pos_scores.mean()), 4) if pos_scores.size else 0.0,
            "p05": round(float(np.percentile(pos_scores, 5)), 4) if pos_scores.size else 0.0,
        },
    }
    auc_overall = _auc(fused, y)
    notes.append(f"Overall fused AUC on synthetic split: {auc_overall:.3f}")

    report = CalibrationReport(
        n_samples=n,
        detectors=names,
        auc={k: round(v, 4) for k, v in auc.items()},
        weights=weights,
        thresholds=thresholds,
        score_stats=score_stats,
        notes=notes,
    )
    return report, [p for p, _ in samples]


def main() -> int:
    print("Building synthetic dataset and running detectors…")
    report, _ = calibrate(n_per_class=25, seed=0)
    out = Path(__file__).parent / "calibration_report.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
