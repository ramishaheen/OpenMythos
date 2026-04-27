"""Image forensics — multi-detector fusion with calibrated weights.

Replaces the v1 ELA-only + crude-copy-move pipeline. Now runs the
peer-reviewed detector suite in ``fraud_detection.detectors`` and fuses
their scores with weights calibrated against a synthetic clean/tampered
split (see ``fraud_detection.calibration``).

Each fused report exposes:
  * per-detector score, confidence, evidence (for chain-of-evidence)
  * EXIF and metadata flags
  * an aggregated forgery score with risk band
  * an algorithm-version manifest for reproducibility
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from fraud_detection.detectors import (
    BenfordDCTDetector,
    CFAInconsistencyDetector,
    Detector,
    DetectorResult,
    ErrorLevelAnalysisDetector,
    JPEGQuantizationDetector,
    LightingConsistencyDetector,
    NoiseResidueDetector,
    PerceptualCopyMoveDetector,
)
from fraud_detection.detectors.base import safe_run
from fraud_detection.utils import clamp01, risk_label

# Calibrated fusion weights — see fraud_detection.calibration. Blended from a
# 60% literature prior + 40% synthetic-data AUC. Re-calibrate against real
# labelled data for production deployments and persist the resulting JSON
# alongside the report (chain-of-custody requirement).
DEFAULT_WEIGHTS: dict[str, float] = {
    "ela": 0.4834,
    "jpeg_qtable": 0.0960,
    "benford_dct": 0.0600,
    "cfa_inconsistency": 0.1326,
    "copy_move_phash": 0.0960,
    "prnu_consistency": 0.1020,
    "lighting_consistency": 0.0300,
}

# Empirical risk thresholds learned on the synthetic split. Conservative.
# A real deployment should re-run calibration and update these.
DEFAULT_THRESHOLDS = {"high": 0.32, "medium": 0.22, "low": 0.17}


@dataclass
class ImageForensicsResult:
    path: str
    detector_results: list[dict[str, Any]]
    metadata: dict[str, Any]
    metadata_flags: list[str]
    fusion_weights: dict[str, float]
    overall_score: float
    risk: str
    notes: list[str] = field(default_factory=list)
    algorithm_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageForensics:
    """Court-aware image-forensics pipeline. Stateless and deterministic."""

    def __init__(
        self,
        detectors: list[Detector] | None = None,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.detectors: list[Detector] = detectors or [
            ErrorLevelAnalysisDetector(),
            JPEGQuantizationDetector(),
            BenfordDCTDetector(),
            CFAInconsistencyDetector(),
            PerceptualCopyMoveDetector(),
            NoiseResidueDetector(),
            LightingConsistencyDetector(),
        ]
        self.weights = weights or DEFAULT_WEIGHTS
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    # ------------------------------------------------------------ EXIF
    def metadata_inspection(
        self, path: str | Path
    ) -> tuple[dict[str, Any], list[str]]:
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
        for needle in ("photoshop", "gimp", "lightroom", "affinity", "snapseed", "midjourney", "dall-e", "stable diffusion"):
            if needle in software:
                flags.append(f"editor_signature:{needle}")

        if not (meta.get("Make") and meta.get("Model")):
            flags.append("missing_camera_make_model")

        if "DateTime" in meta and "DateTimeOriginal" in meta:
            if meta["DateTime"] != meta["DateTimeOriginal"]:
                flags.append("datetime_mismatch")

        return meta, flags

    # ---------------------------------------------------------- compose
    def analyze(self, path: str | Path) -> ImageForensicsResult:
        meta, meta_flags = self.metadata_inspection(path)
        results: list[DetectorResult] = [safe_run(d, path) for d in self.detectors]

        # Confidence-weighted fusion.
        weighted_sum = 0.0
        weight_total = 0.0
        for r in results:
            w = self.weights.get(r.name, 0.0) * r.confidence
            if w <= 0:
                continue
            weighted_sum += w * r.score
            weight_total += w
        forensic_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Metadata penalty: capped at 0.15 so EXIF alone can't dominate.
        meta_penalty = min(0.15, 0.05 * sum(1 for f in meta_flags if "editor_signature" in f) + 0.02 * len(meta_flags))
        score = clamp01(0.85 * forensic_score + meta_penalty)

        notes: list[str] = []
        for r in results:
            for n in r.notes:
                notes.append(f"[{r.name}] {n}")
        for f in meta_flags:
            notes.append(f"[metadata] flag: {f}")

        return ImageForensicsResult(
            path=str(path),
            detector_results=[r.to_dict() for r in results],
            metadata=meta,
            metadata_flags=meta_flags,
            fusion_weights=dict(self.weights),
            overall_score=round(score, 4),
            risk=_risk(score, self.thresholds),
            notes=notes,
            algorithm_versions={r.name: r.version for r in results},
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


def _risk(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds.get("high", 0.62):
        return "high"
    if score >= thresholds.get("medium", 0.40):
        return "medium"
    if score >= thresholds.get("low", 0.18):
        return "low"
    return "minimal"


# Back-compat shim: export the old risk_label name from utils.
__all__ = ["ImageForensics", "ImageForensicsResult", "DEFAULT_WEIGHTS", "DEFAULT_THRESHOLDS", "risk_label"]
