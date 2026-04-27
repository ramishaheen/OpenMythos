"""Detector base type. Each detector returns a calibrated 0-1 score plus
structured evidence the agent can cite verbatim.

Court-grade caveat: scores are calibrated against synthetic data. Real
forensic deployments should re-calibrate against a domain-specific labeled
set and document the calibration as part of the chain of evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class DetectorResult:
    name: str
    version: str
    score: float
    confidence: float  # 0-1 — how trustworthy the score is on this input
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Detector(Protocol):
    name: str
    version: str

    def run(self, image_path: str | Path) -> DetectorResult: ...


def safe_run(detector: Detector, image_path: str | Path) -> DetectorResult:
    """Run a detector and capture exceptions as a structured failure rather
    than a crash — important for forensic pipelines so one bad input doesn't
    sink the whole report."""
    try:
        return detector.run(image_path)
    except Exception as e:  # noqa: BLE001
        return DetectorResult(
            name=getattr(detector, "name", detector.__class__.__name__),
            version=getattr(detector, "version", "unknown"),
            score=0.0,
            confidence=0.0,
            evidence={"input": str(image_path)},
            notes=[f"Detector raised {type(e).__name__}; treating as no-evidence."],
            error=f"{type(e).__name__}: {e}",
        )
