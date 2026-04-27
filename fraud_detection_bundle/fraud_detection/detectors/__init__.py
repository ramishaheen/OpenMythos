"""Forensic detectors — peer-reviewed primitives, each returning a 0-1 score
plus structured evidence. Detectors are intentionally independent so they can
be unit-tested and re-weighted without touching the orchestrator.

Each detector exposes:

    name: str
    version: str
    def run(image_path) -> DetectorResult

DetectorResult.score is calibrated so that 0 = pristine, 1 = highly suspicious.
"""

from fraud_detection.detectors.base import DetectorResult, Detector
from fraud_detection.detectors.benford import BenfordDCTDetector
from fraud_detection.detectors.cfa import CFAInconsistencyDetector
from fraud_detection.detectors.copy_move import PerceptualCopyMoveDetector
from fraud_detection.detectors.ela import ErrorLevelAnalysisDetector
from fraud_detection.detectors.jpeg_qtable import JPEGQuantizationDetector
from fraud_detection.detectors.lighting import LightingConsistencyDetector
from fraud_detection.detectors.noise_residue import NoiseResidueDetector

__all__ = [
    "Detector",
    "DetectorResult",
    "BenfordDCTDetector",
    "CFAInconsistencyDetector",
    "PerceptualCopyMoveDetector",
    "ErrorLevelAnalysisDetector",
    "JPEGQuantizationDetector",
    "LightingConsistencyDetector",
    "NoiseResidueDetector",
]
