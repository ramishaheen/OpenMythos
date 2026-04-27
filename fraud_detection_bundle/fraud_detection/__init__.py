"""Fraud Detection — document, image, video, and signature forensics.

A Claude-powered agent that orchestrates a suite of forensic analyzers to
detect falsification and manipulation. Designed to be dropped into the
MythosBanking codebase as a self-contained module.

Inspired by:
  - IFAKE (image forgery detection)
  - sainipankaj15/Signature-Forgery-Detection
  - GitHub topic: signature-detection
"""

from fraud_detection.agent import FraudDetectionAgent, FraudInput, FraudReport
from fraud_detection.document_analyzer import DocumentAnalyzer, DocumentFinding
from fraud_detection.image_forensics import ImageForensics, ImageForensicsResult
from fraud_detection.signature_verifier import SignatureVerifier, SignatureVerdict
from fraud_detection.video_analyzer import VideoAnalyzer, VideoForensicsResult

__all__ = [
    "FraudDetectionAgent",
    "FraudInput",
    "FraudReport",
    "DocumentAnalyzer",
    "DocumentFinding",
    "ImageForensics",
    "ImageForensicsResult",
    "SignatureVerifier",
    "SignatureVerdict",
    "VideoAnalyzer",
    "VideoForensicsResult",
]

__version__ = "0.1.0"
