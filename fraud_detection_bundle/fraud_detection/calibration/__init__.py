"""Synthetic calibration for fusion weights and risk thresholds.

Run as a module:

    python -m fraud_detection.calibration

Generates a labeled synthetic dataset, runs every detector, and prints
calibrated weights + thresholds + per-detector AUC. The script also writes
``calibration_report.json`` next to itself so the result is auditable.
"""

from fraud_detection.calibration.run import calibrate, main

__all__ = ["calibrate", "main"]
