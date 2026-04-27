"""Tiny sanity check that the calibration pipeline produces a valid report."""

from fraud_detection.calibration import calibrate


def test_calibrate_smoke() -> None:
    report, _ = calibrate(n_per_class=4, seed=0)
    assert report.n_samples == 16
    # Weights sum to ~1 and every detector got at least the floor weight.
    assert abs(sum(report.weights.values()) - 1.0) < 1e-3
    for w in report.weights.values():
        assert w >= 0.029  # MIN_WEIGHT - rounding
    # Thresholds are ordered.
    t = report.thresholds
    assert t["low"] <= t["medium"] <= t["high"]
