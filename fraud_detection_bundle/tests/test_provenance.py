"""Verify that the agent's report includes forensic provenance.

Every report must carry:
  * input SHA-256 + MIME + size
  * a per-detector score breakdown
  * an algorithm-version manifest
  * a deterministic reproducibility hash that changes when inputs change
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fraud_detection import FraudDetectionAgent, FraudInput


@pytest.fixture
def img(tmp_path: Path) -> Path:
    arr = np.random.default_rng(7).integers(0, 255, size=(160, 160, 3), dtype="uint8")
    p = tmp_path / "noise.jpg"
    Image.fromarray(arr).save(p, "JPEG", quality=90)
    return p


def test_report_has_provenance(img: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = FraudDetectionAgent(api_key=None)
    report = agent.run([FraudInput(path=str(img), kind="image")])

    assert report.reproducibility_hash and len(report.reproducibility_hash) == 64
    assert report.generated_at
    assert report.algorithm_versions
    assert report.detector_results, "detector_results should be populated"
    assert report.chain_of_evidence
    entry = report.chain_of_evidence[0]
    assert "sha256" in entry
    assert "size_bytes" in entry
    assert "detectors" in entry
    detectors = entry["detectors"]
    assert {"ela", "jpeg_qtable"} <= {d["name"] for d in detectors}


def test_reproducibility_hash_changes_with_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = FraudDetectionAgent(api_key=None)
    rng = np.random.default_rng(1)
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype="uint8")).save(a, "JPEG")
    Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype="uint8")).save(b, "JPEG")
    r1 = agent.run([FraudInput(path=str(a), kind="image")])
    r2 = agent.run([FraudInput(path=str(b), kind="image")])
    assert r1.reproducibility_hash != r2.reproducibility_hash
