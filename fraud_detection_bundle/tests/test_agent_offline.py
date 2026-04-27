"""Verify the agent runs end-to-end without network access (offline fallback)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fraud_detection import FraudDetectionAgent, FraudInput


@pytest.fixture
def img(tmp_path: Path) -> Path:
    arr = np.random.default_rng(7).integers(0, 255, size=(128, 128, 3), dtype="uint8")
    p = tmp_path / "noise.jpg"
    Image.fromarray(arr).save(p, format="JPEG", quality=90)
    return p


def test_agent_offline_image(img: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force offline path even if anthropic is installed in the env.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = FraudDetectionAgent(api_key=None)
    report = agent.run([FraudInput(path=str(img), kind="image")])
    assert report.verdict in ("authentic", "suspicious", "manipulated", "inconclusive")
    assert any(e["source"] == "image_forensics" for e in report.evidence)
    assert report.model is None  # offline path
