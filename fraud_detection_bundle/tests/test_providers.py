"""Provider abstraction tests — uses fakes for the LLM providers so we
don't need an API key to run them."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from fraud_detection import FraudDetectionAgent, FraudInput
from fraud_detection.providers import (
    AnthropicProvider,
    DeepSeekProvider,
    OfflineProvider,
    Provider,
    ProviderInput,
    ProviderResult,
    build_provider,
)


@pytest.fixture
def img(tmp_path: Path) -> Path:
    arr = np.random.default_rng(7).integers(0, 255, size=(160, 160, 3), dtype="uint8")
    p = tmp_path / "noise.jpg"
    Image.fromarray(arr).save(p, "JPEG", quality=90)
    return p


# ---------------------------------------------------------------- offline
def test_offline_provider_runs(img: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    a = FraudDetectionAgent(provider="offline")
    r = a.run([FraudInput(path=str(img), kind="image")])
    assert r.provider == "offline"
    assert r.verdict in ("authentic", "suspicious", "manipulated", "inconclusive")
    assert r.evidence
    assert r.algorithm_versions
    assert r.reproducibility_hash and len(r.reproducibility_hash) == 64


# ---------------------------------------------------------------- factory
def test_factory_falls_back_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = build_provider("anthropic", api_key=None)
    assert isinstance(p, OfflineProvider), "should fall back when no key"
    p = build_provider("deepseek", api_key=None)
    assert isinstance(p, OfflineProvider), "should fall back when no key"


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("not-a-provider")


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown anthropic model"):
        build_provider("anthropic", api_key="sk-ant-x", model="claude-1-anything")


# ---------------------------------------------------------------- supports_vision
def test_supports_vision_flag() -> None:
    assert AnthropicProvider().supports_vision is True
    assert DeepSeekProvider().supports_vision is False
    assert OfflineProvider().supports_vision is False


# ---------------------------------------------------------------- fake provider
class _FakeProvider:
    """Capture the ProviderInput so we can assert on what the agent built."""

    name = "fake"
    supports_vision = False

    def __init__(self) -> None:
        self.captured: ProviderInput | None = None

    def configured(self) -> bool:
        return True

    def summarize(self, payload: ProviderInput) -> ProviderResult:
        self.captured = payload
        return ProviderResult(
            verdict="suspicious",
            score=0.42,
            summary="(fake) summary",
            evidence=[{"source": "fake", "claim": "synthetic", "support": "n/a"}],
            recommendations=["fake-rec"],
            raw={"model": "fake"},
        )


def test_agent_passes_manifest_to_provider(
    img: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = _FakeProvider()
    a = FraudDetectionAgent(provider=fake)
    r = a.run([FraudInput(path=str(img), kind="image", label="passport")])

    # Provider received an InputManifest with detector breakdown.
    assert fake.captured is not None
    assert len(fake.captured.inputs) == 1
    m = fake.captured.inputs[0]
    assert m.kind == "image"
    assert m.label == "passport"
    assert m.sha256
    assert m.size_bytes
    assert isinstance(m.detectors, list) and m.detectors
    # Vision flag respected — fake doesn't support vision.
    assert m.image_b64 is None

    # Agent surfaces provider output verbatim.
    assert r.verdict == "suspicious"
    assert r.score == 0.42
    assert r.evidence[0]["source"] == "fake"
    assert "fake-rec" in r.recommendations
    assert r.provider == "fake"


def test_agent_attaches_image_for_vision_provider(
    img: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = _FakeProvider()
    fake.supports_vision = True  # type: ignore[assignment]
    a = FraudDetectionAgent(provider=fake)
    a.run([FraudInput(path=str(img), kind="image")])
    assert fake.captured is not None
    m = fake.captured.inputs[0]
    assert m.image_b64 is not None
    assert m.image_media_type == "image/jpeg"


def test_agent_recovers_from_provider_crash(
    img: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misbehaving provider should not poison the report — the agent
    falls back to the deterministic offline summary and notes the failure."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class Boom:
        name = "boom"
        supports_vision = False

        def configured(self) -> bool:
            return True

        def summarize(self, payload: ProviderInput) -> ProviderResult:
            raise RuntimeError("synthetic provider failure")

    a = FraudDetectionAgent(provider=Boom())
    r = a.run([FraudInput(path=str(img), kind="image")])
    assert r.verdict in ("authentic", "suspicious", "manipulated", "inconclusive")
    assert any("Provider boom failed" in rec for rec in r.recommendations)
