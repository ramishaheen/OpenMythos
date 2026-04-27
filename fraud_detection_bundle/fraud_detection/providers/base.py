"""Provider interface.

A provider takes a structured ProviderInput (detector evidence + manifest)
and returns a ProviderResult (verdict / score / summary / evidence /
recommendations). It does NOT run the detectors itself — those are
already deterministic Python and run before the provider is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Verdict = Literal["authentic", "suspicious", "manipulated", "inconclusive"]


@dataclass
class InputManifest:
    """One row per analyzed input — kind, label, sha256, scores, detectors."""

    path: str
    kind: str
    label: str | None
    sha256: str | None
    size_bytes: int | None
    overall_score: float | None
    risk: str | None
    detectors: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    image_b64: str | None = None  # only attached for vision providers
    image_media_type: str | None = None


@dataclass
class ProviderInput:
    """Everything the LLM layer gets to look at."""

    inputs: list[InputManifest]
    aggregate_score: float
    aggregate_risk: str
    context: str | None = None
    algorithm_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """LLM output, normalised."""

    verdict: Verdict
    score: float
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None  # provider-native response, for debugging


class Provider(Protocol):
    """Provider protocol."""

    name: str
    supports_vision: bool

    def configured(self) -> bool: ...
    def summarize(self, payload: ProviderInput) -> ProviderResult: ...
