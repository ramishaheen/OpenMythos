"""Fraud-detection agent — provider-agnostic orchestrator.

Runs the local forensic detectors (deterministic), assembles a
ProviderInput, and delegates the reasoning step to a pluggable provider:

  * AnthropicProvider — Claude with vision corroboration.
  * DeepSeekProvider  — DeepSeek text-only over the detector JSON.
  * OfflineProvider   — deterministic fusion, no LLM.

The public surface is:
    agent = FraudDetectionAgent(provider="anthropic", api_key=..., model=...)
    report = agent.run([FraudInput(path="x.jpg", kind="image")], context="...")
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from fraud_detection.document_analyzer import DocumentAnalyzer
from fraud_detection.image_forensics import ImageForensics
from fraud_detection.providers import (
    InputManifest,
    Provider,
    ProviderInput,
    ProviderResult,
    build_provider,
)
from fraud_detection.providers.factory import PROVIDER_DEFAULTS
from fraud_detection.signature_verifier import SignatureVerifier
from fraud_detection.utils import (
    MediaKind,
    clamp01,
    detect_kind,
    encode_image_b64,
    file_meta,
    risk_label,
)
from fraud_detection.video_analyzer import VideoAnalyzer

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = PROVIDER_DEFAULTS["anthropic"]["default_model"]


# --------------------------------------------------------------------- types
@dataclass
class FraudInput:
    path: str
    kind: MediaKind = "unknown"
    reference_path: str | None = None
    label: str | None = None

    def resolved_kind(self) -> MediaKind:
        return detect_kind(self.path, self.kind)


@dataclass
class FraudReport:
    verdict: Literal["authentic", "suspicious", "manipulated", "inconclusive"]
    score: float
    risk: str
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    chain_of_evidence: list[dict[str, Any]] = field(default_factory=list)
    algorithm_versions: dict[str, str] = field(default_factory=dict)
    detector_results: list[dict[str, Any]] = field(default_factory=list)
    reproducibility_hash: str | None = None
    generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------- the agent
class FraudDetectionAgent:
    """Provider-agnostic orchestrator. Use ``run`` for synchronous analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | Provider | None = None,
        image_forensics: ImageForensics | None = None,
        document_analyzer: DocumentAnalyzer | None = None,
        video_analyzer: VideoAnalyzer | None = None,
        signature_verifier: SignatureVerifier | None = None,
    ) -> None:
        # Resolve provider. A pre-built Provider instance wins; otherwise
        # build one from the (name, api_key, model) tuple.
        if isinstance(provider, str) or provider is None:
            self.provider_name = (provider or DEFAULT_PROVIDER).lower()
            # api_key heuristics: explicit > env (per-provider) > none.
            resolved_key = api_key or _env_key_for(self.provider_name)
            self.provider: Provider = build_provider(
                self.provider_name,
                api_key=resolved_key,
                model=model,
            )
        else:
            self.provider = provider
            self.provider_name = provider.name

        # Model is informational on the FraudReport — actual model used
        # lives inside the provider.
        self.model = model or PROVIDER_DEFAULTS.get(
            self.provider_name, {"default_model": "local-fusion"}
        )["default_model"]

        self.image_forensics = image_forensics or ImageForensics()
        self.document_analyzer = document_analyzer or DocumentAnalyzer(self.image_forensics)
        self.video_analyzer = video_analyzer or VideoAnalyzer(forensics=self.image_forensics)
        self.signature_verifier = signature_verifier or SignatureVerifier()

    # ---------------------------------------------------- public entry
    def run(
        self,
        inputs: Iterable[FraudInput | str | dict[str, Any]],
        context: str | None = None,
    ) -> FraudReport:
        normalised = [self._normalise(i) for i in inputs]
        if not normalised:
            raise ValueError("At least one input is required.")

        # 1. Run analyzers locally — provider-independent.
        manifests, chain, det_results, versions, evidence_seed, scores, trace = (
            self._run_detectors(normalised)
        )

        # 2. Aggregate score.
        agg = max(scores, default=0.0)

        # 3. Hand to the provider for reasoning.
        payload = ProviderInput(
            inputs=manifests,
            aggregate_score=agg,
            aggregate_risk=risk_label(agg),
            context=context,
            algorithm_versions=versions,
        )
        try:
            pr: ProviderResult = self.provider.summarize(payload)
        except Exception as e:  # noqa: BLE001
            # Last-resort: fall back to offline so a misbehaving provider
            # never blocks producing *some* report.
            from fraud_detection.providers import OfflineProvider

            pr = OfflineProvider().summarize(payload)
            pr.recommendations.insert(0, f"Provider {self.provider.name} failed: {e}")

        # 4. Compose the report.
        repro = self._reproducibility_hash(normalised, versions, self.provider.name)
        return FraudReport(
            verdict=pr.verdict,
            score=pr.score,
            risk=risk_label(pr.score),
            summary=pr.summary,
            evidence=pr.evidence or evidence_seed,
            recommendations=pr.recommendations,
            tool_trace=trace,
            inputs=[asdict(i) for i in normalised],
            model=self.model if self.provider.name != "offline" else None,
            provider=self.provider.name,
            chain_of_evidence=chain,
            algorithm_versions=versions,
            detector_results=det_results,
            reproducibility_hash=repro,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ---------------------------------------------------- internals
    def _normalise(self, item: FraudInput | str | dict[str, Any]) -> FraudInput:
        if isinstance(item, FraudInput):
            return item
        if isinstance(item, str):
            return FraudInput(path=item)
        if isinstance(item, dict):
            return FraudInput(**item)
        raise TypeError(f"Unsupported input type: {type(item).__name__}")

    def _run_detectors(self, inputs: list[FraudInput]):
        """Run analyzers per input and assemble:

        Returns:
          manifests       list[InputManifest]      — for the provider
          chain           list[dict]               — chain_of_evidence
          det_results     list[dict]               — detector_results
          versions        dict[str, str]           — algorithm_versions
          evidence_seed   list[dict]               — fallback evidence list
          scores          list[float]              — for aggregate
          trace           list[dict]               — tool_trace
        """
        manifests: list[InputManifest] = []
        chain: list[dict[str, Any]] = []
        det_results: list[dict[str, Any]] = []
        versions: dict[str, str] = {}
        evidence_seed: list[dict[str, Any]] = []
        scores: list[float] = []
        trace: list[dict[str, Any]] = []

        for inp in inputs:
            kind = inp.resolved_kind()
            chain_entry: dict[str, Any] = {
                "input": inp.path,
                "kind": kind,
                "label": inp.label,
            }
            sha256: str | None = None
            size_bytes: int | None = None
            mime: str | None = None
            try:
                fm = file_meta(inp.path)
                sha256, size_bytes, mime = fm.sha256, fm.size_bytes, fm.mime
                chain_entry.update(
                    {"sha256": sha256, "size_bytes": size_bytes, "mime": mime}
                )
            except Exception:  # noqa: BLE001
                pass

            try:
                detectors_for_provider: list[dict[str, Any]] = []
                overall: float | None = None
                risk: str | None = None
                extra: dict[str, Any] = {}
                image_b64: str | None = None
                image_mt: str | None = None

                if kind == "image":
                    r = self.image_forensics.analyze(inp.path)
                    overall, risk = r.overall_score, r.risk
                    versions.update(r.algorithm_versions)
                    detectors_for_provider = r.detector_results
                    chain_entry.update(
                        {"overall_score": overall, "risk": risk, "detectors": r.detector_results}
                    )
                    det_results.append(
                        {"input": inp.path, "kind": kind, "results": r.detector_results}
                    )
                    if self.provider.supports_vision:
                        try:
                            image_b64, image_mt = encode_image_b64(inp.path)
                        except Exception:  # noqa: BLE001
                            pass
                    evidence_seed.append(
                        {
                            "source": "image_forensics",
                            "claim": f"{inp.path}: risk={risk} (score={overall})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "run_image_forensics", "args": {"path": inp.path}, "ok": True})

                elif kind == "document":
                    r = self.document_analyzer.analyze(inp.path)
                    overall, risk = r.overall_score, r.risk
                    for pr in r.page_forensics:
                        versions.update(pr.algorithm_versions)
                    extra = {
                        "page_count": r.page_count,
                        "text_anomalies": r.text_anomalies,
                    }
                    chain_entry.update({"overall_score": overall, "risk": risk, **extra})
                    evidence_seed.append(
                        {
                            "source": "document_analyzer",
                            "claim": f"{inp.path}: risk={risk} (score={overall})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "analyze_document", "args": {"path": inp.path}, "ok": True})

                elif kind == "video":
                    r = self.video_analyzer.analyze(inp.path)
                    overall, risk = r.overall_score, r.risk
                    versions.update(r.algorithm_versions)
                    extra = {
                        "temporal_score": r.temporal_score,
                        "motion_anomaly_score": r.motion_anomaly_score,
                        "duplicate_frames": r.duplicate_frames,
                    }
                    chain_entry.update({"overall_score": overall, "risk": risk, **extra})
                    evidence_seed.append(
                        {
                            "source": "video_analyzer",
                            "claim": f"{inp.path}: risk={risk} (score={overall})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "analyze_video", "args": {"path": inp.path}, "ok": True})

                elif kind == "signature":
                    if inp.reference_path:
                        r = self.signature_verifier.compare(inp.path, inp.reference_path)
                    else:
                        r = self.signature_verifier.analyze_single(inp.path)
                    overall, risk = r.forgery_score, r.risk
                    versions.update(r.algorithm_versions)
                    chain_entry.update({"forgery_score": overall, "risk": risk})
                    if self.provider.supports_vision:
                        try:
                            image_b64, image_mt = encode_image_b64(inp.path)
                        except Exception:  # noqa: BLE001
                            pass
                    evidence_seed.append(
                        {
                            "source": "signature_verifier",
                            "claim": f"{inp.path}: risk={risk} (score={overall})",
                            "support": "; ".join(r.notes),
                        }
                    )
                    trace.append({"tool": "verify_signature", "args": {"path": inp.path}, "ok": True})

                else:
                    chain_entry["error"] = "unsupported_kind"
                    evidence_seed.append(
                        {
                            "source": "router",
                            "claim": f"Unsupported file kind for {inp.path}",
                            "support": f"extension={Path(inp.path).suffix}",
                        }
                    )
                    trace.append(
                        {"tool": "router", "args": {"path": inp.path}, "ok": False, "error": "unsupported_kind"}
                    )

                if overall is not None:
                    scores.append(overall)
                manifests.append(
                    InputManifest(
                        path=inp.path,
                        kind=kind,
                        label=inp.label,
                        sha256=sha256,
                        size_bytes=size_bytes,
                        overall_score=overall,
                        risk=risk,
                        detectors=detectors_for_provider,
                        extra=extra,
                        image_b64=image_b64,
                        image_media_type=image_mt,
                    )
                )
            except Exception as e:  # noqa: BLE001
                chain_entry["error"] = f"{type(e).__name__}: {e}"
                evidence_seed.append(
                    {
                        "source": "error",
                        "claim": f"Analyzer failed for {inp.path}",
                        "support": str(e),
                    }
                )
                trace.append({"tool": "dispatch", "args": {"path": inp.path}, "ok": False, "error": str(e)})
                manifests.append(
                    InputManifest(
                        path=inp.path,
                        kind=kind,
                        label=inp.label,
                        sha256=sha256,
                        size_bytes=size_bytes,
                        overall_score=None,
                        risk=None,
                    )
                )

            chain.append(chain_entry)

        return manifests, chain, det_results, versions, evidence_seed, scores, trace

    # ----------------------------- forensic provenance helpers
    def _reproducibility_hash(
        self,
        inputs: list[FraudInput],
        versions: dict[str, str],
        provider_name: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(b"fraud_detection_v0.3\n")
        h.update(f"provider={provider_name}\n".encode())
        for inp in inputs:
            try:
                meta = file_meta(inp.path)
                h.update(f"{inp.path}|{meta.sha256}|{meta.size_bytes}\n".encode())
            except Exception:  # noqa: BLE001
                h.update(f"{inp.path}|missing\n".encode())
            if inp.reference_path:
                try:
                    rmeta = file_meta(inp.reference_path)
                    h.update(f"REF|{inp.reference_path}|{rmeta.sha256}\n".encode())
                except Exception:  # noqa: BLE001
                    h.update(f"REF|{inp.reference_path}|missing\n".encode())
        for k in sorted(versions):
            h.update(f"{k}={versions[k]}\n".encode())
        return h.hexdigest()


# ---------------------------------------------------------------- helpers
def _env_key_for(provider_name: str) -> str | None:
    """Per-provider env var fallback."""
    if provider_name == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider_name == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY")
    return None
