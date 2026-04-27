"""Claude-powered fraud-detection agent.

Orchestrates the forensic analyzers via Anthropic tool use:

    user input (paths + context)
        │
        ▼
    Claude (sonnet-4-6, vision-enabled, prompt-cached system)
        │
        │  tool_use → run_image_forensics / analyze_document /
        │             analyze_video / verify_signature / ...
        ▼
    Local analyzers return structured JSON evidence
        │
        ▼
    Claude weighs evidence + visual inspection
        │
        ▼
    submit_report  →  FraudReport(verdict, score, evidence, recommendations)

The agent supports the Anthropic SDK if installed; otherwise it falls back to
deterministic local-only scoring so the pipeline still runs in offline tests.
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
from fraud_detection.signature_verifier import SignatureVerifier
from fraud_detection.utils import (
    MediaKind,
    detect_kind,
    encode_image_b64,
    file_meta,
    risk_label,
)
from fraud_detection.video_analyzer import VideoAnalyzer

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOOL_BUDGET = 8


SYSTEM_PROMPT = """You are a forensic fraud analyst for MythosBanking.
You receive media (documents, images, videos, signatures) plus the output of
local forensic analyzers (ELA, metadata, copy-move, ORB, SSIM, OCR text
anomalies, temporal video residuals).

Your job:
  1. Decide which analyzer tools to run for each input. Run the minimum set.
  2. After each tool result, inspect any attached image visually for cues the
     local pipeline cannot catch (font swaps, alignment, lighting, shadows,
     pen-pressure breaks in signatures, watermark damage).
  3. When you have enough evidence, call `submit_report` exactly once with a
     verdict, a 0-1 fraud score, the list of concrete evidence items you used,
     and remediation recommendations.

Calibration:
  - Treat ELA, copy-move, and temporal scores as evidence — not verdicts.
  - A high local score with no visual corroboration should be downgraded.
  - A low local score with clear visual tampering should be upgraded.
  - Be explicit about uncertainty. If the input is too low-quality to judge,
    say so in the verdict reason.
"""

TOOLS = [
    {
        "name": "run_image_forensics",
        "description": "Run ELA, EXIF, copy-move, and noise analysis on a single image.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "analyze_document",
        "description": "Render document pages, run image forensics on each page, and OCR for text anomalies.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "analyze_video",
        "description": "Sample frames from a video and run per-frame + temporal forensics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "sample_count": {"type": "integer", "minimum": 2, "maximum": 32},
            },
            "required": ["path"],
        },
    },
    {
        "name": "verify_signature",
        "description": "Compare a questioned signature against a reference signature image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "questioned": {"type": "string"},
                "reference": {"type": "string"},
            },
            "required": ["questioned", "reference"],
        },
    },
    {
        "name": "analyze_signature",
        "description": "Score intrinsic anomalies in a signature when no reference is available.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "submit_report",
        "description": "Submit the final consolidated fraud report and stop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["authentic", "suspicious", "manipulated", "inconclusive"],
                },
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "summary": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "claim": {"type": "string"},
                            "support": {"type": "string"},
                        },
                        "required": ["source", "claim"],
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["verdict", "score", "summary", "evidence"],
        },
    },
]


# --------------------------------------------------------------------- types
@dataclass
class FraudInput:
    path: str
    kind: MediaKind = "unknown"
    reference_path: str | None = None  # used for signature comparison
    label: str | None = None  # human-readable role, e.g. "passport_front"

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
    """High-level orchestrator. Use ``run`` for synchronous analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tool_budget: int = DEFAULT_TOOL_BUDGET,
        image_forensics: ImageForensics | None = None,
        document_analyzer: DocumentAnalyzer | None = None,
        video_analyzer: VideoAnalyzer | None = None,
        signature_verifier: SignatureVerifier | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.tool_budget = tool_budget
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

        client = self._build_client()
        if client is None:
            return self._offline_report(normalised, context)
        return self._claude_loop(client, normalised, context)

    # --------------------------------------------------- internals
    def _normalise(self, item: FraudInput | str | dict[str, Any]) -> FraudInput:
        if isinstance(item, FraudInput):
            return item
        if isinstance(item, str):
            return FraudInput(path=item)
        if isinstance(item, dict):
            return FraudInput(**item)
        raise TypeError(f"Unsupported input type: {type(item).__name__}")

    def _build_client(self):
        if not self.api_key:
            return None
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError:
            return None
        return Anthropic(api_key=self.api_key)

    # ------------------------------- tool dispatch (used by both paths)
    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "run_image_forensics":
            return self.image_forensics.analyze(args["path"]).to_dict()
        if name == "analyze_document":
            return self.document_analyzer.analyze(args["path"]).to_dict()
        if name == "analyze_video":
            va = self.video_analyzer
            if "sample_count" in args:
                va = VideoAnalyzer(
                    sample_count=int(args["sample_count"]),
                    forensics=self.image_forensics,
                )
            return va.analyze(args["path"]).to_dict()
        if name == "verify_signature":
            return self.signature_verifier.compare(
                args["questioned"], args["reference"]
            ).to_dict()
        if name == "analyze_signature":
            return self.signature_verifier.analyze_single(args["path"]).to_dict()
        raise KeyError(f"Unknown tool: {name}")

    # -------------------------------------------------- Claude tool loop
    def _claude_loop(
        self,
        client: Any,
        inputs: list[FraudInput],
        context: str | None,
    ) -> FraudReport:
        # Initial user turn: paths + previewable images for vision corroboration.
        user_blocks: list[dict[str, Any]] = []
        if context:
            user_blocks.append({"type": "text", "text": f"Context: {context}"})

        manifest_lines = ["Inputs to analyze:"]
        for i, inp in enumerate(inputs):
            kind = inp.resolved_kind()
            meta = file_meta(inp.path)
            manifest_lines.append(
                f"  [{i}] kind={kind} role={inp.label or '-'} "
                f"path={inp.path} sha256={meta.sha256[:12]} size={meta.size_bytes}"
            )
            if inp.reference_path:
                manifest_lines.append(f"        reference={inp.reference_path}")
        user_blocks.append({"type": "text", "text": "\n".join(manifest_lines)})

        # Attach previewable images (image inputs + signature pairs) for vision.
        for inp in inputs:
            kind = inp.resolved_kind()
            if kind in ("image", "signature") and Path(inp.path).suffix.lower() != ".pdf":
                try:
                    b64, mt = encode_image_b64(inp.path)
                    user_blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mt, "data": b64},
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            if inp.reference_path:
                try:
                    b64, mt = encode_image_b64(inp.reference_path)
                    user_blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mt, "data": b64},
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        tool_trace: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None

        for _ in range(self.tool_budget):
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break

            tool_results: list[dict[str, Any]] = []
            stop_after = False
            for tu in tool_uses:
                name = tu.name
                args = tu.input or {}
                if name == "submit_report":
                    final = dict(args)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": "report_received",
                        }
                    )
                    stop_after = True
                    continue
                try:
                    result = self._dispatch_tool(name, args)
                    payload = json.dumps(result, default=str)
                    tool_trace.append({"tool": name, "args": args, "ok": True})
                except Exception as e:  # noqa: BLE001
                    payload = json.dumps({"error": str(e)})
                    tool_trace.append(
                        {"tool": name, "args": args, "ok": False, "error": str(e)}
                    )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": payload}
                )
            messages.append({"role": "user", "content": tool_results})
            if stop_after:
                break

        if final is None:
            return self._offline_report(
                inputs,
                context,
                extra_note="Claude exited without submit_report; falling back to local fusion.",
                tool_trace=tool_trace,
            )

        score = float(final.get("score", 0.0))
        chain, versions, det_results = self._build_provenance(inputs, tool_trace)
        repro = self._reproducibility_hash(inputs, versions)
        return FraudReport(
            verdict=final.get("verdict", "inconclusive"),
            score=round(score, 4),
            risk=risk_label(score),
            summary=final.get("summary", ""),
            evidence=list(final.get("evidence", [])),
            recommendations=list(final.get("recommendations", [])),
            tool_trace=tool_trace,
            inputs=[asdict(i) for i in inputs],
            model=self.model,
            chain_of_evidence=chain,
            algorithm_versions=versions,
            detector_results=det_results,
            reproducibility_hash=repro,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ----------------------------- forensic provenance helpers
    def _build_provenance(
        self,
        inputs: list[FraudInput],
        tool_trace: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        """Assemble chain-of-evidence + algorithm versions from the inputs.

        We re-derive provenance by reading file metadata (sha256, size, mime).
        Algorithm versions come from the analyzer modules; detector_results
        is populated from a single local re-run so the report always
        contains the structured per-detector evidence even if Claude only
        returned a free-text summary.
        """
        chain: list[dict[str, Any]] = []
        det_results: list[dict[str, Any]] = []
        versions: dict[str, str] = {}
        for inp in inputs:
            try:
                meta = file_meta(inp.path)
            except Exception as e:  # noqa: BLE001
                chain.append(
                    {"input": inp.path, "error": f"file_meta failed: {e}"}
                )
                continue
            kind = inp.resolved_kind()
            entry: dict[str, Any] = {
                "input": inp.path,
                "kind": kind,
                "label": inp.label,
                "size_bytes": meta.size_bytes,
                "mime": meta.mime,
                "sha256": meta.sha256,
            }
            if inp.reference_path:
                try:
                    rmeta = file_meta(inp.reference_path)
                    entry["reference_sha256"] = rmeta.sha256
                except Exception:  # noqa: BLE001
                    pass
            try:
                if kind == "image":
                    r = self.image_forensics.analyze(inp.path)
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["detectors"] = r.detector_results
                    versions.update(r.algorithm_versions)
                    det_results.append(
                        {"input": inp.path, "kind": kind, "results": r.detector_results}
                    )
                elif kind == "document":
                    r = self.document_analyzer.analyze(inp.path)
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["page_count"] = r.page_count
                    entry["text_anomalies"] = r.text_anomalies
                    if r.page_forensics:
                        for pr in r.page_forensics:
                            versions.update(pr.algorithm_versions)
                elif kind == "video":
                    r = self.video_analyzer.analyze(inp.path)
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["temporal_score"] = r.temporal_score
                    entry["motion_anomaly_score"] = r.motion_anomaly_score
                    entry["duplicate_frames"] = r.duplicate_frames
                    versions.update(r.algorithm_versions)
                elif kind == "signature":
                    if inp.reference_path:
                        r = self.signature_verifier.compare(inp.path, inp.reference_path)
                    else:
                        r = self.signature_verifier.analyze_single(inp.path)
                    entry["forgery_score"] = r.forgery_score
                    entry["risk"] = r.risk
                    versions.update(r.algorithm_versions)
            except Exception as e:  # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {e}"
            chain.append(entry)
        return chain, versions, det_results

    def _reproducibility_hash(
        self, inputs: list[FraudInput], versions: dict[str, str]
    ) -> str:
        h = hashlib.sha256()
        h.update(b"fraud_detection_v0.2\n")
        for inp in inputs:
            try:
                meta = file_meta(inp.path)
                h.update(f"{inp.path}|{meta.sha256}|{meta.size_bytes}\n".encode())
            except Exception:  # noqa: BLE001
                h.update(f"{inp.path}|missing\n".encode())
            if inp.reference_path:
                try:
                    rmeta = file_meta(inp.reference_path)
                    h.update(
                        f"REF|{inp.reference_path}|{rmeta.sha256}\n".encode()
                    )
                except Exception:  # noqa: BLE001
                    h.update(f"REF|{inp.reference_path}|missing\n".encode())
        for k in sorted(versions):
            h.update(f"{k}={versions[k]}\n".encode())
        return h.hexdigest()

    # ------------------------------------------- offline fallback path
    def _offline_report(
        self,
        inputs: list[FraudInput],
        context: str | None,
        extra_note: str | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> FraudReport:
        evidence: list[dict[str, Any]] = []
        scores: list[float] = []
        trace = list(tool_trace or [])
        det_results: list[dict[str, Any]] = []
        versions: dict[str, str] = {}
        chain: list[dict[str, Any]] = []

        for inp in inputs:
            kind = inp.resolved_kind()
            entry: dict[str, Any] = {"input": inp.path, "kind": kind, "label": inp.label}
            try:
                meta = file_meta(inp.path)
                entry.update({"sha256": meta.sha256, "size_bytes": meta.size_bytes, "mime": meta.mime})
            except Exception:  # noqa: BLE001
                pass
            try:
                if kind == "image":
                    r = self.image_forensics.analyze(inp.path)
                    scores.append(r.overall_score)
                    versions.update(r.algorithm_versions)
                    det_results.append(
                        {"input": inp.path, "kind": kind, "results": r.detector_results}
                    )
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["detectors"] = r.detector_results
                    evidence.append(
                        {
                            "source": "image_forensics",
                            "claim": f"{inp.path}: risk={r.risk} (score={r.overall_score})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "run_image_forensics", "args": {"path": inp.path}, "ok": True})
                elif kind == "document":
                    r = self.document_analyzer.analyze(inp.path)
                    scores.append(r.overall_score)
                    for pr in r.page_forensics:
                        versions.update(pr.algorithm_versions)
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["page_count"] = r.page_count
                    entry["text_anomalies"] = r.text_anomalies
                    evidence.append(
                        {
                            "source": "document_analyzer",
                            "claim": f"{inp.path}: risk={r.risk} (score={r.overall_score})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "analyze_document", "args": {"path": inp.path}, "ok": True})
                elif kind == "video":
                    r = self.video_analyzer.analyze(inp.path)
                    scores.append(r.overall_score)
                    versions.update(r.algorithm_versions)
                    entry["overall_score"] = r.overall_score
                    entry["risk"] = r.risk
                    entry["temporal_score"] = r.temporal_score
                    entry["motion_anomaly_score"] = r.motion_anomaly_score
                    entry["duplicate_frames"] = r.duplicate_frames
                    evidence.append(
                        {
                            "source": "video_analyzer",
                            "claim": f"{inp.path}: risk={r.risk} (score={r.overall_score})",
                            "support": "; ".join(r.notes) or "no notable issues",
                        }
                    )
                    trace.append({"tool": "analyze_video", "args": {"path": inp.path}, "ok": True})
                elif kind == "signature":
                    if inp.reference_path:
                        r = self.signature_verifier.compare(inp.path, inp.reference_path)
                    else:
                        r = self.signature_verifier.analyze_single(inp.path)
                    scores.append(r.forgery_score)
                    versions.update(r.algorithm_versions)
                    entry["forgery_score"] = r.forgery_score
                    entry["risk"] = r.risk
                    evidence.append(
                        {
                            "source": "signature_verifier",
                            "claim": f"{inp.path}: risk={r.risk} (score={r.forgery_score})",
                            "support": "; ".join(r.notes),
                        }
                    )
                    trace.append({"tool": "verify_signature", "args": {"path": inp.path}, "ok": True})
                else:
                    evidence.append(
                        {
                            "source": "router",
                            "claim": f"Unsupported file kind for {inp.path}",
                            "support": f"extension={Path(inp.path).suffix}",
                        }
                    )
            except Exception as e:  # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {e}"
                evidence.append(
                    {
                        "source": "error",
                        "claim": f"Analyzer failed for {inp.path}",
                        "support": str(e),
                    }
                )
                trace.append({"tool": "dispatch", "args": {"path": inp.path}, "ok": False, "error": str(e)})
            chain.append(entry)

        # Use the calibrated thresholds from image_forensics for verdict mapping.
        from fraud_detection.image_forensics import DEFAULT_THRESHOLDS
        agg = max(scores, default=0.0)
        verdict: Literal["authentic", "suspicious", "manipulated", "inconclusive"]
        if agg >= DEFAULT_THRESHOLDS["high"]:
            verdict = "manipulated"
        elif agg >= DEFAULT_THRESHOLDS["medium"]:
            verdict = "suspicious"
        elif agg > 0.0:
            verdict = "authentic"
        else:
            verdict = "inconclusive"

        recs = [
            "Re-run analysis with the Anthropic SDK installed and ANTHROPIC_API_KEY set "
            "to enable Claude's vision-based corroboration.",
        ]
        if agg >= 0.40:
            recs.append("Escalate to a human reviewer with the listed evidence.")
        if extra_note:
            recs.append(extra_note)
        if context:
            recs.append(f"Original context: {context}")

        summary = (
            f"Local-only fusion across {len(inputs)} input(s). "
            f"Aggregate risk score {round(agg, 4)} ({risk_label(agg)})."
        )

        repro = self._reproducibility_hash(inputs, versions)

        return FraudReport(
            verdict=verdict,
            score=round(agg, 4),
            risk=risk_label(agg),
            summary=summary,
            evidence=evidence,
            recommendations=recs,
            tool_trace=trace,
            inputs=[asdict(i) for i in inputs],
            model=None,
            chain_of_evidence=chain,
            algorithm_versions=versions,
            detector_results=det_results,
            reproducibility_hash=repro,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
