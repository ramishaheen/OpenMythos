"""Anthropic / Claude provider — vision-capable.

Uses Claude's tool-use loop. Claude sees the detector JSON AND the actual
image bytes (when available) so it can corroborate (or downgrade) detector
firings against what's visually present.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fraud_detection.providers.base import Provider, ProviderInput, ProviderResult, Verdict
from fraud_detection.utils import risk_label

DEFAULT_MODEL = "claude-sonnet-4-6"

SUBMIT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final consolidated fraud report. Call exactly once.",
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
}

SYSTEM_PROMPT = """You are a forensic fraud analyst for MythosBank.

You receive:
  * Per-input metadata and a per-detector breakdown (score, confidence,
    version, evidence).
  * The actual image bytes for each image / signature input.

Your job:
  1. Cross-check the local detectors against the image visually — would a
    competent reviewer agree this scores look right?
  2. Downgrade detectors that fire on benign artefacts (compression noise,
    legitimate scans).
  3. Upgrade when you see evidence the local detectors miss (font swaps,
    lighting mismatch, pen-pressure breaks).
  4. Call submit_report exactly once with verdict, score (0-1), summary,
    structured evidence, and remediation recommendations.

Be explicit about uncertainty. If the image quality is too low for visual
corroboration, say so in the verdict reason."""


class AnthropicProvider:
    name = "anthropic"
    supports_vision = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        tool_budget: int = 6,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.tool_budget = tool_budget

    def configured(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401  type: ignore
        except ImportError:
            return False
        return True

    def summarize(self, payload: ProviderInput) -> ProviderResult:
        if not self.configured():
            raise RuntimeError(
                "AnthropicProvider not configured: missing API key or SDK."
            )
        from anthropic import Anthropic  # type: ignore

        client = Anthropic(api_key=self.api_key)
        user_blocks: list[dict[str, Any]] = []
        if payload.context:
            user_blocks.append({"type": "text", "text": f"Context: {payload.context}"})
        user_blocks.append(
            {
                "type": "text",
                "text": "Manifest + detector evidence:\n"
                + json.dumps(_manifest_for_prompt(payload), indent=2),
            }
        )
        for inp in payload.inputs:
            if inp.image_b64 and inp.image_media_type:
                user_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inp.image_media_type,
                            "data": inp.image_b64,
                        },
                    }
                )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        final: dict[str, Any] | None = None
        raw_resp: Any = None

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
                tools=[SUBMIT_TOOL],
                messages=messages,
            )
            raw_resp = resp
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break
            tool_results = []
            for tu in tool_uses:
                if tu.name == "submit_report":
                    final = dict(tu.input or {})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": "report_received",
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": "unknown tool — call submit_report instead",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            if final is not None:
                break

        if final is None:
            return _fallback(payload, "Claude returned no submit_report")

        score = float(final.get("score", payload.aggregate_score))
        verdict = _coerce_verdict(final.get("verdict"), score)
        return ProviderResult(
            verdict=verdict,
            score=round(score, 4),
            summary=final.get("summary", ""),
            evidence=list(final.get("evidence", [])),
            recommendations=list(final.get("recommendations", [])),
            raw={"model": self.model, "stop_reason": getattr(raw_resp, "stop_reason", None)},
        )


def _manifest_for_prompt(payload: ProviderInput) -> dict[str, Any]:
    return {
        "aggregate_score": round(payload.aggregate_score, 4),
        "aggregate_risk": payload.aggregate_risk,
        "context": payload.context,
        "algorithm_versions": payload.algorithm_versions,
        "inputs": [
            {
                k: v
                for k, v in {
                    "path": inp.path,
                    "kind": inp.kind,
                    "label": inp.label,
                    "sha256": inp.sha256,
                    "size_bytes": inp.size_bytes,
                    "overall_score": inp.overall_score,
                    "risk": inp.risk,
                    "detectors": inp.detectors,
                    **inp.extra,
                }.items()
                if v is not None
            }
            for inp in payload.inputs
        ],
    }


def _coerce_verdict(v: Any, score: float) -> Verdict:
    valid = ("authentic", "suspicious", "manipulated", "inconclusive")
    if v in valid:
        return v  # type: ignore[return-value]
    if score >= 0.4434:
        return "manipulated"
    if score >= 0.2412:
        return "suspicious"
    if score > 0.0:
        return "authentic"
    return "inconclusive"


def _fallback(payload: ProviderInput, reason: str) -> ProviderResult:
    return ProviderResult(
        verdict=_coerce_verdict(None, payload.aggregate_score),
        score=round(payload.aggregate_score, 4),
        summary=f"Provider fallback ({reason}). Local fusion: "
        f"{round(payload.aggregate_score, 4)} ({risk_label(payload.aggregate_score)}).",
        evidence=[],
        recommendations=[reason],
        raw={"fallback": reason},
    )
