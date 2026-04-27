"""OpenAI provider — vision-capable.

Uses the OpenAI Python SDK against ``https://api.openai.com/v1``. We pick
models that support both **vision** (image_url content) and **JSON-mode**
output, so the agent can attach the image bytes alongside the detector
JSON and get back a structured verdict.

Default model: ``gpt-4o``. ``gpt-4o-mini`` is the cheaper option;
``gpt-4-turbo`` is the older vision-capable model.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fraud_detection.providers.base import Provider, ProviderInput, ProviderResult, Verdict
from fraud_detection.utils import risk_label

DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """You are a forensic fraud analyst for MythosBank.

You receive a JSON manifest of analysed inputs plus a per-detector
breakdown (ELA, JPEG-Q, DCT-Benford, CFA, copy-move pHash, PRNU,
lighting). For images and signatures, the actual bytes are attached.

Your job:
  1. Cross-check the detectors against what you can SEE in the image —
    would a competent reviewer agree these scores look right?
  2. Downgrade detectors that fire on benign artefacts (compression
    noise, legitimate scans).
  3. Upgrade when you see evidence the detectors miss (font swaps,
    lighting mismatch, pen-pressure breaks, watermark damage).
  4. Return a single JSON object with EXACTLY these keys:
       verdict          one of "authentic" | "suspicious" | "manipulated" | "inconclusive"
       score            number in [0, 1]
       summary          one paragraph, citing detector names
       evidence         array of { source, claim, support } objects
       recommendations  array of strings (operational next steps)

Be explicit about uncertainty. If the image quality is too low to
visually corroborate, say so in the verdict reason."""


class OpenAIProvider:
    name = "openai"
    supports_vision = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def configured(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401  type: ignore
        except ImportError:
            return False
        return True

    def summarize(self, payload: ProviderInput) -> ProviderResult:
        if not self.configured():
            raise RuntimeError(
                "OpenAIProvider not configured: missing API key or `openai` SDK."
            )
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        manifest = _manifest_for_prompt(payload)

        # Build a multi-content user message: text manifest first, then any
        # available images as image_url with data URIs.
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    (f"Context: {payload.context}\n\n" if payload.context else "")
                    + "Manifest + detector evidence (JSON below). Return the "
                    "JSON verdict object as instructed.\n\n"
                    + json.dumps(manifest, indent=2)
                ),
            }
        ]
        for inp in payload.inputs:
            if inp.image_b64 and inp.image_media_type:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{inp.image_media_type};base64,{inp.image_b64}"
                        },
                    }
                )

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:  # noqa: BLE001
            return _fallback(payload, f"OpenAI call failed: {type(e).__name__}: {e}")

        text = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _fallback(payload, "OpenAI returned non-JSON content")

        score = _safe_float(parsed.get("score"), payload.aggregate_score)
        return ProviderResult(
            verdict=_coerce_verdict(parsed.get("verdict"), score),
            score=round(score, 4),
            summary=str(parsed.get("summary", "")),
            evidence=list(parsed.get("evidence", []) or []),
            recommendations=list(parsed.get("recommendations", []) or []),
            raw={
                "model": self.model,
                "finish_reason": resp.choices[0].finish_reason,
                "usage": getattr(resp, "usage", None).__dict__
                if getattr(resp, "usage", None)
                else None,
            },
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
                    "path": inp.path.split("/")[-1],
                    "kind": inp.kind,
                    "label": inp.label,
                    "sha256": (inp.sha256 or "")[:16],
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


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _coerce_verdict(v: Any, score: float) -> Verdict:
    valid = ("authentic", "suspicious", "manipulated", "inconclusive")
    if isinstance(v, str) and v.lower() in valid:
        return v.lower()  # type: ignore[return-value]
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
        summary=(
            f"OpenAI fallback ({reason}). Local fusion: "
            f"{round(payload.aggregate_score, 4)} "
            f"({risk_label(payload.aggregate_score)})."
        ),
        evidence=[],
        recommendations=[reason],
        raw={"fallback": reason},
    )
