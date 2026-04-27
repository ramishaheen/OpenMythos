"""DeepSeek provider — text-only reasoning over detector evidence.

DeepSeek's chat API is OpenAI-compatible (https://api.deepseek.com/v1).
It does NOT have vision; we feed the detector JSON + manifest as text and
ask it to produce a structured verdict via JSON-mode output.

Models on api.deepseek.com:
  * deepseek-chat       — general-purpose, fast, JSON-mode capable.
  * deepseek-reasoner   — reasoning-tuned, slower, higher quality.

Why not vision: the agent's existing visual corroboration step (font
swaps, lighting, pen pressure) is *not* available with this provider.
Treat the DeepSeek path as text-only fusion + reasoning.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fraud_detection.providers.base import Provider, ProviderInput, ProviderResult, Verdict
from fraud_detection.utils import risk_label

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

SYSTEM_PROMPT = """You are a forensic fraud analyst for MythosBank.

You receive a JSON manifest of inputs plus the per-detector breakdown
from our forensic pipeline (ELA, JPEG-Q, DCT-Benford, CFA, copy-move
pHash, PRNU, lighting consistency). You do NOT have vision in this mode
— reason ONLY from the structured evidence.

Return a single JSON object with exactly these keys:
  verdict          one of "authentic" | "suspicious" | "manipulated" | "inconclusive"
  score            number in [0, 1]
  summary          one-paragraph justification, citing detectors by name
  evidence         array of { source, claim, support } objects
  recommendations  array of strings (operational next steps)

Calibration tips:
  * High ELA + high PRNU + non-zero copy-move usually means real tampering.
  * High Benford-DCT alone with no other firings often just means recompression.
  * Low aggregate (<0.2) with quiet detectors → authentic.
  * If the detector set looks too thin to judge → "inconclusive" with a
    note that vision corroboration is needed.
"""


class DeepSeekProvider:
    name = "deepseek"
    supports_vision = False

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
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
                "DeepSeekProvider not configured: missing API key or `openai` SDK."
            )
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        manifest = _manifest_for_prompt(payload)

        user = (
            "Manifest + detector evidence (JSON below). "
            "Produce the JSON verdict object as instructed.\n\n"
            f"{json.dumps(manifest, indent=2)}"
        )

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:  # noqa: BLE001
            return _fallback(payload, f"DeepSeek call failed: {type(e).__name__}: {e}")

        text = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _fallback(payload, "DeepSeek returned non-JSON content")

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
                "usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
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
            f"DeepSeek fallback ({reason}). Local fusion: "
            f"{round(payload.aggregate_score, 4)} "
            f"({risk_label(payload.aggregate_score)})."
        ),
        evidence=[],
        recommendations=[reason],
        raw={"fallback": reason},
    )
