"""Deterministic provider — no LLM, fuses scores locally.

This is the fallback when no API key is configured, or for offline
testing. It produces the same ProviderResult shape as the cloud
providers so the agent's output is uniform.
"""

from __future__ import annotations

from fraud_detection.providers.base import Provider, ProviderInput, ProviderResult, Verdict
from fraud_detection.utils import risk_label


class OfflineProvider:
    name = "offline"
    supports_vision = False
    version = "1.0"

    def configured(self) -> bool:
        return True

    def summarize(self, payload: ProviderInput) -> ProviderResult:
        score = payload.aggregate_score
        verdict = _verdict_from_score(score)
        evidence = []
        kind_to_source = {
            "image": "image_forensics",
            "document": "document_analyzer",
            "video": "video_analyzer",
            "signature": "signature_verifier",
        }
        for inp in payload.inputs:
            entry = {
                "source": kind_to_source.get(inp.kind, f"{inp.kind}_analyzer"),
                "claim": (
                    f"{inp.path.split('/')[-1]}: risk={inp.risk} "
                    f"(score={inp.overall_score})"
                ),
                "support": "; ".join(
                    [
                        f"{d['name']}@{d['version']}={d['score']:.3f} "
                        f"(c={d.get('confidence', 1.0):.2f})"
                        for d in inp.detectors[:8]
                    ]
                )
                or "no detectors fired",
            }
            evidence.append(entry)

        recs = [
            "Deterministic local fusion — no LLM corroboration. Configure "
            "an LLM provider in Settings for richer reasoning over the "
            "detector evidence.",
        ]
        if score >= 0.40:
            recs.append("Escalate to a human reviewer with the listed evidence.")

        summary = (
            f"Local fusion across {len(payload.inputs)} input(s). "
            f"Aggregate {round(score, 4)} ({risk_label(score)})."
        )
        return ProviderResult(
            verdict=verdict,
            score=round(score, 4),
            summary=summary,
            evidence=evidence,
            recommendations=recs,
            raw=None,
        )


def _verdict_from_score(score: float) -> Verdict:
    # Aligned with the calibrated thresholds in image_forensics.
    if score >= 0.4434:
        return "manipulated"
    if score >= 0.2412:
        return "suspicious"
    if score > 0.0:
        return "authentic"
    return "inconclusive"
