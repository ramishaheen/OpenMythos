"""CLI: ``python -m fraud_detection <paths...>``

Examples:

    python -m fraud_detection statement.pdf
    python -m fraud_detection -k signature -r ref.png questioned.png
    python -m fraud_detection clip.mp4 --json
"""

from __future__ import annotations

import argparse
import sys

from fraud_detection.agent import FraudDetectionAgent, FraudInput


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fraud_detection",
        description="Detect falsification and manipulation of documents, images, videos, signatures.",
    )
    p.add_argument("paths", nargs="+", help="One or more file paths to analyze.")
    p.add_argument(
        "-k",
        "--kind",
        choices=["image", "document", "video", "signature", "auto"],
        default="auto",
        help="Force a media kind. Default: auto-detect by extension.",
    )
    p.add_argument(
        "-r",
        "--reference",
        help="Reference signature path (only used when --kind signature).",
    )
    p.add_argument(
        "-c",
        "--context",
        help="Free-form context passed to Claude (e.g. 'KYC submission for account X').",
    )
    p.add_argument(
        "--label",
        help="Human-readable role label, e.g. 'passport_front' or 'wire_authorization'.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Anthropic model ID (default: claude-sonnet-4-6).",
    )
    p.add_argument("--json", action="store_true", help="Print JSON instead of a summary.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kind = "unknown" if args.kind == "auto" else args.kind

    inputs = []
    for path in args.paths:
        inputs.append(
            FraudInput(
                path=path,
                kind=kind,  # type: ignore[arg-type]
                reference_path=args.reference if kind == "signature" else None,
                label=args.label,
            )
        )

    agent_kwargs: dict[str, object] = {}
    if args.model:
        agent_kwargs["model"] = args.model
    agent = FraudDetectionAgent(**agent_kwargs)
    report = agent.run(inputs, context=args.context)

    if args.json:
        print(report.to_json())
    else:
        print(_render(report))
    return 0 if report.verdict in ("authentic", "inconclusive") else 1


def _render(report) -> str:
    lines = [
        f"Verdict:         {report.verdict}",
        f"Risk:            {report.risk}",
        f"Score:           {report.score:.3f}",
        f"Model:           {report.model or 'offline'}",
        "",
        "Summary:",
        f"  {report.summary}",
        "",
        "Evidence:",
    ]
    for e in report.evidence:
        lines.append(f"  - [{e.get('source','?')}] {e.get('claim','')}")
        if e.get("support"):
            lines.append(f"      {e['support']}")
    if report.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for r in report.recommendations:
            lines.append(f"  - {r}")
    if report.tool_trace:
        lines.append("")
        lines.append("Tool trace:")
        for t in report.tool_trace:
            ok = "ok" if t.get("ok") else f"err: {t.get('error')}"
            lines.append(f"  - {t.get('tool')} ({ok})")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
