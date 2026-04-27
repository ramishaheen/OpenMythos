# Fraud Detection — for MythosBanking

A Claude-powered agent that detects falsification and manipulation of
**documents**, **images**, **videos**, and **signatures**. Designed as a
self-contained drop-in module for the
[MythosBanking](https://github.com/ramishaheen/MythosBanking) backend.

The agent fuses classical forensic primitives with Claude's vision
reasoning, drawing technique inspiration from:

- [shraddhavijay/IFAKE](https://github.com/shraddhavijay/IFAKE) — image forgery
  detection (ELA, metadata, noise residue).
- [GitHub `signature-detection` topic](https://github.com/topics/signature-detection)
  — signature localization and verification approaches.
- [sainipankaj15/Signature-Forgery-Detection](https://github.com/sainipankaj15/Signature-Forgery-Detection)
  — feature-based signature forgery detection (geometric + ORB matching).

## How it works

```
                ┌────────────────────────────────────────────────┐
                │ FraudDetectionAgent (Claude sonnet-4-6)        │
                │  • cached system prompt                        │
                │  • vision input for every image / signature    │
                │  • tool-use loop (max N iterations)            │
                └────────────────────────────────────────────────┘
                            │ tool_use            │ submit_report
                            ▼                     ▼
   ┌──────────────────────────────────────────┐  FraudReport
   │  ImageForensics  — ELA, EXIF, copy-move │   verdict, score,
   │  DocumentAnalyzer — pages + OCR + text  │   evidence,
   │  VideoAnalyzer    — frames + temporal   │   recommendations,
   │  SignatureVerifier— features + ORB+SSIM │   tool trace
   └──────────────────────────────────────────┘
```

Claude decides *which* analyzers to run, inspects the attached imagery
visually, and weighs the local evidence. The pipeline still works without
the Anthropic SDK / API key — it falls back to a deterministic local fusion,
useful for unit tests and air-gapped deployments.

## Install

```bash
pip install -e .            # core only (image forensics + signatures)
pip install -e .[all]       # adds video, SSIM, PDF, OCR
export ANTHROPIC_API_KEY=...
```

## Quickstart

```python
from fraud_detection import FraudDetectionAgent, FraudInput

agent = FraudDetectionAgent()  # uses ANTHROPIC_API_KEY from env
report = agent.run(
    [
        FraudInput(path="passport.jpg", kind="image", label="passport_front"),
        FraudInput(path="statement.pdf", kind="document", label="proof_of_funds"),
        FraudInput(
            path="sig_questioned.png",
            kind="signature",
            reference_path="sig_reference.png",
            label="account_application",
        ),
        FraudInput(path="liveness.mp4", kind="video", label="liveness_check"),
    ],
    context="High-net-worth onboarding bundle.",
)

print(report.verdict, report.risk, report.score)
print(report.to_json())
```

## CLI

```bash
fraud-detect statement.pdf
fraud-detect -k signature -r ref.png questioned.png --context "wire authorization"
fraud-detect liveness.mp4 --json
```

## Integrating into MythosBanking

1. Copy the `fraud_detection/` folder into your service tree (e.g.
   `mythosbanking/services/fraud_detection/`), or `pip install` this bundle.
2. Add `ANTHROPIC_API_KEY` to your secret store / env config.
3. Wire it into the relevant flows:
   - **KYC / onboarding** — call `agent.run` on uploaded ID photos and
     proof-of-address PDFs before approving an application.
   - **Wire / payment authorization** — call `verify_signature` or the
     full agent on signed instructions.
   - **Claims / disputes** — analyze submitted evidence images and videos.
4. Persist `FraudReport.to_dict()` alongside the underlying record. The
   `tool_trace` makes the decision auditable.

Recommended pattern for a FastAPI route:

```python
from fastapi import UploadFile, File
from fraud_detection import FraudDetectionAgent, FraudInput

agent = FraudDetectionAgent()  # build once, reuse across requests

@app.post("/kyc/review")
async def review(passport: UploadFile = File(...), proof: UploadFile = File(...)):
    paths = [save(passport), save(proof)]
    inputs = [
        FraudInput(path=paths[0], kind="image", label="passport_front"),
        FraudInput(path=paths[1], kind="document", label="proof_of_address"),
    ]
    report = agent.run(inputs, context="KYC submission")
    return report.to_dict()
```

## Modules

| File | Purpose |
| --- | --- |
| `fraud_detection/agent.py` | Claude tool-use orchestrator + offline fallback |
| `fraud_detection/image_forensics.py` | ELA, EXIF flags, copy-move, noise residue |
| `fraud_detection/document_analyzer.py` | PDF rendering, per-page forensics, OCR text checks |
| `fraud_detection/video_analyzer.py` | Frame sampling, per-frame ELA, temporal residual |
| `fraud_detection/signature_verifier.py` | Otsu + skeleton features, ORB, SSIM |
| `fraud_detection/utils.py` | File metadata, base64 image encoding, scoring helpers |
| `fraud_detection/cli.py` | `fraud-detect` console script |

## Risk bands

| Score | Risk | Suggested action |
| --- | --- | --- |
| `≥ 0.75` | high | Block flow, escalate to human review with `tool_trace`. |
| `≥ 0.45` | medium | Hold + secondary verification (callback, doc re-upload). |
| `≥ 0.20` | low | Log + allow. |
| `< 0.20` | minimal | Allow. |

## Tests

```bash
pip install -e .[dev]
pytest -q
```

The test suite uses synthetic images and forces the offline path, so it
runs without an API key.

## Notes & caveats

- The classical detectors here are **screening** tools, not court-grade
  forensics. They flag candidates for review; final calls belong to a human.
- Adversaries who control re-encoding can suppress ELA. Combine with
  signed-image / verified-camera attestations where possible.
- Video analysis samples a handful of frames — for high-stakes deepfake
  detection, replace `VideoAnalyzer` with a dedicated model (the agent's
  tool interface is stable; just swap the implementation).
