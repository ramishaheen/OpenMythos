# Fraud Detection — for MythosBanking

A Claude-powered agent that detects falsification and manipulation of
**documents**, **images**, **videos**, and **signatures**, fusing a suite
of peer-reviewed forensic primitives with Claude's vision reasoning.
Designed as a self-contained drop-in module for the
[MythosBanking](https://github.com/ramishaheen/MythosBanking) backend.

Inspired by:

- [shraddhavijay/IFAKE](https://github.com/shraddhavijay/IFAKE) — image forgery detection.
- [GitHub `signature-detection` topic](https://github.com/topics/signature-detection).
- [sainipankaj15/Signature-Forgery-Detection](https://github.com/sainipankaj15/Signature-Forgery-Detection).

## What's inside

### Image-forensics detector suite (`fraud_detection/detectors/`)

Each detector is a pure-Python module exposing
`run(path) -> DetectorResult(score, confidence, evidence, notes)`.
Scores are calibrated to `[0, 1]` and fused with weights that come from a
60% literature prior + 40% synthetic-AUC blend (re-runnable; see below).

| Detector | Primitive | Reference |
| --- | --- | --- |
| `ela` | Error Level Analysis with hotspot localization | Krawetz / IFAKE |
| `jpeg_qtable` | JPEG quantization-table fingerprint + DCT-histogram double-quantization | Lukáš & Fridrich 2003; Pevný & Fridrich 2008 |
| `benford_dct` | First-digit Benford deviation on AC-DCT magnitudes | Fu, Shi & Su 2007 |
| `cfa_inconsistency` | Bayer demosaicing residual variance ratio | Popescu & Farid 2005 |
| `copy_move_phash` | 64-bit DCT pHash with shift-vector consensus | Zauner 2010, Christlein et al. 2012 |
| `prnu_consistency` | Internal sensor-noise residue consistency (block z-score) | Lukáš, Fridrich & Goljan 2006 |
| `lighting_consistency` | Block-wise dominant gradient direction dispersion | Johnson & Farid 2005 |

### Signature verifier upgrades (`fraud_detection/signature_verifier.py`)

- Vectorised Zhang-Suen skeletonization (≈ 10× faster than the v1 loop).
- Geometric features: aspect, density, centroid, contour count.
- Topological features: Euler number, loop count.
- Stroke-direction histogram (8-bin HOG of skeleton tangents).
- Stroke thickness *variance* — proxy for pen-pressure consistency.
- Reference vs questioned comparator fuses geometric/topological/direction
  distances + ORB descriptors + SSIM.

### Video analyzer upgrades (`fraud_detection/video_analyzer.py`)

- Per-frame run of the full image-forensics suite.
- Frame-duplicate detection via 64-bit perceptual hash.
- Optical-flow magnitude divergence (Farnebäck) for splice / face-swap cues.
- Temporal coherence score from inter-frame forensic-score deltas.

### Agent orchestrator (`fraud_detection/agent.py`)

- Claude `claude-sonnet-4-6` with prompt-cached system prompt and vision
  inputs for every image / signature.
- Tool-use loop: `run_image_forensics`, `analyze_document`, `analyze_video`,
  `verify_signature`, `analyze_signature`, `submit_report`.
- **Forensic provenance** in every report:
  - per-input SHA-256, MIME, size,
  - per-detector score / confidence / evidence,
  - algorithm-version manifest,
  - deterministic reproducibility hash,
  - UTC `generated_at` timestamp.
- Deterministic offline fallback when `anthropic` / API key is unavailable.

## Calibration

Run as a module — produces `calibration_report.json` next to the source:

```bash
python -m fraud_detection.calibration
```

The script:
1. Synthesises a labelled dataset (camera-origin pristine + splice / copy-move
   / re-compressed classes) using a Bayer-mosaic + demosaic pipeline so the
   CFA / ELA / double-JPEG signals exist on the synthetic data.
2. Runs every detector on every sample.
3. Computes per-detector AUC using Mann-Whitney U.
4. Blends AUC-derived weights with a literature-informed prior (60/40
   split, with a 3% floor so no detector is ever zero-weighted).
5. Selects risk thresholds by Youden's J on the fused score.

The current baked weights (in `image_forensics.DEFAULT_WEIGHTS`) and
thresholds (in `DEFAULT_THRESHOLDS`) are from this synthetic baseline.
**Real production deployments should re-run this against labelled domain
data** (e.g. CASIA, CoMoFoD, or your own annotated KYC / claims corpus)
and persist the resulting JSON as part of chain-of-custody.

## Honest caveat about "court-grade"

A pipeline becomes admissible in court when, in combination, it satisfies:

1. **Validation** on a labelled, domain-representative dataset, with
   reported false-positive / false-negative rates per detector and at
   the fusion level.
2. **Reproducibility**: deterministic outputs, versioned algorithms,
   pinned dependencies, signed evidence packages.
3. **Chain of custody**: cryptographic hashes of inputs, time-stamped
   audit trails, append-only log of analyzer invocations.
4. **Expert testimony**: a qualified examiner explaining the methodology
   and limitations to the trier of fact.
5. **Daubert / Frye admissibility**: peer-reviewed methodology, known
   error rates, general acceptance in the forensic community.

This module gives you (1) the *infrastructure* for items 1–3 (per-detector
scores with versions, deterministic fallback, reproducibility hash), and
(2) implementations of *peer-reviewed primitives* whose admissibility
question is well-trodden in the literature. **It does not by itself
constitute court-admissible evidence.** Use it as high-quality screening
that surfaces candidates for an expert reviewer, who will run dedicated
tools (e.g. Amped Authenticate, Belkasoft, X-Ways) and provide testimony.

## Install

**Fastest — one-command launcher** (creates a venv, installs deps, opens
the browser):

```bash
./run_local.sh        # macOS / Linux / WSL
./run_local.ps1       # Windows PowerShell
```

See [QUICKSTART.md](QUICKSTART.md) for the full local recipe.

**Manual:**

```bash
pip install -e .            # core: anthropic + Pillow + numpy
pip install -e .[all]       # adds opencv, scikit-image, pypdfium2, pytesseract
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
print("Reproducibility:", report.reproducibility_hash)
print("Algorithm versions:", report.algorithm_versions)

# Persist for chain-of-custody:
import json
json.dump(report.to_dict(), open("report.json", "w"), indent=2, default=str)
```

## CLI

```bash
fraud-detect statement.pdf
fraud-detect -k signature -r ref.png questioned.png --context "wire authorization"
fraud-detect liveness.mp4 --json
```

## Web UI

A polished, animated front-end ships in `web/`:

```bash
pip install -r web/requirements.txt
uvicorn web.app:app --reload --port 8000
# open http://localhost:8000
```

It gives you drag-and-drop uploads, per-file controls, signature
reference picking, an animated verdict ring, per-detector bar charts,
copy-to-clipboard reproducibility hashes, and a one-click JSON
report download. See `web/README.md` for production deployment notes.

## Risk bands (synthetic-calibrated; re-tune for production)

| Score | Risk | Suggested action |
| --- | --- | --- |
| `≥ 0.32` | high | Block flow, escalate to human review with `chain_of_evidence`. |
| `≥ 0.22` | medium | Hold + secondary verification. |
| `≥ 0.17` | low | Log + allow. |
| `< 0.17` | minimal | Allow. |

## Tests

```bash
pip install -e .[dev]
pytest -q
# 18 tests, ~1.5s; runs entirely offline (no API key needed).
```

## Module map

| File | Purpose |
| --- | --- |
| `fraud_detection/agent.py` | Claude tool-use orchestrator + offline fallback + provenance |
| `fraud_detection/detectors/*.py` | Per-detector forensic primitives |
| `fraud_detection/image_forensics.py` | Multi-detector fusion + EXIF flags |
| `fraud_detection/document_analyzer.py` | PDF rendering + per-page forensics + OCR text checks |
| `fraud_detection/video_analyzer.py` | Frame sampling + per-frame fusion + temporal + motion |
| `fraud_detection/signature_verifier.py` | Otsu + skeleton + topology + ORB + SSIM |
| `fraud_detection/calibration/` | Synthetic dataset generator + AUC-based weight tuning |
| `fraud_detection/cli.py` | `fraud-detect` console script |

## Roadmap toward stronger forensics

If you need to push closer to court-grade in your domain:

1. **Replace synthetic calibration with a labelled production set.**
   Re-run `python -m fraud_detection.calibration` (extended with a
   labelled-data loader) and check the resulting JSON into version
   control alongside the model.
2. **Plug in a deepfake CNN** for video face manipulation (FaceForensics++,
   X-CLIP). Add it as a new detector — the agent's tool contract is stable.
3. **Add jpegio** for direct DCT-coefficient reading instead of the
   spatial-domain double-quantization approximation used today.
4. **Per-camera PRNU fingerprints**: add a reference-fingerprint store
   keyed by claimed device, then do correlation-based verification.
5. **Ingest expert reviewer feedback** — turn screen-out / approve labels
   into a calibration-update loop.
