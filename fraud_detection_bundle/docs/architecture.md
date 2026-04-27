# Architecture

## Why an agent and not a classifier?

Falsification spans heterogeneous media (PDF statements, JPEG IDs, MP4
liveness clips, scanned signatures). A monolithic classifier doesn't fit
because:

1. The right *evidence* differs by modality — ELA matters for JPEGs;
   temporal noise jumps matter for video; geometric features matter for
   signatures.
2. A score alone is not actionable. Operations needs an auditable
   justification (what was checked, what was found).

The agent pattern lets Claude:

- Pick the right tool per input,
- Visually corroborate or downgrade local detector hits,
- Render a structured `FraudReport` that humans and downstream systems can
  consume.

## Tool-use contract

Each tool returns a JSON-serializable result. The agent appends each tool
result to the conversation, and Claude is instructed to terminate by
calling `submit_report`. The orchestrator caps the loop at
`tool_budget` iterations to bound cost.

Prompt caching is applied to the system prompt so repeat invocations within
the cache window pay only the input-token delta.

## Forensic primitives — one-liners

- **ELA** — JPEG re-encode + per-pixel residual; localized splices appear
  as bright regions because pristine areas re-encode at the original loss
  level while edited areas don't.
- **EXIF flags** — missing camera Make/Model on JPEGs from phones,
  Photoshop/GIMP signatures in `Software`, mismatched DateTime vs
  DateTimeOriginal.
- **Copy-move** — DCT-style block hashing; matched non-adjacent blocks
  imply clone-stamped regions.
- **Noise residue** — Laplacian high-pass + block variance heterogeneity;
  splices disrupt sensor noise consistency.
- **Signature features** — Otsu threshold → bbox → density, centroid,
  stroke thickness from skeletonization, contour count.
- **ORB / SSIM** — keypoint distance + structural similarity for
  reference-vs-questioned pairs.
- **Temporal video** — per-frame ELA score; large jumps between adjacent
  samples indicate cuts / face-swaps / re-encoded segments.

## Calibration

The fusion weights in `ImageForensics.analyze` and the document/video
combinators are intentionally conservative. The agent's role is to refine
these — high-precision rejections come from Claude's vision corroboration,
not from the local score alone.

## Extending

To plug in a stronger model (e.g. a deepfake classifier or a
signature-verification CNN):

1. Implement an analyzer with an `analyze(...)` method returning a
   dataclass with a `to_dict()`.
2. Add a tool definition in `agent.TOOLS` and a dispatch branch in
   `FraudDetectionAgent._dispatch_tool`.
3. Update the system prompt to mention when the new tool should be used.

The rest of the pipeline (vision attachment, report shape, offline
fallback) remains unchanged.
