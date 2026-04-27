# Forensic Integrity Console

Animated, classy, dark-themed front-end for the fraud-detection agent.
FastAPI backend + a dependency-free vanilla-JS single-page client.

```
web/
├── app.py              FastAPI service (POST /api/analyze, GET /api/health)
├── requirements.txt    fastapi · uvicorn · python-multipart
└── static/
    ├── index.html
    ├── styles.css      dark theme, glassmorphism, animations
    └── app.js          drag-drop, upload, animated render of the report
```

## Run

```bash
# from fraud_detection_bundle/
pip install -e .[all]
pip install -r web/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; runs in offline mode without it
uvicorn web.app:app --reload --port 8000
```

Open `http://localhost:8000`.

## What you get

- Drag-and-drop upload zone with file-kind auto-detection
- Per-file controls: image / document / video / signature, plus a reference
  picker for signature-vs-signature comparisons
- Free-form context box (passed straight to Claude when online)
- Live animated forensic report:
  - Verdict tile with animated score ring (green → gold → amber → red)
  - One card per input with badge + per-detector animated bars
  - Recommendations list
  - Provenance block: SHA-256, algorithm versions, reproducibility hash
    (click-to-copy), generated_at timestamp
  - Collapsible raw-JSON view + one-click download as a `.json` audit file
- Status pill shows whether the backend is connected to Claude or running
  in deterministic offline mode

## Production deployment notes

The `app.py` here is intentionally minimal. Before exposing it externally:

1. Put it behind your existing banking-app auth + CSRF.
2. Add CORS rules tuned to your domain.
3. Cap request body size at the reverse proxy (e.g. 50 MB).
4. Persist `FraudReport.to_dict()` to your audit store keyed by
   `reproducibility_hash`, instead of returning it raw to the client.
5. Run uvicorn with `--workers N` behind a TLS-terminating gateway.
