"""FastAPI front-end for the fraud-detection agent.

Run:
    pip install -r web/requirements.txt
    uvicorn web.app:app --reload --port 8000

Then open http://localhost:8000

The backend:
  - Saves uploads to a per-request tempdir.
  - Routes them through fraud_detection.FraudDetectionAgent.
  - Returns the structured FraudReport as JSON.
  - Cleans up the tempdir after the request.

There is no auth here. For a production deploy put it behind your normal
banking-app auth, set CORS / CSRF policies, and persist reports to your
audit store rather than returning raw JSON to the client.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fraud_detection import FraudDetectionAgent, FraudInput

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

app = FastAPI(title="MythosBank Fraud Detection", version="0.2.0")

# Cache one agent instance — analyzers are stateless and re-using the
# instance gives us prompt-cache benefits when running with the SDK.
_agent: FraudDetectionAgent | None = None


def get_agent() -> FraudDetectionAgent:
    global _agent
    if _agent is None:
        _agent = FraudDetectionAgent()
    return _agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Health probe used by the UI to confirm the backend is alive."""
    agent = get_agent()
    return {
        "status": "ok",
        "model": agent.model,
        "online": agent._build_client() is not None,
    }


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    kinds: list[str] = Form(...),
    labels: Optional[list[str]] = Form(None),
    references: Optional[list[str]] = Form(None),
    context: Optional[str] = Form(None),
) -> JSONResponse:
    """Accept N files plus parallel arrays describing each one.

    `kinds` / `labels` / `references` are equal-length to `files`. A
    reference of "" or "-" means "no reference". Reference uploads are
    appended to `files` AFTER the questioned signatures and addressed by
    name in `references`.
    """
    if len(files) != len(kinds):
        raise HTTPException(400, "kinds[] must be the same length as files[]")
    if labels and len(labels) != len(files):
        raise HTTPException(400, "labels[] must match files[] length when provided")

    work = Path(tempfile.mkdtemp(prefix=f"mb_fraud_{uuid.uuid4().hex[:8]}_"))
    try:
        # Stage all uploads first, build a name -> on-disk path map.
        on_disk: dict[str, Path] = {}
        for upload in files:
            target = work / _sanitize(upload.filename or f"upload_{uuid.uuid4().hex[:6]}")
            with target.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            on_disk[upload.filename or target.name] = target

        # Build FraudInput list. Reference signatures are matched by
        # filename — the UI sends the original filename in `references[i]`
        # for signature inputs.
        agent_inputs: list[FraudInput] = []
        for i, upload in enumerate(files):
            kind = kinds[i].strip().lower()
            if kind not in ("image", "document", "video", "signature", "auto", "unknown"):
                raise HTTPException(400, f"Unknown kind '{kind}' for input {i}")
            if kind == "auto":
                kind = "unknown"

            label = (labels[i].strip() if labels and labels[i] else None) or None
            ref_path: Optional[str] = None
            if references and i < len(references):
                ref_name = (references[i] or "").strip()
                if ref_name and ref_name not in ("-", "none"):
                    if ref_name not in on_disk:
                        raise HTTPException(
                            400,
                            f"Reference '{ref_name}' for input {i} was not uploaded.",
                        )
                    ref_path = str(on_disk[ref_name])

            input_path = on_disk[upload.filename or list(on_disk)[i]]
            agent_inputs.append(
                FraudInput(
                    path=str(input_path),
                    kind=kind,  # type: ignore[arg-type]
                    reference_path=ref_path,
                    label=label,
                )
            )

        # Filter out reference-only files (those that appear in references[])
        # so we don't double-analyze them.
        ref_filenames = {
            (references[i] or "").strip()
            for i in range(len(files))
            if references and i < len(references)
        }
        agent_inputs = [
            ai
            for ai, upload in zip(agent_inputs, files)
            if (upload.filename or "") not in ref_filenames
        ]
        if not agent_inputs:
            raise HTTPException(400, "No analyzable inputs after filtering references.")

        report = get_agent().run(agent_inputs, context=context)
        return JSONResponse(json.loads(report.to_json()))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _sanitize(name: str) -> str:
    keep = "._-"
    cleaned = "".join(c for c in name if c.isalnum() or c in keep)
    return cleaned or f"upload_{uuid.uuid4().hex[:6]}"


# Mount the static assets last so /api/* and / take precedence.
app.mount("/static", StaticFiles(directory=STATIC), name="static")
