"""FastAPI front-end for the fraud-detection agent.

Run (dev):
    uvicorn web.app:app --reload --port 8000

Run (prod, single-host):
    FRAUD_AUTH_TOKEN=$(openssl rand -hex 32) \\
    FRAUD_ALLOWED_ORIGINS=https://console.mythosbank.example.com \\
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 4

Configuration (all optional):
    FRAUD_AUTH_TOKEN          bearer token required for /api/analyze
    FRAUD_ALLOWED_ORIGINS     comma-separated CORS origins (default: same-origin)
    FRAUD_MAX_BODY_MB         per-request body cap, default 50
    FRAUD_REPORT_DIR          if set, every report is persisted there
                              keyed by reproducibility_hash (audit trail)
    ANTHROPIC_API_KEY         passed straight through to the agent
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fraud_detection import FraudDetectionAgent, FraudInput

# ---------------------------------------------------------------- config
ROOT = Path(__file__).parent
STATIC = ROOT / "static"

AUTH_TOKEN = os.environ.get("FRAUD_AUTH_TOKEN", "").strip() or None
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("FRAUD_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
MAX_BODY_MB = int(os.environ.get("FRAUD_MAX_BODY_MB", "50"))
MAX_BODY_BYTES = MAX_BODY_MB * 1024 * 1024
REPORT_DIR_ENV = os.environ.get("FRAUD_REPORT_DIR", "").strip()
REPORT_DIR = Path(REPORT_DIR_ENV).expanduser() if REPORT_DIR_ENV else None
if REPORT_DIR:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- logging
logging.basicConfig(
    level=os.environ.get("FRAUD_LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s","name":"%(name)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("fraud.web")

# ---------------------------------------------------------------- app
app = FastAPI(
    title="MythosBank Fraud Detection",
    version="0.2.0",
    docs_url="/api/docs" if os.environ.get("FRAUD_ENABLE_DOCS") else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if os.environ.get("FRAUD_ENABLE_DOCS") else None,
)
app.add_middleware(GZipMiddleware, minimum_size=2048)
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

# Cache one agent instance — analyzers are stateless and re-use lets the
# Anthropic SDK's prompt cache hit across requests.
_agent: FraudDetectionAgent | None = None


def get_agent() -> FraudDetectionAgent:
    global _agent
    if _agent is None:
        _agent = FraudDetectionAgent()
    return _agent


# ---------------------------------------------------------------- middleware
@app.middleware("http")
async def access_log_and_size_guard(request: Request, call_next):
    t0 = time.perf_counter()
    rid = uuid.uuid4().hex[:12]
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        log.warning(f"req={rid} 413 body too large ({cl} bytes)")
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds {MAX_BODY_MB} MB cap."},
        )
    try:
        response = await call_next(request)
    except Exception:
        log.exception(f"req={rid} unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        f"req={rid} {request.method} {request.url.path} "
        f"-> {response.status_code} ({elapsed_ms:.1f}ms)"
    )
    response.headers["X-Request-ID"] = rid
    return response


def require_auth(authorization: Optional[str]) -> None:
    if AUTH_TOKEN is None:
        return  # auth disabled (dev mode)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    presented = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(presented, AUTH_TOKEN):
        raise HTTPException(401, "invalid bearer token")


# ---------------------------------------------------------------- routes
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Health probe used by the UI and any external uptime checker."""
    agent = get_agent()
    return {
        "status": "ok",
        "version": "0.2.0",
        "model": agent.model,
        "online": agent._build_client() is not None,
        "auth_required": AUTH_TOKEN is not None,
        "max_body_mb": MAX_BODY_MB,
        "reports_persisted": REPORT_DIR is not None,
    }


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    kinds: list[str] = Form(...),
    labels: Optional[list[str]] = Form(None),
    references: Optional[list[str]] = Form(None),
    context: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    require_auth(authorization)

    if len(files) != len(kinds):
        raise HTTPException(400, "kinds[] must be the same length as files[]")
    if labels and len(labels) != len(files):
        raise HTTPException(400, "labels[] must match files[] length when provided")

    work = Path(tempfile.mkdtemp(prefix=f"mb_fraud_{uuid.uuid4().hex[:8]}_"))
    try:
        # Stage all uploads, enforcing the size cap as we stream.
        on_disk: dict[str, Path] = {}
        total_bytes = 0
        for upload in files:
            target = work / _sanitize(upload.filename or f"upload_{uuid.uuid4().hex[:6]}")
            with target.open("wb") as f:
                while chunk := await upload.read(1 << 16):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_BODY_BYTES:
                        raise HTTPException(
                            413,
                            f"Cumulative upload exceeds {MAX_BODY_MB} MB cap.",
                        )
                    f.write(chunk)
            on_disk[upload.filename or target.name] = target

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

        # Drop reference-only files so we don't double-analyze them.
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
        payload = json.loads(report.to_json())

        # Persist for audit if configured.
        if REPORT_DIR and report.reproducibility_hash:
            audit_path = REPORT_DIR / f"{report.reproducibility_hash}.json"
            audit_path.write_text(json.dumps(payload, indent=2))
            log.info(
                f"persisted report sha={report.reproducibility_hash[:12]} "
                f"verdict={report.verdict} score={report.score}"
            )

        return JSONResponse(payload)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- helpers
def _sanitize(name: str) -> str:
    keep = "._-"
    cleaned = "".join(c for c in name if c.isalnum() or c in keep)
    return cleaned or f"upload_{uuid.uuid4().hex[:6]}"


# Mount the static assets last so /api/* and / take precedence.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- startup banner
@app.on_event("startup")
async def _startup() -> None:
    log.info(
        f"fraud-detection web up: auth={'on' if AUTH_TOKEN else 'OFF (dev)'} "
        f"cors={len(ALLOWED_ORIGINS)} body_cap={MAX_BODY_MB}MB "
        f"persist={'on' if REPORT_DIR else 'off'}"
    )
