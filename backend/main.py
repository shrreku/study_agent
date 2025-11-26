from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import time
import json
from dotenv import load_dotenv, find_dotenv
from metrics import MetricsCollector
from api.resources import router as resources_router
from api.notes import router as notes_router
from api.embeddings import router as embeddings_router
from api.search import router as search_router
from api.llm_endpoints import router as llm_router
from api.agent import router as agent_router
from api.tutor_mdp import router as tutor_mdp_router
from api.mdp_router import router as mdp_router
from api.analytics import router as analytics_router
from api.metrics_endpoints import router as metrics_router
# Disabled: depends on deleted tutor modules (critic, etc.)
# from api.rl_tools import router as rl_router
from api.bench import router as bench_router
from api.kg import router as kg_router
from api.auth import router as auth_router
from core.db import ensure_schema
from kg_pipeline import ensure_neo4j_constraints
from core.auth import decode_token, AUTH_DEV_TOKEN

load_dotenv(find_dotenv(), override=True)

# Configure basic logging for the app; allow override via LOG_LEVEL env
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

app = FastAPI(title="StudyAgent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular routers after app and middleware are initialized
app.include_router(resources_router)
app.include_router(notes_router)
app.include_router(embeddings_router)
app.include_router(search_router)
app.include_router(llm_router)
app.include_router(agent_router)
app.include_router(tutor_mdp_router)
app.include_router(mdp_router)
app.include_router(analytics_router)
app.include_router(metrics_router)
app.include_router(bench_router)
app.include_router(kg_router)
# Disabled: depends on deleted tutor modules
# app.include_router(rl_router)
app.include_router(auth_router)

## security handled in core.auth; routers declare dependencies


## Legacy resources endpoints removed in favor of api/resources.py


## Legacy create_chunks removed (see api/resources.py)


## Legacy parse endpoint removed (see api/resources.py)

## Legacy list_chunks removed (see api/resources.py)


## Legacy get_job_status removed (see api/resources.py)


# Moved: /api/llm/preview is now in api/llm_endpoints.py


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


## Agent endpoints moved to api/agent.py


## Quiz answer moved to api/agent.py


## Analytics endpoint moved to api/analytics.py


## Local ensure_schema removed; using core.db.ensure_schema()


# Moved: /api/llm/smoke is now in api/llm_endpoints.py


# Moved: /api/metrics is now in api/metrics_endpoints.py


# Moved: KG merge helper is now in core/kg.py


@app.on_event("startup")
def on_startup():
    # Ensure DB schema exists for local/dev testing
    try:
        ensure_schema()
    except Exception:
        logging.exception("Error ensuring schema on startup")

    try:
        ensure_neo4j_constraints()
    except Exception:
        logging.exception("Error ensuring Neo4j constraints on startup")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    user_id = None
    session_id = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")

    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
        if AUTH_DEV_TOKEN and token == AUTH_DEV_TOKEN:
            user_id = os.getenv("TEST_USER_ID") or "00000000-0000-0000-0000-000000000001"
        else:
            try:
                payload = decode_token(token)
                user_id = str(payload.get("sub")) if payload.get("sub") is not None else None
            except Exception:
                user_id = None

    response = await call_next(request)
    duration_ms = int((time.time() - start_time) * 1000)

    log_record = {
        "event": "request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "user_id": user_id,
        "session_id": session_id,
    }

    if os.getenv("ENVIRONMENT") == "prod":
        logging.info(json.dumps(log_record))
    else:
        logging.info(
            "request method=%s path=%s status_code=%s duration_ms=%s user_id=%s session_id=%s",
            log_record["method"],
            log_record["path"],
            log_record["status_code"],
            log_record["duration_ms"],
            log_record["user_id"],
            log_record["session_id"],
        )

    return response


# Moved: /api/embeddings/upsert is now in api/embeddings.py


# Moved: /api/search is now in api/search.py


# Moved: /api/admin/recompute-search-tsv is now in api/search.py


# Optional bench endpoint removed from main (can be re-added under /api if needed)


# Moved: /api/resources/{resource_id}/reindex is now in api/resources.py
