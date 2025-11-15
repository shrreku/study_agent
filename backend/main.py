from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv, find_dotenv
from metrics import MetricsCollector
from api.resources import router as resources_router
from api.embeddings import router as embeddings_router
from api.search import router as search_router
from api.llm_endpoints import router as llm_router
from api.agent import router as agent_router
from api.analytics import router as analytics_router
from api.metrics_endpoints import router as metrics_router
from api.rl_tools import router as rl_router
from api.bench import router as bench_router
from api.kg import router as kg_router
from core.db import ensure_schema
from kg_pipeline import ensure_neo4j_constraints

load_dotenv(find_dotenv(), override=True)

# Configure basic logging for the app; allow override via LOG_LEVEL env
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")

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
app.include_router(embeddings_router)
app.include_router(search_router)
app.include_router(llm_router)
app.include_router(agent_router)
app.include_router(analytics_router)
app.include_router(metrics_router)
app.include_router(bench_router)
app.include_router(kg_router)
app.include_router(rl_router)

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


# Moved: /api/embeddings/upsert is now in api/embeddings.py


# Moved: /api/search is now in api/search.py


# Moved: /api/admin/recompute-search-tsv is now in api/search.py


# Optional bench endpoint removed from main (can be re-added under /api if needed)


# Moved: /api/resources/{resource_id}/reindex is now in api/resources.py
