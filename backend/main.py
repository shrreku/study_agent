from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from minio import Minio
import io
import os
import tempfile
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from redis import Redis
from rq import Queue
import psycopg2.extras
import logging
from psycopg2 import sql
from pydantic import BaseModel
from typing import List, Optional
import embed as embed_service
import requests
from dotenv import load_dotenv, find_dotenv
from llm import tag_and_extract, call_llm_for_tagging
from agents import orchestrator_dispatch
from agents.retrieval import hybrid_search

load_dotenv(find_dotenv(), override=True)

app = FastAPI(title="StudyAgent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


def get_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def get_db_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        # build DSN from parts
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "app")
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    conn = psycopg2.connect(dsn)
    # Register pgvector adapter on each connection if available
    try:
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
    except Exception:
        pass
    return conn


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Placeholder: accept any Bearer token for MVP
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth")
    return credentials.credentials


@app.post("/api/resources/upload")
async def upload_resource(file: UploadFile = File(...), title: str = "", token: str = Depends(require_auth)):
    # Enforce max size 100MB
    MAX_BYTES = 100 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    minio_client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET", "resources")
    # create bucket if doesn't exist
    try:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
    except Exception:
        # ignore errors for now
        pass

    object_name = f"{uuid.uuid4()}_{file.filename}"
    try:
        # MinIO client expects a file-like object; wrap bytes in BytesIO
        minio_client.put_object(bucket, object_name, data=io.BytesIO(contents), length=len(contents), content_type=file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store object: {e}")

    # write DB row
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO resource (id, title, filename, content_type, size_bytes, storage_path, created_at) VALUES (%s,%s,%s,%s,%s,%s,now()) RETURNING id, title, filename, size_bytes",
                (str(uuid.uuid4()), title or file.filename, file.filename, file.content_type, len(contents), f"{bucket}/{object_name}")
            )
            row = cur.fetchone()
            conn.commit()
    finally:
        conn.close()

    # enqueue parse job placeholder
    try:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis = Redis.from_url(redis_url)
        q = Queue("parse", connection=redis)
        job_id = str(uuid.uuid4())
        payload = {"resource_id": row["id"], "storage_path": f"{bucket}/{object_name}"}
        # create job row
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO job (id, resource_id, type, status, payload, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,now(),now())",
                            (job_id, row["id"], "parse", "queued", psycopg2.extras.Json(payload)))
                conn.commit()
        finally:
            conn.close()

        q.enqueue_call(func="backend.worker.process_parse_job", args=(job_id, row["id"], payload["storage_path"]))
    except Exception:
        # non-fatal: queueing failed
        job_id = None

    return {"resource_id": row["id"], "title": row["title"], "size": row["size_bytes"], "job_id": job_id}


@app.post("/api/resources/{resource_id}/chunk")
async def create_chunks(resource_id: str, token: str = Depends(require_auth)):
    # Validate resource_id
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")
    # fetch resource storage_path from DB
    conn = get_db_conn()
    storage = None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT storage_path FROM resource WHERE id=%s", (resource_id,))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="resource not found")
            storage = r["storage_path"]
    finally:
        conn.close()

    # resolve local path (sample dir or direct path). If missing, try MinIO download
    local_path = None
    sample_dir = os.path.join(os.getcwd(), "sample")
    if os.path.isdir(sample_dir):
        fname = storage.split("/")[-1]
        for root, dirs, files in os.walk(sample_dir):
            if fname in files:
                local_path = os.path.join(root, fname)
                break
    if not local_path and os.path.exists(storage):
        local_path = storage

    tmp_download_path = None
    if not local_path:
        # attempt to download from MinIO
        try:
            minio_client = get_minio_client()
            bucket, obj = storage.split("/", 1)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(obj)[1] or "")
            tf.close()
            tmp_download_path = tf.name
            minio_client.fget_object(bucket, obj, tmp_download_path)
            local_path = tmp_download_path
        except Exception as e:
            # cleanup and return helpful error
            if tmp_download_path and os.path.exists(tmp_download_path):
                try:
                    os.unlink(tmp_download_path)
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"resource not available locally and MinIO download failed: {e}")

    from chunker import structural_chunk_resource

    chunks = structural_chunk_resource(local_path)

    conn = get_db_conn()
    inserted = 0
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for c in chunks:
                # run LLM-based tagging with fallback heuristics
                try:
                    tags = tag_and_extract(c["full_text"])
                except Exception:
                    tags = {"chunk_type": None, "concepts": [], "math_expressions": []}

                chunk_type = tags.get("chunk_type")
                concepts = tags.get("concepts")
                math_expressions = tags.get("math_expressions")

                cur.execute(
                    "INSERT INTO chunk (id, resource_id, page_number, source_offset, full_text, chunk_type, concepts, math_expressions, created_at) VALUES (uuid_generate_v4(),%s,%s,%s,%s,%s,%s,%s,now()) RETURNING id",
                    (resource_id, c["page_number"], c["source_offset"], c["full_text"], chunk_type, concepts, math_expressions)
                )
                new_id = cur.fetchone()["id"]
                try:
                    if concepts:
                        merge_concepts_in_neo4j(concepts, new_id, c["full_text"][:160], resource_id)
                except Exception:
                    logging.exception("kg_merge_failed")
                inserted += 1
        conn.commit()
    finally:
        conn.close()

    # cleanup any temporary downloaded file
    if tmp_download_path:
        try:
            os.unlink(tmp_download_path)
        except Exception:
            pass

    return {"chunks_created": inserted}


@app.get("/api/resources/{resource_id}/chunks")
async def list_chunks(resource_id: str, limit: int = 25, offset: int = 0, token: str = Depends(require_auth)):
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text, page_number, chunk_type, concepts, math_expressions,
                       LEFT(full_text, 240) AS snippet, (embedding IS NOT NULL) AS has_embedding,
                       embedding_version, created_at
                FROM chunk
                WHERE resource_id=%s::uuid
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (resource_id, limit, offset),
            )
            rows = cur.fetchall()
        return {"chunks": rows, "limit": limit, "offset": offset}
    finally:
        conn.close()


class LlmPreviewRequest(BaseModel):
    text: str
    prompt_override: Optional[str] = None


@app.post("/api/llm/preview")
async def llm_preview(req: LlmPreviewRequest, token: str = Depends(require_auth)):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text required")
    try:
        out = call_llm_for_tagging(req.text, prompt_override=req.prompt_override)
        return out
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


class AgentRequest(BaseModel):
    # flexible payload forwarded to agents
    target_concepts: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    count: Optional[int] = None
    question: Optional[str] = None
    context_chunk_ids: Optional[List[str]] = None


@app.post("/api/agent/{agent_name}")
async def agent_endpoint(agent_name: str, body: AgentRequest, token: str = Depends(require_auth)):
    try:
        # convert pydantic model to dict and strip None fields
        payload = {k: v for k, v in body.dict().items() if v is not None}
        result = orchestrator_dispatch(agent_name, payload)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logging.exception("agent_error")
        raise HTTPException(status_code=500, detail="agent_error")


def ensure_schema():
    ddls = [
        """
CREATE EXTENSION IF NOT EXISTS vector;
""",
        """
CREATE TABLE IF NOT EXISTS app_user (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE,
  password_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS resource (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES app_user(id) NULL,
  title TEXT,
  filename TEXT,
  content_type TEXT,
  size_bytes BIGINT,
  storage_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS extracted_page (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  resource_id UUID REFERENCES resource(id) ON DELETE CASCADE,
  page_number INT,
  raw_text TEXT,
  ocr_confidence NUMERIC NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS job (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NULL,
  resource_id UUID REFERENCES resource(id) NULL,
  type TEXT,
  status TEXT,
  payload JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS chunk (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  resource_id UUID REFERENCES resource(id) ON DELETE CASCADE,
  page_number INT,
  source_offset INT,
  chunk_type TEXT,
  concepts TEXT[],
  text_snippet TEXT,
  full_text TEXT,
  math_expressions TEXT[],
  figure_ids UUID[],
  difficulty SMALLINT,
  embedding vector(384),
  embedding_version TEXT,
  search_tsv tsvector,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
""",
    ]
    conn = None
    try:
        conn = get_db_conn()
        # register pgvector adapter if available so psycopg2 can pass Python lists
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
        except Exception:
            # optional dependency; continue without adapter if not installed
            pass
        with conn.cursor() as cur:
            for ddl in ddls:
                cur.execute(ddl)
        conn.commit()
    except Exception as e:
        logging.exception("Failed to ensure DB schema: %s", e)
    finally:
        if conn:
            conn.close()


@app.get("/api/llm/smoke")
async def llm_smoke(token: str = Depends(require_auth)):
    """Simple AimlAPI/OpenAI-compatible smoke test."""
    base = os.getenv("OPENAI_API_BASE") or os.getenv("AIMLAPI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    model = os.getenv("LLM_MODEL_MINI", "openai/gpt-5-mini-2025-08-07")
    if not base:
        raise HTTPException(status_code=400, detail="OPENAI_API_BASE not set")
    if not key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    url = base.rstrip("/") + ("/chat/completions" if base.rstrip("/").endswith("/v1") else "/v1/chat/completions")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_http_error: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:512])
    try:
        j = r.json()
        txt = j["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=502, detail=f"invalid llm response: {r.text[:256]}")
    return {"ok": True, "reply": txt, "model": model}


def merge_concepts_in_neo4j(concepts: List[str], chunk_id: str, snippet: str, resource_id: str):
    # Lazy import to avoid import error when dependency not yet installed in image
    try:
        from neo4j import GraphDatabase
    except Exception:
        logging.exception("neo4j_driver_import_failed")
        return
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        def _tx(tx):
            for cname in concepts:
                tx.run(
                    """
                    MERGE (c:Concept {name: $cname})
                    MERGE (ch:Chunk {id: $chunk_id})
                    SET ch.snippet = $snippet
                    MERGE (r:Resource {id: $resid})
                    MERGE (c)-[:EXPLAINED_BY]->(ch)
                    MERGE (ch)-[:PART_OF]->(r)
                    """,
                    cname=cname, chunk_id=chunk_id, snippet=snippet[:300], resid=resource_id,
                )
        with driver.session() as session:
            session.execute_write(_tx)
    except Exception:
        logging.exception("neo4j_merge_failed")


@app.on_event("startup")
def on_startup():
    # Ensure DB schema exists for local/dev testing
    try:
        ensure_schema()
    except Exception:
        logging.exception("Error ensuring schema on startup")


class UpsertRequest(BaseModel):
    chunk_ids: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    embedding_version: Optional[str] = None


@app.post("/api/embeddings/upsert")
async def embeddings_upsert(req: UpsertRequest, token: str = Depends(require_auth)):
    """Compute embeddings for provided texts or for existing chunk_ids and upsert into DB."""
    if not req.texts and not req.chunk_ids:
        raise HTTPException(status_code=400, detail="provide texts or chunk_ids to embed")

    conn = get_db_conn()
    try:
        if req.texts:
            texts = req.texts
            vecs = embed_service.embed_texts(texts)
            with conn.cursor() as cur:
                for i, v in enumerate(vecs):
                    # insert a placeholder chunk row with generated id
                    cur.execute("SELECT uuid_generate_v4()::text")
                    new_id = cur.fetchone()[0]
                    cur.execute("INSERT INTO chunk (id, full_text, embedding, embedding_version, created_at) VALUES (%s,%s,%s,%s,now())",
                                (new_id, texts[i], v, req.embedding_version or os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")))
            conn.commit()
            return {"inserted": len(vecs)}

        if req.chunk_ids:
            # Normalize to UUID strings and cast to uuid[] in query
            normalized_ids = []
            for cid in req.chunk_ids:
                try:
                    normalized_ids.append(str(uuid.UUID(str(cid))))
                except Exception:
                    continue

            if not normalized_ids:
                raise HTTPException(status_code=400, detail="no valid chunk_ids provided")

            # fetch chunk texts for given ids
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, full_text FROM chunk WHERE id = ANY(%s::uuid[])", (normalized_ids,))
                rows = cur.fetchall()
            texts = [r["full_text"] for r in rows]
            ids = [str(r["id"]) for r in rows]
            vecs = embed_service.embed_texts(texts)
            with conn.cursor() as cur:
                for cid, vec in zip(ids, vecs):
                    cur.execute("UPDATE chunk SET embedding=%s, embedding_version=%s, updated_at=now() WHERE id=%s::uuid",
                                (vec, req.embedding_version or os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09"), cid))
            conn.commit()
            return {"updated": len(vecs)}
    finally:
        conn.close()


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10


@app.post("/api/search")
async def search(req: SearchRequest, token: str = Depends(require_auth)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    try:
        rows = hybrid_search(req.query, k=int(req.k or 10))
        return {"results": rows}
    except Exception as e:
        logging.exception("search_failed")
        raise HTTPException(status_code=502, detail="search_failed")
