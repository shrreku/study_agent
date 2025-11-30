# StudyAgent Deployment Requirements (Pilot: 5–20 Users)

This document describes what is needed to deploy the StudyAgent **agentic tutor system** for a small pilot (roughly 5–20 users). It is based on the current repo structure and code paths.

The goal is to get a **stable, debuggable deployment**, not yet a hardened internet-scale SaaS.

---

## 1. High-Level Architecture

- **Backend**
  - FastAPI app (`backend/main.py`).
  - Exposes:
    - Generic agents (`/api/agent/tutor`, `/api/agent/doubt`, etc.).
    - MDP v2 tutor API: `/api/mdp/v2/*` (session, chat, transitions).
    - Tutor environment / pedagogy endpoints (three-layer MDP; wired via `backend/agents/tutor`).
    - Resource ingestion & search: `/api/resources/*`, `/api/search/*`, `/api/embeddings/*`.
    - Auth: `/api/auth/*`.
  - Runs under `uvicorn main:app` (see `backend/Dockerfile`).

- **Frontend**
  - Next.js 14 app (`frontend/`).
  - Key UI entrypoints:
    - `app/agents/tutor/page.js`: Tutor agent testing dashboard (calls `/api/agent/tutor`).
    - `app/mdp-chat/page.tsx`: MDP v2 chat UI (calls `/api/mdp/v2/*`).
    - `app/components/tutor/EnvironmentTutorPage.tsx` and related routes: environment-style tutor using `/api/tutor/*` endpoints.
  - Frontend talks to backend via `API_BASE` from `NEXT_PUBLIC_API_BASE_URL` (see `app/lib/api.js`).

- **Core Infra Services** (see `docker-compose.yml` and code):
  - **Postgres + pgvector** (primary persistent store).
  - **Neo4j** (concept / pedagogy knowledge graph).
  - **MinIO or GCS** (resource file storage).
  - **Redis + RQ worker** (background parsing/ingestion jobs).
  - **LLM provider** (OpenAI-compatible HTTP API configured via `OPENAI_API_BASE` + `OPENAI_API_KEY`, or compatible AimlAPI).

- **Supporting Components**
  - Ingestion pipeline: `backend/ingestion/pipeline.py` and `scripts/fast_ingest.py`.
  - Tutor MDP v2 + policies: `backend/mdp/*`, `backend/api/mdp_v2.py`.
  - Legacy/conversational tutor agents: `backend/agents/tutor/*`, `backend/agents/tutor_mdp.py`.

---

## 2. Required Services & Hosting Model

For a 5–20 user pilot you can run everything in a **single small cluster or VM group**, but keep services logically separate.

### 2.1 Minimum Services

- **Backend API**
  - Container built from `backend/Dockerfile`.
  - Exposes HTTP on port `8080` (mapped to `8000` locally via Docker Compose).

- **Frontend Web App**
  - Container built from `frontend/Dockerfile`.
  - Serves Next.js (SSR) on port `3000`.
  - Should be deployed behind the same or separate HTTPS endpoint and configured with `NEXT_PUBLIC_API_BASE_URL` pointing at backend.

- **Postgres (with pgvector)**
  - Use `pgvector/pgvector:pg16` or managed Postgres with pgvector.
  - Load schema from `infra/db/init/*.sql`:
    - `00-extensions.sql`: `uuid-ossp`, `vector`.
    - `01-schema.sql`, `02-chunks.sql`, `03-user-doubt.sql`, `04-tutor-session.sql`.
  - Backend auto-runs `ensure_schema()` on startup, but **you should run the SQL migrations explicitly** for clean environments.

- **Neo4j**
  - Version 5, configured with password from `NEO4J_PASSWORD`.
  - Backend ensures constraints at startup via `kg_pipeline.ensure_neo4j_constraints()`.

- **Object Storage**
  - MVP: MinIO (see `docker-compose.yml`).
  - Production-ish: prefer GCS by setting `NOTES_GCS_BUCKET` and related env.
  - Backend uses `core/storage.py` to abstract MinIO vs GCS.

- **Redis + RQ Worker**
  - Redis 7 container.
  - Background worker process running `backend/worker.py` (e.g., its own container built from backend image but with different CMD).

- **LLM Provider**
  - Any OpenAI-compatible endpoint.
  - Must support `chat/completions` with JSON output for prompts used in ingestion and tutoring.
  - For some scripts (e.g., `fast_ingest.py`) there is special handling for `google/gemini-2.5-flash-lite` via `call_json_chat`.

### 2.2 Recommended Hosting Setup

For a small pilot:

- **Option A – Single Docker host / VM (fastest to set up)**
  - Use `docker-compose.yml` largely as-is for a pilot behind a reverse proxy that terminates TLS.
  - Expose only frontend and backend ports externally; keep Postgres, Redis, Neo4j, MinIO internal to the Docker network.

- **Option B – Managed DBs + Cloud Run / ECS**
  - Build and deploy `backend` image to a managed container runtime (Cloud Run, ECS, etc.).
  - Use managed Postgres and managed Neo4j (or a self-hosted instance / AuraDB) reachable via private networking.
  - Run ingestion worker as a separate service using the same backend image but `CMD python worker.py`.
  - Use managed Redis (or an alternative queue) or keep Redis container on a small VM.

---

## 3. Environment Variables & Secrets

Use `.env.example` as the baseline; you will need a real `.env` in production.

### 3.1 Backend Essentials

Set at **backend container level** (see `.env.example` and `docker-compose.yml`):

- **LLM / Provider**
  - `OPENAI_API_BASE` – Base URL for OpenAI-compatible API (e.g. `https://api.openai.com/v1` or AimlAPI endpoint sans `/v1`).
  - `OPENAI_API_KEY` – Secret API key.
  - `LLM_MODEL_MINI` – Default main model (e.g. `gpt-4o` or `anthropic/claude-haiku-4.5`).
  - `LLM_MODEL_NANO` – Cheaper model for light tasks.
  - Optional: `LLM_RESPONSE_FORMAT_JSON`, `LLM_PREVIEW_MAX_TOKENS`.

- **Database**
  - Either `DATABASE_URL` (single DSN) **or** the `POSTGRES_*` vars:
    - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`.
  - For Docker network use host `postgres`; for external DB adjust accordingly.

- **Neo4j**
  - `NEO4J_URI` (e.g. `bolt://neo4j:7687` or managed URI).
  - `NEO4J_USER`, `NEO4J_PASSWORD`.

- **Object Storage**
  - Prod (GCS-first):
    - `NOTES_GCS_BUCKET`, `EXPORTS_GCS_BUCKET` and **GCP ADC** credentials via runtime environment.
  - Dev/MinIO:
    - `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`.
    - `MINIO_ENDPOINT` (e.g. `minio:9000`).
    - `MINIO_SECURE` (likely `false` for pilot).
    - `MINIO_BUCKET` (default `resources`).

- **Redis / RQ**
  - `REDIS_URL` (e.g. `redis://redis:6379/0`).

- **Auth / JWT**
  - `JWT_SECRET` – **must be strong and non-default** for pilot.
  - `JWT_ALGORITHM` – `HS256`.
  - `JWT_EXPIRES_MINUTES` – 7 days default, can reduce for pilot.
  - `AUTH_DEV_TOKEN` – remove or change from `test-token` in real environments (otherwise anyone can use the dev token).
  - `TEST_USER_ID` – optional default user id for dev.

- **Tutor / RL / SRL flags** (tune as needed for pilot)
  - `TUTOR_LLM_MODEL` – override tutor model if needed.
  - `TUTOR_LLM_POLICY_ENABLED` – leave `false` initially unless you want policy LLM active.
  - `TUTOR_SRL_MODE`, `TUTOR_SRL_PLANNING_ENABLED` etc. – keep default unless you’re explicitly running SRL/RL experiments.
  - Mastery tuning: `MASTERY_STEP_CORRECT`, `MASTERY_STEP_WRONG`, `MASTERY_DECAY_LAMBDA`.

- **Ingestion & KG**
  - `ENHANCED_CHUNKING_ENABLED`, `ENHANCED_TAGGING_ENABLED`, `PEDAGOGY_LLM_CLASSIFICATION`. For a small pilot, you can enable them if LLM budget allows.
  - `KG_ENHANCED_EXTRACTION_ENABLED` and associated thresholds to control Graph size and quality.
  - `EMBED_MODEL`, `EMBED_VERSION` for embedding pipeline.

- **Runtime**
  - `ENVIRONMENT=prod` for production-like logging (JSON logs from middleware).
  - `PROMPT_SET=baseline` (already default and used throughout tutor MDP.
  - `OTEL_ENABLE` if you plug into OpenTelemetry later.

### 3.2 Frontend Essentials

Set at **frontend build/runtime**:

- `NEXT_PUBLIC_API_BASE_URL` – must point to backend base URL (e.g. `https://studyagent-backend.myorg.com`).
- (Auth-related) Next.js app relies on `useAuth` hook and backend `/api/auth/*`; it doesn’t need secrets itself beyond these public URLs.

### 3.3 Secrets Management

- Do **not** store real `OPENAI_API_KEY`, `JWT_SECRET`, `NEO4J_PASSWORD` or DB passwords in the repo.
- Use your cloud provider’s secret manager or environment configuration.

---

## 4. Database & Storage Preparation

### 4.1 Postgres

1. **Create DB instance** with pgvector:
   - If using Compose: `pgvector/pgvector:pg16` image already sets this up.
   - If using managed DB, manually install `pgvector` and `uuid-ossp`.

2. **Apply schema** (`infra/db/init/*.sql`):
   - Run all scripts in order: `00-extensions.sql` → `01-schema.sql` → `02-chunks.sql` → `03-user-doubt.sql` → `04-tutor-session.sql`.
   - For managed environments, use your migration/DDL tool or run them manually.

3. **Verify**:
   - `SELECT extname FROM pg_extension;` should include `vector`.
   - Tables: `app_user`, `resource`, `chunk`, `job`, `tutor_session`, `tutor_turn`, `tutor_event`, etc.

### 4.2 Neo4j

1. Start Neo4j instance with a secure password.
2. Ensure backend can reach it via `NEO4J_URI`.
3. Backend startup will call `ensure_neo4j_constraints()`; watch logs for failure.

### 4.3 MinIO / GCS

- **MinIO (dev / simple pilot)**:
  - Start container from `docker-compose.yml`.
  - Access UI at `http://localhost:9000` in dev; in prod, keep it internal.
  - Ensure bucket specified by `MINIO_BUCKET` exists or let backend create on demand.

- **GCS (more production-like)**:
  - Create bucket `NOTES_GCS_BUCKET` (and `EXPORTS_GCS_BUCKET` if needed).
  - Configure service account and ADC for the backend environment.
  - Backend automatically chooses GCS when `NOTES_GCS_BUCKET` is non-empty.

### 4.4 Redis + Worker

1. Start Redis service.
2. Run worker process using backend environment:
   - As a container: same image as backend, with `CMD python worker.py`.
   - Ensure `REDIS_URL` and DB envs are shared with backend.
3. Confirm jobs:
   - Upload a resource via `/api/resources/upload` and verify a `job` row is created and processed by the worker.

---

## 5. LLM Configuration & Limits

For a 5–20 user pilot you need:

- **Reliable LLM endpoint** – OpenAI or compatible.
- Ensure sufficient **rate limits** for:
  - MDP v2 tutoring (`/api/mdp/v2/*`) – roughly 1–2 LLM calls per student turn.
  - Tutor environment endpoints.
  - Ingestion pipeline (LLM-heavy if `STANDARD`/`FULL` or `fast_ingest.py` with LLM prereqs).

**Recommended pilot defaults**:

- Set `USE_LLM_MOCK=0` in production.
- Keep `INGEST_MODE=standard` or `fast` initially; run `FULL` selectively on key resources.
- For ingestion scripts (`scripts/fast_ingest.py`), ensure `FAST_INGEST_MODEL` is supported by your provider.

---

## 6. Deployment Steps (End-to-End)

### 6.1 Prepare Configuration

- **Step 1** – Copy `.env.example` → `.env` and customise:
  - Set all secrets and external service URLs.
  - Turn `ENVIRONMENT=prod` for the deployed backend.
- **Step 2** – Decide storage mode (MinIO vs GCS) and update env accordingly.
- **Step 3** – Decide ingestion mode (`INGEST_MODE`) and retrieval thresholds.

### 6.2 Build & Run Locally (Final Smoke)

- From repo root:
  - `docker-compose up --build -d`
  - Visit `http://localhost:3000` and test:
    - Auth registration/login.
    - Upload a PDF in the UI (if the resource upload UI is wired).
    - Tutor agent at `/agents/tutor`.
    - MDP chat at `/mdp-chat`.
    - Environment tutor pages.
  - Check backend health: `curl http://localhost:8000/health`.

### 6.3 Production-like Deployment

Adapt to your platform, but the main steps:

- **Backend**
  - Build image `studyagent-backend` from `backend/Dockerfile`.
  - Deploy with env vars from `.env` (or secret manager) and `PORT` set by platform.
  - Ensure network access to Postgres, Neo4j, Redis, MinIO/GCS.

- **Worker**
  - Reuse the same backend image.
  - Command: `python worker.py`.
  - Same env profile as backend.

- **Frontend**
  - Build image `studyagent-frontend` from `frontend/Dockerfile`.
  - Set `NEXT_PUBLIC_API_BASE_URL` to backend URL.
  - Expose over HTTPS behind reverse proxy / load balancer.

- **Reverse Proxy / TLS**
  - Terminate TLS at gateway (NGINX/Envoy/Cloud Load Balancing/etc.).
  - Route `/` → frontend, `/api/*` → backend.

---

## 7. Operational Concerns for 5–20 Users

### 7.1 Capacity & Performance

- A single small VM/container for backend (e.g., 2 vCPU, 4–8 GB RAM) is typically enough for 5–20 concurrent users, assuming LLM latency dominates.
- Ensure:
  - Postgres has reasonable connection limits; backend uses simple connections (not async pool), plus ingestion pipeline’s pool.
  - Neo4j instance sized to hold concepts/edges for your pilot resources.

### 7.2 Logging & Metrics

- **Backend logs**
  - HTTP middleware logs requests; in `ENVIRONMENT=prod` logs are JSON-like; ensure your log pipeline collects them.
  - MDP v2 logs transitions and actions (`backend/mdp/*`); useful for debugging and RL data.

- **Metrics**
  - In-memory only (`backend/metrics.py`) for now.
  - For pilot, rely on logs + simple counters; for more serious deployment, plan to integrate Prometheus/Otel.

### 7.3 Error Handling & Recovery

- Monitor logs for:
  - DB connection errors (`psycopg2.OperationalError`).
  - Neo4j driver errors (will cause reduced KG functionality but tutor can still operate using RAG from chunks).
  - LLM errors / HTTP 4xx/5xx from provider.

- For ingestion failures:
  - `resource.status` and `job.status` columns capture phases; use them to surface errors.
  - Re-run ingestion via scripts or `/api/resources/{id}/reindex`.

### 7.4 Backups

- **Postgres**
  - Daily backup snapshot (managed service or cron `pg_dump`).
  - For a small pilot, nightly dumps may be enough.

- **Neo4j**
  - Use Neo4j backup tooling or snapshot the disk volume.

- **Object Storage**
  - If using MinIO, snapshot volume; if GCS, rely on bucket-level retention/versioning if required.

### 7.5 Access Control & Multi-User

- Users register via `/api/auth/register` and login via `/api/auth/login`;
  - Frontend uses `useAuth` hook (token-based).
  - Ensure HTTPS so JWT is not exposed in plaintext.
- For a 5–20 user closed pilot, you can:
  - Pre-create test accounts, or
  - Allow self-registration but restrict join link distribution.

---

## 8. Data Seeding & Ingestion Workflow

For a good experience, preload some study materials.

### 8.1 Option A – In-App Upload

- Use UI to upload PDFs, which:
  - Stores file in MinIO/GCS.
  - Creates `resource` row.
  - Queues parse job processed by `worker.py` (parse pages → `extracted_page`).
  - Run ingestion pipeline to create `chunk` rows + KG edges.

### 8.2 Option B – Scripts

- Use `scripts/fast_ingest.py` for fast ingestion & KG building:
  - `python scripts/fast_ingest.py path/to/document.pdf --title "Heat Transfer"`.
  - Requires direct DB/Neo4j access and LLM configuration.

- Alternatively, integrate the new `IngestPipeline` (`backend/ingestion/pipeline.py`) in a management script or API.

### 8.3 Verifying RAG

- Confirm:
  - `chunk.embedding` and `search_tsv` are populated.
  - Searching via `/api/search` returns relevant chunks.
  - Tutor agent returns context-backed explanations (see debug `source_chunk_ids`).

---

## 9. Tutor / MDP Entry Points to Expose

For the pilot, decide which flows you want users to try:

- **Classic Tutor Agent**
  - Frontend: `/agents/tutor`.
  - Backend: `/api/agent/tutor` → `backend/agents/tutor/agent.py`.
  - Uses resources/ingestion + KG for mastery and concept paths.

- **MDP v2 Tutor Playground**
  - Frontend: `/mdp-chat`.
  - Backend: `/api/mdp/v2/start`, `/api/mdp/v2/chat`, `/api/mdp/v2/button`, `/api/mdp/v2/transitions`.
  - Good for testing action selection and RL logging.

- **Environment Tutor Page**
  - Frontend: Environment tutor pages using `EnvironmentTutorPage`.
  - Backend: `/api/tutor/session/start`, `/api/tutor/pedagogy/session/*` wired into three-layer environment (see `ENVIRONMENT_IMPLEMENTATION.md`).

Ensure these routes are reachable and tested end-to-end before inviting users.

---

## 10. Pre-Pilot Checklist

Use this as a final gate before giving access to 5–20 people:

- **[ ]** Backend reachable over HTTPS at `{BACKEND_URL}`.
- **[ ]** Frontend reachable over HTTPS at `{FRONTEND_URL}` with `NEXT_PUBLIC_API_BASE_URL={BACKEND_URL}`.
- **[ ]** `GET {BACKEND_URL}/health` returns `{ "status": "ok" }`.
- **[ ]** Postgres schema applied; `chunk`, `tutor_session`, `tutor_turn` tables present.
- **[ ]** Neo4j reachable; `ensure_neo4j_constraints()` logs no fatal errors.
- **[ ]** MinIO or GCS configured and tested via `/api/resources/upload`.
- **[ ]** Redis reachable; `worker.py` process running and processing jobs.
- **[ ]** LLM provider configured; test `/api/llm/preview` and a simple tutor turn.
- **[ ]** At least one course PDF ingested and verified via `/api/resources/*` and `/api/search`.
- **[ ]** Tutor UI (`/agents/tutor`) produces sensible, grounded responses using that material.
- **[ ]** MDP v2 chat (`/mdp-chat`) works for at least one concept, and transitions appear at `/api/mdp/v2/transitions/{session_id}`.
- **[ ]** Basic auth flows (`/api/auth/register`, `/api/auth/login`, `/api/auth/me`) work in the UI.
- **[ ]** Logs are flowing to your preferred destination (Cloud logs, ELK, etc.).

Once all of the above are green, you should be in a good position to invite a small group of users to try the agentic system.
