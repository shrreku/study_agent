# StudyAgent — Project skeleton

This repo contains a minimal skeleton for StudyAgent MVP.

Services:

- backend: FastAPI app (http://localhost:8000)
- frontend: Next.js app (http://localhost:3000)
- postgres: Postgres with pgvector extension (5432)
- neo4j: Neo4j (7474/7687)
- minio: MinIO (9000)
- redis: Redis (6379)

Run locally (requires Docker & Docker Compose):

```bash
# build and start
docker-compose up --build -d

# check containers
docker-compose ps

# health check backend
curl -v http://localhost:8000/health
# expected: {"status":"ok"}

# check pgvector extension (inside postgres container)
docker exec -it $(docker ps -qf "ancestor=postgres:15") psql -U postgres -d app -c "SELECT extname FROM pg_extension;"
# expected row contains 'vector'

# check neo4j
curl -u neo4j:neo4j http://localhost:7474/

# check MinIO UI
# open http://localhost:9000 with credentials minioadmin/minioadmin
```

Environment

Copy `.env.example` to `.env` at the repo root and adjust as needed:

```
# see .env.example for a full list; highlights:
# deterministic local dev/tests
USE_LLM_MOCK=1
ALLOW_PREVIEW_TESTS=0
# retrieval tunables
RETRIEVAL_SIM_WEIGHT=0.7
RETRIEVAL_BM25_WEIGHT=0.3
RETRIEVAL_RESOURCE_BOOST=1.0
RETRIEVAL_PAGE_PROXIMITY=false
```

Development (without Docker):

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
# open http://localhost:3000
# Agents UI at http://localhost:3000/agents
```

## API Notes — Sprint 3 Contract Fixes

- Jobs API
  - `GET /api/jobs/{job_id}` (auth required) returns `{job_id, status, payload, created_at, updated_at}`.
  - Quick test:
    ```bash
    curl -H "Authorization: Bearer test-token" http://localhost:8000/api/jobs/$JOB_ID | jq
    ```
- Daily Quiz response
  - `POST /api/agent/daily-quiz` now returns both keys for compatibility: `{ quiz: [...], items: [...] }`.
- Doubt API request
  - `POST /api/agent/doubt` accepts `question_text` as an alias of `question`.

## Sprint 3 — Retrieval Bench & Smoke

- Bench endpoint
  - `POST /api/bench/pk` with body:
    ```json
    {
      "queries": ["heat flux", "boundary layer"],
      "k": 5,
      "sim_weight": 0.7,
      "bm25_weight": 0.3,
      "resource_boost": 1.0,
      "page_proximity_boost": false
    }
    ```
  - Response includes per-query `ids`, `scores`, and `elapsed_ms`.
  - Useful env defaults (can be overridden in request): `RETRIEVAL_SIM_WEIGHT`, `RETRIEVAL_BM25_WEIGHT`, `RETRIEVAL_RESOURCE_BOOST`, `RETRIEVAL_PAGE_PROXIMITY`.

- Smoke script
  - `bash scripts/smoke.sh` exercises:
    - `GET /health`
    - `POST /api/llm/preview` (set `USE_LLM_MOCK=1` locally for stability)
    - `POST /api/admin/recompute-search-tsv`
    - `POST /api/bench/pk`

## CI

- GitHub Actions workflow runs backend tests deterministically with `USE_LLM_MOCK=1`.

For a fuller, compact OpenAPI, see `tickets/s4-a.md` and the technical overview `docs/overview_and_plan.md`.
