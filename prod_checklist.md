## PROD-07 Ops Checklist (Vector DB + KG in Prod)

### 1. Cloud SQL Postgres + pgvector

- **Create / identify Cloud SQL instance**
  - **[ ]** Instance running Postgres 14+ (or version supported by pgvector).
  - **[ ]** Network/Cloud Run connectivity configured (VPC connector or public + IAM).

- **Enable extensions & schema**
  - **[ ]** Connect to the DB (psql / Cloud SQL console) and run:
    - `CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`
    - `CREATE EXTENSION IF NOT EXISTS vector;`
  - **[ ]** Point the backend to this DB and let [ensure_schema()](cci:1://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/core/db.py:31:0-285:24) run on startup:
    - Set `DATABASE_URL` (preferred) or `POSTGRES_*` env vars.
  - **[ ]** Verify schema:
    - Table `chunk` exists.
    - Column `embedding vector(384)` exists.
    - Index `idx_chunk_search_tsv` exists.

- **Configure app env for retrieval**
  - **[ ]** Ensure `DATABASE_URL` points to Cloud SQL.
  - **[ ]** Optionally tune:
    - `RETRIEVAL_SIM_WEIGHT` (default 0.7)
    - `RETRIEVAL_BM25_WEIGHT` (default 0.3)
    - `RETRIEVAL_RESOURCE_BOOST`, `RETRIEVAL_PAGE_PROXIMITY` (optional).

---

### 2. Ingestion + Embedding Verification

- **Configure ingestion worker**
  - **[ ]** Worker service/job uses the same `DATABASE_URL`.
  - **[ ]** `INGEST_LLM_MODEL` is set to the fast model you want in prod.
  - **[ ]** Storage envs are correct (`NOTES_GCS_BUCKET` or MinIO).

- **Run a sample ingestion**
  - **[ ]** In the frontend [/notes](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/frontend/app/notes:0:0-0:0) page:
    - Upload a small PDF.
    - Trigger ingestion.
    - Wait for status to reach `ready`.
  - **[ ]** In DB, confirm:
    - `chunk` rows exist for that `resource_id`.
    - `embedding` column is non-null for those chunks.

---

### 3. Vector Search Smoke Tests

- **Search API smoke test**
  - **[ ]** Call `POST /api/search` with a query you expect to hit your sample notes:
    - Body: `{"query": "conduction", "k": 5, "resource_id": "<your_resource_id>"}`.
  - **[ ]** Verify:
    - 200 response.
    - Non-empty `results`.
    - Reasonable `snippet` text and scores.

- **Bench endpoint smoke test**
  - **[ ]** Call `POST /api/bench/pk`:
    - Body:
      ```json
      {
        "queries": ["conduction", "heat transfer"],
        "k": 5,
        "resource_id": "<your_resource_id>"
      }
      ```
  - **[ ]** Verify:
    - 200 response.
    - For each query: `elapsed_ms` is sane, `ids` and `scores` lists are non-empty.
  - **[ ]** Check Cloud Run logs for:
    - No `hybrid_search_failed` / `pgvector` errors.
    - Metrics increments (`retrieval_hybrid_calls`, etc.) if metrics backend is hooked.

---

### 4. Neo4j / Knowledge Graph

- **Provision Neo4j**
  - **[ ]** Create a Neo4j Aura DB or self-hosted Neo4j reachable from backend.
  - **[ ]** Note its Bolt URI and credentials.

- **Configure env**
  - **[ ]** Set backend env vars:
    - `NEO4J_URI` (e.g., `neo4j+s://xxxx.databases.neo4j.io` or `bolt://...`)
    - `NEO4J_USER`
    - `NEO4J_PASSWORD`
  - **[ ]** Redeploy backend.

- **Validate KG connectivity**
  - **[ ]** On backend startup, check logs for:
    - Successful constraint creation (no `neo4j_driver_init_failed`).
  - **[ ]** If you want, call a small script/endpoint that runs [ensure_neo4j_constraints()](cci:1://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/kg_pipeline/base.py:120:0-124:12) (already called on startup) and check logs again.

- **Run KG pipeline on sample corpus**
  - **[ ]** Ingest or reindex a resource that has formulas/figures/sections:
    - Use existing ingestion/reindex endpoints (no extra ops).
  - **[ ]** In Neo4j Browser, verify:
    - Nodes with labels `Concept`, `Chunk`, `Resource` exist.
    - Basic relationships (e.g., `[:NEXT_CHUNK]`, prereq edges, etc.) exist.
    - Constraints show up under DB `Constraints`.

---

### 5. Tutor & KG Integration (Optional)

- **Feature flag / env configuration**
  - **[ ]** Decide whether KG-enhanced features should be on in prod.
  - **[ ]** Set KG-related env flags accordingly (e.g., any `KG_*` envs your config uses).

- **Failure behavior**
  - **[ ]** Temporarily break Neo4j (wrong password / URI) in a staging environment and confirm:
    - Tutor still works.
    - Ingestion completes.
    - Logs show `neo4j_*` errors but no user-facing 5xx purely due to KG.

---

### 6. Monitoring & Operational Hygiene

- **Monitoring**
  - **[ ]** Ensure Cloud Run logs are captured (GCP logs explorer).
  - **[ ]** Verify periodic metrics for:
    - `search_calls_total`, `retrieval_hybrid_calls`.
    - Ingestion job metrics if present.

- **Backups & safety**
  - **[ ]** Enable automated backups and PITR for Cloud SQL.
  - **[ ]** For Neo4j Aura: make sure backup plan is enabled; for self-hosted, configure backup schedule.
  - **[ ]** Lock down access:
    - Cloud SQL (authorized networks / IAM).
    - Neo4j (strong passwords, IP allowlist or private networking).

---

### 7. Final PROD-07 Validation

- **[ ]** From the frontend, run:
  - Upload + ingest a note.
  - Ask the tutor a question that should be answerable from that note.
  - Confirm the response cites the right parts (via snippet, page, or explanation quality).
- **[ ]** Confirm that if Neo4j env vars are cleared/unset, vector search and tutor still function (vector-only mode).

---

If you’d like, I can next propose concrete `gcloud` / `psql` commands (non-destructive) to run each of these checks in your staging/prod environment.