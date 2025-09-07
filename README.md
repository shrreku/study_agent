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

Create a `.env` at the repo root with:

```
OPENAI_API_BASE=
OPENAI_API_KEY=
LLM_MODEL_MINI=gpt-5-mini
LLM_MODEL_NANO=gpt-5-nano

POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=app
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=minio:9000
MINIO_SECURE=false
MINIO_BUCKET=resources

REDIS_URL=redis://redis:6379/0

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_VERSION=all-MiniLM-L6-v2-2025-09
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
```
