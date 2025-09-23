#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
TOKEN=${TOKEN:-test-token}
SAMPLE_PDF=${SAMPLE_PDF:-sample/Fundamentals\ of\ Heat\ and\ Mass\ Transfer\ Chapter\ 6\ (1).pdf}

hdr=(-H "Authorization: Bearer ${TOKEN}")

echo "==> Uploading sample"
UPLOAD_JSON=$(curl -sS -X POST "${BASE_URL}/api/resources/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@${SAMPLE_PDF}" -F "title=Demo Resource")
echo "$UPLOAD_JSON" | jq . || echo "$UPLOAD_JSON"
RESOURCE_ID=$(echo "$UPLOAD_JSON" | jq -r .resource_id 2>/dev/null || true)
JOB_ID=$(echo "$UPLOAD_JSON" | jq -r .job_id 2>/dev/null || true)

if [[ -z "${RESOURCE_ID}" || "${RESOURCE_ID}" == "null" ]]; then
  echo "Upload did not return resource_id; aborting" >&2
  exit 1
fi

echo "==> Create chunks"
CHUNK_JSON=$(curl -sS -X POST "${BASE_URL}/api/resources/${RESOURCE_ID}/chunk" -H "Authorization: Bearer ${TOKEN}")
echo "$CHUNK_JSON" | jq . || echo "$CHUNK_JSON"

echo "==> Upsert embeddings for new chunks (idempotent)"
# List first page of chunks and upsert by ids
CHUNKS_JSON=$(curl -sS "${BASE_URL}/api/resources/${RESOURCE_ID}/chunks" -H "Authorization: Bearer ${TOKEN}")
IDS=$(echo "$CHUNKS_JSON" | jq -rc '.chunks | map(.id)')
EMB_JSON=$(curl -sS -X POST "${BASE_URL}/api/embeddings/upsert" -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' -d "{\"chunk_ids\": ${IDS}}")
echo "$EMB_JSON" | jq . || echo "$EMB_JSON"

echo "==> Bench (p@k proxy)"
BENCH_JSON=$(curl -sS -X POST "${BASE_URL}/api/bench/pk" -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' -d '{"queries":["heat flux","boundary layer"],"k":5}')
echo "$BENCH_JSON" | jq . || echo "$BENCH_JSON"

echo "==> Reindex (incremental)"
REINDEX_JSON=$(curl -sS -X POST "${BASE_URL}/api/resources/${RESOURCE_ID}/reindex" -H "Authorization: Bearer ${TOKEN}")
echo "$REINDEX_JSON" | jq . || echo "$REINDEX_JSON"

echo "Seed demo complete. RESOURCE_ID=${RESOURCE_ID}"
