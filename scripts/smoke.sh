#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
TOKEN=${TOKEN:-test-token}

hdr=(-H "Authorization: Bearer ${TOKEN}")

say() { printf "\n==> %s\n" "$*"; }

say "Health"
curl -sS "${BASE_URL}/health" | sed -E 's/.{200}/&\n/g'

say "LLM preview (mocked environment recommended)"
curl -sS -X POST "${BASE_URL}/api/llm/preview" -H 'Content-Type: application/json' "${hdr[@]}" \
  -d '{"text":"Short academic text for preview."}' | sed -E 's/.{200}/&\n/g'

say "Admin recompute search_tsv"
curl -sS -X POST "${BASE_URL}/api/admin/recompute-search-tsv" "${hdr[@]}" | sed -E 's/.{200}/&\n/g'

say "Benchmark P@k proxy"
curl -sS -X POST "${BASE_URL}/api/bench/pk" -H 'Content-Type: application/json' "${hdr[@]}" \
  -d '{"queries":["heat flux","boundary layer"],"k":5}' | sed -E 's/.{200}/&\n/g'

say "KG OCCURS_IN backfill (dry-run)"
python3 scripts/kg_backfill_occurs_in.py --dry-run | sed -E 's/.{200}/&\n/g'

echo "\nSmoke completed."
