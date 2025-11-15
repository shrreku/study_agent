#!/usr/bin/env bash
# Quick smoke: toggle example-generation flag and hit tutor agent API
# Usage:
#   scripts/smoke_example_generation.sh on    # enable flag, restart backend, call API
#   scripts/smoke_example_generation.sh off   # disable flag, restart backend, call API
# Env:
#   BASE_URL (default http://localhost:8000)
#   TOKEN (default test-token)
#   USER_ID (default 123e4567-e89b-12d3-a456-426614174000)
#   MESSAGE (default sample tutor message)
# Notes: Requires backend running via docker compose with env_file ./.env
set -euo pipefail

MODE=${1:-on}
FLAG="false"
if [[ "$MODE" == "on" ]]; then FLAG="true"; fi
if [[ "$MODE" == "off" ]]; then FLAG="false"; fi
if [[ "$MODE" != "on" && "$MODE" != "off" ]]; then
  echo "Usage: $0 [on|off]" >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

ENV_FILE=".env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at repo root. Copy .env.example to .env and try again." >&2
  exit 1
fi

# Update flag in .env (macOS-compatible sed with backup)
if grep -q '^TUTOR_EXAMPLE_GENERATION_ENABLED=' "$ENV_FILE"; then
  sed -i.bak -E "s/^TUTOR_EXAMPLE_GENERATION_ENABLED=.*/TUTOR_EXAMPLE_GENERATION_ENABLED=${FLAG}/" "$ENV_FILE"
else
  echo "TUTOR_EXAMPLE_GENERATION_ENABLED=${FLAG}" >> "$ENV_FILE"
fi

# Restart backend to apply env. Try docker compose plugin first, then legacy.
if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    echo "Recreating backend (docker compose) ..."
    docker compose up -d --force-recreate backend >/dev/null
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "Recreating backend (docker-compose) ..."
    docker-compose up -d --force-recreate backend >/dev/null
  else
    echo "WARN: docker compose not found; please restart backend manually to apply .env changes." >&2
  fi
else
  echo "WARN: docker not installed; please restart backend manually." >&2
fi

BASE_URL=${BASE_URL:-http://localhost:8000}
TOKEN=${TOKEN:-test-token}
USER_ID=${USER_ID:-123e4567-e89b-12d3-a456-426614174000}
MESSAGE=${MESSAGE:-"I'm confused about Fourier's law. Could you give an example?"}

HDR=(-H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json')

# Wait for backend readiness
echo "Waiting for backend to be ready at ${BASE_URL}/health ..."
READY=0
for i in {1..60}; do
  if curl -sS "${BASE_URL}/health" >/dev/null; then
    READY=1
    break
  fi
  sleep 1
done
if [[ $READY -ne 1 ]]; then
  echo "ERROR: Backend not ready after 60s; aborting." >&2
  exit 1
fi
# Escape for JSON (macOS bash-compatible)
MSG_ESC=${MESSAGE//\\/\\\\}
MSG_ESC=${MSG_ESC//\"/\\\"}
BODY="{\n  \"message\": \"${MSG_ESC}\",\n  \"user_id\": \"${USER_ID}\",\n  \"target_concepts\": [\"Fourier's law\"]\n}"

echo "\n==> Tutor agent call with TUTOR_EXAMPLE_GENERATION_ENABLED=${FLAG}" 
set +e
RESP=$(curl -sS -X POST "${BASE_URL}/api/agent/tutor" "${HDR[@]}" -d "${BODY}")
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  echo "ERROR: API call failed" >&2
  exit $STATUS
fi
# Pretty print limited output length
python3 - "$RESP" <<'PY'
import json, sys, textwrap
try:
    data = json.loads(sys.argv[1])
except Exception:
    print(sys.argv[1][:800])
    sys.exit(0)
# Print summary fields if present
if isinstance(data, dict):
    print("status:", data.get("status", "ok"))
    print("action:", data.get("action", data.get("action_type")))
    resp = data.get("response_text") or data.get("response") or data.get("text") or ""
    print("response:\n" + "\n".join(textwrap.wrap(resp, width=100)))
else:
    print(json.dumps(data)[:800])
PY

echo "\nDone. (Tip: run again with 'off' to compare.)"
