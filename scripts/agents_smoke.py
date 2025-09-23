#!/usr/bin/env python3
"""
Agent smoke test against a running backend.

Checks:
- Health
- Orchestrator mock
- Study Plan agent (minimal payload)
- Daily Quiz agent (2 items)
- Doubt agent (answer + citations)

Usage:
  python3 scripts/agents_smoke.py [--base http://localhost:8000]

Env defaults:
  SMOKE_BASE (default: http://localhost:8000)
  SMOKE_AUTH (default: "Bearer test-token")

Notes:
- Keep outputs short; do not print full bodies or secrets per dev norms.
- Ensure backend is running (prefer docker compose) and LLM env is configured.
"""
import argparse
import json
import os
from urllib import request, error

DEFAULT_BASE = os.environ.get("SMOKE_BASE", "http://localhost:8000")
AUTH = os.environ.get("SMOKE_AUTH", "Bearer test-token")


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 60.0):
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, method=method)
    req.add_header("Authorization", AUTH)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in ct or raw.startswith(b"{") or raw.startswith(b"["):
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {"_raw": raw.decode("utf-8", errors="ignore")}
            return {"_raw": raw.decode("utf-8", errors="ignore")}
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        return {"_error": f"HTTP {e.code}", "detail": raw[:240]}
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="backend base URL")
    args = ap.parse_args()

    base = args.base.rstrip("/")

    print("[health] GET /health")
    h = http_json("GET", f"{base}/health")
    print({"status": h})

    print("[agent] POST /api/agent/mock")
    r = http_json("POST", f"{base}/api/agent/mock", {"foo": "bar"})
    print({"ok": r.get("ok"), "agent": r.get("agent")})

    print("[agent] POST /api/agent/study-plan")
    sp = http_json(
        "POST",
        f"{base}/api/agent/study-plan",
        {"target_concepts": ["Heat Transfer", "Convection"], "daily_minutes": 30},
    )
    todos = sp.get("todos") if isinstance(sp, dict) else None
    print({
        "plan_id": (sp.get("plan_id") if isinstance(sp, dict) else None),
        "todos_sample": (todos[:2] if isinstance(todos, list) else None),
        "todos_len": (len(todos) if isinstance(todos, list) else None),
    })

    print("[agent] POST /api/agent/daily-quiz")
    dq = http_json(
        "POST",
        f"{base}/api/agent/daily-quiz",
        {"concepts": ["Convection", "Fourier law"], "count": 2},
    )
    items = (dq.get("items") or dq.get("quiz")) if isinstance(dq, dict) else None
    print({
        "returned_items": (len(items) if isinstance(items, list) else None),
        "sample": (items[:1] if isinstance(items, list) else None),
    })

    print("[agent] POST /api/agent/doubt")
    da = http_json(
        "POST",
        f"{base}/api/agent/doubt",
        {"question": "What is heat flux?"},
    )
    cits = da.get("citations") if isinstance(da, dict) else None
    print({
        "has_answer": ("answer" in da if isinstance(da, dict) else False),
        "citations": (cits[:2] if isinstance(cits, list) else None),
        "cit_count": (len(cits) if isinstance(cits, list) else None),
    })

    print("[done]")


if __name__ == "__main__":
    main()
