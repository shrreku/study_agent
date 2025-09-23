#!/usr/bin/env python3
"""
End-to-end smoke test for a single resource_id:
- (Optionally) chunk + tag
- list chunks
- recompute BM25 (tsvector)
- upsert embeddings for those chunks
- run a couple of search queries

Usage:
  python3 scripts/resource_smoke.py --rid <resource_uuid> [--base http://localhost:8000] [--force]

Defaults:
  --rid defaults to the latest provided RID from chat: 9f65b917-1b6c-4ca3-8581-ae35a5ef91f8
  --base defaults to http://localhost:8000
  Authorization header is set to the local dev token per project norms.
"""
import argparse
import json
import os
import sys
import time
from urllib import request, parse, error

DEFAULT_BASE = os.environ.get("SMOKE_BASE", "http://localhost:8000")
DEFAULT_RID = os.environ.get("SMOKE_RID", "9f65b917-1b6c-4ca3-8581-ae35a5ef91f8")
AUTH_HEADER = ("Authorization", os.environ.get("SMOKE_AUTH", "Bearer test-token"))


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 60.0):
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, method=method)
    req.add_header(*AUTH_HEADER)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
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
        return {"_error": f"HTTP {e.code}", "detail": raw}
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", default=DEFAULT_RID, help="resource UUID to smoke test")
    ap.add_argument("--base", default=DEFAULT_BASE, help="backend base URL (default http://localhost:8000)")
    ap.add_argument("--force", action="store_true", help="force re-chunking even if chunks exist")
    ap.add_argument("--limit", type=int, default=200, help="max chunks to fetch for this test")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    rid = (args.rid or "").strip()
    if not rid:
        print("[err] missing --rid", file=sys.stderr)
        sys.exit(2)

    print(f"[health] GET {base}/health")
    print(http_json("GET", f"{base}/health"))

    # list chunks (first) to avoid accidental heavy chunking
    print(f"[chunks] GET /api/resources/{rid}/chunks?limit={args.limit}&offset=0")
    chunks_resp = http_json("GET", f"{base}/api/resources/{rid}/chunks?limit={args.limit}&offset=0")
    chunks = chunks_resp.get("chunks", []) if isinstance(chunks_resp, dict) else []
    print({"chunk_count": len(chunks)})
    for c in chunks[:5]:
        print({
            "id": c.get("id"),
            "page_number": c.get("page_number"),
            "has_embedding": c.get("has_embedding"),
            "snippet": (c.get("snippet") or "")[:160],
        })

    # chunk + tag (only when explicitly forced OR no chunks yet)
    if args.force or not chunks:
        qs = f"?force={'true' if args.force else 'false'}"
        print(f"[chunk] POST /api/resources/{rid}/chunk{qs}")
        # allow long timeout for chunking when needed
        print(http_json("POST", f"{base}/api/resources/{rid}/chunk{qs}", timeout=600.0))
        # refresh chunks after chunking
        print(f"[chunks] GET /api/resources/{rid}/chunks?limit={args.limit}&offset=0 (after chunk)")
        chunks_resp = http_json("GET", f"{base}/api/resources/{rid}/chunks?limit={args.limit}&offset=0")
        chunks = chunks_resp.get("chunks", []) if isinstance(chunks_resp, dict) else []
        print({"chunk_count": len(chunks)})
        for c in chunks[:5]:
            print({
                "id": c.get("id"),
                "page_number": c.get("page_number"),
                "has_embedding": c.get("has_embedding"),
                "snippet": (c.get("snippet") or "")[:160],
            })
    else:
        print("[info] chunks already exist; skipping re-chunking (use --force to re-run)")

    # recompute BM25
    print("[bm25] POST /api/admin/recompute-search-tsv")
    print(http_json("POST", f"{base}/api/admin/recompute-search-tsv", timeout=180.0))

    # upsert embeddings for these chunks
    chunk_ids = [c.get("id") for c in chunks if c.get("id")]
    print(f"[emb] POST /api/embeddings/upsert (ids={len(chunk_ids)})")
    print(http_json("POST", f"{base}/api/embeddings/upsert", {"chunk_ids": chunk_ids}, timeout=300.0))

    # searches
    for q in ("heat flux", "convection"):
        print(f"[search] POST /api/search q='{q}'")
        res = http_json("POST", f"{base}/api/search", {"query": q, "k": 8})
        if isinstance(res, dict):
            results = res.get("results", [])
            print({
                "returned": len(results),
                "sample": results[:2],
            })
        else:
            print(res)

    print("[done]")


if __name__ == "__main__":
    main()
