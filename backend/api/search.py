from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import time

from core.auth import require_auth
from agents.retrieval import hybrid_search, recompute_search_tsv_for_all_chunks
from metrics import MetricsCollector

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    resource_id: Optional[str] = None


@router.post("/api/search")
async def search(req: SearchRequest, token: str = Depends(require_auth)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    try:
        t0 = time.time()
        rows = hybrid_search(req.query, k=int(req.k or 10), resource_id=req.resource_id)
        elapsed_ms = int((time.time() - t0) * 1000)
        try:
            mc = MetricsCollector.get_global()
            mc.increment("search_calls_total")
            mc.timing("search_elapsed_ms", elapsed_ms)
            mc.timing("search_result_count", len(rows) if isinstance(rows, list) else 0)
        except Exception:
            logging.exception("search_metrics_failed")
        return {"results": rows}
    except Exception:
        logging.exception("search_failed")
        raise HTTPException(status_code=502, detail="search_failed")


@router.post("/api/admin/recompute-search-tsv")
async def admin_recompute_search_tsv(token: str = Depends(require_auth)):
    try:
        updated = recompute_search_tsv_for_all_chunks()
        return {"updated": int(updated)}
    except Exception:
        logging.exception("recompute_search_tsv_failed")
        raise HTTPException(status_code=500, detail="recompute_failed")
