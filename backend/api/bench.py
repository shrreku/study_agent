from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

from core.auth import require_auth
from agents.retrieval import hybrid_search

router = APIRouter()


class BenchPkRequest(BaseModel):
    queries: List[str]
    k: Optional[int] = 5
    sim_weight: Optional[float] = None
    bm25_weight: Optional[float] = None
    resource_boost: Optional[float] = None
    page_proximity_boost: Optional[bool] = None
    resource_id: Optional[str] = None


@router.post("/api/bench/pk")
async def bench_pk(req: BenchPkRequest, token: str = Depends(require_auth)):
    if not req.queries or not isinstance(req.queries, list):
        raise HTTPException(status_code=400, detail="queries required")
    k = int(req.k or 5)
    results: List[Dict[str, Any]] = []
    for q in req.queries:
        qtext = (q or "").strip()
        if not qtext:
            results.append({"query": q, "ids": [], "scores": [], "elapsed_ms": 0})
            continue
        t0 = time.time()
        try:
            rows = hybrid_search(
                qtext,
                k=k,
                sim_weight=req.sim_weight,
                bm25_weight=req.bm25_weight,
                resource_boost=req.resource_boost,
                page_proximity_boost=req.page_proximity_boost,
                resource_id=req.resource_id,
            )
        except Exception:
            rows = []
        elapsed_ms = int((time.time() - t0) * 1000)
        ids = [r.get("id") for r in rows]
        scores = [float(r.get("score")) for r in rows if isinstance(r.get("score"), (int, float))]
        results.append({
            "query": qtext,
            "elapsed_ms": elapsed_ms,
            "ids": ids,
            "scores": scores,
        })
    return {"k": k, "results": results}
