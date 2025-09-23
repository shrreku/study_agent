from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException

from core.auth import require_auth
from metrics import MetricsCollector

router = APIRouter()


@router.get("/api/metrics")
async def metrics_snapshot(token: str = Depends(require_auth)):
    try:
        mc = MetricsCollector.get_global()
        return mc.snapshot()
    except Exception:
        raise HTTPException(status_code=500, detail="metrics_error")
