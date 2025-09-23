from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import requests

from core.auth import require_auth
from llm import call_llm_for_tagging

router = APIRouter()


class LlmPreviewRequest(BaseModel):
    text: str
    prompt_override: Optional[str] = None


@router.post("/api/llm/preview")
async def llm_preview(req: LlmPreviewRequest, token: str = Depends(require_auth)):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text required")
    try:
        out = call_llm_for_tagging(req.text, prompt_override=req.prompt_override)
        return out
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/llm/smoke")
async def llm_smoke(token: str = Depends(require_auth)):
    base = os.getenv("OPENAI_API_BASE") or os.getenv("AIMLAPI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    model = os.getenv("LLM_MODEL_MINI", "openai/gpt-5-mini-2025-08-07")
    if not base:
        raise HTTPException(status_code=400, detail="OPENAI_API_BASE not set")
    if not key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    url = base.rstrip("/") + ("/chat/completions" if base.rstrip("/").endswith("/v1") else "/v1/chat/completions")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_http_error: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:512])
    try:
        j = r.json()
        txt = j["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=502, detail=f"invalid llm response: {r.text[:256]}")
    return {"ok": True, "reply": txt, "model": model}
