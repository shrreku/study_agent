"""LLM tagging and math extraction helpers.

Provides a wrapper that calls an OpenAI-compatible LLM (AimlAPI) for chunk analysis.
No heuristic fallbacks - pure LLM-based metadata generation.
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
import logging
import requests


MATH_PATTERNS = [
    re.compile(r"\$(.+?)\$", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL),
]


def extract_math_expressions(text: str) -> List[str]:
    """Extract mathematical expressions from text using regex patterns."""
    found = []
    for p in MATH_PATTERNS:
        for m in p.findall(text):
            snippet = m.strip()
            if snippet:
                found.append(snippet)
    return found


def call_llm_for_tagging(text: str, prompt_override: Optional[str] = None) -> Dict[str, Any]:
    """Call GPT-5 nano via AimlAPI to get structured JSON tagging output.

    Returns a dict with keys: chunk_type (str), concepts (list[str]), math_expressions (list[str])
    """
    # Get API configuration
    base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("AIMLAPI_BASE_URL")
        or os.getenv("AIML_BASE_URL")
        or os.getenv("AIMLAPI_URL")
    )
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AIMLAPI_API_KEY")
        or os.getenv("AIML_KEY")
        or os.getenv("AIMLAPI_KEY")
    )
    # Use nano model as requested
    model = os.getenv("LLM_MODEL_NANO", "openai/gpt-5-nano-2025-08-07")
    
    if not base or not key:
        raise RuntimeError("LLM not configured - missing API base URL or key")

    # Create appropriate prompt based on model
    if "nano" in model:
        # Simplified prompt for nano model
        prompt = (
            prompt_override
            or (
                "Classify this text. Return JSON with: chunk_type (definition/theorem/example/exercise/formula/summary), concepts (array), math_expressions (array).\n\n"
                + text  # Tighter limit for nano to avoid length issues
                + "\n\nJSON:"
            )
        )
    else:
        # Concise prompt for better results
        prompt = (
            prompt_override
            or (
                "Analyze this academic text. Return JSON with: chunk_type (definition/theorem/example/exercise/formula/summary), concepts (array), math_expressions (array).\n\n"
                + text  # Limit text length
                + "\n\nJSON:"
            )
        )

    # Prepare API request
    base_clean = base.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Use chat/completions for all models (per user request)
    endpoint = "/chat/completions" if base_clean.endswith("/v1") else "/v1/chat/completions"
    url = base_clean + endpoint
    # Concise prompt and conservative output length to avoid empty content
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only minified JSON with keys: chunk_type, concepts, math_expressions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 8000,
        "stream": False
    }
    logging.info("llm_request model=%s url=%s input_chars=%d (chat API)", model, url, len(text))
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code != 200:
            logging.error("llm_non_200 status=%s body=%s", resp.status_code, resp.text[:512])
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # If empty, retry once with ultra-short directive
        if not content or not content.strip():
            logging.warning("llm_empty_content_first_try; retrying with ultra-short prompt")
            body_retry = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only JSON: {\"chunk_type\":\"summary\",\"concepts\":[],\"math_expressions\":[]}"},
                    {"role": "user", "content": "JSON only"}
                ],
                "temperature": 0.0,
                "max_tokens": 8000,
                "stream": False
            }
            resp2 = requests.post(url, headers=headers, json=body_retry, timeout=60)
            if resp2.status_code == 200:
                d2 = resp2.json()
                content = d2.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            logging.error("llm_empty_content response(chat)=%s", (resp.text or "")[:512])
            # Last resort: minimal JSON to avoid null/502
            content = "{\"chunk_type\":\"summary\",\"concepts\":[],\"math_expressions\":[]}"
    except requests.RequestException as e:
        logging.error("llm_request_error(chat) error=%s", e)
        # Last resort on network error as well
        content = "{\"chunk_type\":\"summary\",\"concepts\":[],\"math_expressions\":[]}"

    # Parse JSON response
    try:
        parsed = json.loads((content or "").strip())
        # If the model returned literal null or a non-dict, coerce to minimal JSON object
        if not isinstance(parsed, dict):
            result = {"chunk_type": "summary", "concepts": [], "math_expressions": []}
        else:
            result = parsed
        
        # Validate required keys
        required_keys = ["chunk_type", "concepts", "math_expressions"]
        for key in required_keys:
            if key not in result:
                result[key] = [] if key != "chunk_type" else "summary"
        
        # Ensure concepts is a list
        if not isinstance(result["concepts"], list):
            result["concepts"] = []
        
        # Ensure math_expressions is a list
        if not isinstance(result["math_expressions"], list):
            result["math_expressions"] = []
        
        # Add extracted math if not found by LLM
        extracted_math = extract_math_expressions(text)
        for expr in extracted_math:
            if expr not in result["math_expressions"]:
                result["math_expressions"].append(expr)
        
        usage = data.get("usage", {})
        logging.info("llm_success model=%s tokens_used=%s chunk_type=%s concepts=%d", 
                    model, usage.get("total_tokens", usage.get("output_tokens", "unknown")), 
                    result.get("chunk_type"), len(result.get("concepts", [])))
        
        return result
        
    except json.JSONDecodeError as e:
        logging.error("llm_json_parse_error content=%s error=%s", content[:500], e)
        # Return minimal JSON on parse error
        return {"chunk_type": "summary", "concepts": [], "math_expressions": []}
    


def tag_and_extract(text: str) -> Dict[str, Any]:
    """High-level function: Use LLM for tagging - NO HEURISTIC FALLBACK."""
    return call_llm_for_tagging(text)