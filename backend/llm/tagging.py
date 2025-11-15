"""LLM tagging and math extraction helpers.

Provides a wrapper that calls an OpenAI-compatible LLM (AimlAPI) for chunk analysis.
Enhancements:
- Strict JSON guidance and optional JSON mode via response_format (if provider supports it)
- Sentinel markers to aid JSON extraction from model output
- Normalization of fields (chunk_type whitelist, concept de-dup/trim)
- Concepts-only fallback call when the first pass extracts zero concepts
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
import logging
import requests
import time
from metrics import MetricsCollector
from prompts import active_set as prompts_active_set
from llm.common import _extract_json_blob, _repair_json  # type: ignore


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
    # Optional deterministic mock path for CI/local dev
    # When USE_LLM_MOCK is set, bypass any external network calls and
    # return a stable, minimal JSON object derived from the input text.
    try:
        if os.getenv("USE_LLM_MOCK", "0").lower() in ("1", "true", "yes"):  # pragma: no cover - simple env guard
            math_list = extract_math_expressions(text or "")
            # metrics
            try:
                mc = MetricsCollector.get_global()
                mc.increment("llm_calls_total")
                mc.increment("llm_mock_calls")
                mc.timing("llm_elapsed_ms", 0)
                ps = (os.getenv("PROMPT_SET", "baseline").strip() or "baseline").replace("/", "_")
                mc.increment(f"llm_calls_total_ps_{ps}")
            except Exception:
                pass
            return {"chunk_type": "summary", "concepts": ["MockConcept"], "math_expressions": math_list}
    except Exception:
        # Never fail because of mock path; fall through to real logic.
        pass
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
    # Use model from env (nano preferred for preview); do not hardcode in prompts
    # Honor thread-local override first, then env models
    try:
        from llm.common import _get_model_override as _thread_model_override  # type: ignore
        override = _thread_model_override()
    except Exception:
        override = None
    model = override or os.getenv("LLM_MODEL_MINI") or os.getenv("LLM_MODEL_NANO") or "openai/gpt-5-nano-2025-08-07"
    
    if not base or not key:
        raise RuntimeError("LLM not configured - missing API base URL or key")

    # Truncate input aggressively for preview stability (semantic micro-chunk size)
    try:
        max_chars = int(os.getenv("LLM_PREVIEW_MAX_CHARS", "1000"))
    except Exception:
        max_chars = 1000
    input_text = (text or "")[:max_chars]

    # Create appropriate prompt based on model
    if "nano" in model:
        # Simplified prompt for nano model with strict JSON instructions and sentinels
        prompt = (
            prompt_override
            or (
                "You are an information extraction model. "
                "Strictly output only JSON that conforms to: {"
                "\"chunk_type\": one of [definition,theorem,example,exercise,formula,summary,derivation,procedure,fact], "
                "\"concepts\": array of 1-5 short subject-specific terms (strings), "
                "\"math_expressions\": array of LaTeX strings (can be empty) }. "
                "No markdown. No commentary. Output exactly one JSON object.\n\n"
                "Text:\n" + input_text + "\n\n"
                "Wrap the JSON between a single pair of sentinel tags BEGIN_STRICT_JSON and END_STRICT_JSON."
            )
        )
    else:
        # Concise prompt for better results with strict JSON and sentinels
        prompt = (
            prompt_override
            or (
                "Analyze the academic text. set reasoning_effort to minimal. "
                "Strictly output only JSON that conforms to: {"
                "\"chunk_type\": one of [definition,theorem,example,exercise,formula,summary,derivation,procedure,fact], "
                "\"concepts\": array of 1-5 short subject-specific terms (strings), "
                "\"math_expressions\": array of LaTeX strings }. "
                "No markdown. No commentary.\n\n"
                "Text:\n" + input_text + "\n\n"
                "Wrap your JSON strictly between the tags BEGIN_STRICT_JSON and END_STRICT_JSON.\n"
                "BEGIN_STRICT_JSON\n{\n  \"chunk_type\": \"summary\",\n  \"concepts\": [],\n  \"math_expressions\": []\n}\nEND_STRICT_JSON"
            )
        )

    # Prepare API request
    base_clean = base.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Use chat/completions for all models (per user request)
    endpoint = "/chat/completions" if base_clean.endswith("/v1") else "/v1/chat/completions"
    url = base_clean + endpoint
    # Concise prompt and output length
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only minified JSON with keys: chunk_type, concepts, math_expressions. "
                    "Conform to the schema, do not include markdown fences or extra text."
                ),
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "8000")),
        "stream": False
    }

    # Optional JSON mode (if provider supports OpenAI-style response_format)
    try:
        use_json_mode = os.getenv("LLM_RESPONSE_FORMAT_JSON", "1").lower() in ("1", "true", "yes")
    except Exception:
        use_json_mode = True
    if use_json_mode:
        body["response_format"] = {"type": "json_object"}
    logging.info("llm_request model=%s url=%s input_chars=%d (chat API)", model, url, len(input_text))
    resp_data: Dict[str, Any] = {}
    content = ""
    t_start = time.time()
    try:
        # optional OpenTelemetry span
        tracer = None
        try:
            if os.getenv("OTEL_ENABLE", "0").lower() in ("1", "true", "yes"):
                from opentelemetry import trace  # type: ignore
                tracer = trace.get_tracer("backend.llm")
        except Exception:
            tracer = None
        span_ctx = tracer.start_as_current_span("llm.chat.completions") if tracer else None
        if span_ctx:
            span_ctx.__enter__()
        resp = requests.post(url, headers=headers, json=body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if 200 <= resp.status_code < 300:
            resp_data = resp.json()
            content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            logging.error("llm_non_200 status=%s body=%s", resp.status_code, resp.text[:512])
    except requests.RequestException as e:
        logging.error("llm_request_error(chat) error=%s", e)
    finally:
        try:
            if span_ctx:
                span_ctx.__exit__(None, None, None)
        except Exception:
            pass

    # If empty or non-200, retry once with ultra-short directive
    if not content or not content.strip():
        logging.warning("llm_first_try_failed_or_empty; retrying with ultra-short prompt")
        body_retry = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON: {\"chunk_type\":\"summary\",\"concepts\":[],\"math_expressions\":[]}"},
                {"role": "user", "content": "JSON only"}
            ],
            "temperature": 0.0,
            "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "8000")),
            "stream": False
        }
        if use_json_mode:
            body_retry["response_format"] = {"type": "json_object"}
        try:
            span_ctx2 = tracer.start_as_current_span("llm.chat.retry") if 'tracer' in locals() and tracer else None
            if span_ctx2:
                span_ctx2.__enter__()
            resp2 = requests.post(url, headers=headers, json=body_retry, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
            if 200 <= resp2.status_code < 300:
                resp_data = resp2.json()
                content = resp2.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logging.error("llm_retry_non_200 status=%s body=%s", resp2.status_code, resp2.text[:512])
        except requests.RequestException as e:
            logging.error("llm_retry_error(chat) error=%s", e)
        finally:
            try:
                if span_ctx2:
                    span_ctx2.__exit__(None, None, None)
            except Exception:
                pass

    if not content or not content.strip():
        # Last resort: minimal JSON to avoid null/502
        logging.error("llm_empty_content_after_retry; coercing to minimal JSON")
        content = "{\"chunk_type\":\"summary\",\"concepts\":[],\"math_expressions\":[]}"

    # Parse JSON response
    coerced = False
    try:
        json_blob = _extract_json_blob(content)
        parsed = json.loads((json_blob or "").strip())
        # If the model returned literal null or a non-dict, coerce to minimal JSON object
        if not isinstance(parsed, dict):
            result = {"chunk_type": "summary", "concepts": [], "math_expressions": []}
            coerced = True
        else:
            result = parsed
        
        # Validate required keys
        required_keys = ["chunk_type", "concepts", "math_expressions"]
        for key in required_keys:
            if key not in result:
                result[key] = [] if key != "chunk_type" else "summary"
                coerced = True
        # Normalize chunk_type to an allowed set
        allowed_types = {
            "definition",
            "theorem",
            "example",
            "exercise",
            "formula",
            "summary",
            "derivation",
            "procedure",
            "fact",
        }
        try:
            ct = str(result.get("chunk_type") or "summary").strip().lower()
        except Exception:
            ct = "summary"
        if ct not in allowed_types:
            result["chunk_type"] = "summary"
            coerced = True
        else:
            result["chunk_type"] = ct

        # Ensure concepts is a list
        if not isinstance(result["concepts"], list):
            result["concepts"] = []
            coerced = True
        # Clean up concepts: trim, dedup, drop empties, cap at 5
        if result["concepts"]:
            cleaned: List[str] = []
            seen = set()
            for c in result["concepts"]:
                try:
                    cs = str(c).strip()
                except Exception:
                    cs = ""
                if not cs:
                    continue
                if cs.lower() in seen:
                    continue
                seen.add(cs.lower())
                cleaned.append(cs)
                if len(cleaned) >= 5:
                    break
            result["concepts"] = cleaned

        # Ensure math_expressions is a list
        if not isinstance(result["math_expressions"], list):
            result["math_expressions"] = []
            coerced = True
        
        # Add extracted math if not found by LLM
        extracted_math = extract_math_expressions(input_text)
        for expr in extracted_math:
            if expr not in result["math_expressions"]:
                result["math_expressions"].append(expr)
        
        usage = resp_data.get("usage", {}) if isinstance(resp_data, dict) else {}
        total_tokens = usage.get("total_tokens", usage.get("output_tokens", "unknown"))
        elapsed_ms = int((time.time() - t_start) * 1000)
        # Simple success-rate tracking (internal and metrics)
        try:
            _counters["calls_total"] += 1
            if not coerced:
                _counters["calls_success"] += 1
        except NameError:
            pass
        success_rate = 0.0
        try:
            success_rate = (_counters["calls_success"] / max(1, _counters["calls_total"]))
        except Exception:
            success_rate = 0.0
        # emit metrics
        try:
            mc = MetricsCollector.get_global()
            mc.increment("llm_calls_total")
            if not coerced:
                mc.increment("llm_calls_success")
            model_key = str(model).replace("/", "_")
            mc.increment(f"llm_model_{model_key}_calls")
            mc.timing("llm_elapsed_ms", elapsed_ms)
            ps = (os.getenv("PROMPT_SET", "baseline").strip() or "baseline").replace("/", "_")
            mc.increment(f"llm_calls_total_ps_{ps}")
        except Exception:
            pass
        logging.info(
            "llm_success model=%s tokens_used=%s elapsed_ms=%d success_rate=%.2f chunk_type=%s concepts=%d",
            model,
            total_tokens,
            elapsed_ms,
            success_rate,
            result.get("chunk_type"),
            len(result.get("concepts", [])),
        )
        # If concepts are still empty, trigger a second-pass concepts-only extraction
        if not result.get("concepts"):
            try:
                c2 = _extract_concepts_only(input_text, model, url, headers)
                if c2:
                    result["concepts"] = c2
            except Exception:
                logging.exception("concepts_only_fallback_failed")

        return result
        
    except json.JSONDecodeError as e:
        logging.error("llm_json_parse_error content=%s error=%s", content[:500], e)
        # Return minimal JSON on parse error
        try:
            _counters["calls_total"] += 1
        except NameError:
            pass
        return {"chunk_type": "summary", "concepts": [], "math_expressions": []}
def tag_and_extract(text: str) -> Dict[str, Any]:
    """High-level function: Use LLM for tagging with strict JSON guidance and fallbacks."""
    return call_llm_for_tagging(text)

# lightweight module-level counters for success rate logging
_counters = {"calls_total": 0, "calls_success": 0}

def call_llm_json(prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Generic JSON caller: returns arbitrary JSON dictated by the prompt.
    Does not coerce to tagging schema. Honors USE_LLM_MOCK for deterministic dev/tests."""
    try:
        if os.getenv("USE_LLM_MOCK", "0").lower() in ("1", "true", "yes"):
            ps = (os.getenv("PROMPT_SET", "baseline").strip() or "baseline").replace("/", "_")
            try:
                mc = MetricsCollector.get_global()
                mc.increment("llm_json_calls_total")
                mc.increment(f"llm_json_calls_total_ps_{ps}")
            except Exception:
                pass
            return default
    except Exception:
        pass
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
    # Honor thread-local override (from model_override_context) first, then env defaults
    try:
        from llm.common import _get_model_override as _thread_model_override  # type: ignore
        _override = _thread_model_override()
    except Exception:
        _override = None
    model = _override or os.getenv("LLM_MODEL_MINI") or os.getenv("LLM_MODEL_NANO") or "openai/gpt-5-nano-2025-08-07"
    if not base or not key:
        return default
    base_clean = base.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    endpoint = "/chat/completions" if base_clean.endswith("/v1") else "/v1/chat/completions"
    url = base_clean + endpoint
    # Check if this is a reasoning model (gpt-5, o1, etc.)
    is_reasoning_model = "gpt-5" in model.lower() or "o1" in model.lower()
    
    # Reasoning models need explicit instruction to output in the message
    if is_reasoning_model:
        system_msg = "You are a JSON API. You MUST analyze the input and return ONLY valid JSON matching the requested format. Do not engage in conversation. Do not explain. Only output the JSON structure requested."
    else:
        system_msg = "Return only minified JSON. No markdown, no fences."
    
    # Reasoning models need much more tokens (reasoning + output)
    if is_reasoning_model:
        max_tokens = int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "800")) * 3  # 3x tokens for reasoning
    else:
        max_tokens = int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "800"))
    
    user_content = (prompt or "").strip()
    if not user_content:
        user_content = "Provide the requested JSON payload following the StudyAgent schema."

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    try:
        use_json_mode = os.getenv("LLM_RESPONSE_FORMAT_JSON", "1").lower() in ("1", "true", "yes")
    except Exception:
        use_json_mode = True
    
    # Reasoning models don't support json_object mode and need special handling
    if not is_reasoning_model and use_json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if not (200 <= resp.status_code < 300):
            logging.error("llm_json_non_200 status=%s", resp.status_code)
            return default
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        blob = _extract_json_blob(content)
        
        # Try parsing the extracted JSON
        if blob:
            try:
                parsed = json.loads(blob.strip())
            except json.JSONDecodeError:
                # Try repairing the JSON
                repaired = _repair_json(blob)
                try:
                    parsed = json.loads(repaired.strip())
                except json.JSONDecodeError:
                    parsed = None
        else:
            parsed = None
        if isinstance(parsed, dict):
            try:
                mc = MetricsCollector.get_global()
                mc.increment("llm_json_calls_total")
                mc.timing("llm_elapsed_ms", int((time.time() - t0) * 1000))
                ps = (os.getenv("PROMPT_SET", "baseline").strip() or "baseline").replace("/", "_")
                mc.increment(f"llm_json_calls_total_ps_{ps}")
            except Exception:
                pass
            return parsed
        # Retry once with ultra-strict sentinel guidance
        logging.warning("llm_json_parse_failed_first_try; retrying with sentinel-wrapped JSON prompt")
        retry_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return ONLY minified JSON. No markdown, no fences, unless specified."},
                {"role": "user", "content": "Wrap valid JSON strictly between BEGIN_STRICT_JSON and END_STRICT_JSON for: " + prompt + "\nBEGIN_STRICT_JSON {} END_STRICT_JSON"},
            ],
            "temperature": 0.0,
            "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "800")),
            "stream": False,
        }
        if not is_reasoning_model and use_json_mode:
            retry_body["response_format"] = {"type": "json_object"}
        resp2 = requests.post(url, headers=headers, json=retry_body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if not (200 <= resp2.status_code < 300):
            logging.error("llm_json_retry_non_200 status=%s", resp2.status_code)
            return default
        data2 = resp2.json()
        content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        blob2 = _extract_json_blob(content2)
        
        # Try parsing with repair
        if blob2:
            try:
                parsed2 = json.loads(blob2.strip())
            except json.JSONDecodeError:
                repaired2 = _repair_json(blob2)
                try:
                    parsed2 = json.loads(repaired2.strip())
                except json.JSONDecodeError:
                    parsed2 = None
        else:
            parsed2 = None
        if isinstance(parsed2, dict):
            try:
                mc = MetricsCollector.get_global()
                mc.increment("llm_json_calls_total")
                mc.timing("llm_elapsed_ms", int((time.time() - t0) * 1000))
                ps = (os.getenv("PROMPT_SET", "baseline").strip() or "baseline").replace("/", "_")
                mc.increment(f"llm_json_calls_total_ps_{ps}")
            except Exception:
                pass
            return parsed2
        return default
    except Exception:
        logging.exception("llm_json_call_failed")
        return default

def _extract_concepts_only(text: str, model: str, url: str, headers: Dict[str, str]) -> List[str]:
    """Second-pass call to extract concepts only, used when first pass returns none."""
    try:
        max_chars = int(os.getenv("LLM_PREVIEW_MAX_CHARS", "1000"))
    except Exception:
        max_chars = 1000
    input_text = (text or "")[:max_chars]

    # Short and strict prompt with sentinels
    user_msg = (
        "Extract up to 5 short, domain-specific key concepts (nouns) present in the text. "
        "Return only JSON between BEGIN_STRICT_JSON and END_STRICT_JSON with the shape: {\"concepts\":[\"...\"]}.\n\n"
        "Text:\n" + input_text + "\n\nBEGIN_STRICT_JSON\n{\n  \"concepts\": []\n}\nEND_STRICT_JSON"
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only minified JSON with key: concepts (array of strings)."},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "8000")),
        "stream": False,
    }
    try:
        use_json_mode = os.getenv("LLM_RESPONSE_FORMAT_JSON", "1").lower() in ("1", "true", "yes")
    except Exception:
        use_json_mode = True
    if use_json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if resp.status_code != 200:
            logging.error("concepts_only_non_200 status=%s", resp.status_code)
            return []
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        json_blob = _extract_json_blob(content)
        parsed = json.loads((json_blob or "").strip())
        arr = parsed.get("concepts") if isinstance(parsed, dict) else None
        if not isinstance(arr, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for c in arr:
            try:
                cs = str(c).strip()
            except Exception:
                cs = ""
            if not cs:
                continue
            if cs.lower() in seen:
                continue
            seen.add(cs.lower())
            cleaned.append(cs)
            if len(cleaned) >= 5:
                break
        return cleaned
    except Exception:
        logging.exception("concepts_only_call_failed")
        return []