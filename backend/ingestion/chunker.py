from typing import List, Dict, Tuple, Any
import os
import logging
import time
import re
from .parse_utils import extract_text_by_type
from . import embed as embed_service
from metrics import MetricsCollector
from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json
from . import semantic_chunker
from . import hierarchical_tagger
from . import context_builder
from . import chunk_linker

# create module logger
logger = logging.getLogger("backend.chunker")

# default metrics collector (in-memory) - can be swapped in production
metrics = MetricsCollector.get_global()


def _sentence_token_counts(sentences: List[str]) -> List[int]:
    """Return approximate token counts for sentences (words ~= tokens)."""
    return [max(1, len(s.split())) for s in sentences]


def _join_sentences(sentences: List[str], start: int, end: int) -> str:
    return " ".join(s.strip() for s in sentences[start:end])


def semantic_chunk_sentences(sentences: List[str],
                             threshold: float = None,
                             min_tokens: int = None,
                             max_tokens: int = None,
                             overlap: int = None) -> List[Tuple[int,int]]:
    """
    Take a list of sentences and return list of (start_idx, end_idx) pairs
    representing chunks. Uses adjacent-sentence embedding cosine similarity
    cuts with token budget constraints. This is a lightweight in-process
    implementation that encodes sentences via `embed.embed_texts`.
    """
    if not sentences:
        return []

    # read env-driven defaults lazily to avoid module import ordering issues
    if threshold is None:
        try:
            threshold = float(os.getenv("SEM_SPLIT_TAU", "0.65"))
        except Exception:
            threshold = 0.65
    if min_tokens is None:
        try:
            min_tokens = int(os.getenv("SEM_MIN_TOKENS", "60"))
        except Exception:
            min_tokens = 60
    if max_tokens is None:
        try:
            max_tokens = int(os.getenv("SEM_MAX_TOKENS", "240"))
        except Exception:
            max_tokens = 240
    if overlap is None:
        try:
            overlap = int(os.getenv("SEM_OVERLAP", "1"))
        except Exception:
            overlap = 1

    # compute embeddings in batches using the new embed encode_sentences API
    batch_env = os.getenv("SEM_BATCH_SIZE")
    batch_size = int(batch_env) if batch_env and batch_env.isdigit() else None
    t0 = time.time()
    vecs = embed_service.encode_sentences(sentences, batch_size=batch_size)
    t_ms = int((time.time() - t0) * 1000)
    logger.info("encoded %d sentences in %dms for semantic split", len(sentences), t_ms)
    metrics.increment("semantic_encode_calls")
    metrics.timing("semantic_encode_ms", t_ms)

    # cosine similarity between adjacent sentences
    def cosine(a, b):
        # both are lists
        sa = sum(x * x for x in a) ** 0.5
        sb = sum(x * x for x in b) ** 0.5
        if sa == 0 or sb == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (sa * sb)

    sims = [0.0] * (len(sentences) - 1)
    for i in range(len(sims)):
        sims[i] = cosine(vecs[i], vecs[i + 1])

    token_counts = _sentence_token_counts(sentences)

    chunks: List[Tuple[int,int]] = []
    cur_start = 0
    cur_tokens = 0
    for i, s in enumerate(sentences):
        cur_tokens += token_counts[i]
        # if token budget exceeded, or similarity below threshold -> cut
        should_cut = False
        if cur_tokens >= max_tokens:
            should_cut = True
        elif i < len(sims) and sims[i] < threshold and cur_tokens >= min_tokens:
            should_cut = True

        if should_cut:
            cur_end = i + 1
            # enforce overlap
            chunks.append((cur_start, cur_end))
            # set next start with overlap sentences
            cur_start = cur_end - overlap if cur_end - overlap >= 0 else cur_end
            cur_tokens = sum(token_counts[cur_start:i+1]) if cur_start <= i else 0

    # flush remaining
    if cur_start < len(sentences):
        chunks.append((cur_start, len(sentences)))

    return chunks


def split_text_into_chunks(text: str,
                           threshold: float = 0.65,
                           min_tokens: int = 60,
                           max_tokens: int = 240,
                           overlap: int = 1) -> List[Dict]:
    """
    Semantic splitter: splits input text into sentences, encodes them,
    and cuts where adjacent-sentence similarity < threshold while enforcing
    token budgets. Returns list of dicts with source_offset (char) and full_text.
    """
    # crude sentence split: split on sentence-ending punctuation

    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
    if not sentences:
        return []

    spans = semantic_chunk_sentences(sentences, threshold=threshold, min_tokens=min_tokens, max_tokens=max_tokens, overlap=overlap)

    chunks = []
    char_offset = 0
    # precompute cumulative char lengths of sentences to compute source_offset
    cum_chars = []
    running = 0
    for s in sentences:
        running += len(s) + 1  # account for a space/newline
        cum_chars.append(running)

    for start, end in spans:
        full = _join_sentences(sentences, start, end)
        source_offset = cum_chars[start-1] if start > 0 else 0
        chunks.append({"source_offset": source_offset, "full_text": full})
    return chunks


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _trim_page_text(text: str) -> str:
    limit = _env_int("INGEST_PAGE_MAX_CHARS", 6000)
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit]


def _llm_page_structure(page_text: str, page_number: int) -> Dict[str, Any]:
    tmpl = prompt_get("ingest.page_structure")
    prompt = prompt_render(
        tmpl,
        {
            "page_number": page_number,
            "page_text": _trim_page_text(page_text),
        },
    )
    default = {"sections": [], "chunks": []}
    try:
        return call_llm_json(prompt, default) or default
    except Exception:
        logging.exception("page_structure_llm_failed", extra={"page_number": page_number})
        return default


def _llm_chunk_tags(text: str) -> List[str]:
    tmpl = prompt_get("ingest.chunk_tags")
    prompt = prompt_render(tmpl, {"chunk_text": text})
    default = {"tags": []}
    try:
        resp = call_llm_json(prompt, default)
        tags = resp.get("tags") or []
    except Exception:
        logging.exception("chunk_tags_llm_failed")
        tags = []
    cleaned: List[str] = []
    seen = set()
    for t in tags:
        try:
            val = str(t).strip()
        except Exception:
            val = ""
        if not val:
            continue
        low = val.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(val)
        if len(cleaned) >= _env_int("INGEST_TAGS_PER_CHUNK", 6):
            break
    return cleaned


def _section_path(section_number: str, section_title: str) -> List[str]:
    if section_number:
        parts = [p for p in section_number.replace(" ", "").split(".") if p]
        if parts:
            return parts
    if section_title:
        return [section_title]
    return []


def _find_section_for_chunk(sections: List[Dict[str, Any]], chunk_meta: Dict[str, Any], start: int) -> Dict[str, Any]:
    sec_number = str(chunk_meta.get("section_number") or "").strip()
    if sec_number:
        for sec in sections:
            if str(sec.get("number") or "").strip() == sec_number:
                return sec
    for sec in sections:
        try:
            sec_start = int(sec.get("start") or 0)
            sec_end = int(sec.get("end") or 0)
        except Exception:
            continue
        if sec_end <= 0:
            continue
        if sec_start <= start < sec_end:
            return sec
    return {}


def _sanitize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    try:
        return bool(value)
    except Exception:
        return False


def structural_chunk_resource(resource_path: str) -> List[Dict]:
    """Extract pages and produce LLM-structured chunks with semantic refinement fallback."""
    _tau = float(os.getenv("SEM_SPLIT_TAU", 0.65))
    _min = int(os.getenv("SEM_MIN_TOKENS", 60))
    _max = int(os.getenv("SEM_MAX_TOKENS", 240))
    _over = int(os.getenv("SEM_OVERLAP", 1))

    pages = extract_text_by_type(resource_path, None)
    all_chunks: List[Dict[str, Any]] = []
    for i, page_text in enumerate(pages, start=1):
        text = page_text or ""
        if not text.strip():
            continue

        structure = _llm_page_structure(text, i)
        sections = structure.get("sections") or []
        for sec in sections:
            for key in ("start", "end", "level"):
                if key in sec:
                    try:
                        sec[key] = int(sec[key])
                    except Exception:
                        sec[key] = 0

        chunk_entries = structure.get("chunks") or []
        if not chunk_entries:
            # fallback: semantic split entire page
            sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
            spans = semantic_chunk_sentences(sentences, threshold=_tau, min_tokens=_min, max_tokens=_max, overlap=_over)
            for start_idx, end_idx in spans:
                full = _join_sentences(sentences, start_idx, end_idx)
                norm = full.strip()
                if len(norm) < 20:
                    continue
                out = {
                    "page_number": i,
                    "page_start": i,
                    "page_end": i,
                    "source_offset": 0,
                    "full_text": norm,
                    "section_title": sections[0].get("title") if sections else "",
                    "section_number": sections[0].get("number") if sections else "",
                    "section_level": sections[0].get("level") if sections else None,
                    "section_path": _section_path(
                        str(sections[0].get("number") or "") if sections else "",
                        str(sections[0].get("title") or "") if sections else "",
                    ),
                    "token_count": len(norm.split()),
                    "has_figure": False,
                    "has_equation": False,
                    "figure_labels": [],
                    "equation_labels": [],
                    "caption": None,
                    "tags": _llm_chunk_tags(norm),
                    "text_snippet": norm[:300],
                }
                # Add pedagogy role classification
                section_ctx = {'title': sections[0].get("title") if sections else ""}
                out['pedagogy_role'] = hierarchical_tagger.classify_pedagogy_role(norm, section_ctx if section_ctx['title'] else None)
                all_chunks.append(out)
            continue

        text_len = len(text)
        for entry in chunk_entries:
            try:
                start = int(entry.get("start") or entry.get("offset") or 0)
                end = int(entry.get("end") or 0)
            except Exception:
                start, end = 0, 0
            if end <= 0 or end <= start:
                continue
            start = max(0, min(start, text_len - 1))
            end = max(start + 1, min(end, text_len))
            snippet_text = text[start:end]
            norm = snippet_text.strip()
            if len(norm) < 20:
                continue

            section = _find_section_for_chunk(sections, entry, start)
            section_title = str(entry.get("section_title") or section.get("title") or "").strip()
            section_number = str(entry.get("section_number") or section.get("number") or "").strip()
            try:
                section_level = int(entry.get("section_level") or section.get("level") or 0)
            except Exception:
                section_level = None
            section_path = entry.get("section_path") or _section_path(section_number, section_title)

            tags = entry.get("tags") or []
            if not tags:
                tags = _llm_chunk_tags(norm)

            has_figure = _sanitize_bool(entry.get("has_figure"))
            has_equation = _sanitize_bool(entry.get("has_equation"))
            figure_labels = entry.get("figure_labels") or []
            equation_labels = entry.get("equation_labels") or []
            caption = entry.get("caption")
            chunk_type = entry.get("type")

            out_chunk = {
                "page_number": i,
                "page_start": i,
                "page_end": i,
                "source_offset": start,
                "full_text": norm,
                "section_title": section_title,
                "section_number": section_number,
                "section_level": section_level,
                "section_path": section_path,
                "token_count": len(norm.split()),
                "has_figure": has_figure,
                "has_equation": has_equation,
                "figure_labels": figure_labels,
                "equation_labels": equation_labels,
                "caption": caption,
                "tags": tags,
                "chunk_type_hint": chunk_type,
                "text_snippet": norm[:300],
            }
            # Add pedagogy role classification
            section_ctx = {'title': section_title, 'number': section_number, 'level': section_level}
            out_chunk['pedagogy_role'] = hierarchical_tagger.classify_pedagogy_role(norm, section_ctx if section_title else None)
            all_chunks.append(out_chunk)

    try:
        metrics.timing("chunks_per_resource", len(all_chunks))
        if all_chunks:
            token_counts = [max(1, c.get("token_count") or len((c.get("full_text") or "").split())) for c in all_chunks]
            avg_tokens = int(sum(token_counts) / max(1, len(token_counts)))
            metrics.timing("avg_tokens_per_chunk", avg_tokens)
    except Exception:
        pass
    return all_chunks


def enhanced_structural_chunk_resource(resource_path: str) -> List[Dict]:
    """Enhanced chunker with semantic awareness, formula preservation, and hierarchical tagging.
    
    This is the new implementation that uses semantic_chunker module for:
    - Content-type aware chunking (concept, derivation, example, etc.)
    - Formula preservation with context
    - Multi-page concept support
    - Hierarchical educational metadata
    
    Controlled by environment variables:
    - ENHANCED_CHUNKING_ENABLED: Enable this enhanced chunker (default: false)
    - ENHANCED_TAGGING_ENABLED: Enable hierarchical tagging (default: false)
    - FORMULA_EXTRACTION_ENABLED: Enable formula metadata extraction (default: false)
    """
    # Feature flags
    enhanced_enabled = os.getenv("ENHANCED_CHUNKING_ENABLED", "false").lower() in ("true", "1", "yes")
    tagging_enabled = os.getenv("ENHANCED_TAGGING_ENABLED", "false").lower() in ("true", "1", "yes")
    formula_enabled = os.getenv("FORMULA_EXTRACTION_ENABLED", "false").lower() in ("true", "1", "yes")
    
    if not enhanced_enabled:
        # Fall back to legacy chunker
        logger.info("Enhanced chunking disabled, using legacy structural_chunk_resource")
        return structural_chunk_resource(resource_path)
    
    logger.info("Using enhanced semantic chunker", extra={"resource": resource_path})
    
    # Extract pages
    pages = extract_text_by_type(resource_path, None)
    
    # Create semantic chunks with content-aware sizing and formula preservation
    all_chunks = semantic_chunker.create_semantic_chunks(
        pages,
        content_aware=True,
        preserve_formulas=True
    )
    
    # Enhance chunks with hierarchical tags and formula metadata if enabled
    if tagging_enabled or formula_enabled:
        logger.info("Applying enhanced tagging (hierarchical=%s, formulas=%s)", tagging_enabled, formula_enabled)
        
        # Process chunks with context awareness
        previous_tags = []
        for idx, chunk in enumerate(all_chunks):
            try:
                # Build section context
                section_context = {
                    'title': chunk.get('section_title', ''),
                    'number': chunk.get('section_number', ''),
                    'level': chunk.get('section_level')
                }
                
                # Use full hierarchical tagger if both features enabled
                if tagging_enabled and formula_enabled:
                    # Complete tagging and formula extraction
                    enhanced_chunk = hierarchical_tagger.tag_and_extract_formulas(
                        chunk,
                        section_context=section_context if section_context['title'] else None,
                        previous_tags=previous_tags if previous_tags else None
                    )
                    # Update chunk in place
                    chunk.update(enhanced_chunk)
                    
                elif tagging_enabled:
                    # Only hierarchical tagging
                    hierarchical_tags = hierarchical_tagger.extract_hierarchical_tags(
                        chunk.get("full_text", ""),
                        section_context=section_context if section_context['title'] else None,
                        previous_tags=previous_tags[-1] if previous_tags else None
                    )
                    chunk.update(hierarchical_tags)
                    
                    # Classify content type and estimate difficulty
                    chunk['content_type'] = hierarchical_tagger.classify_content_type(chunk)
                    difficulty, details = hierarchical_tagger.estimate_difficulty(chunk)
                    chunk['difficulty'] = difficulty
                    chunk['difficulty_details'] = details
                    
                elif formula_enabled:
                    # Only formula extraction
                    if chunk.get("has_equation"):
                        chunk_text = chunk.get("full_text", "")
                        latex_formulas = hierarchical_tagger.parse_latex(chunk_text)
                        if latex_formulas:
                            formula_metadata = hierarchical_tagger.extract_formula_metadata(
                                latex_formulas,
                                chunk_text,
                                chunk_context=section_context if section_context['title'] else None
                            )
                            chunk["formulas"] = formula_metadata
                            chunk["formula_count"] = len(formula_metadata)
                
                # Store this chunk's tags for next chunk's prerequisites
                if tagging_enabled:
                    previous_tags.append({
                        'key_concepts': chunk.get('key_concepts', []),
                        'domain': chunk.get('domain', []),
                        'topic': chunk.get('topic', [])
                    })
                
            except Exception as e:
                logger.exception("Failed to enhance chunk", extra={"page": chunk.get("page_number"), "chunk_idx": idx})
    else:
        # Even if enhanced tagging is disabled, classify pedagogy_role for all chunks
        logger.info("Applying basic pedagogy role classification")
        for chunk in all_chunks:
            try:
                section_context = {
                    'title': chunk.get('section_title', ''),
                    'number': chunk.get('section_number', ''),
                    'level': chunk.get('section_level')
                }
                pedagogy_role = hierarchical_tagger.classify_pedagogy_role(
                    chunk.get("full_text", ""),
                    section_context=section_context if section_context.get('title') else None
                )
                chunk['pedagogy_role'] = pedagogy_role
            except Exception as e:
                logger.exception("Failed to classify pedagogy role", extra={"page": chunk.get("page_number")})
                chunk['pedagogy_role'] = "explanation"
    
    # Add extended context windows and figure metadata (INGEST-05)
    context_enabled = os.environ.get("EXTENDED_CONTEXT_ENABLED", "false").lower() in ("true", "1", "yes")
    if context_enabled:
        logger.info("Adding extended context windows and figure metadata")
        for idx, chunk in enumerate(all_chunks):
            try:
                tags = {
                    'prerequisites': chunk.get('prerequisites', []),
                    'domain': chunk.get('domain', []),
                    'topic': chunk.get('topic', [])
                } if tagging_enabled else None
                
                context_builder.enhance_chunk_with_context(chunk, all_chunks, idx, tags)
            except Exception as e:
                logger.exception("Failed to add context", extra={"page": chunk.get("page_number"), "chunk_idx": idx})
    
    # Build chunk relationships and learning sequences (INGEST-06)
    linking_enabled = os.environ.get("CHUNK_LINKING_ENABLED", "false").lower() in ("true", "1", "yes")
    if linking_enabled:
        logger.info("Building chunk relationships and learning sequences")
        try:
            all_chunks = chunk_linker.link_all_relationships(all_chunks)
        except Exception as e:
            logger.exception("Failed to link chunk relationships")
    
    # Collect metrics
    try:
        metrics.timing("enhanced_chunks_per_resource", len(all_chunks))
        if all_chunks:
            token_counts = [c.get("token_count", 0) for c in all_chunks]
            avg_tokens = int(sum(token_counts) / max(1, len(token_counts)))
            metrics.timing("enhanced_avg_tokens_per_chunk", avg_tokens)
            
            # Count chunks by content type
            content_types = {}
            for chunk in all_chunks:
                ct = chunk.get("content_type", "unknown")
                content_types[ct] = content_types.get(ct, 0) + 1
            logger.info("Chunk content type distribution", extra={"distribution": content_types})
            
            # Count formula-containing chunks
            formula_chunks = sum(1 for c in all_chunks if c.get("has_equation"))
            logger.info(f"Formula-containing chunks: {formula_chunks}/{len(all_chunks)}")
    except Exception:
        pass
    
    return all_chunks


