from typing import List, Dict, Tuple
import os
import logging
import time
import re
from parse_utils import extract_text_by_type
import embed as embed_service
from metrics import MetricsCollector

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


def structural_chunk_resource(resource_path: str) -> List[Dict]:
    """Extract pages/slides and produce semantic-refined chunks per page.

    This function seeds splits by headings (simple heuristic) and then
    applies semantic splitting per page. Tunable parameters can be set
    via environment variables: `SEM_SPLIT_TAU`, `SEM_MIN_TOKENS`,
    `SEM_MAX_TOKENS`, `SEM_OVERLAP`.
    """
    _tau = float(os.getenv("SEM_SPLIT_TAU", 0.65))
    _min = int(os.getenv("SEM_MIN_TOKENS", 60))
    _max = int(os.getenv("SEM_MAX_TOKENS", 240))
    _over = int(os.getenv("SEM_OVERLAP", 1))

    pages = extract_text_by_type(resource_path, None)
    all_chunks: List[Dict] = []
    for i, p in enumerate(pages, start=1):
        # naive heading split: split at lines that look like headings (all caps or numbered)
        blocks = []
        current = []
        for line in p.splitlines():
            if line.strip().isupper() or re.match(r'^\d+\.', line.strip()):
                if current:
                    blocks.append('\n'.join(current))
                    current = [line]
                else:
                    current.append(line)
            else:
                current.append(line)
        if current:
            blocks.append('\n'.join(current))

        for b in blocks:
            subchunks = split_text_into_chunks(b, threshold=_tau, min_tokens=_min, max_tokens=_max, overlap=_over)
            for s in subchunks:
                full = s["full_text"] or ""
                text_norm = full.strip()
                # Filter tiny or low-signal chunks
                if len(text_norm) < 20:
                    continue
                alpha = sum(1 for ch in text_norm if ch.isalpha())
                ratio = alpha / max(1, len(text_norm))
                if ratio < 0.35:
                    continue
                all_chunks.append({
                    "page_number": i,
                    "source_offset": s["source_offset"],
                    "full_text": full,
                })
    # Observability: per-resource chunk stats
    try:
        metrics.timing("chunks_per_resource", len(all_chunks))
        if all_chunks:
            token_counts = [max(1, len((c.get("full_text") or "").split())) for c in all_chunks]
            avg_tokens = int(sum(token_counts) / max(1, len(token_counts)))
            metrics.timing("avg_tokens_per_chunk", avg_tokens)
    except Exception:
        pass
    return all_chunks


