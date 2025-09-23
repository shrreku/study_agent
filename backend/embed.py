"""Embedding service wrapper.

This module defers heavy ML imports until runtime and falls back to a
lightweight pseudo-embedding when `sentence_transformers` or its
dependencies are unavailable or incompatible in the runtime image.

Public API:
- embed_texts(texts: List[str]) -> List[List[float]]
- embed_text(text: str) -> List[float]
"""
import os
from typing import List, Optional
import time
import logging

# Configurable embedding model name and batch size via env
_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_DEFAULT_BATCH = int(os.getenv("SEM_BATCH_SIZE", "32"))

# internal singleton model reference
_model = None


def _load_model() -> Optional[object]:
    """Lazily load and cache the SentenceTransformer model. Returns None on failure.

    Keeps imports local to avoid heavy import-time overhead during startup.
    """
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import pytorch_cos_sim  # noqa: F401

        _model = SentenceTransformer(_MODEL_NAME)
        return _model
    except Exception:
        # Import or runtime failure — leave _model as None and allow fallback
        return None


def encode_sentences(sentences: List[str], batch_size: int = None, normalize: bool = True) -> List[List[float]]:
    """Encode a list of sentences into embeddings.

    - Uses a singleton cached SentenceTransformer when available.
    - Supports batching to reduce memory pressure and improve throughput.
    - Falls back to a deterministic pseudo-embedding when model is unavailable.

    Args:
        sentences: list of input strings
        batch_size: size for batching; defaults to env SEM_BATCH_SIZE or 32
        normalize: whether to return numpy floats (model returns numpy arrays)
    Returns:
        List of vectors (list of float lists)
    """
    if batch_size is None:
        batch_size = _DEFAULT_BATCH

    model = _load_model()
    if model is not None:
        try:
            out = []
            start = 0
            t0 = time.time()
            while start < len(sentences):
                end = min(len(sentences), start + batch_size)
                batch = sentences[start:end]
                vecs = model.encode(batch, show_progress_bar=False)
                for v in vecs:
                    out.append(v.tolist())
                start = end
            # small perf log
            t_ms = int((time.time() - t0) * 1000)
            if len(sentences) > 0:
                avg_ms_per = t_ms / len(sentences)
                logging.getLogger("backend.embed").info(
                    "encoded %d sentences in %dms (%.1fms/sent)", len(sentences), t_ms, avg_ms_per
                )
            return out
        except Exception:
            # Fallthrough to pseudo embedding on runtime error
            pass

    # deterministic fallback embedding (not for production retrieval)
    out = []
    for t in sentences:
        word_count = max(1, len(t.split()))
        val = (word_count % 100) / 100.0
        vec = [val] * 384
        out.append(vec)
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Backward-compatible wrapper around encode_sentences."""
    return encode_sentences(texts)


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


