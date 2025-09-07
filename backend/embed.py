"""Embedding service wrapper.

This module defers heavy ML imports until runtime and falls back to a
lightweight pseudo-embedding when `sentence_transformers` or its
dependencies are unavailable or incompatible in the runtime image.

Public API:
- embed_texts(texts: List[str]) -> List[List[float]]
- embed_text(text: str) -> List[float]
"""
import os
from typing import List

_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def _get_model():
    """Lazily load SentenceTransformer. Return None on any failure."""
    global _model
    if _model is not None:
        return _model
    try:
        # import locally to avoid import-time failures during container startup
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
        return _model
    except Exception:
        # If import fails (version mismatch, missing deps), return None
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for texts. Uses real model when available,
    otherwise returns a deterministic pseudo-embedding for testing.
    """
    model = _get_model()
    if model is not None:
        try:
            vecs = model.encode(texts, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception:
            # fallthrough to pseudo-embeddings on any runtime error
            pass

    # Fallback deterministic embedding: uses token/word counts to create a
    # reproducible 384-dim vector. Not suitable for production retrieval.
    out = []
    for t in texts:
        word_count = max(1, len(t.split()))
        val = (word_count % 100) / 100.0
        vec = [val] * 384
        out.append(vec)
    return out


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


