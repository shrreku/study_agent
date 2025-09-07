from typing import List, Dict
import os
from parse_utils import extract_text_by_type


def split_text_into_chunks(text: str, max_tokens: int = 800) -> List[Dict]:
    """
    Naive splitter: split by paragraphs and then join until approx max_tokens.
    Tokens approximated by words for simplicity.
    Returns list of chunks with source_offset and full_text.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    current_len = 0
    offset = 0
    for p in paragraphs:
        plen = len(p.split())
        if current_len + plen > max_tokens and current:
            full = '\n\n'.join(current)
            chunks.append({"source_offset": offset, "full_text": full})
            offset += len(full)
            current = [p]
            current_len = plen
        else:
            current.append(p)
            current_len += plen
    if current:
        full = '\n\n'.join(current)
        chunks.append({"source_offset": offset, "full_text": full})
    return chunks


def structural_chunk_resource(resource_path: str) -> List[Dict]:
    """Extract pages/slides and produce structural chunks per page."""
    pages = extract_text_by_type(resource_path, None)
    all_chunks = []
    for i, p in enumerate(pages, start=1):
        subchunks = split_text_into_chunks(p)
        for s in subchunks:
            all_chunks.append({
                "page_number": i,
                "source_offset": s["source_offset"],
                "full_text": s["full_text"],
            })
    return all_chunks


