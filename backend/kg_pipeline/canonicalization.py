from typing import Dict, List, Optional, Set
import re
from difflib import SequenceMatcher
import os

# Acronym expansions (extendable via env later)
ACRONYM_MAP = {
    "DFT": "Discrete Fourier Transform",
    "FFT": "Fast Fourier Transform",
    "PDE": "Partial Differential Equation",
    "ODE": "Ordinary Differential Equation",
    "BVP": "Boundary Value Problem",
    "IVP": "Initial Value Problem",
}

# Common variants / synonyms (seed list; can be extended by domain)
SYNONYM_GROUPS = [
    {"Heat Transfer", "heat transfer", "Heat Xfer", "thermal transfer"},
    {"Fourier Law", "Fourier's Law", "Fourier's law of heat conduction"},
    {"Conduction", "Heat Conduction", "thermal conduction"},
]


class ConceptCanonicalizer:
    """Canonicalize concept names for deduplication.

    Steps:
      1. Expand acronyms
      2. Normalize whitespace and punctuation
      3. Map through synonym groups
      4. Fuzzy match to existing canonical forms (threshold configurable)
    """

    def __init__(self, fuzzy_threshold: Optional[float] = None):
        try:
            env_thresh = float(os.getenv("KG_FUZZY_MATCH_THRESHOLD", "0.85"))
        except Exception:
            env_thresh = 0.85
        self.fuzzy_threshold = float(fuzzy_threshold) if fuzzy_threshold is not None else env_thresh
        self.canonical_map = self._build_canonical_map()

    def _build_canonical_map(self) -> Dict[str, str]:
        canonical: Dict[str, str] = {}
        for group in SYNONYM_GROUPS:
            canonical_form = sorted(group)[0]
            for variant in group:
                canonical[variant.lower()] = canonical_form
        return canonical

    def canonicalize(self, concept: str) -> str:
        if not concept:
            return concept

        # Acronym expansion
        stripped_upper = concept.strip().upper()
        if stripped_upper in ACRONYM_MAP:
            return ACRONYM_MAP[stripped_upper]

        # Normalize
        normalized = concept.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[''′]", "'", normalized)

        lower = normalized.lower()
        if lower in self.canonical_map:
            return self.canonical_map[lower]

        # Fuzzy to existing canonical forms
        best = self._fuzzy_match(normalized)
        if best:
            return best

        return normalized

    def _fuzzy_match(self, concept: str) -> Optional[str]:
        best_score = 0.0
        best_canonical = None
        canonical_forms: Set[str] = set(self.canonical_map.values())
        for canonical_form in canonical_forms:
            score = SequenceMatcher(None, concept.lower(), canonical_form.lower()).ratio()
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_canonical = canonical_form
        return best_canonical

    def merge_concepts(self, concepts: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for c in concepts:
            mapping[c] = self.canonicalize(c)
        return mapping
