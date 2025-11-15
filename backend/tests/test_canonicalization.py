import os
from backend.kg_pipeline.canonicalization import ConceptCanonicalizer


def test_acronym_expansion():
    c = ConceptCanonicalizer()
    assert c.canonicalize("FFT") == "Fast Fourier Transform"


def test_synonym_mapping():
    c = ConceptCanonicalizer()
    out1 = c.canonicalize("heat transfer")
    out2 = c.canonicalize("Heat Xfer")
    assert out1 == out2


def test_fuzzy_match_threshold_env():
    os.environ["KG_FUZZY_MATCH_THRESHOLD"] = "0.80"
    c = ConceptCanonicalizer()
    # Close match should pass with 0.80 threshold
    assert c.canonicalize("Fourier's Law") == c.canonicalize("Fourier Law")
    del os.environ["KG_FUZZY_MATCH_THRESHOLD"]
