from __future__ import annotations

import re
from typing import List

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


def _sentence_lengths(text: str) -> List[int]:
    sentences = re.split(r"[.!?]+\s*", text.strip())
    lengths = [len(sentence.split()) for sentence in sentences if sentence]
    return lengths or [len(text.split())]


def style_check(context: ValidatorContext, config: ValidatorConfig) -> ValidatorComponentResult:
    response_text = context.response_text
    words = response_text.split()
    word_count = len(words)
    sentences = _sentence_lengths(response_text)
    avg_sentence = sum(sentences) / len(sentences)

    score = 1.0
    flags: List[str] = []

    if word_count < config.min_words:
        shortfall_ratio = (config.min_words - word_count) / config.min_words
        score -= min(0.5, shortfall_ratio)
        flags.append("response_too_short")
    if word_count > config.max_words:
        overflow_ratio = (word_count - config.max_words) / config.max_words
        score -= min(0.4, overflow_ratio)
        flags.append("response_too_long")

    if avg_sentence > 32:
        score -= 0.1
        flags.append("long_sentences")

    lowered = response_text.lower()
    banned_hits = [phrase for phrase in config.banned_phrases if phrase in lowered]
    if banned_hits:
        score = min(score, 0.2)
        flags.append("banned_phrase")

    score = max(min(score, 1.0), 0.0)

    details = {
        "word_count": word_count,
        "avg_sentence_length": avg_sentence,
        "banned_hits": banned_hits,
    }

    return ValidatorComponentResult(name="style", score=round(score, 4), details=details, flags=flags)


__all__ = ["style_check"]

