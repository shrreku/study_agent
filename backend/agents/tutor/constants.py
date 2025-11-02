from __future__ import annotations

import logging
from typing import List, Tuple

ALLOWED_INTENTS = {"question", "answer", "reflection", "off_topic", "greeting", "unknown"}
ALLOWED_AFFECTS = {"confused", "unsure", "engaged", "frustrated", "neutral"}

LEVEL_BUCKETS: List[Tuple[str, float, float]] = [
    ("beginner", 0.0, 0.3),
    ("developing", 0.3, 0.6),
    ("proficient", 0.6, 0.8),
    ("mastering", 0.8, 1.01),
]

logger = logging.getLogger(__name__)
