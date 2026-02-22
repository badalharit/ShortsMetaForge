from __future__ import annotations

"""Keyword extraction and lightweight expansion for SEO generation."""

from collections import Counter
import re
from typing import List


STOP_WORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "at", "from", "by",
    "is", "it", "this", "that", "short", "video", "frame", "showing",
}


class KeywordEngine:
    def expand_keywords(self, caption: str, scene: str, mood: str) -> List[str]:
        """Create dynamic keyword list from context + caption terms."""
        # Seed list with mandatory scene/mood and Shorts-centric phrases.
        base = [scene, mood, f"{scene} shorts", f"{mood} vibes", "youtube shorts", "viral shorts"]

        # Extract alphabetic tokens from caption and drop weak/common terms.
        tokens = re.findall(r"[a-zA-Z]+", caption.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
        common = [word for word, _ in Counter(filtered).most_common(6)]

        # Preserve insertion order while removing duplicates.
        expanded = []
        for item in base + common:
            if item not in expanded:
                expanded.append(item)
        return expanded
