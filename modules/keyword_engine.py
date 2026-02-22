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
        """Create richer, less-generic keyword expansions from caption context."""
        # Keyword expansion logic: start from scene/mood anchors, then add semantic variants.
        base = [
            scene,
            mood,
            f"{scene} shorts",
            f"{mood} vibes",
            f"{scene} video",
            f"{scene} aesthetic",
            "youtube shorts",
            "cinematic shorts",
        ]

        scene_semantic = {
            "nature": ["nature views", "scenic nature", "relaxing nature", "outdoor serenity"],
            "city": ["urban visuals", "city vibes", "street atmosphere", "city aesthetic"],
            "people": ["human moments", "daily life", "real emotions", "lifestyle short"],
            "travel": ["travel vibes", "wanderlust", "destination views", "travel aesthetic"],
            "food": ["food visuals", "foodie short", "cooking vibes", "tasty moments"],
            "abstract": ["abstract visuals", "aesthetic edit", "visual art short", "creative visuals"],
            "other": ["visual story", "short clip", "immersive visuals", "trending short"],
        }
        mood_semantic = {
            "peaceful": ["calming vibes", "mindful moments", "soothing visuals"],
            "dramatic": ["intense visuals", "high tension", "powerful atmosphere"],
            "energetic": ["high energy", "dynamic visuals", "fast paced"],
            "mysterious": ["mystic vibes", "curious mood", "enigmatic visuals"],
            "emotional": ["heartfelt moments", "emotional vibes", "deep feelings"],
        }

        # Pull meaningful caption tokens for long-tail relevance.
        tokens = re.findall(r"[a-zA-Z]+", caption.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
        common = [word for word, _ in Counter(filtered).most_common(8)]

        # Preserve insertion order and deduplicate for stable, reusable output.
        expanded: List[str] = []
        for item in base + scene_semantic.get(scene, []) + mood_semantic.get(mood, []) + common:
            if item not in expanded:
                expanded.append(item)
        return expanded
