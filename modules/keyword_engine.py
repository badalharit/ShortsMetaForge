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
        """Create scalable keyword expansions with long-tail candidate injection."""
        # Long-tail seeds combine scene+mood, scene+cinematic, and mood+niche.
        long_tail_by_scene = {
            "nature": ["forest walk", "mountain landscape", "sunset visuals", "nature retreat"],
            "city": ["urban exploration", "night street", "city aesthetic", "street rhythm"],
            "people": ["human moments", "daily life", "real reactions", "social vibes"],
            "travel": ["wanderlust journey", "destination views", "travel diary", "adventure route"],
            "food": ["flavor journey", "street food", "mouthwatering plate", "cooking moment"],
            "abstract": ["artistic motion", "creative visuals", "color flow", "visual abstraction"],
            "other": ["visual story", "short sequence", "aesthetic clip", "immersive moment"],
        }

        mood_niche = {
            "peaceful": ["relaxing visuals", "soothing atmosphere", "tranquil mood"],
            "dramatic": ["powerful atmosphere", "bold visuals", "striking mood"],
            "energetic": ["dynamic visuals", "lively atmosphere", "fast paced mood"],
            "mysterious": ["enigmatic atmosphere", "shadowy mood", "surreal tone"],
            "emotional": ["heartfelt atmosphere", "reflective mood", "soulful visuals"],
        }

        base = [
            scene,
            mood,
            f"{scene} shorts",
            f"{mood} vibes",
            f"{scene} cinematic",
            "youtube shorts",
            "cinematic shorts",
            "visual storytelling",
        ]

        # Add semantic scene/mood terms for richer relevance and less generic repetition.
        semantic = long_tail_by_scene.get(scene, []) + mood_niche.get(mood, [])
        long_tail_combo = [
            f"{mood} {scene}",
            f"cinematic {scene}",
            f"{mood} {scene} visuals",
            f"{mood} cinematic {scene}",
        ]

        # Pull caption tokens for contextual long-tail variation.
        tokens = re.findall(r"[a-zA-Z]+", caption.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
        common = [word for word, _ in Counter(filtered).most_common(8)]

        expanded: List[str] = []
        for item in base + semantic + long_tail_combo + common:
            if item not in expanded:
                expanded.append(item)
        return expanded
