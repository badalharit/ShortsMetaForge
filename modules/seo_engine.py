from __future__ import annotations

"""SEO metadata generation for YouTube Shorts."""

import hashlib
import re
from typing import Dict, List

from modules.keyword_engine import KeywordEngine


class SEOEngine:
    """Generate structured, high-CTR, low-repetition metadata at scale."""

    _MOOD_TO_EMOTION = {
        "peaceful": "calming",
        "dramatic": "intense",
        "energetic": "electric",
        "mysterious": "mystical",
        "emotional": "heartfelt",
    }

    _SCENE_TO_BENEFIT = {
        "nature": "instant calm",
        "city": "a fresh perspective",
        "people": "real connection",
        "travel": "wanderlust energy",
        "food": "pure craving",
        "abstract": "visual inspiration",
        "other": "a surprising mood shift",
    }

    def __init__(self, max_tags: int, title_max_length: int) -> None:
        self.max_tags = max(8, min(15, int(max_tags)))
        self.title_max_length = title_max_length
        self.keyword_engine = KeywordEngine()
        self._seen_titles: set[str] = set()

    def build_metadata(
        self,
        filename: str,
        caption: str,
        scene: str,
        mood: str,
        virality_score: int,
        scene_confidence: float,
        caption_confidence: float,
    ) -> Dict[str, str]:
        """Build metadata payload with A/B title variants and strict formatting."""
        digest = self._digest(filename, caption, scene, mood, str(virality_score))
        keywords = self.keyword_engine.expand_keywords(caption=caption, scene=scene, mood=mood)
        primary_keyword = self._clean_phrase(f"{scene} {mood} shorts")
        secondary_keyword = self._pick_secondary_keyword(keywords, primary_keyword)
        focus = self._focus_phrase(caption, scene, mood)

        low_confidence = scene_confidence < 0.45 or caption_confidence < 0.4

        title_a, title_b = self._build_title_variants(
            scene=scene,
            mood=mood,
            primary_keyword=primary_keyword,
            focus=focus,
            digest=digest,
            low_confidence=low_confidence,
        )
        description = self._build_description(
            scene=scene,
            mood=mood,
            primary_keyword=primary_keyword,
            secondary_keyword=secondary_keyword,
            focus=focus,
            keywords=keywords,
            digest=digest,
        )
        tags = self._build_tags(scene=scene, mood=mood, primary_keyword=primary_keyword, keywords=keywords, focus=focus)
        hashtags = self._build_hashtags(scene=scene, mood=mood, focus=focus, digest=digest)

        return {
            "title": title_a,
            "title_a": title_a,
            "title_b": title_b,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
        }

    def _build_title_variants(
        self,
        scene: str,
        mood: str,
        primary_keyword: str,
        focus: str,
        digest: int,
        low_confidence: bool,
    ) -> tuple[str, str]:
        """Hook logic: rotate proven patterns while keeping main keyword near the start."""
        emotion = self._MOOD_TO_EMOTION.get(mood, "captivating")
        benefit = self._SCENE_TO_BENEFIT.get(scene, "a better scroll stop")
        scene_title = self._title_case(scene)
        seconds = 8 + (digest % 13)

        # Hook template rotation avoids repetitive-feeling titles in large batches.
        patterns = [
            f"{scene_title}: Watch This For {self._title_case(benefit)}",
            f"{scene_title}: POV That Feels {self._title_case(emotion)}",
            f"{scene_title}: {seconds} Seconds Of {self._title_case(emotion)}",
            f"{scene_title}: You Won't Believe This View",
            f"{scene_title}: {self._title_case(focus)} That Feels {self._title_case(emotion)}",
            f"{scene_title}: Watch This {self._title_case(primary_keyword)}",
        ]
        if low_confidence:
            patterns.extend(
                [
                    f"{scene_title}: Cinematic Shorts With {self._title_case(emotion)} Vibes",
                    f"{scene_title}: A Visual Mood Worth Watching",
                ]
            )

        first = self._to_unique_title(patterns[digest % len(patterns)])
        second = self._to_unique_title(patterns[(digest // 5) % len(patterns)], allow_duplicate_of=first)
        return first, second

    def _build_description(
        self,
        scene: str,
        mood: str,
        primary_keyword: str,
        secondary_keyword: str,
        focus: str,
        keywords: List[str],
        digest: int,
    ) -> str:
        """Enforce strict 4-line structure with natural keyword placement (2-3 times)."""
        emotion = self._MOOD_TO_EMOTION.get(mood, mood)
        scene_phrase = self._clean_phrase(scene)
        niche_phrase = self._pick_niche_phrase(scene, mood, digest)
        related_terms = ", ".join(self._dedupe_lower(keywords)[:3])

        line1_options = [
            f"A {emotion} visual hook that keeps you watching.",
            f"This {scene_phrase} short opens with a strong {emotion} mood.",
            f"An eye-catching {scene_phrase} moment with a {emotion} finish.",
        ]
        line1 = line1_options[digest % len(line1_options)]
        line2 = f"{self._title_case(primary_keyword)} meets {self._clean_phrase(secondary_keyword)} in this {self._clean_phrase(focus)}."
        line3 = f"Built for {niche_phrase}: {self._title_case(primary_keyword)}, {related_terms}."
        line4 = f"Follow for more premium {self._clean_phrase(scene)} stories. #Shorts"

        # Formatting enforcement: compact spaces and punctuation normalization.
        lines = [self._normalize_line(v) for v in [line1, line2, line3, line4]]
        return "\n".join(lines)

    def _build_tags(self, scene: str, mood: str, primary_keyword: str, keywords: List[str], focus: str) -> str:
        """Keyword expansion logic for 8-15 lowercase, deduplicated, relevant tags."""
        base_tags = [
            primary_keyword.lower(),
            scene.lower(),
            mood.lower(),
            "youtube shorts",
            "shorts",
            "cinematic shorts",
            f"{scene.lower()} vibe",
            f"{mood.lower()} atmosphere",
            focus.lower(),
        ]
        semantic = [
            f"{scene.lower()} aesthetic",
            f"{scene.lower()} video",
            f"{mood.lower()} vibes",
            "visual storytelling",
            "short form video",
        ]
        expanded = self._dedupe_lower(base_tags + semantic + keywords)
        selected = expanded[: self.max_tags]
        if len(selected) < 8:
            fillers = ["trending shorts", "viral shorts", f"{scene.lower()} shorts", f"{mood.lower()} short"]
            selected = self._dedupe_lower(selected + fillers)[:8]
        # Tags are comma+space separated and guaranteed without trailing comma.
        return ", ".join(selected)

    def _build_hashtags(self, scene: str, mood: str, focus: str, digest: int) -> str:
        """Formatting enforcement: #Shorts first, CamelCase, 3-6 total, space separated."""
        focus_words = re.findall(r"[A-Za-z]+", focus)[:2]
        dynamic = ["#" + "".join(w.capitalize() for w in focus_words)] if focus_words else []
        base = [
            "#Shorts",
            self._to_camel_hashtag(scene),
            self._to_camel_hashtag(mood),
            "#CinematicShorts",
            "#VisualStorytelling",
            "#TrendingShorts",
        ]
        pool = base + dynamic
        deduped = self._dedupe_preserve(pool)
        rotated_tail = deduped[3 + (digest % 2) : 6 + (digest % 2)]
        final_tags = deduped[:3] + rotated_tail
        return " ".join(self._dedupe_preserve(final_tags)[:6])

    def _pick_secondary_keyword(self, keywords: List[str], primary_keyword: str) -> str:
        primary = primary_keyword.lower()
        for kw in keywords:
            token = self._clean_phrase(kw).lower()
            if token and token != primary and token not in primary:
                return token
        return "cinematic short"

    def _pick_niche_phrase(self, scene: str, mood: str, digest: int) -> str:
        if scene == "nature" and mood == "peaceful":
            phrases = ["nature relaxation", "calming scenery", "mindful viewing"]
        else:
            phrases = ["cinematic short-form", "immersive visual storytelling", "high-retention short content"]
        return phrases[digest % len(phrases)]

    def _focus_phrase(self, caption: str, scene: str, mood: str) -> str:
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [w for w in words if len(w) > 2 and w not in {scene.lower(), mood.lower(), "with", "this", "that"}]
        return " ".join(filtered[:4]) if filtered else f"{scene} moment"

    def _to_unique_title(self, title: str, allow_duplicate_of: str | None = None) -> str:
        cleaned = self._normalize_line(title)
        cleaned = self._title_case(cleaned)[: self.title_max_length].rstrip(" :,-")
        if cleaned == allow_duplicate_of:
            cleaned = f"{cleaned[: max(1, self.title_max_length - 4)]} Alt"
        if cleaned in self._seen_titles:
            suffix = " II"
            cleaned = f"{cleaned[: max(1, self.title_max_length - len(suffix))]}{suffix}"
        self._seen_titles.add(cleaned)
        return cleaned

    def _title_case(self, text: str) -> str:
        words = self._normalize_line(text).split()
        out: List[str] = []
        for word in words:
            if "'" in word:
                parts = word.split("'")
                parts = [p[:1].upper() + p[1:].lower() if p else "" for p in parts]
                out.append("'".join(parts))
            else:
                out.append(word[:1].upper() + word[1:].lower() if word else "")
        return " ".join(out)

    def _clean_phrase(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _normalize_line(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        clean = re.sub(r"\s+([,.:;!?])", r"\1", clean)
        return clean.rstrip(", ")

    def _to_camel_hashtag(self, phrase: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", phrase)
        if not words:
            return "#Shorts"
        return "#" + "".join(w.capitalize() for w in words)

    def _dedupe_lower(self, values: List[str]) -> List[str]:
        out: List[str] = []
        for value in values:
            token = self._clean_phrase(value).lower()
            if token and token not in out:
                out.append(token)
        return out

    def _dedupe_preserve(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            key = value.lower()
            if key not in seen:
                out.append(value)
                seen.add(key)
        return out

    def _digest(self, *parts: str) -> int:
        return int(hashlib.md5("|".join(parts).encode("utf-8")).hexdigest(), 16)
