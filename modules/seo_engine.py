from __future__ import annotations

"""Premium SEO metadata generation for YouTube Shorts."""

import hashlib
import re
from typing import Dict, List

from modules.keyword_engine import KeywordEngine


class SEOEngine:
    """Generate polished, non-clickbaity, niche-aligned metadata."""

    _NOISE = {
        "a",
        "an",
        "the",
        "with",
        "from",
        "this",
        "that",
        "video",
        "short",
        "frame",
        "showing",
    }

    _NATURE_PACKS = [
        ["nature relaxation", "calming scenery", "peaceful views", "mindful moments", "ambient nature"],
        ["serene landscape", "stress relief visuals", "quiet nature", "slow living", "healing visuals"],
        ["relaxing nature shorts", "meditative scenery", "peaceful atmosphere", "nature therapy", "visual calm"],
    ]

    _GENERAL_PACKS = [
        ["cinematic shorts", "visual storytelling", "aesthetic video", "trending shorts", "viral shorts"],
        ["short form content", "scroll stopping clip", "immersive visuals", "engaging shorts", "youtube shorts"],
    ]

    def __init__(self, max_tags: int, title_max_length: int) -> None:
        self.max_tags = max_tags
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
        """Build production metadata with A/B titles and confidence-aware copy."""
        digest = self._digest(filename, caption, scene, mood)
        focus = self._focus_phrase(caption, scene, mood, 5)
        low_confidence = scene_confidence < 0.45 or caption_confidence < 0.4
        keywords = self.keyword_engine.expand_keywords(caption=caption, scene=scene, mood=mood)
        pack = self._keyword_pack(scene, mood, digest)
        merged_keywords = self._dedupe_lower(pack + keywords)

        if low_confidence:
            title_a, title_b = self._fallback_titles(focus, scene, digest)
            description = self._fallback_description(scene, merged_keywords)
        else:
            title_a, title_b = self._premium_titles(focus, scene, mood, digest)
            description = self._premium_description(focus, caption, scene, mood, merged_keywords, digest)

        title_a = self._to_unique_title(title_a)
        title_b = self._to_unique_title(title_b, allow_duplicate_of=title_a)
        tags = self._build_tags(scene, mood, merged_keywords, caption)
        hashtags = self._build_hashtags(scene, mood, caption, digest)

        return {
            "title": title_a,
            "title_a": title_a,
            "title_b": title_b,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
        }

    def _premium_titles(self, focus: str, scene: str, mood: str, digest: int) -> tuple[str, str]:
        mood_word = {
            "peaceful": "calm",
            "dramatic": "intense",
            "energetic": "vibrant",
            "mysterious": "mystic",
            "emotional": "soulful",
        }.get(mood, "captivating")

        if scene == "nature" and mood == "peaceful":
            options_a = [
                f"20 Seconds of Pure Calm in Nature",
                f"POV: The Most Peaceful View You'll See Today",
                f"A Quiet Nature Moment to Reset Your Mind",
                f"This Nature Scene Feels Like a Deep Breath",
            ]
            options_b = [
                f"Serene Nature Short for Instant Calm",
                f"Gentle Nature Vibes in One Beautiful Shot",
                f"If You Needed Calm Today, Watch This",
                f"Nature Therapy in One Short Clip",
            ]
        else:
            options_a = [
                f"{scene.title()} Visuals with a {mood_word} Finish",
                f"Short, Cinematic, and {mood_word} from Start to End",
                f"This {scene.title()} Clip Holds Attention Instantly",
                f"One {focus} Moment That Feels Premium",
            ]
            options_b = [
                f"Polished {scene.title()} Short with {mood_word} Energy",
                f"{focus} in a Scroll-Stopping Short",
                f"A Clean {scene.title()} Sequence with Strong Atmosphere",
                f"Watch This {scene.title()} Moment in Full",
            ]

        return options_a[digest % len(options_a)], options_b[(digest // 5) % len(options_b)]

    def _premium_description(
        self,
        focus: str,
        caption: str,
        scene: str,
        mood: str,
        keywords: List[str],
        digest: int,
    ) -> str:
        clean_caption = self._clean(caption)
        hook_lines = [
            "A calm visual pause designed for a busy day.",
            "Cinematic short-form storytelling with a soothing finish.",
            "A refined visual moment that slows the scroll.",
            "A clean, immersive short built for mood and atmosphere.",
        ]
        if scene == "nature" and mood == "peaceful":
            hook_lines = [
                "A peaceful nature pause to reset your mind.",
                "Soft scenery, steady pacing, and pure calm energy.",
                "A gentle visual reset in under a minute.",
                "A soothing nature moment worth replaying.",
            ]

        context_lines = [
            f"Scene: {clean_caption}.",
            f"Visual focus: {focus}.",
            f"In this short: {clean_caption}.",
            f"Frame story: {focus}.",
        ]
        seo_lines = [
            f"Discover via: {', '.join(keywords[:5])}.",
            f"Related searches: {', '.join(keywords[:5])}.",
            f"Best match topics: {', '.join(keywords[:5])}.",
        ]
        cta_lines = [
            "If this vibe fits your feed, save it for later.",
            "Follow for more premium shorts in this style.",
            "Share this with someone who needs a calm moment.",
            "Replay once and notice the details.",
        ]

        return "\n".join(
            [
                hook_lines[digest % len(hook_lines)],
                context_lines[(digest // 3) % len(context_lines)],
                seo_lines[(digest // 7) % len(seo_lines)],
                cta_lines[(digest // 11) % len(cta_lines)],
                "#shorts",
            ]
        )

    def _fallback_titles(self, focus: str, scene: str, digest: int) -> tuple[str, str]:
        safe_a = [
            f"Beautiful {scene.title()} Short with a Clean Visual Style",
            f"Immersive {scene.title()} Clip for Your Feed",
            f"Atmospheric {scene.title()} Visual in One Short",
        ]
        safe_b = [
            f"Quick {scene.title()} Moment with Strong Visual Mood",
            f"Aesthetic {scene.title()} Short Worth Rewatching",
            f"{focus} | Visual Short",
        ]
        return safe_a[digest % len(safe_a)], safe_b[(digest // 5) % len(safe_b)]

    def _fallback_description(self, scene: str, keywords: List[str]) -> str:
        return "\n".join(
            [
                "A polished short clip with strong visual atmosphere.",
                f"Theme: {scene.title()} visual storytelling.",
                f"Discover via: {', '.join(keywords[:5])}.",
                "Follow for more high-quality short-form visuals.",
                "#shorts",
            ]
        )

    def _build_tags(self, scene: str, mood: str, keywords: List[str], caption: str) -> str:
        caption_terms = self._caption_terms(caption)
        base = [
            "shorts",
            "youtube shorts",
            f"{scene} shorts",
            f"{mood} vibes",
            scene,
            mood,
            "cinematic shorts",
        ]
        merged = self._dedupe_lower(base + keywords + caption_terms)
        return ", ".join(merged[: self.max_tags])

    def _build_hashtags(self, scene: str, mood: str, caption: str, digest: int) -> str:
        terms = self._caption_terms(caption)[:2]
        dynamic = [f"#{t.replace(' ', '')}" for t in terms if t]
        pool = ["#shorts", f"#{scene}", f"#{mood}", "#visualstorytelling", "#cinematic", "#trending", *dynamic]
        deduped = []
        for tag in pool:
            low = tag.lower()
            if low not in [d.lower() for d in deduped]:
                deduped.append(tag)
        start = digest % 2
        fixed = deduped[:4]
        tail = deduped[4 + start : 6 + start]
        return " ".join((fixed + tail)[:6])

    def _keyword_pack(self, scene: str, mood: str, digest: int) -> List[str]:
        if scene == "nature" and mood == "peaceful":
            return self._NATURE_PACKS[digest % len(self._NATURE_PACKS)]
        return self._GENERAL_PACKS[digest % len(self._GENERAL_PACKS)]

    def _to_unique_title(self, title: str, allow_duplicate_of: str | None = None) -> str:
        trimmed = title[: self.title_max_length].rstrip(" :,-")
        if trimmed == allow_duplicate_of:
            trimmed = f"{trimmed[: max(1, self.title_max_length - 4)]} Alt"
        if trimmed in self._seen_titles:
            suffix = " II"
            trimmed = f"{trimmed[: max(1, self.title_max_length - len(suffix))]}{suffix}"
        self._seen_titles.add(trimmed)
        return trimmed

    def _focus_phrase(self, caption: str, scene: str, mood: str, max_words: int) -> str:
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [w for w in words if len(w) > 2 and w not in self._NOISE and w not in {scene, mood}]
        phrase = " ".join(filtered[:max_words]).strip()
        return phrase.title() if phrase else f"{scene.title()} {mood.title()} Moment"

    def _caption_terms(self, caption: str) -> List[str]:
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [w for w in words if len(w) > 2 and w not in self._NOISE]
        unique = []
        for w in filtered:
            if w not in unique:
                unique.append(w)
        return unique[:8]

    def _clean(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return "visual sequence"
        clean = clean[0].upper() + clean[1:]
        return clean.rstrip(".!?")

    def _dedupe_lower(self, values: List[str]) -> List[str]:
        out: List[str] = []
        for v in values:
            token = v.strip().lower()
            if token and token not in out:
                out.append(token)
        return out

    def _digest(self, *parts: str) -> int:
        return int(hashlib.md5("|".join(parts).encode("utf-8")).hexdigest(), 16)
