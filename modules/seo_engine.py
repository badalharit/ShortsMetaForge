from __future__ import annotations

"""SEO metadata generation for Shorts titles, descriptions, tags, and hashtags."""

import hashlib
import re
from typing import Dict, List

from modules.keyword_engine import KeywordEngine


class SEOEngine:
    """Generate more human, varied, and CTR-friendly metadata from vision outputs."""

    _NOISE_WORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "with",
        "from",
        "this",
        "that",
        "there",
        "here",
        "video",
        "short",
        "frame",
        "showing",
    }

    _MOOD_WORD = {
        "peaceful": "calming",
        "dramatic": "intense",
        "energetic": "high-energy",
        "mysterious": "mysterious",
        "emotional": "heartfelt",
    }

    _SCENE_STYLE = {
        "nature": ["scenery", "landscape", "outdoor"],
        "city": ["street", "urban", "citylife"],
        "people": ["human moments", "real reactions", "daily life"],
        "travel": ["wanderlust", "new places", "adventure"],
        "food": ["foodie", "cooking", "flavor"],
        "abstract": ["visual art", "motion aesthetic", "creative edit"],
        "other": ["visual story", "quick moment", "clip"],
    }

    def __init__(self, max_tags: int, title_max_length: int) -> None:
        self.max_tags = max_tags
        self.title_max_length = title_max_length
        self.keyword_engine = KeywordEngine()
        self._seen_titles: set[str] = set()

    def build_metadata(self, filename: str, caption: str, scene: str, mood: str, virality_score: int) -> Dict[str, str]:
        """Build complete metadata payload for one video."""
        keywords = self.keyword_engine.expand_keywords(caption=caption, scene=scene, mood=mood)
        title = self._build_title(filename, caption, scene, mood)
        description = self._build_description(filename, caption, scene, mood, keywords, virality_score)
        tags = self._build_tags(caption, keywords, scene, mood)
        hashtags = self._build_hashtags(caption, scene, mood)

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
        }

    def _build_title(self, filename: str, caption: str, scene: str, mood: str) -> str:
        """Compose a hook-heavy, curiosity-led title with variation."""
        digest = self._digest(filename, caption, scene, mood)
        focus_short = self._focus_phrase(caption, scene, mood, max_words=3)
        mood_word = self._MOOD_WORD.get(mood, "captivating")

        candidates = [
            f"The {focus_short} Moment You Didn't Expect",
            f"Why This {scene.title()} Clip Feels So {mood_word}",
            f"POV: {focus_short} Hits Different",
            f"Watch This {scene.title()} Scene to the End",
            f"This {focus_short} Shot Is Weirdly Addictive",
            f"Can You Explain This {mood_word} {scene.title()} Moment?",
            f"One {focus_short} Detail Changes Everything",
            f"This {scene.title()} Short Has a {mood_word} Twist",
            f"Not Ready for This {focus_short} Reveal",
            f"{focus_short} in 20 Seconds | {scene.title()} Shorts",
        ]

        return self._pick_unique_title(candidates, digest)

    def _build_description(
        self,
        filename: str,
        caption: str,
        scene: str,
        mood: str,
        keywords: List[str],
        virality_score: int,
    ) -> str:
        """Create 4 polished lines + #shorts footer."""
        digest = self._digest(filename, caption, scene, mood, str(virality_score))
        focus_phrase = self._focus_phrase(caption, scene, mood, max_words=6)
        clean_caption = self._clean_sentence(caption)
        mood_word = self._MOOD_WORD.get(mood, mood)
        style_token = self._pick(self._SCENE_STYLE.get(scene, ["clip"]), digest // 7)
        top_searches = ", ".join(keywords[:5])
        momentum = self._momentum_text(virality_score)

        hook_lines = [
            f"{scene.title()} visuals with a {mood_word} edge. This one stays with you.",
            f"Instant scroll-stopper: {mood_word} {scene} vibes in one clean shot.",
            f"If you like {style_token}, this short lands hard.",
            f"This clip starts simple and ends with a {mood_word} punch.",
            f"Short, sharp, and {mood_word} from the first second.",
        ]
        context_lines = [
            f"On screen: {clean_caption}.",
            f"Story in one frame: {focus_phrase}.",
            f"Context: {clean_caption}.",
            f"What makes it hit: {focus_phrase}.",
        ]
        seo_lines = [
            f"Related searches: {top_searches}.",
            f"SEO focus: {top_searches}.",
            f"Keywords: {top_searches}.",
            f"Best match topics: {top_searches}.",
        ]
        cta_lines = [
            f"Would you watch this again? Replay potential: {virality_score}/100 ({momentum}).",
            f"Drop your rating below. Momentum score: {virality_score}/100 ({momentum}).",
            f"Share with someone who loves {scene} edits. Score: {virality_score}/100 ({momentum}).",
            f"Save this for later if this vibe is your style. Score: {virality_score}/100 ({momentum}).",
        ]

        line1 = self._pick(hook_lines, digest)
        line2 = self._pick(context_lines, digest // 3)
        line3 = self._pick(seo_lines, digest // 5)
        line4 = self._pick(cta_lines, digest // 11)
        return "\n".join([line1, line2, line3, line4, "#shorts"])

    def _build_tags(self, caption: str, keywords: List[str], scene: str, mood: str) -> str:
        """Build keyword-rich, non-spammy comma-separated tags."""
        caption_terms = self._caption_terms(caption)
        base = [
            "shorts",
            "youtube shorts",
            "viral shorts",
            f"{scene} shorts",
            f"{mood} shorts",
            f"{scene} video",
            f"{mood} vibes",
            scene,
            mood,
        ]

        merged: List[str] = []
        for item in base + keywords + caption_terms:
            token = item.strip().lower()
            if not token or token in merged:
                continue
            merged.append(token)
            if len(merged) >= self.max_tags:
                break
        return ", ".join(merged)

    def _build_hashtags(self, caption: str, scene: str, mood: str) -> str:
        """Build concise hashtags with mandatory #shorts."""
        terms = self._caption_terms(caption)[:2]
        dynamic = [f"#{t.replace(' ', '')}" for t in terms if t]
        tags = [
            "#shorts",
            f"#{scene.replace(' ', '')}",
            f"#{mood.replace(' ', '')}",
            "#viralshorts",
            *dynamic,
            "#trending",
        ]
        deduped: List[str] = []
        for tag in tags:
            low = tag.lower()
            if low not in [d.lower() for d in deduped]:
                deduped.append(tag)
        return " ".join(deduped[:6])

    def _pick_unique_title(self, candidates: List[str], digest: int) -> str:
        """Pick deterministic title with uniqueness fallback."""
        for i in range(len(candidates)):
            candidate = candidates[(digest + i) % len(candidates)]
            candidate = candidate[: self.title_max_length].rstrip(" :,-")
            if candidate not in self._seen_titles:
                self._seen_titles.add(candidate)
                return candidate

        fallback = candidates[digest % len(candidates)]
        suffix = f" | {digest % 97}"
        allowed = self.title_max_length - len(suffix)
        titled = f"{fallback[:max(1, allowed)].rstrip(' :,-')}{suffix}"
        self._seen_titles.add(titled)
        return titled

    def _focus_phrase(self, caption: str, scene: str, mood: str, max_words: int) -> str:
        """Extract a natural focus phrase from caption."""
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [
            w
            for w in words
            if len(w) > 2 and w not in self._NOISE_WORDS and w not in {scene.lower(), mood.lower()}
        ]
        phrase = " ".join(filtered[:max_words]).strip()
        return phrase.title() if phrase else f"{scene.title()} {mood.title()} Vibe"

    def _caption_terms(self, caption: str) -> List[str]:
        """Return useful single-word and two-word caption terms for tags/hashtags."""
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [w for w in words if len(w) > 2 and w not in self._NOISE_WORDS]
        singles: List[str] = []
        for word in filtered:
            if word not in singles:
                singles.append(word)

        bigrams: List[str] = []
        for i in range(len(filtered) - 1):
            phrase = f"{filtered[i]} {filtered[i + 1]}"
            if phrase not in bigrams:
                bigrams.append(phrase)

        return (singles + bigrams)[:10]

    def _clean_sentence(self, text: str) -> str:
        """Normalize caption sentence casing/punctuation for description line."""
        cleaned = " ".join(text.strip().split())
        cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
        cleaned = cleaned.rstrip(".!?")
        return cleaned

    def _momentum_text(self, score: int) -> str:
        """Map numeric virality score to a human-friendly descriptor."""
        if score >= 85:
            return "very high"
        if score >= 70:
            return "strong"
        if score >= 55:
            return "solid"
        return "niche"

    def _digest(self, *parts: str) -> int:
        """Deterministic hash helper for stable template variation."""
        return int(hashlib.md5("|".join(parts).encode("utf-8")).hexdigest(), 16)

    def _pick(self, values: List[str], seed: int) -> str:
        """Pick stable list item from hash-derived seed."""
        return values[seed % len(values)]
