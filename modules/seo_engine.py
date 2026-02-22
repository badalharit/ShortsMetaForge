from __future__ import annotations

"""SEO metadata generation for YouTube Shorts."""

import hashlib
import re
from typing import Dict, List, Tuple

from modules.keyword_engine import KeywordEngine


class SEOEngine:
    """Generate scalable metadata with low repetition across large batches."""

    _EMOTION_SYNONYMS = {
        # Synonym rotation pools: mood-specific language avoids repetitive tone.
        "peaceful": ["calm", "serene", "soothing", "tranquil", "relaxing", "still"],
        "dramatic": ["intense", "powerful", "striking", "bold", "epic", "gripping"],
        "energetic": ["vibrant", "dynamic", "lively", "electric", "fast-paced"],
        "mysterious": ["haunting", "surreal", "hidden", "shadowy", "enigmatic"],
        "emotional": ["touching", "heartfelt", "soulful", "moving", "reflective"],
    }

    _NICHE_BIAS = {
        # Niche bias influences title, description, and tags.
        "nature": {
            "benefits": ["instant calm", "a mindful reset", "visual peace"],
            "niche": ["cinematic nature", "peaceful landscape", "relaxing nature visuals"],
            "cta": ["follow for daily calm visuals", "save this for your reset break"],
        },
        "city": {
            "benefits": ["urban inspiration", "a fresh city mood", "street energy"],
            "niche": ["urban exploration", "city aesthetic", "cinematic city visuals"],
            "cta": ["follow for more city shorts", "share with someone who loves city vibes"],
        },
        "people": {
            "benefits": ["human connection", "relatable energy", "real emotion"],
            "niche": ["human-centered story", "real life moments", "emotional storytelling"],
            "cta": ["follow for human stories", "share this with someone who relates"],
        },
        "travel": {
            "benefits": ["wanderlust energy", "adventure mood", "escape inspiration"],
            "niche": ["travel aesthetic", "journey visuals", "adventure storytelling"],
            "cta": ["follow for travel escapes", "save this for your next trip mood"],
        },
        "food": {
            "benefits": ["instant craving", "flavor excitement", "sensory delight"],
            "niche": ["mouthwatering visuals", "food storytelling", "delicious moments"],
            "cta": ["follow for more food shorts", "send this to your foodie friend"],
        },
        "abstract": {
            "benefits": ["creative inspiration", "visual curiosity", "artistic mood"],
            "niche": ["artistic visuals", "creative motion", "abstract storytelling"],
            "cta": ["follow for visual art shorts", "share this with a creative friend"],
        },
        "other": {
            "benefits": ["a unique vibe", "a visual shift", "fresh short-form energy"],
            "niche": ["cinematic short", "aesthetic visual", "immersive storytelling"],
            "cta": ["follow for more premium shorts", "save this for later"],
        },
    }

    def __init__(self, max_tags: int, title_max_length: int) -> None:
        self.max_tags = max(8, min(15, int(max_tags)))
        self.title_max_length = title_max_length
        self.keyword_engine = KeywordEngine()
        self._seen_titles: set[str] = set()
        self._used_title_structures: set[str] = set()
        self._used_long_tails: set[str] = set()

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
        digest = self._digest(filename, caption, scene, mood, str(virality_score))
        keyword_pool = self.keyword_engine.expand_keywords(caption=caption, scene=scene, mood=mood)
        long_tails = self._select_long_tails(keyword_pool, scene=scene, mood=mood, digest=digest, limit=4)
        primary_keyword = self._clean_phrase(f"{mood} {scene} shorts")
        secondary_keyword = self._pick_secondary(keyword_pool, primary_keyword)
        focus = self._focus_phrase(caption, scene, mood)

        title_a, structure_a = self._build_title(
            scene=scene,
            mood=mood,
            focus=focus,
            primary_keyword=primary_keyword,
            virality_score=virality_score,
            digest=digest,
            variant_seed=0,
        )
        title_b, structure_b = self._build_title(
            scene=scene,
            mood=mood,
            focus=focus,
            primary_keyword=primary_keyword,
            virality_score=virality_score,
            digest=digest,
            variant_seed=1,
        )

        # Track used structures for low repetition across large batches.
        self._used_title_structures.add(structure_a)
        self._used_title_structures.add(structure_b)

        title_a = self._to_unique_title(title_a)
        title_b = self._to_unique_title(title_b, allow_duplicate_of=title_a)
        description = self._build_description(
            scene=scene,
            mood=mood,
            focus=focus,
            primary_keyword=primary_keyword,
            secondary_keyword=secondary_keyword,
            long_tails=long_tails,
            virality_score=virality_score,
            digest=digest,
        )
        tags = self._build_tags(scene=scene, mood=mood, primary_keyword=primary_keyword, keyword_pool=keyword_pool, long_tails=long_tails)
        hashtags = self._build_hashtags(scene=scene, mood=mood, focus=focus, long_tails=long_tails, digest=digest)

        return {
            "title": title_a,
            "title_a": title_a,
            "title_b": title_b,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
        }

    def _build_title(
        self,
        scene: str,
        mood: str,
        focus: str,
        primary_keyword: str,
        virality_score: int,
        digest: int,
        variant_seed: int,
    ) -> Tuple[str, str]:
        # Virality scaling: hook intensity and word choice adapt by score band.
        intensity = self._intensity_band(virality_score)
        emotion = self._pick_emotion_word(mood, digest + variant_seed)
        benefit = self._pick_niche(scene, "benefits", digest + variant_seed)
        scene_tc = self._title_case(scene)
        seconds = 8 + ((digest + variant_seed) % 19)
        outcome = self._pick_outcome(scene, mood, digest + variant_seed)

        power_prefix = ""
        if intensity == "high":
            power_prefix = self._pick(["Pure", "Unbelievable", "Ultimate"], digest + variant_seed)
        elif intensity == "low":
            power_prefix = self._pick(["Soft", "Gentle", "Subtle"], digest + variant_seed)

        # Mutation logic: rotate grammar structures, not just words.
        structures = [
            ("A", f"{scene_tc}: Watch This For {self._title_case(benefit)}"),
            ("B", f"{self._title_case(benefit)} Begins With This {scene_tc}"),
            ("C", f"This {scene_tc} Will {self._title_case(outcome)}"),
            ("D", f"{seconds} Seconds Of {self._title_case(emotion)} Inside {scene_tc}"),
            ("E", f"POV: {scene_tc} That Feels {self._title_case(emotion)}"),
            ("F", f"{scene_tc}: Watch This {self._title_case(primary_keyword)}"),
        ]

        # Prefer structures not yet used in current batch for lower repetition.
        ordered = structures[(digest + variant_seed) % len(structures):] + structures[: (digest + variant_seed) % len(structures)]
        chosen_key, chosen_title = ordered[0]
        for key, title in ordered:
            if key not in self._used_title_structures:
                chosen_key, chosen_title = key, title
                break

        if power_prefix:
            chosen_title = f"{power_prefix} {chosen_title}"
        return self._normalize_line(chosen_title), chosen_key

    def _build_description(
        self,
        scene: str,
        mood: str,
        focus: str,
        primary_keyword: str,
        secondary_keyword: str,
        long_tails: List[str],
        virality_score: int,
        digest: int,
    ) -> str:
        # Virality scaling: stronger/softer tone by score band.
        intensity = self._intensity_band(virality_score)
        emotion = self._pick_emotion_word(mood, digest)
        niche = self._pick_niche(scene, "niche", digest)

        if intensity == "high":
            hook_options = [
                f"A {emotion} hook with pure visual impact.",
                f"An unbelievable {scene} moment that grabs attention instantly.",
                f"Ultimate short-form {scene} energy with a {emotion} finish.",
            ]
            cta_options = [
                "Follow now for premium shorts that hit this hard. #Shorts",
                "Share this if you want more powerful visuals like this. #Shorts",
            ]
        elif intensity == "mid":
            hook_options = [
                f"A {emotion} visual hook that keeps you watching.",
                f"This {scene} short opens with a strong {emotion} mood.",
                f"An eye-catching {scene} moment with balanced cinematic tone.",
            ]
            cta_options = [
                f"Follow for more {scene} shorts in this style. #Shorts",
                "Save this and come back when you need this vibe. #Shorts",
            ]
        else:
            hook_options = [
                f"A soft {emotion} visual moment designed to slow the scroll.",
                f"A calm {scene} atmosphere with gentle cinematic detail.",
                f"A soothing short that highlights pure visual mood.",
            ]
            cta_options = [
                "Save this for a calm visual reset. #Shorts",
                "Follow for more relaxing aesthetic shorts. #Shorts",
            ]

        # Strict 4-part structure with natural 2-3 keyword placements.
        line1 = self._pick(hook_options, digest)
        line2 = f"{self._title_case(primary_keyword)} with {self._clean_phrase(secondary_keyword)} around {self._clean_phrase(focus)}."
        line3 = f"Built for {niche}: {self._clean_phrase(long_tails[0])}, {self._clean_phrase(long_tails[1])}, {self._clean_phrase(primary_keyword)}."
        line4 = self._pick(cta_options, digest // 5)

        return "\n".join([self._normalize_line(line1), self._normalize_line(line2), self._normalize_line(line3), self._normalize_line(line4)])

    def _build_tags(self, scene: str, mood: str, primary_keyword: str, keyword_pool: List[str], long_tails: List[str]) -> str:
        # Long-tail injection into tags for higher search specificity.
        niche_terms = self._NICHE_BIAS.get(scene, self._NICHE_BIAS["other"])["niche"]
        tags = [
            primary_keyword.lower(),
            scene.lower(),
            mood.lower(),
            "shorts",
            "youtube shorts",
            "cinematic shorts",
            *[t.lower() for t in niche_terms[:2]],
            *[t.lower() for t in long_tails],
            *[k.lower() for k in keyword_pool[:8]],
        ]
        deduped = self._dedupe_lower(tags)
        selected = deduped[: self.max_tags]
        if len(selected) < 8:
            selected = self._dedupe_lower(selected + [f"{scene.lower()} shorts", f"{mood.lower()} vibes", "visual storytelling"])[:8]
        return ", ".join(selected)

    def _build_hashtags(self, scene: str, mood: str, focus: str, long_tails: List[str], digest: int) -> str:
        focus_words = re.findall(r"[A-Za-z]+", focus)[:2]
        long_tail_words = re.findall(r"[A-Za-z]+", long_tails[0])[:2] if long_tails else []
        dynamic = []
        if focus_words:
            dynamic.append("#" + "".join(w.capitalize() for w in focus_words))
        if long_tail_words:
            dynamic.append("#" + "".join(w.capitalize() for w in long_tail_words))

        # Hashtag rules: #Shorts first, CamelCase, 3-6 total, no duplicates.
        pool = [
            "#Shorts",
            self._to_camel_hashtag(scene),
            self._to_camel_hashtag(mood),
            "#CinematicShorts",
            "#VisualStorytelling",
            "#TrendingShorts",
            *dynamic,
        ]
        deduped = self._dedupe_preserve(pool)
        start = digest % 2
        tail = deduped[3 + start : 6 + start]
        return " ".join(self._dedupe_preserve(deduped[:3] + tail)[:6])

    def _select_long_tails(self, keyword_pool: List[str], scene: str, mood: str, digest: int, limit: int) -> List[str]:
        candidates = []
        # Long-tail creation from requested combinations.
        candidates.extend(
            [
                f"{mood} {scene} visuals",
                f"cinematic {scene} {self._pick_emotion_word(mood, digest)}",
                f"{self._pick_emotion_word(mood, digest + 3)} {scene} atmosphere",
                f"{scene} cinematic short",
            ]
        )
        candidates.extend([k for k in keyword_pool if " " in k and len(k.split()) >= 2])

        selected: List[str] = []
        for phrase in candidates:
            clean = self._clean_phrase(phrase).lower()
            if not clean or clean in self._used_long_tails:
                continue
            selected.append(clean)
            self._used_long_tails.add(clean)
            if len(selected) >= limit:
                break

        # Fallback ensures 2-4 long-tail phrases per video.
        if len(selected) < 2:
            fallback = [f"{mood} {scene}", f"cinematic {scene} visuals", f"{mood} niche visuals"]
            for phrase in fallback:
                clean = self._clean_phrase(phrase).lower()
                if clean not in selected:
                    selected.append(clean)
                if len(selected) >= 2:
                    break
        return selected[:max(2, min(4, limit))]

    def _pick_secondary(self, keywords: List[str], primary_keyword: str) -> str:
        primary = primary_keyword.lower()
        for item in keywords:
            token = self._clean_phrase(item).lower()
            if token and token != primary and token not in primary:
                return token
        return "cinematic short"

    def _pick_emotion_word(self, mood: str, seed: int) -> str:
        pool = self._EMOTION_SYNONYMS.get(mood, ["captivating"])
        # Synonym rotation is hash-seeded for stable but varied output.
        return self._pick(pool, seed)

    def _pick_outcome(self, scene: str, mood: str, seed: int) -> str:
        options = {
            "nature": ["reset your mood", "slow your scroll", "calm your feed"],
            "city": ["shift your perspective", "elevate your vibe", "hold your attention"],
            "people": ["feel more relatable", "hit emotionally", "feel deeply real"],
            "travel": ["spark wanderlust", "feel like an escape", "pull you into adventure"],
            "food": ["trigger cravings", "feel mouthwatering", "look unbelievably good"],
            "abstract": ["spark visual curiosity", "feel artistically bold", "look creatively fresh"],
            "other": ["change your mood", "feel visually rich", "stand out instantly"],
        }
        return self._pick(options.get(scene, options["other"]), seed)

    def _pick_niche(self, scene: str, key: str, seed: int) -> str:
        return self._pick(self._NICHE_BIAS.get(scene, self._NICHE_BIAS["other"])[key], seed)

    def _intensity_band(self, score: int) -> str:
        if score >= 75:
            return "high"
        if score >= 50:
            return "mid"
        return "low"

    def _focus_phrase(self, caption: str, scene: str, mood: str) -> str:
        words = re.findall(r"[A-Za-z]+", caption.lower())
        filtered = [w for w in words if len(w) > 2 and w not in {scene.lower(), mood.lower(), "with", "this", "that"}]
        return " ".join(filtered[:4]) if filtered else f"{scene} moment"

    def _to_unique_title(self, title: str, allow_duplicate_of: str | None = None) -> str:
        cleaned = self._title_case(self._normalize_line(title))
        cleaned = cleaned[: self.title_max_length].rstrip(" :,-")
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

    def _normalize_line(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        clean = re.sub(r"\s+([,.:;!?])", r"\1", clean)
        return clean.rstrip(", ")

    def _clean_phrase(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _to_camel_hashtag(self, phrase: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", phrase)
        return "#Shorts" if not words else "#" + "".join(w.capitalize() for w in words)

    def _pick(self, items: List[str], seed: int) -> str:
        return items[seed % len(items)]

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
