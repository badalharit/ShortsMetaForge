from __future__ import annotations

"""SEO metadata generation for YouTube Shorts."""

from collections import deque
from difflib import SequenceMatcher
import hashlib
import re
from typing import Deque, Dict, List, Tuple

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

    _NATURE_WORDS = {
        "landscape": ["meadow", "alpine", "valley", "woodland", "coastal", "horizon", "sunrise", "dusk"],
        "atmosphere": ["golden light", "morning mist", "drifting clouds", "gentle breeze", "still water"],
        "sensory": ["whispering leaves", "flowing stream", "soft sunlight", "rustling trees"],
        "calm_cta": ["Take a moment.", "Let this breathe.", "Pause and reset."],
        "hashtags": ["#NatureLovers", "#ScenicViews", "#CalmVibes", "#MindfulMoments"],
    }

    def __init__(self, max_tags: int, title_max_length: int, specialization_mode: str = "general") -> None:
        self.max_tags = max(8, min(15, int(max_tags)))
        self.title_max_length = title_max_length
        self.specialization_mode = specialization_mode.strip().lower()
        self.keyword_engine = KeywordEngine()
        self._seen_titles: set[str] = set()
        # Pattern memory system: rolling windows enforce anti-repetition across big batches.
        self._recent_title_structures: Deque[str] = deque(maxlen=25)
        self._recent_titles: Deque[str] = deque(maxlen=40)
        self._recent_phrases: Deque[str] = deque(maxlen=20)  # long-tail, CTA, emotional adjectives.
        # First-comment anti-pattern memory.
        self._recent_comment_openings: Deque[str] = deque(maxlen=10)
        self._recent_comment_phrases: Deque[str] = deque(maxlen=20)
        self._recent_comments: Deque[str] = deque(maxlen=30)
        self._emoji_cursor = 0

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

        self._remember_structure(structure_a)
        self._remember_structure(structure_b)

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
        first_comment = self._build_first_comment(
            scene=scene,
            mood=mood,
            focus=focus,
            virality_score=virality_score,
            digest=digest,
        )

        return {
            "title": title_a,
            "title_a": title_a,
            "title_b": title_b,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
            "first_comment": first_comment,
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
        # Virality tier logic: distinct tones by score band.
        tier = self._virality_tier(virality_score)
        emotion = self._pick_emotion_word(mood, digest + variant_seed)
        benefit = self._pick_niche(scene, "benefits", digest + variant_seed)
        scene_tc = self._title_case(scene)
        seconds = 8 + ((digest + variant_seed) % 19)
        outcome = self._pick_outcome(scene, mood, digest + variant_seed)

        prefix = ""
        if tier == "elite":
            prefix = self._pick(["Pure", "Unbelievable", "Ultimate"], digest + variant_seed)
        elif tier == "low":
            prefix = self._pick(["Soft", "Quiet", "Gentle"], digest + variant_seed)

        # Mutation logic: rotate grammar structures, then rebalance by recent structure memory.
        structures = [
            ("A", f"{scene_tc}: Watch This For {self._title_case(benefit)}"),
            ("B", f"{self._title_case(benefit)} Begins With This {scene_tc}"),
            ("C", f"This {scene_tc} Will {self._title_case(outcome)}"),
            ("D", f"{seconds} Seconds Of {self._title_case(emotion)} Inside {scene_tc}"),
            ("E", f"POV: {scene_tc} That Feels {self._title_case(emotion)}"),
            ("F", f"{scene_tc}: Watch This {self._title_case(primary_keyword)}"),
        ]
        ordered = structures[(digest + variant_seed) % len(structures):] + structures[: (digest + variant_seed) % len(structures)]
        chosen_key, chosen_title = ordered[0]
        for key, title in ordered:
            if self._structure_count(key) <= 4:
                chosen_key, chosen_title = key, title
                break

        # Micro-entropy injection: rare human-like variation for anti-pattern defense.
        if ((digest + variant_seed) % 13) == 0:
            chosen_title = chosen_title.replace(": ", ": Just ", 1)
        if ((digest + variant_seed) % 29) == 0 and tier in {"elite", "high"}:
            chosen_title = f"{chosen_title} {self._pick(['✨', '🌿'], digest)}"

        if prefix:
            chosen_title = f"{prefix} {chosen_title}"

        # Light semantic filter: regenerate if too similar to recent titles.
        if self._is_title_too_similar(chosen_title):
            chosen_key = f"{chosen_key}_alt"
            chosen_title = f"{scene_tc}: {self._title_case(focus)} For {self._title_case(emotion)}"

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
        # Virality tier logic: tone clearly differs by performance band.
        tier = self._virality_tier(virality_score)
        emotion = self._pick_emotion_word(mood, digest)
        niche = self._pick_niche(scene, "niche", digest)

        if tier == "elite":
            hook_options = [
                f"A {emotion} hook with pure visual impact.",
                f"An unbelievable {scene} moment that locks attention instantly.",
                f"Ultimate short-form {scene} energy with a {emotion} finish.",
            ]
            cta_options = [
                "Follow now for premium shorts with this level of impact. #Shorts",
                "Share this now if you want more visuals like this. #Shorts",
            ]
        elif tier == "high":
            hook_options = [
                f"A {emotion} visual hook that keeps you watching.",
                f"This {scene} short opens with strong {emotion} tone.",
                f"Bold {scene} storytelling with balanced cinematic energy.",
            ]
            cta_options = [
                f"Follow for more {scene} shorts in this style. #Shorts",
                "Save this and come back when you need this vibe. #Shorts",
            ]
        elif tier == "mid":
            hook_options = [
                f"A calm {emotion} visual moment with aesthetic depth.",
                f"This {scene} sequence carries a steady cinematic mood.",
                f"A polished short that highlights visual atmosphere.",
            ]
            cta_options = [
                "Save this visual for later. #Shorts",
                f"Follow for more aesthetic {scene} shorts. #Shorts",
            ]
        else:
            hook_options = [
                f"A soft {emotion} visual pause for a quieter scroll.",
                f"A gentle {scene} atmosphere with subtle cinematic texture.",
                "A soothing short designed for calm viewing.",
            ]
            cta_options = [
                "Take a calm pause and enjoy the visuals. #Shorts",
                "Save this for a quiet reset. #Shorts",
            ]

        # Nature specialization logic: immersive language and calm-biased CTA.
        if self.specialization_mode == "nature":
            nature_atmosphere = self._pick(self._NATURE_WORDS["atmosphere"], digest)
            hook_options = [
                f"A {emotion} nature moment shaped by {nature_atmosphere}.",
                f"Cinematic nature with {nature_atmosphere} and quiet depth.",
                f"A peaceful visual reset with {nature_atmosphere}.",
            ]
            cta_options = [f"{self._pick(self._NATURE_WORDS['calm_cta'], digest)} #Shorts"]

        line1 = self._pick_with_memory(hook_options, digest)
        line2 = f"{self._title_case(primary_keyword)} with {self._clean_phrase(secondary_keyword)} around {self._clean_phrase(focus)}."
        line3 = f"Built for {niche}: {self._clean_phrase(long_tails[0])}, {self._clean_phrase(long_tails[1])}, {self._clean_phrase(primary_keyword)}."
        line4 = self._pick_with_memory(cta_options, digest // 5)

        return "\n".join([self._normalize_line(line1), self._normalize_line(line2), self._normalize_line(line3), self._normalize_line(line4)])

    def _build_tags(self, scene: str, mood: str, primary_keyword: str, keyword_pool: List[str], long_tails: List[str]) -> str:
        # Tag diversity control: remove semantically near-duplicate tags within the same row.
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
        if self.specialization_mode == "nature":
            tags.extend(["nature lovers", "scenic views", "calm vibes", "mindful moments"])

        deduped = self._dedupe_lower(tags)
        filtered = self._filter_similar_tags(deduped)
        selected = filtered[: self.max_tags]
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
        if self.specialization_mode == "nature":
            pool.extend(self._NATURE_WORDS["hashtags"])

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
        if self.specialization_mode == "nature":
            # Nature specialization long-tail generation.
            landscape = self._pick(self._NATURE_WORDS["landscape"], digest)
            atmosphere = self._pick(self._NATURE_WORDS["atmosphere"], digest + 2)
            candidates.extend(
                [
                    f"cinematic {landscape} sunrise",
                    f"serene {landscape} escape",
                    f"peaceful {atmosphere}",
                ]
            )
        candidates.extend([k for k in keyword_pool if " " in k and len(k.split()) >= 2])

        selected: List[str] = []
        for phrase in candidates:
            clean = self._clean_phrase(phrase).lower()
            if not clean or clean in self._recent_phrases:
                continue
            selected.append(clean)
            self._remember_phrase(clean)
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
        # Synonym rotation is hash-seeded, with recent-phrase avoidance.
        ordered = pool[seed % len(pool):] + pool[: seed % len(pool)]
        for word in ordered:
            if word not in self._recent_phrases:
                self._remember_phrase(word)
                return word
        return ordered[0]

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

    def _virality_tier(self, score: int) -> str:
        if score >= 85:
            return "elite"
        if score >= 70:
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
        self._recent_titles.append(cleaned.lower())
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

    def _structure_count(self, key: str) -> int:
        return sum(1 for item in self._recent_title_structures if item == key)

    def _remember_structure(self, key: str) -> None:
        self._recent_title_structures.append(key)

    def _remember_phrase(self, phrase: str) -> None:
        self._recent_phrases.append(phrase.lower())

    def _pick_with_memory(self, options: List[str], seed: int) -> str:
        ordered = options[seed % len(options):] + options[: seed % len(options)]
        for opt in ordered:
            low = opt.lower()
            if low not in self._recent_phrases:
                self._remember_phrase(low)
                return opt
        self._remember_phrase(ordered[0].lower())
        return ordered[0]

    def _is_title_too_similar(self, candidate: str) -> bool:
        normalized = candidate.lower().strip()
        for recent in self._recent_titles:
            if SequenceMatcher(a=normalized, b=recent).ratio() >= 0.86:
                return True
        return False

    def _filter_similar_tags(self, tags: List[str]) -> List[str]:
        def token_set(tag: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", tag.lower()))

        filtered: List[str] = []
        for tag in tags:
            current = token_set(tag)
            near_duplicate = False
            for kept in filtered:
                kept_set = token_set(kept)
                inter = len(current & kept_set)
                union = len(current | kept_set) or 1
                # Lightweight semantic-near check via token overlap.
                if (inter / union) > 0.8:
                    near_duplicate = True
                    break
            if not near_duplicate:
                filtered.append(tag)
        return filtered

    def _build_first_comment(self, scene: str, mood: str, focus: str, virality_score: int, digest: int) -> str:
        """Generate a 1-2 line engagement comment with anti-repetition safeguards."""
        score_norm = virality_score / 100.0
        emotion = self._pick_emotion_word(mood, digest + 17)
        scene_label = self._title_case(scene)
        focus_short = self._title_case(" ".join(re.findall(r"[A-Za-z]+", focus)[:3]) or f"{scene} moment")

        # Rotate emoji sets (max 2) to avoid repetitive comment signatures.
        emoji_sets = [("🌿", "✨"), ("🍃", "🌅"), ("🫧", "🎧"), ("🌙", "🍂")]
        chosen_emojis = emoji_sets[self._emoji_cursor % len(emoji_sets)]
        self._emoji_cursor += 1
        e1, e2 = chosen_emojis

        strong_hook = score_norm > 0.8
        if self.specialization_mode == "nature":
            # Nature-only mode: calm, reflective, immersive tone.
            frameworks = [
                ("what_did_you_feel", f"What did you feel in this {focus_short.lower()} {e1}"),
                ("which_vibe", f"Which vibe fits this frame best: stillness or wonder?"),
                ("can_you_hear", f"Can you almost hear the breeze in this {scene.lower()} clip? {e2}"),
                ("are_you", f"Are you a sunrise person or a dusk person in scenes like this?"),
                ("one_quiet_line", f"A quiet {scene.lower()} moment, and everything slows down."),
                ("did_you_rewatch", f"Did this loop make you watch one more time?"),
            ]
            if strong_hook:
                frameworks.append(
                    ("felt_this", f"This one feels deeply {emotion} from the first second {e1}")
                )
        else:
            frameworks = [
                ("what_did_you_feel", f"What did you feel first watching this {scene.lower()} short?"),
                ("which_one", f"Which detail stood out more here: color, motion, or mood?"),
                ("can_you_sense", f"Can you sense the {emotion} energy in this frame? {e1}"),
                ("are_you_this", f"Are you someone who rewatches visuals like this?"),
                ("poetic_drop", f"One frame, one mood, and the scroll pauses."),
                ("loop_hook", f"Did the ending make you loop it again? {e2}"),
            ]
            if strong_hook:
                frameworks.append(
                    ("bold_reflect", f"This hit harder than expected, right?")
                )

        # Opening memory: do not reuse same opening key within last 10 generations.
        ordered = frameworks[digest % len(frameworks):] + frameworks[: digest % len(frameworks)]
        selected_key, line1 = ordered[0]
        for key, candidate in ordered:
            if key not in self._recent_comment_openings:
                selected_key, line1 = key, candidate
                break

        line2_options = [
            "Drop one word in the comments.",
            "Curious how you'd describe this in one line.",
            "What would you call this mood?",
            "Would you watch this again tonight?",
        ]
        if self.specialization_mode == "nature":
            line2_options = [
                "Drop one word for this atmosphere.",
                "What would you name this moment?",
                "Would you pause here for a while?",
                "What mood does this leave you with?",
            ]

        line2 = self._pick(line2_options, digest // 3)
        comment = f"{self._normalize_line(line1)}\n{self._normalize_line(line2)}"

        # Phrase memory + semantic dedupe: avoid repeating recent comment intent.
        phrase_key = self._comment_phrase_key(comment)
        if phrase_key in self._recent_comment_phrases or self._is_comment_too_similar(comment):
            alt_idx = (digest // 5) % len(ordered)
            _, alt_line1 = ordered[alt_idx]
            alt_line2 = self._pick(line2_options, digest // 7)
            comment = f"{self._normalize_line(alt_line1)}\n{self._normalize_line(alt_line2)}"
            phrase_key = self._comment_phrase_key(comment)

        self._recent_comment_openings.append(selected_key)
        self._recent_comment_phrases.append(phrase_key)
        self._recent_comments.append(comment.lower())
        return comment

    def _comment_phrase_key(self, comment: str) -> str:
        words = re.findall(r"[a-z]+", comment.lower())
        return " ".join(words[:6])

    def _is_comment_too_similar(self, candidate: str) -> bool:
        normalized = candidate.lower().strip()
        for recent in self._recent_comments:
            if SequenceMatcher(a=normalized, b=recent).ratio() >= 0.84:
                return True
        return False
