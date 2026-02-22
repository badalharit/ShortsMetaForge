from __future__ import annotations

"""Vision pipeline: BLIP captioning, CLIP scene classification, mood and virality."""

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class VisionResult:
    """Structured output from visual analysis."""

    caption: str
    scene: str
    mood: str
    virality_score: int


class VisionEngine:
    """Singleton ML inference engine to avoid repeated model loads."""

    _instance: "VisionEngine | None" = None

    SCENE_LABELS = ["nature", "city", "people", "travel", "food", "abstract", "other"]
    SCENE_PROMPTS = {
        "nature": "a short video frame showing nature landscape",
        "city": "a short video frame showing city streets and buildings",
        "people": "a short video frame focused on people",
        "travel": "a short video frame from travel adventure",
        "food": "a short video frame of food or cooking",
        "abstract": "a short video frame with abstract visuals",
        "other": "a short video frame with mixed content",
    }

    MOOD_LABELS = ["peaceful", "dramatic", "energetic", "mysterious", "emotional"]

    def __new__(cls, device: torch.device) -> "VisionEngine":
        # Load heavy models only once per process.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models(device)
        return cls._instance

    def _init_models(self, device: torch.device) -> None:
        """Initialize BLIP and CLIP models/processors on selected device."""
        self.device = device
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def analyze(self, frame_bgr: "cv2.typing.MatLike") -> VisionResult:
        """Run full visual analysis pipeline for one extracted frame."""
        # Convert OpenCV BGR frame to RGB PIL image for HF pipelines.
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Inference stages: caption -> scene -> mood -> score.
        caption = self._caption(pil_image)
        scene_scores = self._scene_scores(pil_image)
        scene = max(scene_scores, key=scene_scores.get)
        mood = self._infer_mood(frame_bgr, scene)
        virality_score = self._virality_score(scene, mood, caption)

        return VisionResult(
            caption=caption,
            scene=scene,
            mood=mood,
            virality_score=virality_score,
        )

    def _caption(self, image: Image.Image) -> str:
        """Generate a concise natural-language caption with BLIP."""
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=30)
        return self.blip_processor.decode(out[0], skip_special_tokens=True).strip()

    def _scene_scores(self, image: Image.Image) -> Dict[str, float]:
        """Score predefined scene classes via CLIP image-text similarity."""
        labels: List[str] = self.SCENE_LABELS
        prompts = [self.SCENE_PROMPTS[label] for label in labels]
        encoded = self.clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**encoded)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy().tolist()

        return {label: float(score) for label, score in zip(labels, probs)}

    def _infer_mood(self, frame_bgr: "cv2.typing.MatLike", scene: str) -> str:
        """Infer mood using lightweight image statistics and scene hints."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2].astype(np.float32)
        sat_channel = hsv[:, :, 1].astype(np.float32)

        brightness = float(np.mean(value_channel))
        contrast = float(np.std(value_channel))
        saturation = float(np.mean(sat_channel))

        if contrast > 65 and brightness < 110:
            return "dramatic"
        if saturation > 130 and brightness > 120:
            return "energetic"
        if brightness < 85:
            return "mysterious"
        if scene in {"people", "travel"}:
            return "emotional"
        return "peaceful"

    def _virality_score(self, scene: str, mood: str, caption: str) -> int:
        """Compute bounded virality heuristic from scene, mood, and caption cues."""
        # Baseline score keeps all results in a usable range.
        score = 50

        # Scene and mood bonuses bias toward high-retention short-form patterns.
        scene_bonus = {
            "people": 12,
            "travel": 10,
            "food": 9,
            "city": 7,
            "nature": 6,
            "abstract": 5,
            "other": 3,
        }
        mood_bonus = {
            "energetic": 16,
            "dramatic": 14,
            "emotional": 12,
            "mysterious": 10,
            "peaceful": 7,
        }

        score += scene_bonus.get(scene, 0)
        score += mood_bonus.get(mood, 0)

        # Small bonus when caption includes strong emotional trigger words.
        caption_lower = caption.lower()
        trigger_words = ["amazing", "secret", "unexpected", "insane", "epic", "stunning"]
        if any(word in caption_lower for word in trigger_words):
            score += 8

        # Clamp score for consistent downstream output.
        return max(1, min(100, score))
