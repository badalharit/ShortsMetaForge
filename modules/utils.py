from __future__ import annotations

"""Shared utilities: config loading, logging, path helpers, and row shaping."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict
import json
import logging

import torch
import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    input_dir: Path
    processed_dir: Path
    output_csv: Path


@dataclass(frozen=True)
class ProcessingConfig:
    """Runtime processing options."""

    extract_frame_second: int
    device: str


@dataclass(frozen=True)
class SEOConfig:
    """SEO generation constraints."""

    max_tags: int
    title_max_length: int


@dataclass(frozen=True)
class AppConfig:
    """Top-level typed config container."""

    paths: PathsConfig
    processing: ProcessingConfig
    seo: SEOConfig


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        # Standard log envelope.
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        # Optional custom fields used across the app.
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "filename_ctx"):
            payload["filename"] = record.filename_ctx
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON lines to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def load_config(config_path: Path) -> AppConfig:
    """Parse YAML config file into typed dataclasses."""
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Coerce configured paths into pathlib objects.
    paths = PathsConfig(
        input_dir=Path(raw["paths"]["input_dir"]),
        processed_dir=Path(raw["paths"]["processed_dir"]),
        output_csv=Path(raw["paths"]["output_csv"]),
    )
    # Normalize primitive types for downstream strictness.
    processing = ProcessingConfig(
        extract_frame_second=int(raw["processing"]["extract_frame_second"]),
        device=str(raw["processing"]["device"]),
    )
    seo = SEOConfig(
        max_tags=int(raw["seo"]["max_tags"]),
        title_max_length=int(raw["seo"]["title_max_length"]),
    )
    return AppConfig(paths=paths, processing=processing, seo=seo)


def resolve_device(device_pref: str) -> torch.device:
    """Return CUDA when requested and available, otherwise CPU."""
    pref = device_pref.strip().lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_directories(config: AppConfig) -> None:
    """Create runtime directories if they do not exist."""
    config.paths.input_dir.mkdir(parents=True, exist_ok=True)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    config.paths.output_csv.parent.mkdir(parents=True, exist_ok=True)


def enforce_local_runtime_cache_paths(project_root: Path) -> None:
    """Route runtime caches to project-local folders, even without shell activation."""
    cache_root = project_root / ".cache"
    hf_home = cache_root / "huggingface"
    hf_hub = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    torch_home = cache_root / "torch"
    pycache_prefix = project_root / ".pycache"

    for path in [cache_root, hf_home, hf_hub, transformers_cache, torch_home, pycache_prefix]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)


def build_csv_row(
    filename: str,
    seo_payload: Dict[str, str],
    scene: str,
    mood: str,
    virality_score: int,
) -> Dict[str, object]:
    """Shape metadata into the canonical CSV row schema."""
    return {
        "filename": filename,
        "title": seo_payload["title"],
        "description": seo_payload["description"],
        "tags": seo_payload["tags"],
        "hashtags": seo_payload["hashtags"],
        "scene": scene,
        "mood": mood,
        "virality_score": virality_score,
    }


def resolve_project_root() -> Path:
    """Return fixed project root path used by this deployment."""
    return Path("D:/ShortsMetaForge")
