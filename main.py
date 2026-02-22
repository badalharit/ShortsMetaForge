from __future__ import annotations

"""Application entrypoint for ShortsMetaForge."""

import logging
from pathlib import Path

from modules.csv_writer import CSVWriter
from modules.seo_engine import SEOEngine
from modules.utils import (
    build_csv_row,
    classify_content_strategy,
    enforce_local_runtime_cache_paths,
    ensure_directories,
    load_config,
    resolve_device,
    resolve_project_root,
    setup_logging,
)
from modules.video_processor import VideoProcessor


def run(config_path: Path) -> None:
    """Execute the end-to-end batch metadata pipeline."""
    # Initialize structured logging before any work starts.
    setup_logging()
    logger = logging.getLogger("shorts_meta_forge")
    project_root = resolve_project_root()
    enforce_local_runtime_cache_paths(project_root)
    # Import VisionEngine only after cache env vars are enforced.
    from modules.vision_engine import VisionEngine

    # Load runtime configuration and ensure required folders exist.
    config = load_config(config_path)
    ensure_directories(config)
    if config.processing.wipe_output_csv_on_start and config.paths.output_csv.exists():
        config.paths.output_csv.unlink()
        logger.info("Wiped previous CSV for fresh run", extra={"event": "wipe_csv", "filename_ctx": str(config.paths.output_csv)})

    # Resolve runtime device with automatic CUDA fallback behavior.
    device = resolve_device(config.processing.device)
    logger.info("Initialized device", extra={"event": "init", "filename_ctx": str(device)})

    # Build service objects once and reuse for the full batch.
    video_processor = VideoProcessor(
        input_dir=config.paths.input_dir,
        processed_dir=config.paths.processed_dir,
        frame_second=config.processing.extract_frame_second,
    )
    vision_engine = VisionEngine(device=device)
    seo_engine = SEOEngine(max_tags=config.seo.max_tags, title_max_length=config.seo.title_max_length)
    csv_writer = CSVWriter(output_csv=config.paths.output_csv)

    videos = video_processor.scan_videos()
    total = len(videos)
    logger.info("Discovered videos", extra={"event": "scan", "filename_ctx": str(total)})

    # Process each file independently so one failure does not stop the batch.
    for index, video_path in enumerate(videos, start=1):
        filename = video_path.name

        # Skip duplicates only when running in append mode.
        if (not config.processing.wipe_output_csv_on_start) and csv_writer.is_duplicate(filename):
            logger.info("Skipped duplicate", extra={"event": "skip_duplicate", "filename_ctx": filename})
            moved = video_processor.move_to_processed(video_path)
            logger.info("Moved duplicate video", extra={"event": "move_duplicate", "filename_ctx": str(moved)})
            continue

        logger.info(
            "Processing video",
            extra={"event": "progress", "filename_ctx": f"{index}/{total} {filename}"},
        )

        try:
            # 1) Extract early/middle/late frames for context-aware analysis.
            frames = video_processor.extract_frames(video_path, ratios=config.processing.extract_frame_ratios)
            duration_sec = video_processor.get_duration_seconds(video_path)
            # 2) Run aggregated multi-frame vision analysis.
            vision = vision_engine.analyze_frames(frames)
            strategy, priority_bucket = classify_content_strategy(
                duration_sec=duration_sec,
                scene_confidence=vision.scene_confidence,
                caption_confidence=vision.caption_confidence,
                virality_score=vision.virality_score,
            )

            # 3) Generate SEO-ready metadata payload.
            seo_payload = seo_engine.build_metadata(
                filename=filename,
                caption=vision.caption,
                scene=vision.scene,
                mood=vision.mood,
                virality_score=vision.virality_score,
                scene_confidence=vision.scene_confidence,
                caption_confidence=vision.caption_confidence,
            )

            # 4) Persist metadata and move video after successful write.
            row = build_csv_row(
                filename=filename,
                seo_payload=seo_payload,
                duration_sec=duration_sec,
                scene=vision.scene,
                mood=vision.mood,
                scene_confidence=vision.scene_confidence,
                caption_confidence=vision.caption_confidence,
                virality_score=vision.virality_score,
                strategy=strategy,
                priority_bucket=priority_bucket,
            )
            csv_writer.append_row(row)

            destination = video_processor.move_to_processed(video_path)
            logger.info("Processed successfully", extra={"event": "processed", "filename_ctx": str(destination)})
        except Exception as exc:
            # Log and continue so remaining files are still processed.
            logger.exception(
                "Video processing failed",
                extra={"event": "error", "filename_ctx": f"{filename}: {exc}"},
            )


def main() -> None:
    """Resolve default config path and run the pipeline."""
    project_root = resolve_project_root()
    config_path = project_root / "config.yaml"
    run(config_path=config_path)


if __name__ == "__main__":
    main()
