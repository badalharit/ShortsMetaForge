# ShortsMetaForge

Production-grade AI metadata generation pipeline for YouTube Shorts.

## What It Does

- Scans `incoming/` for `.mp4` and `.mov`
- Extracts three contextual frames per video (early/middle/late)
- Generates visual caption (BLIP)
- Classifies scene (CLIP)
- Infers mood + virality score
- Generates YouTube-ready title, description, tags, hashtags
- Generates A/B titles (`title_a`, `title_b`) for optional testing
- Rebuilds CSV from incoming videos when fresh-run mode is enabled
- Flags clips as `standalone` or `merge_candidate` using duration + confidence + score
- Moves processed videos to `processed/`
- Continues batch on per-video errors

No upload logic is included.

## Project Structure

```text
D:/ShortsMetaForge
├── config.yaml
├── main.py
├── requirements.txt
├── pip.ini
├── scripts/
│   ├── setup_local_env.ps1
│   └── activate_local_env.ps1
├── modules/
│   ├── video_processor.py
│   ├── vision_engine.py
│   ├── seo_engine.py
│   ├── keyword_engine.py
│   ├── csv_writer.py
│   └── utils.py
├── data/
├── incoming/
└── processed/
```

## Local-Only Environment (No C: cache/deps)

This project is configured to keep environment, packages, and caches under `D:/ShortsMetaForge`.

Locations used:

- Virtual environment: `D:/ShortsMetaForge/.venv`
- Pip cache: `D:/ShortsMetaForge/.cache/pip`
- Hugging Face cache: `D:/ShortsMetaForge/.cache/huggingface`
- Transformers cache: `D:/ShortsMetaForge/.cache/huggingface/transformers`
- Hub cache: `D:/ShortsMetaForge/.cache/huggingface/hub`
- Torch cache: `D:/ShortsMetaForge/.cache/torch`
- Python bytecode cache: `D:/ShortsMetaForge/.pycache`

### One-time setup

```powershell
cd D:\ShortsMetaForge
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local_env.ps1
```
You do not need to activate the environment in PowerShell.
Run directly with the project venv Python:

```powershell
cd D:\ShortsMetaForge
.\.venv\Scripts\python.exe main.py
```

## Requirements

- Python 3.11+
- NVIDIA GPU + CUDA-enabled PyTorch for GPU acceleration (optional)

## CUDA / Device Behavior

`config.yaml` has:

```yaml
processing:
  device: "cuda"
```

Runtime behavior:

- If `device` is `"cuda"` and CUDA is available, models run on GPU
- Otherwise system falls back to CPU automatically

To force CPU:

```yaml
processing:
  device: "cpu"
```

## Configuration

Default `config.yaml`:

```yaml
paths:
  input_dir: "D:/ShortsMetaForge/incoming"
  processed_dir: "D:/ShortsMetaForge/processed"
  output_csv: "D:/ShortsMetaForge/data/youtube_metadata.csv"

processing:
  extract_frame_second: 3
  extract_frame_ratios: [0.15, 0.5, 0.85]
  wipe_output_csv_on_start: true
  device: "cuda"

seo:
  max_tags: 15
  title_max_length: 80
```

Notes:

- Keep paths explicitly on `D:/ShortsMetaForge`
- `extract_frame_ratios` defines contextual snapshots used for final scene/mood decision
- `wipe_output_csv_on_start: true` clears old CSV rows and regenerates from current `incoming/` only

## Run

1. Put videos in `D:/ShortsMetaForge/incoming`
2. Execute:

```powershell
.\.venv\Scripts\python.exe main.py
```

3. Output CSV: `D:/ShortsMetaForge/data/youtube_metadata.csv`

## CSV Schema

Current CSV columns:

```text
filename,title,title_a,title_b,description,tags,hashtags,duration_sec,scene,mood,scene_confidence,caption_confidence,virality_score,strategy,priority_bucket
```

Example row:

```csv
city_clip_01.mp4,"Cinematic City Visuals with Intense Energy","Cinematic City Visuals with Intense Energy","Watch This Polished City Sequence in Full","A refined visual moment built for short-form viewing.\nScene: Neon street at night with fast motion.\nRelated searches: city, dramatic, city shorts, cinematic shorts, visual storytelling.\nFollow for more premium shorts in this style.\n#shorts","shorts, youtube shorts, city shorts, dramatic vibes, city, dramatic, cinematic shorts","#shorts #city #dramatic #visualstorytelling #cinematic #trending",23.1,city,dramatic,0.79,0.71,78,standalone,high_priority
```

## Logging

Structured JSON logs are emitted to stdout with event metadata for:

- initialization
- scan count
- per-video progress
- duplicate skips
- processing success
- per-video errors

## Operational Notes

- First run downloads BLIP/CLIP model weights from Hugging Face
- Runtime cache paths for Hugging Face/Torch/Python bytecode are enforced to project-local directories by the app
- CSV is wiped at startup when `wipe_output_csv_on_start` is enabled
- Duplicate filename entries are skipped
- Processed files are moved to `processed/` and auto-renamed on name collisions
