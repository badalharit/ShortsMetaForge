from __future__ import annotations

"""Video I/O utilities for scanning, frame extraction, and file moving."""

from pathlib import Path
from typing import Iterable, List
import shutil

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov"}


class VideoProcessor:
    """Handles local video discovery and lifecycle operations."""

    def __init__(self, input_dir: Path, processed_dir: Path, frame_second: int) -> None:
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.frame_second = frame_second

    def scan_videos(self) -> List[Path]:
        """Return sorted candidate video files from input directory."""
        files: Iterable[Path] = self.input_dir.iterdir()
        return sorted([f for f in files if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS])

    def extract_frame(self, video_path: Path) -> "cv2.typing.MatLike":
        """Extract a single representative frame based on configured second."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        # Compute target frame using fps; fallback to first frame when fps is unavailable.
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            target_frame = 0
        else:
            target_frame = int(self.frame_second * fps)
        # Clamp index to valid range for shorter clips.
        if frame_count > 0:
            target_frame = min(target_frame, max(frame_count - 1, 0))

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to extract frame from {video_path.name}")
        return frame

    def extract_frames(self, video_path: Path, ratios: List[float]) -> List["cv2.typing.MatLike"]:
        """Extract multiple contextual frames (for example early/middle/late)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        target_frames: List[int] = []
        if frame_count > 0:
            max_index = max(frame_count - 1, 0)
            for ratio in ratios:
                normalized = max(0.0, min(1.0, float(ratio)))
                target_frames.append(int(round(normalized * max_index)))
        else:
            # Fallback when frame_count is unavailable.
            base_index = int(self.frame_second * fps) if fps > 0 else 0
            target_frames = [base_index for _ in ratios]

        frames: List["cv2.typing.MatLike"] = []
        for idx in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError(f"Unable to extract contextual frames from {video_path.name}")
        return frames

    def get_duration_seconds(self, video_path: Path) -> float:
        """Return video duration in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for duration check: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0:
            return 0.0
        return float(frame_count / fps)

    def move_to_processed(self, video_path: Path) -> Path:
        """Move file into processed folder with collision-safe renaming."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        destination = self.processed_dir / video_path.name
        counter = 1
        while destination.exists():
            destination = self.processed_dir / f"{video_path.stem}_{counter}{video_path.suffix}"
            counter += 1
        shutil.move(str(video_path), str(destination))
        return destination
