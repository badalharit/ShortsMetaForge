from __future__ import annotations

"""CSV persistence layer with duplicate filename protection."""

from pathlib import Path
from typing import Dict, Set

import pandas as pd


CSV_COLUMNS = [
    "filename",
    "title",
    "title_a",
    "title_b",
    "description",
    "first_comment",
    "tags",
    "hashtags",
    "duration_sec",
    "scene",
    "mood",
    "scene_confidence",
    "caption_confidence",
    "virality_score",
    "strategy",
    "priority_bucket",
]


class CSVWriter:
    """Append metadata rows to CSV while avoiding duplicate filenames."""

    def __init__(self, output_csv: Path) -> None:
        self.output_csv = output_csv
        # Load existing names once for O(1) duplicate checks during batch run.
        self._known_filenames: Set[str] = self._load_known_filenames()

    def _load_known_filenames(self) -> Set[str]:
        """Read existing CSV and return known filenames."""
        if not self.output_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.output_csv)
            if "filename" not in df.columns:
                return set()
            return set(df["filename"].dropna().astype(str).tolist())
        except Exception:
            # Start clean if CSV is malformed or unreadable.
            return set()

    def is_duplicate(self, filename: str) -> bool:
        """Check if filename already exists in current or prior CSV data."""
        return filename in self._known_filenames

    def append_row(self, row: Dict[str, object]) -> None:
        """Append one row with schema normalization and header-once behavior."""
        if "filename" not in row:
            raise ValueError("Row must include 'filename'.")

        filename = str(row["filename"])
        if self.is_duplicate(filename):
            return

        # Ensure CSV directory exists before writing.
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Preserve canonical column order regardless of input row order.
        frame = pd.DataFrame([{col: row.get(col, "") for col in CSV_COLUMNS}], columns=CSV_COLUMNS)
        write_header = not self.output_csv.exists()
        frame.to_csv(self.output_csv, mode="a", header=write_header, index=False)
        self._known_filenames.add(filename)
