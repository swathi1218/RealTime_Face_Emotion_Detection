"""
CSV emotion logging module.

Buffers prediction rows in memory and flushes them to a CSV file
at configurable intervals to avoid excessive disk I/O.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from utils.config import LOGGING_CONFIG, LoggingConfig

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES: List[str] = ["timestamp", "face_id", "emotion", "confidence"]


class EmotionLogger:
    """
    Thread-safe CSV logger for emotion prediction events.

    Rows are accumulated in an in-memory buffer and flushed to disk
    every *flush_interval* rows.  When the log reaches *max_rows* the
    file is rotated (old file renamed with a timestamp suffix).

    Args:
        cfg: Logging configuration dataclass.
    """

    def __init__(self, cfg: LoggingConfig = LOGGING_CONFIG) -> None:
        self._cfg = cfg
        self._log_path = Path(cfg.log_path)
        self._buffer: List[dict] = []
        self._total_rows: int = 0
        self._ensure_log_file()

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        face_id: int,
        emotion: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a single emotion prediction.

        Args:
            face_id:    Numeric identifier for the detected face.
            emotion:    Predicted emotion label.
            confidence: Prediction confidence in range [0, 1].
            timestamp:  Event time; defaults to ``datetime.now()``.
        """
        ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self._buffer.append({
            "timestamp": ts,
            "face_id": face_id,
            "emotion": emotion,
            "confidence": f"{confidence:.4f}",
        })

        if len(self._buffer) >= self._cfg.flush_interval:
            self._flush()

    def flush(self) -> None:
        """Force-flush the remaining buffer to disk."""
        self._flush()

    def close(self) -> None:
        """Flush the buffer and finalise the log file."""
        self._flush()
        logger.info("EmotionLogger closed. Total rows written: %d.", self._total_rows)

    @property
    def total_rows(self) -> int:
        """Total number of rows written to disk (excludes buffered rows)."""
        return self._total_rows

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_log_file(self) -> None:
        """Create the log directory and file with headers if needed."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._log_path.exists():
            with self._log_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()
            logger.info("Created emotion log at '%s'.", self._log_path)

    def _flush(self) -> None:
        """Write buffered rows to disk, rotating if necessary."""
        if not self._buffer:
            return

        if self._should_rotate():
            self._rotate()

        try:
            with self._log_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writerows(self._buffer)

            self._total_rows += len(self._buffer)
            self._buffer.clear()

        except OSError as exc:
            logger.error("Failed to write emotion log: %s", exc)

    def _should_rotate(self) -> bool:
        """Return True if the current log file has reached max_rows."""
        try:
            with self._log_path.open("r", encoding="utf-8") as fh:
                # Subtract 1 for the header row
                row_count = sum(1 for _ in fh) - 1
            return row_count >= self._cfg.max_rows
        except OSError:
            return False

    def _rotate(self) -> None:
        """Rename the current log file and create a fresh one."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = self._log_path.with_stem(f"{self._log_path.stem}_{ts}")
        try:
            self._log_path.rename(rotated_path)
            logger.info("Rotated log to '%s'.", rotated_path)
        except OSError as exc:
            logger.error("Log rotation failed: %s", exc)
        finally:
            self._ensure_log_file()
