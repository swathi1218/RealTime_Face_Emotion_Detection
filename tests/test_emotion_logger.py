"""
Tests for the EmotionLogger CSV logging module.
"""

import csv
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from logs.emotion_logger import EmotionLogger
from utils.config import LoggingConfig


def _make_logger(tmp_path: Path, flush_interval: int = 1) -> EmotionLogger:
    cfg = LoggingConfig(
        log_path=str(tmp_path / "test_log.csv"),
        flush_interval=flush_interval,
        max_rows=10_000,
    )
    return EmotionLogger(cfg)


class TestEmotionLogger:

    def test_creates_log_file_with_header(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        log_path = Path(logger._cfg.log_path)
        assert log_path.exists()
        with log_path.open() as f:
            header = f.readline().strip()
        assert "timestamp" in header
        assert "face_id" in header
        assert "emotion" in header
        assert "confidence" in header

    def test_log_writes_row_after_flush(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=1)
        logger.log(face_id=1, emotion="Happy", confidence=0.92)
        rows = _read_rows(logger._cfg.log_path)
        assert len(rows) == 1
        assert rows[0]["emotion"] == "Happy"

    def test_confidence_formatted_correctly(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=1)
        logger.log(face_id=1, emotion="Sad", confidence=0.7777)
        rows = _read_rows(logger._cfg.log_path)
        assert rows[0]["confidence"] == "0.7777"

    def test_multiple_faces_logged(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=1)
        logger.log(face_id=1, emotion="Happy", confidence=0.9)
        logger.log(face_id=2, emotion="Sad", confidence=0.6)
        rows = _read_rows(logger._cfg.log_path)
        assert len(rows) == 2
        assert {r["face_id"] for r in rows} == {"1", "2"}

    def test_flush_writes_buffer(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=5)
        logger.log(face_id=1, emotion="Neutral", confidence=0.5)
        # Not flushed yet
        rows = _read_rows(logger._cfg.log_path)
        assert len(rows) == 0
        logger.flush()
        rows = _read_rows(logger._cfg.log_path)
        assert len(rows) == 1

    def test_total_rows_counter(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=1)
        for i in range(5):
            logger.log(face_id=i, emotion="Happy", confidence=0.8)
        assert logger.total_rows == 5

    def test_close_flushes_buffer(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, flush_interval=100)
        logger.log(face_id=1, emotion="Angry", confidence=0.4)
        logger.close()
        rows = _read_rows(logger._cfg.log_path)
        assert len(rows) == 1

    def test_log_rotation(self, tmp_path: Path) -> None:
        cfg = LoggingConfig(
            log_path=str(tmp_path / "test_log.csv"),
            flush_interval=1,
            max_rows=2,
        )
        logger = EmotionLogger(cfg)
        for _ in range(4):
            logger.log(face_id=1, emotion="Happy", confidence=0.9)

        # At least one rotated file should exist
        log_files = list(tmp_path.glob("test_log_*.csv"))
        assert len(log_files) >= 1


def _read_rows(log_path: str) -> list:
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
