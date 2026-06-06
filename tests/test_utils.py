"""
Tests for utility modules: FPSCounter and drawing helpers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import numpy as np
import pytest

from utils.fps import FPSCounter
from utils.drawing import draw_fps, draw_no_face_message, draw_face_annotation
from utils.config import DRAWING_CONFIG


class TestFPSCounter:

    def test_initial_fps_is_zero(self) -> None:
        counter = FPSCounter()
        assert counter.fps == 0.0

    def test_fps_increases_after_ticks(self) -> None:
        counter = FPSCounter(window_size=10)
        for _ in range(5):
            counter.tick()
            time.sleep(0.01)
        assert counter.fps > 0

    def test_fps_is_approximately_correct(self) -> None:
        counter = FPSCounter(window_size=20)
        target = 50  # ticks per second
        interval = 1 / target
        for _ in range(15):
            counter.tick()
            time.sleep(interval)
        # Allow ±25% tolerance
        assert target * 0.75 <= counter.fps <= target * 1.25

    def test_reset_zeroes_fps(self) -> None:
        counter = FPSCounter()
        for _ in range(5):
            counter.tick()
        counter.reset()
        assert counter.fps == 0.0

    def test_tick_returns_fps(self) -> None:
        counter = FPSCounter()
        result = counter.tick()
        assert isinstance(result, float)


class TestDrawingHelpers:

    def _blank_frame(self, h: int = 480, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_draw_fps_does_not_raise(self) -> None:
        frame = self._blank_frame()
        draw_fps(frame, 29.7)

    def test_draw_fps_modifies_frame(self) -> None:
        frame = self._blank_frame()
        original = frame.copy()
        draw_fps(frame, 30.0)
        assert not np.array_equal(frame, original)

    def test_draw_no_face_does_not_raise(self) -> None:
        frame = self._blank_frame()
        draw_no_face_message(frame)

    def test_draw_face_annotation_does_not_raise(self) -> None:
        frame = self._blank_frame()
        draw_face_annotation(frame, [50, 50, 100, 100], "Happy", 0.9, 1)

    def test_draw_face_annotation_modifies_frame(self) -> None:
        frame = self._blank_frame()
        original = frame.copy()
        draw_face_annotation(frame, [50, 50, 100, 100], "Happy", 0.9, 1)
        assert not np.array_equal(frame, original)

    def test_draw_face_annotation_unknown_emotion(self) -> None:
        frame = self._blank_frame()
        # Should fall back to default color without raising
        draw_face_annotation(frame, [50, 50, 100, 100], "Unknown", 0.5, 1)

    def test_draw_face_annotation_edge_position(self) -> None:
        frame = self._blank_frame()
        # BBox at top-left corner — label y clamp must not go negative
        draw_face_annotation(frame, [0, 0, 80, 80], "Sad", 0.6, 1)
