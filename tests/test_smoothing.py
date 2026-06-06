"""
Tests for the temporal emotion smoothing module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from emotion.smoothing import EmotionSmoother, SmootherPool, SmoothedPrediction
from utils.config import SmoothingConfig


_DUMMY_SCORES = {
    "Happy": 0.7, "Neutral": 0.1, "Sad": 0.05,
    "Angry": 0.05, "Surprise": 0.05, "Fear": 0.03, "Disgust": 0.02,
}


class TestEmotionSmoother:
    """Unit tests for EmotionSmoother."""

    def _make_smoother(self, window: int = 5, ema: bool = True) -> EmotionSmoother:
        cfg = SmoothingConfig(window_size=window, use_ema=ema, ema_alpha=0.3)
        return EmotionSmoother(face_id=1, cfg=cfg)

    def test_single_update_returns_prediction(self) -> None:
        s = self._make_smoother()
        result = s.update("Happy", 0.9, _DUMMY_SCORES)
        assert isinstance(result, SmoothedPrediction)
        assert result.emotion == "Happy"
        assert 0.0 <= result.confidence <= 1.0

    def test_majority_vote_dominates(self) -> None:
        s = self._make_smoother(window=5)
        for _ in range(4):
            s.update("Sad", 0.8, {**_DUMMY_SCORES, "Sad": 0.8, "Happy": 0.1})
        result = s.update("Happy", 0.9, _DUMMY_SCORES)
        assert result.emotion == "Sad"

    def test_scores_contain_all_emotions(self) -> None:
        s = self._make_smoother()
        result = s.update("Happy", 0.9, _DUMMY_SCORES)
        for key in _DUMMY_SCORES:
            assert key in result.scores

    def test_is_stable_after_half_window(self) -> None:
        s = self._make_smoother(window=10)
        for _ in range(4):
            s.update("Happy", 0.9, _DUMMY_SCORES)
        result = s.update("Happy", 0.9, _DUMMY_SCORES)
        assert result.is_stable is True

    def test_is_not_stable_initially(self) -> None:
        s = self._make_smoother(window=10)
        result = s.update("Happy", 0.9, _DUMMY_SCORES)
        assert result.is_stable is False

    def test_reset_clears_history(self) -> None:
        s = self._make_smoother()
        for _ in range(5):
            s.update("Happy", 0.9, _DUMMY_SCORES)
        s.reset()
        result = s.update("Sad", 0.8, _DUMMY_SCORES)
        assert result.emotion == "Sad"

    def test_ema_scores_are_in_range(self) -> None:
        s = self._make_smoother(ema=True)
        for _ in range(3):
            result = s.update("Happy", 0.9, _DUMMY_SCORES)
        for prob in result.scores.values():
            assert 0.0 <= prob <= 1.0

    def test_confidence_rounded(self) -> None:
        s = self._make_smoother()
        result = s.update("Happy", 0.123456789, _DUMMY_SCORES)
        assert len(str(result.confidence).split(".")[-1]) <= 4


class TestSmootherPool:
    """Unit tests for SmootherPool."""

    def test_creates_smoother_on_demand(self) -> None:
        pool = SmootherPool()
        s = pool.get_or_create(face_id=1)
        assert isinstance(s, EmotionSmoother)

    def test_returns_same_instance_for_same_id(self) -> None:
        pool = SmootherPool()
        s1 = pool.get_or_create(1)
        s2 = pool.get_or_create(1)
        assert s1 is s2

    def test_evicts_idle_smoother(self) -> None:
        pool = SmootherPool(max_idle_frames=3)
        pool.get_or_create(face_id=42)
        for _ in range(4):
            pool.tick_idle(active_ids=[])
        assert 42 not in pool._smoothers

    def test_does_not_evict_active_face(self) -> None:
        pool = SmootherPool(max_idle_frames=3)
        pool.get_or_create(face_id=1)
        for _ in range(10):
            pool.tick_idle(active_ids=[1])
        assert 1 in pool._smoothers

    def test_reset_all(self) -> None:
        pool = SmootherPool()
        pool.get_or_create(1)
        pool.get_or_create(2)
        pool.reset_all()
        assert len(pool._smoothers) == 0
