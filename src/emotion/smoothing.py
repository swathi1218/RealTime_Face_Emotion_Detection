"""
Temporal smoothing for emotion predictions.

Prevents flickering by maintaining a rolling history of predictions
and applying majority-vote or EMA stabilisation strategies.
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from utils.config import SMOOTHING_CONFIG, SmoothingConfig

logger = logging.getLogger(__name__)


@dataclass
class SmoothedPrediction:
    """
    Output of the :class:`EmotionSmoother` for a single face.

    Attributes:
        emotion:    Stabilised emotion label.
        confidence: Smoothed confidence score for the winning emotion.
        scores:     EMA-smoothed probability distribution over all emotions.
        is_stable:  True once the history window is at least half-filled.
    """
    emotion: str
    confidence: float
    scores: Dict[str, float]
    is_stable: bool


class EmotionSmoother:
    """
    Per-face temporal smoother that combines majority voting with EMA.

    Args:
        face_id:   Identifier of the face this smoother tracks.
        cfg:       Smoothing configuration.
    """

    def __init__(
        self,
        face_id: int,
        cfg: SmoothingConfig = SMOOTHING_CONFIG,
    ) -> None:
        self._face_id = face_id
        self._cfg = cfg

        # Rolling history of (emotion_label, confidence) tuples
        self._history: Deque[Tuple[str, float]] = deque(maxlen=cfg.window_size)

        # EMA state for the full score distribution
        self._ema_scores: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        emotion: str,
        confidence: float,
        scores: Dict[str, float],
    ) -> SmoothedPrediction:
        """
        Add a new prediction and return the smoothed result.

        Args:
            emotion:    Raw predicted emotion label.
            confidence: Raw prediction confidence.
            scores:     Full probability distribution from the model.

        Returns:
            :class:`SmoothedPrediction` with stabilised label and scores.
        """
        self._history.append((emotion, confidence))
        self._update_ema(scores)

        smoothed_emotion, smoothed_confidence = self._majority_vote()
        is_stable = len(self._history) >= self._cfg.window_size // 2

        return SmoothedPrediction(
            emotion=smoothed_emotion,
            confidence=round(smoothed_confidence, 4),
            scores={k: round(v, 4) for k, v in self._ema_scores.items()},
            is_stable=is_stable,
        )

    def reset(self) -> None:
        """Clear all history (e.g. when a face disappears between sessions)."""
        self._history.clear()
        self._ema_scores.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _majority_vote(self) -> Tuple[str, float]:
        """
        Determine the dominant emotion from the rolling history.

        Ties are broken by taking the emotion with the highest average
        confidence among those tied.
        """
        if not self._history:
            return "Neutral", 0.0

        vote_counter: Counter = Counter(e for e, _ in self._history)
        max_votes = vote_counter.most_common(1)[0][1]
        candidates = [e for e, cnt in vote_counter.items() if cnt == max_votes]

        # Break ties by average confidence
        best_emotion = max(
            candidates,
            key=lambda e: sum(c for em, c in self._history if em == e) / max_votes,
        )

        avg_confidence = sum(c for em, c in self._history if em == best_emotion) / max_votes

        return best_emotion, avg_confidence

    def _update_ema(self, scores: Dict[str, float]) -> None:
        """Apply exponential moving average to the score distribution."""
        alpha = self._cfg.ema_alpha if self._cfg.use_ema else 1.0

        if not self._ema_scores:
            # Initialise on first update
            self._ema_scores = dict(scores)
        else:
            for emotion, prob in scores.items():
                prev = self._ema_scores.get(emotion, 0.0)
                self._ema_scores[emotion] = alpha * prob + (1 - alpha) * prev


class SmootherPool:
    """
    Manages per-face :class:`EmotionSmoother` instances.

    Creates smoothers on demand and evicts stale ones after a configurable
    number of frames without a detection.

    Args:
        cfg:             Smoothing configuration.
        max_idle_frames: Frames of absence before a smoother is evicted.
    """

    def __init__(
        self,
        cfg: SmoothingConfig = SMOOTHING_CONFIG,
        max_idle_frames: int = 30,
    ) -> None:
        self._cfg = cfg
        self._max_idle = max_idle_frames
        self._smoothers: Dict[int, EmotionSmoother] = {}
        self._idle_counters: Dict[int, int] = {}

    def get_or_create(self, face_id: int) -> EmotionSmoother:
        """Return the smoother for *face_id*, creating one if necessary."""
        if face_id not in self._smoothers:
            self._smoothers[face_id] = EmotionSmoother(face_id, self._cfg)
            logger.debug("Created EmotionSmoother for face_id=%d", face_id)
        self._idle_counters[face_id] = 0
        return self._smoothers[face_id]

    def tick_idle(self, active_ids: List[int]) -> None:
        """
        Increment idle counters for absent faces and evict stale smoothers.

        Args:
            active_ids: Face IDs that were detected in the current frame.
        """
        for face_id in list(self._smoothers.keys()):
            if face_id not in active_ids:
                self._idle_counters[face_id] = self._idle_counters.get(face_id, 0) + 1
                if self._idle_counters[face_id] >= self._max_idle:
                    del self._smoothers[face_id]
                    del self._idle_counters[face_id]
                    logger.debug("Evicted EmotionSmoother for face_id=%d", face_id)

    def reset_all(self) -> None:
        """Clear all smoothers (e.g. on camera restart)."""
        self._smoothers.clear()
        self._idle_counters.clear()
