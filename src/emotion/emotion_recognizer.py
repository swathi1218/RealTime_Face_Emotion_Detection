"""
Emotion recognition module backed by HSEmotion (EfficientNet-based).

Handles model loading, preprocessing, inference, and result formatting.
Designed for CPU-only deployment with batch_size=1.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.config import EMOTION_CONFIG, EmotionConfig, EMOTION_LABELS

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """
    Output of a single emotion inference call.

    Attributes:
        emotion:    Top-ranked emotion label.
        confidence: Probability of the top emotion (0–1).
        scores:     Full probability distribution over all emotion classes.
    """
    emotion: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


class EmotionRecognizer:
    """
    Wraps the HSEmotion library to provide a simple ``predict`` interface.

    The model is downloaded automatically by HSEmotion on first use and
    cached locally.  Inference runs on CPU by default.

    Args:
        cfg: Emotion configuration dataclass.
    """

    def __init__(self, cfg: EmotionConfig = EMOTION_CONFIG) -> None:
        self._cfg = cfg
        self._model = None
        self._model_loaded: bool = False
        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the model has been loaded successfully."""
        return self._model_loaded

    def predict(self, face_bgr: np.ndarray) -> Optional[EmotionResult]:
        """
        Run emotion inference on a single face crop.

        Args:
            face_bgr: BGR image of a cropped face (any size; will be resized).

        Returns:
            :class:`EmotionResult` on success, or ``None`` if inference fails.
        """
        if not self._model_loaded or face_bgr is None or face_bgr.size == 0:
            return None

        try:
            return self._run_inference(face_bgr)
        except Exception as exc:
            logger.exception("Emotion inference failed: %s", exc)
            return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the HSEmotion model; log any errors without crashing."""
        try:
            import sys, importlib, pkgutil
            import timm.models.layers
            import timm.models.efficientnet_blocks
            for _importer, _modname, _ispkg in pkgutil.iter_modules(timm.models.layers.__path__):
                _old = 'timm.models.layers.' + _modname
                _new     = 'timm.layers.' + _modname
                try:
                    importlib.import_module(_old)
                    sys.modules[_new] = sys.modules[_old]
                except Exception:
                    pass
            sys.modules['timm.layers'] = timm.models.layers
            sys.modules['timm.models._efficientnet_blocks'] = timm.models.efficientnet_blocks

            from hsemotion.facial_emotions import HSEmotionRecognizer

            self._model = HSEmotionRecognizer(
                model_name=self._cfg.model_name,
                device=self._cfg.device,
            )
            self._model_loaded = True
            logger.info("HSEmotion model '%s' loaded on %s.", self._cfg.model_name, self._cfg.device)
        except ImportError:
            logger.error(
                "hsemotion package is not installed. Run: pip install hsemotion",
                )
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _run_inference(self, face_bgr: np.ndarray) -> EmotionResult:
        """
        Preprocess the face image and call the HSEmotion predict method.

        HSEmotion's ``predict_emotions`` accepts a BGR uint8 array and
        returns ``(emotion_label, scores_array)``.
        """
        # HSEmotion handles its own resizing internally
        emotion_label, scores = self._model.predict_emotions(face_bgr, logits=False)

        # Build a mapping from label → probability
        score_map: Dict[str, float] = {}
        if scores is not None and len(scores) > 0:
            flat = np.asarray(scores).flatten()
            labels = self._model.labels if hasattr(self._model, "labels") else EMOTION_LABELS
            for label, prob in zip(labels, flat):
                score_map[label] = float(prob)

        confidence = score_map.get(emotion_label, 1.0)

        return EmotionResult(
            emotion=str(emotion_label),
            confidence=round(float(confidence), 4),
            scores=score_map,
        )


class InferenceThread:
    """
    Frame-skip manager for throttled inference.

    Maintains a counter so that expensive emotion inference is only
    triggered every ``frame_skip`` frames, improving throughput on CPU.

    Args:
        frame_skip: Run inference on every N-th frame (1 = every frame).
    """

    def __init__(self, frame_skip: int = EMOTION_CONFIG.frame_skip) -> None:
        self._skip = max(1, frame_skip)
        self._counter: int = 0

    def should_infer(self) -> bool:
        """Return True if this frame should trigger inference."""
        self._counter += 1
        if self._counter >= self._skip:
            self._counter = 0
            return True
        return False

    def reset(self) -> None:
        """Reset the frame counter."""
        self._counter = 0
