"""
Centralised drawing utilities for EmotionLens visualisation.

All OpenCV rendering operations are defined here to ensure a consistent
visual style across the application.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from utils.config import DRAWING_CONFIG, DrawingConfig


def _get_emotion_color(emotion: str, cfg: DrawingConfig = DRAWING_CONFIG) -> Tuple[int, int, int]:
    """Return the BGR color assigned to *emotion*."""
    return cfg.bbox_color_map.get(emotion, cfg.default_bbox_color)


def draw_face_annotation(
    frame: np.ndarray,
    bbox: List[int],
    emotion: str,
    confidence: float,
    face_id: int,
    cfg: DrawingConfig = DRAWING_CONFIG,
) -> None:
    """
    Draw a bounding box with emotion label and confidence on *frame* in-place.

    Args:
        frame:      BGR image array to draw on.
        bbox:       [x, y, w, h] in pixel coordinates.
        emotion:    Predicted emotion string.
        confidence: Confidence in range [0, 1].
        face_id:    Numeric face identifier for multi-face display.
        cfg:        Drawing configuration dataclass.
    """
    x, y, w, h = bbox
    color = _get_emotion_color(emotion, cfg)

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, cfg.bbox_thickness)

    # ── Corner accents ────────────────────────────────────────────────────────
    corner_len = max(12, min(w, h) // 6)
    thickness = cfg.bbox_thickness + 1
    for px, py, dx, dy in [
        (x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)
    ]:
        cv2.line(frame, (px, py), (px + dx * corner_len, py), color, thickness)
        cv2.line(frame, (px, py), (px, py + dy * corner_len), color, thickness)

    # ── Label background ──────────────────────────────────────────────────────
    label = f"#{face_id}  {emotion}  {confidence * 100:.0f}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, cfg.font_scale, cfg.font_thickness
    )
    label_y = max(y - text_h - baseline - 6, 0)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, label_y),
        (x + text_w + 8, label_y + text_h + baseline + 6),
        color,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, cfg.label_bg_alpha, frame, 1 - cfg.label_bg_alpha, 0, frame)

    # ── Label text ────────────────────────────────────────────────────────────
    cv2.putText(
        frame,
        label,
        (x + 4, label_y + text_h + 2),
        cv2.FONT_HERSHEY_DUPLEX,
        cfg.font_scale,
        (255, 255, 255),
        cfg.font_thickness,
        cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float, cfg: DrawingConfig = DRAWING_CONFIG) -> None:
    """
    Overlay the FPS counter in the top-right corner of *frame*.

    Args:
        frame: BGR image array.
        fps:   Current frames-per-second value.
        cfg:   Drawing configuration.
    """
    text = f"FPS: {fps:.1f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - text_w - 10
    cv2.putText(
        frame, text, (x, text_h + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.fps_color, 2, cv2.LINE_AA
    )


def draw_no_face_message(frame: np.ndarray) -> None:
    """
    Draw a subtle 'No face detected' hint at the bottom of *frame*.

    Args:
        frame: BGR image array.
    """
    h, w = frame.shape[:2]
    text = "No face detected"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = (w - text_w) // 2
    cv2.putText(
        frame, text, (x, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA
    )




