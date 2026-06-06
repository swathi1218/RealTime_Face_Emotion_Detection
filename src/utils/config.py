"""
EmotionLens Configuration Module.

Centralised configuration for all system parameters.
Modify values here to tune performance vs accuracy trade-offs.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture configuration."""
    width: int = 640
    height: int = 480
    fps_target: int = 30
    device_index: int = 0
    buffer_size: int = 1  # Minimise latency by keeping buffer small


@dataclass(frozen=True)
class DetectionConfig:
    """MediaPipe face detection configuration."""
    model_selection: int = 0          # 0 = short range (<2m), 1 = full range
    min_detection_confidence: float = 0.5
    padding_ratio: float = 0.15       # Extra padding around bounding box
    min_face_size: int = 30           # Minimum face crop size in pixels


@dataclass(frozen=True)
class EmotionConfig:
    """Emotion recognition configuration."""
    model_name: str = "enet_b0_8_best_afew"
    device: str = "cpu"
    batch_size: int = 1
    input_size: Tuple[int, int] = (224, 224)
    frame_skip: int = 2              # Run inference every N-th frame


@dataclass(frozen=True)
class SmoothingConfig:
    """Temporal smoothing configuration."""
    window_size: int = 10
    use_ema: bool = True
    ema_alpha: float = 0.3           # EMA weight for latest prediction


@dataclass(frozen=True)
class LoggingConfig:
    """CSV logging configuration."""
    log_path: str = "data/emotion_logs.csv"
    flush_interval: int = 10         # Flush to disk every N rows
    max_rows: int = 100_000          # Rotate log after this many rows


@dataclass(frozen=True)
class DrawingConfig:
    """Visualisation drawing configuration."""
    bbox_color_map: dict = field(default_factory=lambda: {
        "Happiness":    (0, 220, 100),
        "Neutral":  (0, 180, 180),
        "Sadness":  (80, 80, 255),
        "Angry":    (0, 0, 220),
        "Surprise ": (0, 200, 255),
        "Fear":     (160, 0, 200),
        "Disgust":  (0, 160, 80),
    })
    default_bbox_color: Tuple[int, int, int] = (200, 200, 200)
    bbox_thickness: int = 2
    font_scale: float = 0.65
    font_thickness: int = 2
    label_bg_alpha: float = 0.6
    fps_color: Tuple[int, int, int] = (255, 255, 0)


# ── Singleton instances ──────────────────────────────────────────────────────

CAMERA_CONFIG = CameraConfig()
DETECTION_CONFIG = DetectionConfig()
EMOTION_CONFIG = EmotionConfig()
SMOOTHING_CONFIG = SmoothingConfig()
LOGGING_CONFIG = LoggingConfig()
DRAWING_CONFIG = DrawingConfig()

# ── Flat aliases kept for backwards-compatible imports ────────────────────────

CAMERA_WIDTH: int = CAMERA_CONFIG.width
CAMERA_HEIGHT: int = CAMERA_CONFIG.height
FACE_CONFIDENCE_THRESHOLD: float = DETECTION_CONFIG.min_detection_confidence
SMOOTHING_WINDOW: int = SMOOTHING_CONFIG.window_size
FRAME_SKIP: int = EMOTION_CONFIG.frame_skip
HSEMOTION_MODEL: str = EMOTION_CONFIG.model_name

EMOTION_LABELS: list[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
]
