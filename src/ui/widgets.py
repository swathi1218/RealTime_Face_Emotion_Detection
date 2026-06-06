"""
Custom PyQt6 widgets for the EmotionLens UI.

Contains reusable components for the statistics panel, emotion bars,
status indicators, and the video display label.
"""

import time
from typing import Dict

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QImage
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from utils.config import EMOTION_LABELS


# ── Palette ───────────────────────────────────────────────────────────────────

_EMOTION_COLORS: Dict[str, str] = {
    "Happiness":    "#DC00DC",
    "Neutral":  "#B4B4B4",
    "Sadmess":      "#5050FF",
    "Angry":    "#FF3030",
    "Surprise ": "#BFFF00",
    "Fear":     "#A000C8",
    "Disgust":  "#00A050",
}


# ── VideoDisplay ──────────────────────────────────────────────────────────────

class VideoDisplay(QLabel):
    """
    Full-resolution camera feed display.

    Scales the pixmap to fill the label while maintaining aspect ratio.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)
        self.setStyleSheet(
            "background: #0A0A0F; border: 1px solid #1E1E2E; border-radius: 6px;"
        )
        self._placeholder()

    def _placeholder(self) -> None:
        """Show a branded placeholder before the camera opens."""
        self.setText("◉  Camera Initialising…")
        self.setStyleSheet(
            "color: #4040A0; font-size: 16px; background: #0A0A0F;"
            "border: 1px solid #1E1E2E; border-radius: 6px;"
        )

    def update_frame(self, pixmap: QPixmap) -> None:
        """Scale and set a new camera frame pixmap."""
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setStyleSheet(
            "background: #0A0A0F; border: 1px solid #1E1E2E; border-radius: 6px;"
        )


# ── EmotionBar ────────────────────────────────────────────────────────────────

class EmotionBar(QWidget):
    """Single emotion progress bar with label and percentage."""

    def __init__(self, emotion: str, parent=None) -> None:
        super().__init__(parent)
        self._emotion = emotion
        color = _EMOTION_COLORS.get(emotion)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self._label = QLabel(emotion)
        self._label.setFixedWidth(64)
        self._label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1A1A2E;
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 5px;
            }}
        """)

        self._pct = QLabel("0%")
        self._pct.setFixedWidth(36)
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._pct.setStyleSheet("color: #888; font-size: 11px;")

        layout.addWidget(self._label)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._pct)

    def set_value(self, fraction: float) -> None:
        """Update bar to *fraction* in [0, 1]."""
        pct = int(fraction * 100)
        self._bar.setValue(pct)
        self._pct.setText(f"{pct}%")


# ── StatsPanel ────────────────────────────────────────────────────────────────

class StatsPanel(QFrame):
    """
    Right-side statistics panel.

    Displays emotion distribution bars, session counters, and
    aggregate metrics that update in real time.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #0D0D1A;
                border: 1px solid #1E1E2E;
                border-radius: 8px;
            }
        """)
        self.setFixedWidth(260)

        self._session_start: float = time.monotonic()
        self._total_detections: int = 0
        self._confidence_sum: float = 0.0
        self._emotion_counts: Dict[str, int] = {e: 0 for e in EMOTION_LABELS}

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_session_time)
        self._timer.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Title
        title = QLabel("EMOTION ANALYTICS")
        title.setStyleSheet(
            "color: #6060FF; font-size: 11px; font-weight: 700; letter-spacing: 2px;"
        )
        root.addWidget(title)

        root.addWidget(self._divider())

        # Emotion bars
        self._bars: Dict[str, EmotionBar] = {}
        for emotion in EMOTION_LABELS:
            bar = EmotionBar(emotion, self)
            self._bars[emotion] = bar
            root.addWidget(bar)

        root.addWidget(self._divider())

        # Counters grid
        grid = QGridLayout()
        grid.setSpacing(6)

        self._lbl_detections = self._stat_value("0")
        self._lbl_session = self._stat_value("00:00")
        self._lbl_avg_conf = self._stat_value("—")

        grid.addWidget(self._stat_label("Total Detections"), 0, 0)
        grid.addWidget(self._lbl_detections, 0, 1)
        grid.addWidget(self._stat_label("Session Time"), 1, 0)
        grid.addWidget(self._lbl_session, 1, 1)
        grid.addWidget(self._stat_label("Avg Confidence"), 2, 0)
        grid.addWidget(self._lbl_avg_conf, 2, 1)

        root.addLayout(grid)
        root.addStretch()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_prediction(self, emotion: str, confidence: float) -> None:
        """Update internal tallies with a new prediction result."""
        self._total_detections += 1
        self._confidence_sum += confidence
        self._emotion_counts[emotion] = self._emotion_counts.get(emotion, 0) + 1

        total = sum(self._emotion_counts.values()) or 1
        for emo, bar in self._bars.items():
            bar.set_value(self._emotion_counts.get(emo, 0) / total)

        avg = self._confidence_sum / self._total_detections
        self._lbl_detections.setText(str(self._total_detections))
        self._lbl_avg_conf.setText(f"{avg * 100:.1f}%")

    def reset_stats(self) -> None:
        """Reset all counters (called on camera restart)."""
        self._session_start = time.monotonic()
        self._total_detections = 0
        self._confidence_sum = 0.0
        self._emotion_counts = {e: 0 for e in EMOTION_LABELS}
        for bar in self._bars.values():
            bar.set_value(0.0)
        self._lbl_detections.setText("0")
        self._lbl_avg_conf.setText("—")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_session_time(self) -> None:
        elapsed = int(time.monotonic() - self._session_start)
        mm, ss = divmod(elapsed, 60)
        hh, mm = divmod(mm, 60)
        if hh:
            self._lbl_session.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")
        else:
            self._lbl_session.setText(f"{mm:02d}:{ss:02d}")

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background: #1E1E2E; max-height: 1px;")
        return line

    @staticmethod
    def _stat_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #606070; font-size: 11px;")
        return lbl

    @staticmethod
    def _stat_value(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("color: #C0C0D0; font-size: 12px; font-weight: 600;")
        return lbl


# ── StatusBar ─────────────────────────────────────────────────────────────────

class StatusBar(QFrame):
    """Bottom status strip showing FPS, camera state, and detection state."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #080810;
                border-top: 1px solid #1E1E2E;
            }
        """)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(20)

        self._fps_lbl = self._make_label("FPS: —")
        self._cam_lbl = self._make_label("● Camera: Stopped")
        self._det_lbl = self._make_label("Detections: —")
        self._model_lbl = self._make_label("Model: Loading…")

        layout.addWidget(self._fps_lbl)
        layout.addWidget(self._cam_lbl)
        layout.addWidget(self._det_lbl)
        layout.addStretch()
        layout.addWidget(self._model_lbl)

    def set_fps(self, fps: float) -> None:
        self._fps_lbl.setText(f"FPS: {fps:.1f}")

    def set_camera_state(self, running: bool) -> None:
        if running:
            self._cam_lbl.setText("● Camera: Running")
            self._cam_lbl.setStyleSheet("color: #00DC64; font-size: 11px;")
        else:
            self._cam_lbl.setText("● Camera: Stopped")
            self._cam_lbl.setStyleSheet("color: #FF4040; font-size: 11px;")

    def set_detection_count(self, count: int) -> None:
        self._det_lbl.setText(f"Faces: {count}")

    def set_model_state(self, ready: bool) -> None:
        if ready:
            self._model_lbl.setText("Model: Ready ✓")
            self._model_lbl.setStyleSheet("color: #00DC64; font-size: 11px;")
        else:
            self._model_lbl.setText("Model: Error ✗")
            self._model_lbl.setStyleSheet("color: #FF4040; font-size: 11px;")

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #606070; font-size: 11px;")
        return lbl
