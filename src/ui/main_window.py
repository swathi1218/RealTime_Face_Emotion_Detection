"""
EmotionLens Main Window.

Orchestrates the full pipeline:
  WebcamThread → FaceDetector → EmotionRecognizer → SmootherPool → Drawing → GUI
All heavy work runs in InferenceWorker (QThread); the GUI thread only renders.
"""

import logging
import sys
from typing import Dict, List, Optional
from unittest import result

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QFont, QIcon, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from camera.webcam import WebcamThread
from detection.face_detector import FaceDetection, FaceDetector, crop_face
from emotion.emotion_recognizer import EmotionRecognizer, InferenceThread
from emotion.smoothing import SmootherPool
from logs.emotion_logger import EmotionLogger
from ui.widgets import StatusBar, StatsPanel, VideoDisplay
from utils.config import (
    CAMERA_CONFIG,
    DETECTION_CONFIG,
    EMOTION_CONFIG,
    LOGGING_CONFIG,
    SMOOTHING_CONFIG,
)
from utils.drawing import draw_face_annotation, draw_fps, draw_no_face_message

logger = logging.getLogger(__name__)


# ── Inference Worker ──────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    """
    Background thread that handles face detection and emotion recognition.

    Receives raw frames from the camera thread, performs the full inference
    pipeline, and emits annotated frames back to the main window.

    Signals:
        result_ready (np.ndarray, list, float):
            Emitted with (annotated_frame, list_of_detections, fps).
        model_ready (bool):
            Emitted once the emotion model finishes loading.
    """

    result_ready = pyqtSignal(np.ndarray, list, dict, float)
    model_ready = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._detector = FaceDetector(DETECTION_CONFIG)
        self._recognizer = EmotionRecognizer(EMOTION_CONFIG)
        self._smoother_pool = SmootherPool(SMOOTHING_CONFIG)
        self._logger = EmotionLogger(LOGGING_CONFIG)
        self._frame_gate = InferenceThread(EMOTION_CONFIG.frame_skip)

        # Latest frame buffer (shared with camera thread via assignment)
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_fps: float = 0.0
        self._running: bool = False

        # Cache of last known predictions for frames that skip inference
        self._last_detections: List[FaceDetection] = []
        self._last_results: Dict[int, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_frame(self, frame: np.ndarray, fps: float) -> None:
        """Accept a new frame from the camera thread."""
        self._pending_frame = frame
        self._pending_fps = fps

    def stop(self) -> None:
        """Stop the inference loop and flush the logger."""
        self._running = False
        self._logger.close()
        self.wait(msecs=3000)

    # ── QThread override ─────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self.model_ready.emit(self._recognizer.is_ready)

        while self._running:
            frame = self._pending_frame
            if frame is None:
                self.msleep(5)
                continue

            self._pending_frame = None
            fps = self._pending_fps
            annotated = self._process(frame, fps)
            self.result_ready.emit(annotated, self._last_detections, self._last_results, fps)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Run the full per-frame pipeline and return an annotated copy."""
        output = frame.copy()
        do_infer = self._frame_gate.should_infer()

        if do_infer:
            detections = self._detector.detect(frame)
            self._last_detections = detections
            active_ids = [d.face_id for d in detections]
            self._smoother_pool.tick_idle(active_ids)

            new_results: Dict[int, dict] = {}
            for det in detections:
                crop = crop_face(frame, det.bbox)
                if crop is None:
                    continue

                raw = self._recognizer.predict(crop)
                if raw is None:
                    continue

                smoother = self._smoother_pool.get_or_create(det.face_id)
                smoothed = smoother.update(raw.emotion, raw.confidence, raw.scores)
                new_results[det.face_id] = {
                    "emotion": smoothed.emotion,
                    "confidence": smoothed.confidence,
                    "scores": smoothed.scores,
                    "bbox": det.bbox,
                    "face_id": det.face_id,
                }
                self._logger.log(det.face_id, smoothed.emotion, smoothed.confidence)

            self._last_results = new_results

        # Draw last known results (cached between inference frames)
        for fid, result in self._last_results.items():
            draw_face_annotation(
                output,
                result["bbox"],
                result["emotion"],
                result["confidence"],
                result["face_id"],
            )

        if not self._last_results:
            draw_no_face_message(output)

        draw_fps(output, fps)
        return output


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    EmotionLens application main window.

    Layout
    ------
    Toolbar → [VideoDisplay | StatsPanel] → StatusBar
    """

    def __init__(self) -> None:
        super().__init__()
        self._camera_thread: Optional[WebcamThread] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._camera_running: bool = False

        self._build_ui()
        self._connect_toolbar_actions()
        self._start_pipeline()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("EmotionLens  —  Real-Time Facial Emotion Recognition")
        self.setMinimumSize(900, 600)
        self.resize(1100, 680)
        self.setStyleSheet("""
            QMainWindow {
                background: #050508;
            }
            QToolBar {
                background: #0A0A12;
                border-bottom: 1px solid #1A1A28;
                spacing: 6px;
                padding: 4px 8px;
            }
            QToolBar QLabel {
                color: #6060A0;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 3px;
            }
            QPushButton {
                background: #1A1A2E;
                color: #A0A0C0;
                border: 1px solid #2E2E50;
                border-radius: 5px;
                padding: 5px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #2A2A4E;
                color: #E0E0FF;
            }
            QPushButton:pressed {
                background: #3030A0;
            }
            QPushButton#stop_btn {
                color: #FF6060;
                border-color: #5E2020;
            }
            QPushButton#stop_btn:hover {
                background: #3A1010;
                color: #FF8080;
            }
        """)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        brand = QLabel("EmotionLens")
        brand.setStyleSheet(
            "color: #7070FF; font-size: 14px; font-weight: 800; letter-spacing: 2px;"
        )
        toolbar.addWidget(brand)
        toolbar.addSeparator()

        self._btn_start = QPushButton("▶  Start")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("stop_btn")
        self._btn_reset = QPushButton("↺  Reset Stats")
        toolbar.addWidget(self._btn_start)
        toolbar.addWidget(self._btn_stop)
        toolbar.addSeparator()
        toolbar.addWidget(self._btn_reset)

        # ── Central area ──────────────────────────────────────────────────────
        central = QWidget()
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(8, 8, 8, 0)
        h_layout.setSpacing(8)

        self._video_display = VideoDisplay()
        h_layout.addWidget(self._video_display, 1)

        self._stats_panel = StatsPanel()
        h_layout.addWidget(self._stats_panel, 0)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = StatusBar()
        main_v = QVBoxLayout()
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)
        main_v.addWidget(central, 1)
        main_v.addWidget(self._status_bar, 0)

        wrapper = QWidget()
        wrapper.setLayout(main_v)
        self.setCentralWidget(wrapper)

    def _connect_toolbar_actions(self) -> None:
        self._btn_start.clicked.connect(self._start_pipeline)
        self._btn_stop.clicked.connect(self._stop_pipeline)
        self._btn_reset.clicked.connect(self._reset_stats)

    # ── Pipeline Management ───────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        """Start the camera and inference threads."""
        if self._camera_running:
            return

        self._inference_worker = InferenceWorker()
        self._inference_worker.result_ready.connect(self._on_result_ready)
        self._inference_worker.model_ready.connect(self._on_model_ready)
        self._inference_worker.start()

        self._camera_thread = WebcamThread(CAMERA_CONFIG)
        self._camera_thread.frame_ready.connect(self._on_frame_ready)
        self._camera_thread.camera_error.connect(self._on_camera_error)
        self._camera_thread.camera_opened.connect(self._on_camera_opened)
        self._camera_thread.start()

    def _stop_pipeline(self) -> None:
        """Stop all worker threads gracefully."""
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None

        if self._inference_worker is not None:
            self._inference_worker.stop()
            self._inference_worker = None

        self._camera_running = False
        self._status_bar.set_camera_state(False)
        self._status_bar.set_fps(0.0)
        self._status_bar.set_detection_count(0)

    def _reset_stats(self) -> None:
        self._stats_panel.reset_stats()

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(np.ndarray, float)
    def _on_frame_ready(self, frame: np.ndarray, fps: float) -> None:
        """Forward raw frame to the inference worker."""
        if self._inference_worker is not None:
            self._inference_worker.submit_frame(frame, fps)

    @pyqtSlot(np.ndarray, list, dict, float)
    def _on_result_ready(
        self, annotated: np.ndarray, detections: list, results: dict, fps: float
    ) -> None:
        """Receive annotated frame and update the GUI."""
        # Convert BGR → QPixmap
        h, w, ch = annotated.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._video_display.update_frame(QPixmap.fromImage(qimg))

        self._status_bar.set_fps(fps)
        self._status_bar.set_detection_count(len(detections))
        
        for fid, result in results.items():
            self._stats_panel.record_prediction(result["emotion"], result["confidence"])
    @pyqtSlot(bool)
    def _on_model_ready(self, ready: bool) -> None:
        self._status_bar.set_model_state(ready)
        if not ready:
            QMessageBox.warning(
                self,
                "Model Load Failed",
                "The HSEmotion model could not be loaded.\n\n"
                "Please ensure 'hsemotion' is installed:\n"
                "  pip install hsemotion",
            )

    @pyqtSlot()
    def _on_camera_opened(self) -> None:
        self._camera_running = True
        self._status_bar.set_camera_state(True)

    @pyqtSlot(str)
    def _on_camera_error(self, message: str) -> None:
        self._camera_running = False
        self._status_bar.set_camera_state(False)
        QMessageBox.critical(self, "Camera Error", message)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Ensure threads are stopped before the window closes."""
        self._stop_pipeline()
        event.accept()
