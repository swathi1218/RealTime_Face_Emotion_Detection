"""
Webcam capture manager.

Runs on a dedicated QThread to avoid blocking the GUI event loop.
Emits frames via Qt signals for thread-safe delivery to the main window.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from utils.config import CAMERA_CONFIG, CameraConfig
from utils.fps import FPSCounter

logger = logging.getLogger(__name__)


class WebcamThread(QThread):
    """
    Background thread that continuously captures frames from a webcam.

    Signals:
        frame_ready (np.ndarray, float):
            Emitted for each successfully captured frame.
            Payload is (BGR frame, current FPS).
        camera_error (str):
            Emitted when the camera cannot be opened or a read fails.
        camera_opened ():
            Emitted once the camera is successfully opened.
    """

    frame_ready: pyqtSignal = pyqtSignal(np.ndarray, float)
    camera_error: pyqtSignal = pyqtSignal(str)
    camera_opened: pyqtSignal = pyqtSignal()

    def __init__(
        self,
        cfg: CameraConfig = CAMERA_CONFIG,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._running: bool = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps_counter = FPSCounter(window_size=30)

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the capture loop to exit and wait for the thread to finish."""
        self._running = False
        self.wait(3000)

    # ── QThread override ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Main capture loop; runs in the worker thread."""
        self._running = True
        self._cap = self._open_camera()

        if self._cap is None:
            return  # Error was already emitted inside _open_camera

        self.camera_opened.emit()
        logger.info("Camera opened: %dx%d", self._cfg.width, self._cfg.height)

        consecutive_failures = 0
        max_failures = 10

        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning("Frame read failed (%d/%d)", consecutive_failures, max_failures)
                if consecutive_failures >= max_failures:
                    self.camera_error.emit(
                        "Lost connection to camera after repeated read failures."
                    )
                    break
                continue

            consecutive_failures = 0
            frame = self._preprocess(frame)
            fps = self._fps_counter.tick()
            self.frame_ready.emit(frame, fps)

        self._release_camera()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Attempt to open the webcam; emit an error signal on failure."""
        cap = cv2.VideoCapture(self._cfg.device_index)

        if not cap.isOpened():
            msg = f"Cannot open camera at device index {self._cfg.device_index}."
            logger.error(msg)
            self.camera_error.emit(msg)
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cfg.buffer_size)

        return cap

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution."""
        h, w = frame.shape[:2]
        if (w, h) != (self._cfg.width, self._cfg.height):
            frame = cv2.resize(frame, (self._cfg.width, self._cfg.height))
        return frame

    def _release_camera(self) -> None:
        """Release the OpenCV capture device."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released.")
