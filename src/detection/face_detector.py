"""
Face detection module using MediaPipe Face Detection.

Wraps the MediaPipe API and provides a clean, typed interface
that returns structured detection results.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from utils.config import DETECTION_CONFIG, DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """
    A single face detection result.

    Attributes:
        bbox:        [x, y, w, h] in absolute pixel coordinates.
        confidence:  Detection confidence in range [0, 1].
        face_id:     Index assigned during this frame (0-based).
    """
    bbox: List[int]
    confidence: float
    face_id: int = 0


class FaceDetector:
    """
    MediaPipe-based face detector.

    Detects multiple faces in a single BGR frame and returns structured
    :class:`FaceDetection` objects with validated bounding boxes.

    Args:
        cfg: Detection configuration.
    """

    def __init__(self, cfg: DetectionConfig = DETECTION_CONFIG) -> None:
        self._cfg = cfg
        try:
            import mediapipe.python.solutions.face_detection as _fd_mod
            self._mp_face_detection = _fd_mod
        except ImportError:
            self._mp_face_detection = mp.solutions.face_detection
        self._detector: Optional[object] = None
        self._initialize()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Run face detection on a single BGR frame.

        Args:
            frame: BGR image (H × W × 3, uint8).

        Returns:
            List of :class:`FaceDetection` objects, sorted by x-coordinate
            (left-to-right) for consistent face ID assignment across frames.
        """
        if self._detector is None:
            logger.error("FaceDetector is not initialised.")
            return []

        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        detections: List[FaceDetection] = []
        for idx, detection in enumerate(results.detections):
            score = detection.score[0] if detection.score else 0.0
            if score < self._cfg.min_detection_confidence:
                continue

            bbox = self._extract_bbox(detection.location_data.relative_bounding_box, w, h)
            if bbox is None:
                continue

            detections.append(
                FaceDetection(bbox=bbox, confidence=round(float(score), 4), face_id=idx)
            )

        # Re-assign IDs left-to-right after filtering
        detections.sort(key=lambda d: d.bbox[0])
        for i, det in enumerate(detections):
            det.face_id = i + 1

        return detections

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Instantiate the MediaPipe model."""
        try:
            self._detector = self._mp_face_detection.FaceDetection(
                model_selection=self._cfg.model_selection,
                min_detection_confidence=self._cfg.min_detection_confidence,
            )
            logger.info("MediaPipe FaceDetection initialised (model=%d).", self._cfg.model_selection)
        except Exception as exc:
            logger.exception("Failed to initialise MediaPipe FaceDetection: %s", exc)
            self._detector = None

    def _extract_bbox(
        self,
        rel_bb,
        frame_w: int,
        frame_h: int,
    ) -> Optional[List[int]]:
        """
        Convert a relative MediaPipe bounding box to absolute [x, y, w, h].

        Applies padding and validates that the crop lies within the frame.
        Returns ``None`` if the face crop would be too small.
        """
        pad = self._cfg.padding_ratio

        x = int((rel_bb.xmin - pad) * frame_w)
        y = int((rel_bb.ymin - pad) * frame_h)
        w = int((rel_bb.width + 2 * pad) * frame_w)
        h = int((rel_bb.height + 2 * pad) * frame_h)

        # Clamp to frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w < self._cfg.min_face_size or h < self._cfg.min_face_size:
            return None

        return [x, y, w, h]


def crop_face(frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    """
    Crop a face region from *frame* using the given bounding box.

    Args:
        frame: BGR image.
        bbox:  [x, y, w, h] in pixel coordinates.

    Returns:
        Cropped BGR image, or ``None`` if the crop is invalid.
    """
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]

    if x < 0 or y < 0 or x + w > w_frame or y + h > h_frame or w <= 0 or h <= 0:
        return None

    crop = frame[y: y + h, x: x + w]
    if crop.size == 0:
        return None

    return crop
