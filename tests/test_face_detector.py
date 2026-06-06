"""
Tests for the face detection module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from detection.face_detector import FaceDetector, FaceDetection, crop_face
from utils.config import DetectionConfig


class TestCropFace:
    """Unit tests for the crop_face helper function."""

    def test_valid_crop(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = crop_face(frame, [10, 10, 100, 100])
        assert crop is not None
        assert crop.shape == (100, 100, 3)

    def test_zero_size_returns_none(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert crop_face(frame, [0, 0, 0, 0]) is None

    def test_out_of_bounds_returns_none(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # x + w exceeds frame width
        assert crop_face(frame, [600, 0, 200, 100]) is None

    def test_negative_origin_returns_none(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert crop_face(frame, [-10, 10, 50, 50]) is None

    def test_full_frame_crop(self) -> None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = crop_face(frame, [0, 0, 640, 480])
        assert crop is not None
        np.testing.assert_array_equal(crop, frame)


class TestFaceDetection:
    """Unit tests for the FaceDetection dataclass."""

    def test_default_face_id(self) -> None:
        det = FaceDetection(bbox=[0, 0, 100, 100], confidence=0.9)
        assert det.face_id == 0

    def test_custom_face_id(self) -> None:
        det = FaceDetection(bbox=[0, 0, 100, 100], confidence=0.9, face_id=3)
        assert det.face_id == 3

    def test_confidence_range(self) -> None:
        for conf in [0.0, 0.5, 1.0]:
            det = FaceDetection(bbox=[0, 0, 50, 50], confidence=conf)
            assert 0.0 <= det.confidence <= 1.0


class TestFaceDetectorConfig:
    """Unit tests for detector configuration handling."""

    def test_custom_confidence_threshold(self) -> None:
        cfg = DetectionConfig(min_detection_confidence=0.8)
        assert cfg.min_detection_confidence == 0.8

    def test_custom_min_face_size(self) -> None:
        cfg = DetectionConfig(min_face_size=60)
        assert cfg.min_face_size == 60

    def test_padding_is_non_negative(self) -> None:
        cfg = DetectionConfig(padding_ratio=0.2)
        assert cfg.padding_ratio >= 0
