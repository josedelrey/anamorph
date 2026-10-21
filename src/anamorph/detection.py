"""Replaceable face detection, with a bundled, CPU-only baseline."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .types import Box, Detection, Image, Landmarks, validate_image

YUNET_LANDMARK_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "right_mouth_corner",
    "left_mouth_corner",
)


class FaceDetector(Protocol):
    """Return full-frame boxes from RGB images without mutating the input.

    metadata() must return JSON-serializable settings and model provenance.
    A detector instance is used serially and must be reusable across videos.
    """

    def detect(self, image: Image) -> list[Detection]: ...

    def metadata(self) -> dict: ...


class HaarFaceDetector:
    """OpenCV's frontal-face cascade. No calibrated confidence or landmarks."""

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 24,
        cascade_path: str | Path | None = None,
    ) -> None:
        if not math.isfinite(scale_factor) or scale_factor <= 1:
            raise ValueError("scale_factor must be finite and greater than 1")
        if type(min_neighbors) is not int or min_neighbors < 0:
            raise ValueError("min_neighbors must be a nonnegative integer")
        if type(min_size) is not int or min_size < 1:
            raise ValueError("min_size must be a positive integer")
        path = (
            Path(cascade_path)
            if cascade_path is not None
            else Path(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise ValueError(f"Unable to load cascade: {path}")
        self._model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def metadata(self) -> dict:
        return {
            "name": type(self).__name__,
            "model_sha256": self._model_sha256,
            "scale_factor": self.scale_factor,
            "min_neighbors": self.min_neighbors,
            "min_size": self.min_size,
        }

    def detect(self, image: Image) -> list[Detection]:
        validate_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return [
            Detection(Box(float(x), float(y), float(x + w), float(y + h)))
            for x, y, w, h in sorted(boxes, key=lambda b: tuple(b))
        ]


class YuNetFaceDetector:
    """CPU YuNet inference with five named landmarks and original-frame boxes.

    model_path must point to a local YuNet ONNX file. No implicit downloads.
    max_size caps the longest inference dimension without upscaling. Use None
    for native resolution. Scores are model outputs, not calibrated certainty.
    Instances are reusable serially and are not safe for concurrent inference.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_size: int | None = 640,
    ) -> None:
        for name, value in (("score_threshold", score_threshold), ("nms_threshold", nms_threshold)):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if max_size is not None and (type(max_size) is not int or max_size < 1):
            raise ValueError("max_size must be a positive integer or None")
        path = Path(model_path)
        model = path.read_bytes()
        if model.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("Model is a Git LFS pointer. Download the actual YuNet ONNX weights")
        self._model_sha256 = hashlib.sha256(model).hexdigest()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path),
                "",
                (320, 320),
                score_threshold,
                nms_threshold,
                top_k,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except cv2.error as exc:
            raise ValueError(f"Cannot load YuNet ONNX model: {path}") from exc
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_size = max_size

    def metadata(self) -> dict:
        return {
            "name": type(self).__name__,
            "model_sha256": self._model_sha256,
            "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
            "top_k": self.top_k,
            "max_size": self.max_size,
            "backend": "opencv",
            "target": "cpu",
            "input_color": "BGR",
            "resize": "aspect_preserving_no_upscale",
            "landmark_names": list(YUNET_LANDMARK_NAMES),
        }

    def detect(self, image: Image) -> list[Detection]:
        validate_image(image)
        height, width = image.shape[:2]
        scale = min(1.0, self.max_size / max(width, height)) if self.max_size else 1.0
        size = max(1, round(width * scale)), max(1, round(height * scale))
        resized = (
            cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if size != (width, height)
            else image
        )
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize(size)
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []
        if faces.ndim != 2 or faces.shape[1] != 15 or not np.all(np.isfinite(faces)):
            raise RuntimeError("YuNet returned malformed detection rows")
        sx, sy = width / size[0], height / size[1]
        detections = []
        for row in faces:
            x, y, w, h = (float(value) for value in row[:4])
            points = tuple((float(row[i]) * sx, float(row[i + 1]) * sy) for i in range(4, 14, 2))
            detections.append(
                Detection(
                    Box(x * sx, y * sy, (x + w) * sx, (y + h) * sy),
                    float(row[14]),
                    Landmarks(YUNET_LANDMARK_NAMES, points),
                )
            )
        return sorted(detections, key=lambda detection: detection.box.as_tuple())
