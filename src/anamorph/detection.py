"""Replaceable face detection, with a bundled, CPU-only baseline."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

import cv2

from .types import Box, Detection, Image, validate_image


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
