"""Shared coordinate and image contracts. Public images are always uint8 RGB."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Alpha = NDArray[np.float32]


def validate_image(image: Image) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Expected a nonempty H x W x 3 uint8 RGB image")


@dataclass(frozen=True)
class Box:
    """Continuous full-frame pixel edges [x1, x2) x [y1, y2), x right, y down."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in self.as_tuple()):
            raise ValueError("Box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: Box) -> float:
        intersection = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0.0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )
        return intersection / (self.width * self.height + other.width * other.height - intersection)

    def bounds(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """Enclosing integer crop, clipped to the image, or None when outside."""
        x1, y1 = max(0, math.floor(self.x1)), max(0, math.floor(self.y1))
        x2, y2 = min(width, math.ceil(self.x2)), min(height, math.ceil(self.y2))
        return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence must be None or a finite number in [0, 1]")


@dataclass(frozen=True)
class Face:
    """Tracked box, including an explicit age for held, unobserved detections."""

    track_id: int
    box: Box
    observed: bool = True
    missing_frames: int = 0
    confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        if type(self.missing_frames) is not int or self.missing_frames < 0:
            raise ValueError("missing_frames must be a nonnegative integer")
        if type(self.observed) is not bool or self.observed != (self.missing_frames == 0):
            raise ValueError("observed must agree with missing_frames == 0")
        Detection(self.box, self.confidence)


@dataclass(frozen=True)
class FrameAnalysis:
    index: int
    faces: tuple[Face, ...] = ()
    # None means inference was skipped or raw observations are unavailable.
    # An empty tuple means inference ran and found no faces.
    detections: tuple[Detection, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Frame index must be a nonnegative integer")
        if len({face.track_id for face in self.faces}) != len(self.faces):
            raise ValueError("A frame cannot contain duplicate track IDs")


@dataclass(frozen=True)
class EffectContext:
    """Context for one crop. crop_bounds are integer full-frame pixel edges."""

    frame_index: int
    timestamp: float
    face: Face
    crop_bounds: tuple[int, int, int, int]
