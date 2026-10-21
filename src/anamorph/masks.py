"""Analytic masks sampled at pixel centers, with compact inward feathering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .types import Alpha, Box


@dataclass(frozen=True)
class MaskRegion:
    bounds: tuple[int, int, int, int]
    alpha: Alpha


@dataclass(frozen=True)
class FaceMask:
    """Scale and shift a detector box before rasterization.

    offset_y is a fraction of the original box height. feather is an inward
    fraction of the shape radius, in [0, 1]. Zero gives a hard boundary.
    """

    shape: Literal["ellipse", "rectangle"] = "ellipse"
    scale_x: float = 1.25
    scale_y: float = 1.5
    offset_y: float = -0.05
    feather: float = 0.15

    def __post_init__(self) -> None:
        if self.shape not in ("ellipse", "rectangle"):
            raise ValueError("shape must be ellipse or rectangle")
        if not all(
            math.isfinite(v) for v in (self.scale_x, self.scale_y, self.offset_y, self.feather)
        ):
            raise ValueError("Mask parameters must be finite")
        if self.scale_x <= 0 or self.scale_y <= 0 or not 0 <= self.feather <= 1:
            raise ValueError("Mask scales must be positive and feather must be in [0, 1]")

    def build(self, box: Box, width: int, height: int) -> MaskRegion | None:
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2 + self.offset_y * box.height
        rx, ry = box.width * self.scale_x / 2, box.height * self.scale_y / 2
        bounds = Box(cx - rx, cy - ry, cx + rx, cy + ry).bounds(width, height)
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds
        # Preserve the original geometry when a shape extends beyond the frame.
        x = (np.arange(x1, x2, dtype=np.float64) + 0.5 - cx) / rx
        y = (np.arange(y1, y2, dtype=np.float64) + 0.5 - cy) / ry
        radius = (
            np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
            if self.shape == "ellipse"
            else np.maximum(np.abs(y[:, None]), np.abs(x[None, :]))
        )
        alpha = (
            (radius <= 1).astype(np.float32)
            if self.feather == 0
            else np.clip((1 - radius) / self.feather, 0, 1).astype(np.float32)
        )
        return MaskRegion(bounds, alpha)
