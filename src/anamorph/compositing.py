"""Frame rendering independent of video codecs and face inference."""

from __future__ import annotations

import math

import numpy as np

from .effects import Effect
from .masks import FaceMask
from .types import EffectContext, FrameAnalysis, Image, validate_image


def alpha_composite(original: Image, transformed: Image, alpha: np.ndarray) -> Image:
    """Blend in encoded RGB space with rounding, leaving alpha-zero pixels exact."""
    validate_image(original)
    validate_image(transformed)
    if transformed.shape != original.shape or alpha.shape != original.shape[:2]:
        raise ValueError("Image and alpha shapes must agree")
    if not np.issubdtype(alpha.dtype, np.floating) or not np.all(np.isfinite(alpha)):
        raise ValueError("Alpha must contain finite floating-point values")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("Alpha must be in [0, 1]")
    weight = alpha[..., None]
    return np.rint(original * (1 - weight) + transformed * weight).clip(0, 255).astype(np.uint8)


class FrameRenderer:
    """One sequential render session, with deterministic overlap by track ID.

    Every effect reads a crop of the original frame. In overlapping masks,
    higher track IDs are alpha-composited last. Input frames remain unchanged.
    Construct a fresh renderer for each sequence. Frame indices must increase.
    """

    def __init__(self, effect: Effect, mask: FaceMask | None = None) -> None:
        self._session = effect.create_session()
        self.mask = mask if mask is not None else FaceMask()
        self._last_index = -1

    def render(self, image: Image, analysis: FrameAnalysis, *, timestamp: float) -> Image:
        validate_image(image)
        if analysis.index <= self._last_index:
            raise ValueError("Frame indices must strictly increase in a render session")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        output = image.copy()
        height, width = image.shape[:2]
        active = set()
        for face in sorted(analysis.faces, key=lambda face: face.track_id):
            region = self.mask.build(face.box, width, height)
            if region is None:
                continue
            x1, y1, x2, y2 = region.bounds
            crop = image[y1:y2, x1:x2].copy()
            context = EffectContext(analysis.index, timestamp, face, region.bounds)
            transformed = self._session.apply(crop, context)
            validate_image(transformed)
            if transformed.shape != crop.shape:
                raise ValueError("Effects must preserve the crop shape")
            output[y1:y2, x1:x2] = alpha_composite(output[y1:y2, x1:x2], transformed, region.alpha)
            active.add(face.track_id)
        self._session.end_frame(active)
        self._last_index = analysis.index
        return output
