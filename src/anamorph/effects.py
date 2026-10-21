"""Image operators and isolated streaming sessions for causal temporal effects."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from .types import EffectContext, Image, validate_image


class EffectSession(ABC):
    """Mutable state owned by one render, never shared between renders.

    apply receives a private RGB crop and must return the same shape and dtype.
    It may mutate its input. Coordinates and track identity come from context.
    end_frame is called once per frame, including frames with no visible faces.
    """

    @abstractmethod
    def apply(self, image: Image, context: EffectContext) -> Image: ...

    def end_frame(self, active_track_ids: set[int]) -> None:
        """Release state for tracks absent from this frame when appropriate."""
        return None


class Effect(ABC):
    @abstractmethod
    def create_session(self) -> EffectSession:
        """Return a fresh session for a new video or in-memory sequence."""


class SpatialEffect(Effect):
    """Stateless image operator, also usable directly on full RGB arrays."""

    @abstractmethod
    def apply(self, image: Image) -> Image: ...

    def create_session(self) -> EffectSession:
        return _SpatialSession(self)


@dataclass
class _SpatialSession(EffectSession):
    effect: SpatialEffect

    def apply(self, image: Image, context: EffectContext) -> Image:
        return self.effect.apply(image)


@dataclass(frozen=True)
class GaussianBlur(SpatialEffect):
    """Gaussian sigma in crop pixels, with reflected image boundary extension."""

    sigma: float = 12.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive and finite")

    def apply(self, image: Image) -> Image:
        validate_image(image)
        return cv2.GaussianBlur(image, (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT_101)


@dataclass(frozen=True)
class Pixelate(SpatialEffect):
    """Area downsample to at most `cells` on each axis, then nearest upsample."""

    cells: int = 8

    def __post_init__(self) -> None:
        if type(self.cells) is not int or self.cells < 1:
            raise ValueError("cells must be a positive integer")

    def apply(self, image: Image) -> Image:
        validate_image(image)
        height, width = image.shape[:2]
        small = cv2.resize(
            image, (min(self.cells, width), min(self.cells, height)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


@dataclass(frozen=True, init=False)
class Compose(Effect):
    """Apply effects in the given order and composite the result once.

    Order generally matters. Each stage must obey the same RGB crop contract.
    """

    effects: tuple[Effect, ...]

    def __init__(self, *effects: Effect) -> None:
        if not effects:
            raise ValueError("Compose requires at least one effect")
        object.__setattr__(self, "effects", effects)

    def create_session(self) -> EffectSession:
        return _ComposeSession(tuple(effect.create_session() for effect in self.effects))


@dataclass
class _ComposeSession(EffectSession):
    sessions: tuple[EffectSession, ...]

    def apply(self, image: Image, context: EffectContext) -> Image:
        shape = image.shape
        for session in self.sessions:
            image = session.apply(image, context)
            validate_image(image)
            if image.shape != shape:
                raise ValueError("Every composed effect must preserve the crop shape")
        return image

    def end_frame(self, active_track_ids: set[int]) -> None:
        for session in self.sessions:
            session.end_frame(active_track_ids)


@dataclass(frozen=True)
class TemporalBlend(Effect):
    """Causal exponential blend in crop coordinates, intended as a ghost effect.

    current_weight weights the current crop. Previous crops are resized to the
    current size, without motion compensation. Missing sequence indices reset
    state. Compose after an obscuring operator to blend obscured imagery.
    """

    current_weight: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.current_weight) or not 0 < self.current_weight <= 1:
            raise ValueError("current_weight must be in (0, 1]")

    def create_session(self) -> EffectSession:
        return _TemporalBlendSession(self.current_weight)


class _TemporalBlendSession(EffectSession):
    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.previous: dict[int, tuple[int, np.ndarray]] = {}

    def apply(self, image: Image, context: EffectContext) -> Image:
        validate_image(image)
        current = image.astype(np.float32)
        previous = self.previous.get(context.face.track_id)
        if previous is not None and previous[0] == context.frame_index - 1:
            history = previous[1]
            if history.shape != image.shape:
                history = cv2.resize(history, (image.shape[1], image.shape[0]))
            current = self.weight * current + (1 - self.weight) * history
        self.previous[context.face.track_id] = (context.frame_index, current.copy())
        return np.rint(current).clip(0, 255).astype(np.uint8)

    def end_frame(self, active_track_ids: set[int]) -> None:
        for track_id in self.previous.keys() - active_track_ids:
            del self.previous[track_id]
