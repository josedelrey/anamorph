"""Inspectable face geometry and pixel-center mask rasterization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from .types import Alpha, Box, FrameAnalysis, Landmarks

if TYPE_CHECKING:
    from .analysis import VideoAnalysis


class FrameMaskSession(Protocol):
    def build(self, frame: FrameAnalysis, width: int, height: int) -> dict[int, MaskRegion]: ...


class MaskProvider(Protocol):
    """Extension contract for masks with independent analysis or session state."""

    def create_session(self) -> FrameMaskSession: ...

    def validate_analysis(self, analysis: VideoAnalysis) -> None: ...


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


@dataclass(frozen=True)
class FaceMaskGeometry:
    """Full-frame geometry, with clockwise roll in radians in image coordinates.

    radii describe the base opaque core before chin extension. This is a
    geometric estimate of the face, not a segmentation of skin, hair, or ears.
    """

    center: tuple[float, float]
    radii: tuple[float, float]
    angle: float = 0
    landmark_guided: bool = False

    def __post_init__(self) -> None:
        if len(self.center) != 2 or len(self.radii) != 2:
            raise ValueError("Mask center and radii must each have two coordinates")
        if not all(math.isfinite(v) for v in (*self.center, *self.radii, self.angle)):
            raise ValueError("Mask geometry must be finite")
        if min(self.radii) <= 0:
            raise ValueError("Mask radii must be positive")
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "radii", tuple(float(v) for v in self.radii))


@dataclass(frozen=True)
class LandmarkFaceMask:
    """A rounded forehead and tapered jaw, oriented by five named landmarks.

    Defaults target the face, including forehead and chin. Five points cannot
    identify the hairline or ears exactly. scale_x/y multiply estimated face
    dimensions, offset_y is a fraction of box height, and feather extends
    OUTSIDE the opaque core as a fraction of its radius, using smoothstep.
    stabilization is the current observation weight in a box-relative EMA.
    One disables filtering. A MaskSession owns temporal state, never this object.
    chin_extension lengthens the lower radius by a fraction in [0, 1].
    """

    scale_x: float = 1.12
    scale_y: float = 1.10
    offset_y: float = 0
    feather: float = 0.08
    stabilization: float = 0.65
    chin_extension: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.scale_x,
            self.scale_y,
            self.offset_y,
            self.feather,
            self.stabilization,
            self.chin_extension,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Mask parameters must be finite")
        if self.scale_x <= 0 or self.scale_y <= 0 or not 0 <= self.feather <= 1:
            raise ValueError("Mask scales must be positive and feather must be in [0, 1]")
        if not 0 < self.stabilization <= 1:
            raise ValueError("stabilization must be in (0, 1]")
        if not 0 <= self.chin_extension <= 1:
            raise ValueError("chin_extension must be in [0, 1]")

    def geometry(self, box: Box, landmarks: Landmarks | None = None) -> FaceMaskGeometry:
        """Estimate geometry, falling back to an upright box for implausible points.

        The eye line estimates roll. Eye and mouth spacing constrain dimensions
        within conservative box-relative limits. No dense contour layout is
        assumed. Left/right names are subject-anatomical.
        """
        fallback = FaceMaskGeometry(
            ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2 + self.offset_y * box.height),
            (box.width * self.scale_x / 2, box.height * self.scale_y / 2),
        )
        if landmarks is None:
            return fallback
        try:
            right, left, nose, mr, ml = (
                np.asarray(landmarks[name], dtype=np.float64)
                for name in (
                    "right_eye",
                    "left_eye",
                    "nose_tip",
                    "right_mouth_corner",
                    "left_mouth_corner",
                )
            )
        except KeyError:
            return fallback
        points = np.array([right, left, nose, mr, ml])
        relative = (points - (box.x1, box.y1)) / (box.width, box.height)
        eye_vector = left - right
        distance = float(np.linalg.norm(eye_vector))
        angle = math.atan2(eye_vector[1], eye_vector[0])
        if (
            np.any(relative < -0.2)
            or np.any(relative > 1.2)
            or not 0.12 * box.width <= distance <= 0.9 * box.width
            or abs(angle) > math.pi / 3
        ):
            return fallback
        across = eye_vector / distance
        down = np.array([-across[1], across[0]])
        eyes, mouth = (right + left) / 2, (mr + ml) / 2
        eye_mouth = float((mouth - eyes) @ down)
        nose_depth = float((nose - eyes) @ down)
        if not (0.08 * box.height < eye_mouth < 0.65 * box.height and 0 < nose_depth < eye_mouth):
            return fallback
        # Clamp landmark-derived dimensions to avoid catastrophic changes on profiles.
        width = float(np.clip(distance * 2.7, 0.95 * box.width, box.width))
        height = float(np.clip(eye_mouth * 2.8, box.height, 1.15 * box.height))
        box_center = np.array([(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2])
        delta = eyes + down * (0.13 * height) - box_center
        # Looking down compresses eye/mouth spacing and shifts both below the
        # forehead. Keep the box as a coverage anchor rather than shrinking to
        # just the visible features. Bound lateral drift on yawed faces as well.
        center = (
            box_center
            + across * np.clip(delta @ across, -0.04 * box.width, 0.04 * box.width)
            + down
            * (
                np.clip(delta @ down, -0.04 * box.height, 0.04 * box.height)
                + self.offset_y * box.height
            )
        )
        return FaceMaskGeometry(
            tuple(center), (width * self.scale_x / 2, height * self.scale_y / 2), angle, True
        )

    def rasterize(self, geometry: FaceMaskGeometry, width: int, height: int) -> MaskRegion | None:
        """Sample an opaque core and an outward soft boundary at pixel centers."""
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        cx, cy = geometry.center
        rx, ry = geometry.radii
        ry *= 1 + self.chin_extension
        cosine, sine = math.cos(geometry.angle), math.sin(geometry.angle)
        # A rotated enclosing rectangle is conservative for both halves of the face.
        extent_x = (abs(cosine) * rx + abs(sine) * ry) * (1 + self.feather)
        extent_y = (abs(sine) * rx + abs(cosine) * ry) * (1 + self.feather)
        bounds = Box(cx - extent_x, cy - extent_y, cx + extent_x, cy + extent_y).bounds(
            width, height
        )
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds
        x = np.arange(x1, x2, dtype=np.float64)[None, :] + 0.5
        y = np.arange(y1, y2, dtype=np.float64)[:, None] + 0.5
        return MaskRegion(bounds, self.alpha_at(geometry, x, y))

    def alpha_at(self, geometry: FaceMaskGeometry, x: np.ndarray, y: np.ndarray) -> Alpha:
        """Evaluate continuous full-frame coordinates using NumPy broadcasting.

        Rasterization samples (column + 0.5, row + 0.5). For annotation-to-pixel
        comparisons, sample the center of the pixel containing each annotation.
        """
        cx, cy = geometry.center
        rx, ry = geometry.radii
        cosine, sine = math.cos(geometry.angle), math.sin(geometry.angle)
        dx, dy = np.asarray(x, dtype=np.float64) - cx, np.asarray(y, dtype=np.float64) - cy
        x = (cosine * dx + sine * dy) / rx
        y = (-sine * dx + cosine * dy) / ry
        # A broad upper face, with a gently narrowing lower face toward the chin.
        y = np.where(y > 0, y / (1 + self.chin_extension), y)
        taper = 1 - 0.08 * np.clip(y, 0, 1)
        radius = (np.abs(x / taper) ** 2.6 + np.abs(y) ** 2.6) ** (1 / 2.6)
        if self.feather == 0:
            alpha = (radius <= 1).astype(np.float32)
        else:
            t = np.clip((radius - 1) / self.feather, 0, 1)
            alpha = (1 - t * t * (3 - 2 * t)).astype(np.float32)
        return alpha

    def build(
        self, box: Box, width: int, height: int, *, landmarks: Landmarks | None = None
    ) -> MaskRegion | None:
        """Build one independent mask, without temporal filtering."""
        return self.rasterize(self.geometry(box, landmarks), width, height)


class MaskSession:
    """Sequential per-track mask filtering, reusable for rendering and inspection.

    build returns track-ID keyed regions for one FrameAnalysis. Filtering acts
    on box-relative geometry so it adds no translation or scale lag. The final
    alpha is the maximum of current and filtered masks, preserving all current
    coverage. Gaps, absent IDs, implausible landmarks, and large box jumps reset
    history. Held detections remain stale, just like the underlying analysis.
    """

    def __init__(self, mask: FaceMask | LandmarkFaceMask | MaskProvider | None = None) -> None:
        self.mask = mask if mask is not None else LandmarkFaceMask()
        self._delegate = (
            None
            if isinstance(self.mask, (FaceMask, LandmarkFaceMask))
            else self.mask.create_session()
        )
        self._history: dict[int, tuple[Box, np.ndarray]] = {}
        self._last_index = -1
        self._dimensions: tuple[int, int] | None = None

    def build(self, frame: FrameAnalysis, width: int, height: int) -> dict[int, MaskRegion]:
        if frame.index <= self._last_index:
            raise ValueError("Frame indices must strictly increase in a mask session")
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if self._delegate is not None:
            regions = self._delegate.build(frame, width, height)
            self._last_index = frame.index
            return regions
        if frame.index != self._last_index + 1 or self._dimensions != (width, height):
            self._history.clear()
        regions = {}
        history = {}
        for face in frame.faces:
            if isinstance(self.mask, FaceMask):
                region = self.mask.build(face.box, width, height)
            else:
                box = face.box
                geometry = self.mask.geometry(box, face.landmarks)
                region = self.mask.rasterize(geometry, width, height)
                if geometry.landmark_guided:
                    current = np.array(
                        [
                            (geometry.center[0] - box.x1) / box.width,
                            (geometry.center[1] - box.y1) / box.height,
                            geometry.radii[0] / box.width,
                            geometry.radii[1] / box.height,
                            geometry.angle,
                        ]
                    )
                    previous = self._history.get(face.track_id)
                    filtered = current
                    if previous is not None and previous[0].iou(box) >= 0.1:
                        weight = self.mask.stabilization
                        filtered = (
                            weight * current + (1 - weight) * previous[1]
                            if face.observed
                            else previous[1]
                        )
                        stable = FaceMaskGeometry(
                            (box.x1 + filtered[0] * box.width, box.y1 + filtered[1] * box.height),
                            (filtered[2] * box.width, filtered[3] * box.height),
                            filtered[4],
                            True,
                        )
                        region = _union(region, self.mask.rasterize(stable, width, height))
                    history[face.track_id] = (box, filtered)
            if region is not None:
                regions[face.track_id] = region
        self._history = history
        self._last_index = frame.index
        self._dimensions = (width, height)
        return regions


@dataclass(frozen=True)
class PerFaceMask:
    """Select a mask configuration by cached track ID, with a common default.

    Overrides are copied and immutable. Each distinct configuration object owns
    a fresh session. Providers receive the complete original frame, preserving
    analysis/cache binding. Only the selected track's region is returned.
    Sessions may implement build_selected to limit processing after validation,
    as ParsedMaskSession does. Other providers calculate complete frames.
    This replays mask calculations per configuration, never face inference.
    Track IDs are local to an analysis and may switch during crossings.
    """

    default: FaceMask | LandmarkFaceMask | MaskProvider = field(default_factory=LandmarkFaceMask)
    overrides: Mapping[int, FaceMask | LandmarkFaceMask | MaskProvider] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        copied = dict(self.overrides)
        if any(type(track_id) is not int or track_id < 0 for track_id in copied):
            raise ValueError("Mask override IDs must be nonnegative integers")
        for mask in (self.default, *copied.values()):
            if not isinstance(mask, (FaceMask, LandmarkFaceMask)) and not (
                callable(getattr(mask, "create_session", None))
                and callable(getattr(mask, "validate_analysis", None))
            ):
                raise ValueError("Each override and default must implement the mask contract")
        object.__setattr__(self, "overrides", MappingProxyType(copied))

    def validate_analysis(self, analysis: VideoAnalysis) -> None:
        for mask in (self.default, *self.overrides.values()):
            if not isinstance(mask, (FaceMask, LandmarkFaceMask)):
                mask.validate_analysis(analysis)

    def create_session(self) -> PerFaceMaskSession:
        return PerFaceMaskSession(self)


class PerFaceMaskSession:
    """Independent configuration sessions with selected per-track diagnostics."""

    def __init__(self, mask: PerFaceMask) -> None:
        self.mask = mask
        configs = {id(value): value for value in (mask.default, *mask.overrides.values())}
        self._sessions = {
            key: MaskSession(value)
            if isinstance(value, (FaceMask, LandmarkFaceMask))
            else value.create_session()
            for key, value in configs.items()
        }
        self._last_index = -1
        self.diagnostics: dict[int, str] = {}

    def build(self, frame: FrameAnalysis, width: int, height: int) -> dict[int, MaskRegion]:
        return self._build(frame, width, height, None)

    def build_selected(
        self, frame: FrameAnalysis, width: int, height: int, track_ids: frozenset[int]
    ) -> dict[int, MaskRegion]:
        return self._build(frame, width, height, track_ids)

    def _build(
        self, frame: FrameAnalysis, width: int, height: int, track_ids: frozenset[int] | None
    ) -> dict[int, MaskRegion]:
        if frame.index <= self._last_index:
            raise ValueError("Frame indices must strictly increase in a mask session")
        chosen = {
            face.track_id: id(self.mask.overrides.get(face.track_id, self.mask.default))
            for face in frame.faces
            if track_ids is None or face.track_id in track_ids
        }
        candidates = {}
        for key, session in self._sessions.items():
            selected = frozenset(t for t, config in chosen.items() if config == key)
            candidates[key] = (
                session.build_selected(frame, width, height, selected)
                if hasattr(session, "build_selected")
                else session.build(frame, width, height)
            )
        regions, diagnostics = {}, {}
        for face in frame.faces:
            if face.track_id not in chosen:
                continue
            selected = self.mask.overrides.get(face.track_id, self.mask.default)
            key = id(selected)
            if face.track_id in candidates[key]:
                regions[face.track_id] = candidates[key][face.track_id]
            status = "legacy" if isinstance(selected, FaceMask) else "geometry"
            diagnostics[face.track_id] = getattr(self._sessions[key], "diagnostics", {}).get(
                face.track_id, status
            )
        self.diagnostics = diagnostics
        self._last_index = frame.index
        return regions


def _union(first: MaskRegion | None, second: MaskRegion | None) -> MaskRegion | None:
    if first is None:
        return second
    if second is None:
        return first
    x1, y1 = min(first.bounds[0], second.bounds[0]), min(first.bounds[1], second.bounds[1])
    x2, y2 = max(first.bounds[2], second.bounds[2]), max(first.bounds[3], second.bounds[3])
    alpha = np.zeros((y2 - y1, x2 - x1), np.float32)
    for region in (first, second):
        a, b, c, d = region.bounds
        target = alpha[b - y1 : d - y1, a - x1 : c - x1]
        np.maximum(target, region.alpha, out=target)
    return MaskRegion((x1, y1, x2, y2), alpha)
