"""Semantic mask boundaries with explicit exclusions and per-track state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.optimize import linear_sum_assignment

from .analysis import VideoAnalysis
from .masks import LandmarkFaceMask, MaskRegion
from .parsing import FACE_PARSING_LABELS, FaceParsing, VideoParsing
from .types import Box, Detection, Face, FrameAnalysis

_FACE_LABELS = (
    "skin",
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "glasses",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
)


@dataclass(frozen=True)
class ParsedFaceMask:
    """Use cached semantic boundaries, excluding hair, ears, and neck by default.

    With expansion=0, all selected face pixels stay opaque. feather is an
    OUTWARD distance in full-frame pixels, restricted to predicted background.
    Excluded semantic classes remain zero after feathering and temporal filtering,
    except classes explicitly allowed in boundary_labels. Thus skin/hair boundaries
    can be hard by default. Exclusions follow model predictions and cannot guarantee
    the real anatomical boundary.

    included_labels can change coverage without inference. fallback='geometry'
    uses an unsegmented geometric mask for unusable parses, prioritizing face
    coverage. These fallbacks can include hair and ears. fallback='error'
    refuses them. The default fallback='hold' transports the last accepted alpha
    with the tracked box for up to max_hold_frames without a fresh accepted
    parse. It raises if no accepted shape is available or the limit is exceeded.
    It never inserts geometry. Carried exclusions are estimates of old boundaries.
    Inspect session.diagnostics to distinguish parsed, held, and fallback masks.

    expansion is a signed full-frame pixel distance. Positive values grow the
    opaque core into background. boundary_labels can explicitly permit a limited
    margin into other classes, for example to correct a misplaced hairline. This
    trades semantic exclusion for coverage. Negative expansion erodes the core,
    deliberately reducing face coverage. Disconnected facial components veto growth.
    Output margins extend beyond the parsing crop into unclassified background.
    Temporal support also remains inside the output working region. Distance
    smoothing suppresses ridges from pixel-stepped contours before feathering.
    Quality checks run BEFORE edits, so
    deliberate erosion cannot be mistaken for a parsing failure. fallback_mask
    controls geometric fallbacks independently of accepted semantic masks.
    recovery_frames requires consecutive usable observed parses before updating
    a held shape or leaving geometry. Failures enter the chosen fallback policy
    immediately. Recovery never extends the semantic hold budget.
    """

    parsing: VideoParsing
    feather: float = 2
    stabilization: float = 0.65
    included_labels: tuple[str, ...] = _FACE_LABELS
    fallback: Literal["geometry", "error", "hold"] = "hold"
    expansion: float = 0
    fallback_mask: LandmarkFaceMask = field(default_factory=LandmarkFaceMask)
    boundary_labels: tuple[str, ...] = ("background",)
    recovery_frames: int = 1
    max_hold_frames: int = 15

    def __post_init__(self) -> None:
        if not isinstance(self.parsing, VideoParsing):
            raise ValueError("parsing must be a VideoParsing cache")
        if not math.isfinite(self.feather) or self.feather < 0:
            raise ValueError("feather must be finite and nonnegative")
        if not math.isfinite(self.stabilization) or not 0 < self.stabilization <= 1:
            raise ValueError("stabilization must be in (0, 1]")
        if self.fallback not in ("geometry", "error", "hold"):
            raise ValueError("fallback must be geometry, error, or hold")
        if not math.isfinite(self.expansion):
            raise ValueError("expansion must be finite")
        if type(self.recovery_frames) is not int or self.recovery_frames < 1:
            raise ValueError("recovery_frames must be a positive integer")
        if type(self.max_hold_frames) is not int or self.max_hold_frames < 1:
            raise ValueError("max_hold_frames must be a positive integer")
        if not isinstance(self.fallback_mask, LandmarkFaceMask):
            raise ValueError("fallback_mask must be a LandmarkFaceMask")
        if (
            not self.included_labels
            or isinstance(self.included_labels, str)
            or len(set(self.included_labels)) != len(self.included_labels)
            or any(name not in FACE_PARSING_LABELS[1:] for name in self.included_labels)
        ):
            raise ValueError("included_labels must contain unique non-background class names")
        if not set(_FACE_LABELS).issubset(self.included_labels):
            raise ValueError("Full-face masks must retain all default facial labels")
        object.__setattr__(self, "included_labels", tuple(self.included_labels))
        if (
            isinstance(self.boundary_labels, str)
            or len(set(self.boundary_labels)) != len(self.boundary_labels)
            or any(name not in FACE_PARSING_LABELS for name in self.boundary_labels)
            or set(self.boundary_labels).intersection(self.included_labels)
        ):
            raise ValueError("boundary_labels must be unique class names outside included_labels")
        object.__setattr__(self, "boundary_labels", tuple(self.boundary_labels))

    def validate_analysis(self, analysis: VideoAnalysis) -> None:
        self.parsing.validate_analysis(analysis)

    def create_session(self) -> ParsedMaskSession:
        return ParsedMaskSession(self)


def _labels_in_frame(
    parsed: FaceParsing,
    width: int,
    height: int,
    *,
    padding: int = 0,
    extra_bounds: tuple[int, int, int, int] | None = None,
    clip: bool = True,
) -> tuple[tuple, np.ndarray] | None:
    a, b, c, d = parsed.bounds
    left, top, right, bottom = a - padding, b - padding, c + padding, d + padding
    if extra_bounds is not None:
        left, top = min(left, extra_bounds[0]), min(top, extra_bounds[1])
        right, bottom = max(right, extra_bounds[2]), max(bottom, extra_bounds[3])
    bounds = (
        Box(left, top, right, bottom).bounds(width, height) if clip else (left, top, right, bottom)
    )
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    gh, gw = parsed.labels.shape
    # Pixel-center mapping, including un-clipped crop geometry at image borders.
    ix = np.floor((np.arange(x1, x2) + 0.5 - a) * gw / (c - a)).astype(int)
    iy = np.floor((np.arange(y1, y2) + 0.5 - b) * gh / (d - b)).astype(int)
    labels = parsed.labels[np.ix_(np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1))]
    # Pixels beyond the inference crop have no labels. Treat them as background
    # for the output margin, without replicating facial classes into that margin.
    inside_x = (np.arange(x1, x2) + 0.5 >= a) & (np.arange(x1, x2) + 0.5 < c)
    inside_y = (np.arange(y1, y2) + 0.5 >= b) & (np.arange(y1, y2) + 0.5 < d)
    labels[~(inside_y[:, None] & inside_x[None, :])] = 0
    return bounds, labels


def _component(core: np.ndarray, face: Face, bounds: tuple) -> np.ndarray:
    count, components, _, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    if count == 1:
        return core
    x1, y1, _, _ = bounds
    seeds = []
    if face.landmarks is not None:
        for name in (
            "nose_tip",
            "left_eye",
            "right_eye",
            "left_mouth_corner",
            "right_mouth_corner",
        ):
            if name in face.landmarks.names:
                seeds.append(face.landmarks[name])
    chosen = set()
    for x, y in seeds:
        ix, iy = math.floor(x) - x1, math.floor(y) - y1
        if 0 <= ix < core.shape[1] and 0 <= iy < core.shape[0]:
            label = int(components[iy, ix])
            if label:
                chosen.add(label)
    if not chosen:
        # Prefer the largest component inside this face's box, rather than a
        # larger neighboring face elsewhere in the context crop.
        box = Box(face.box.x1 - x1, face.box.y1 - y1, face.box.x2 - x1, face.box.y2 - y1).bounds(
            core.shape[1], core.shape[0]
        )
        if box is None:
            return np.zeros_like(core)
        a, b, c, d = box
        overlap = np.bincount(components[b:d, a:c].ravel(), minlength=count)
        overlap[0] = 0
        winner = int(np.argmax(overlap))
        if overlap[winner] == 0:
            return np.zeros_like(core)
        chosen.add(winner)
    return np.isin(components, tuple(chosen))


def _warp_previous(region: MaskRegion, old_box: Box, new_box: Box, bounds: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    x = (
        (np.arange(x1, x2, dtype=np.float32) + 0.5 - new_box.x1) * (old_box.width / new_box.width)
        + old_box.x1
        - 0.5
        - region.bounds[0]
    )
    y = (
        (np.arange(y1, y2, dtype=np.float32) + 0.5 - new_box.y1) * (old_box.height / new_box.height)
        + old_box.y1
        - 0.5
        - region.bounds[1]
    )
    map_x, map_y = np.meshgrid(x, y)
    return cv2.remap(region.alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _remap_bounds(bounds: tuple, old_box: Box, new_box: Box) -> tuple[int, int, int, int]:
    a, b, c, d = bounds
    sx, sy = new_box.width / old_box.width, new_box.height / old_box.height
    return (
        math.floor(new_box.x1 + (a - old_box.x1) * sx),
        math.floor(new_box.y1 + (b - old_box.y1) * sy),
        math.ceil(new_box.x1 + (c - old_box.x1) * sx),
        math.ceil(new_box.y1 + (d - old_box.y1) * sy),
    )


def _prior_support(previous: tuple[Face, MaskRegion, FaceParsing], face: Face) -> tuple | None:
    old_face, region, _ = previous
    x, y, w, h = cv2.boundingRect((region.alpha > 0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    # Include the interpolation footprint when transporting the old alpha.
    bounds = (
        region.bounds[0] + x - 1,
        region.bounds[1] + y - 1,
        region.bounds[0] + x + w + 1,
        region.bounds[1] + y + h + 1,
    )
    return _remap_bounds(bounds, old_face.box, face.box)


def _feather_core(core: np.ndarray, feather: float) -> np.ndarray:
    if feather == 0 or not np.any(core):
        return core.astype(np.float32)
    distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # A broad distance ramp exposes ridges from the stepped label contour.
    # Smooth the distance field before tapering, keeping the opaque core exact.
    sigma = feather / 8
    radius = math.ceil(3 * sigma)
    kernel = (2 * radius + 1, 2 * radius + 1)
    distance = cv2.GaussianBlur(distance, kernel, sigma, borderType=cv2.BORDER_REPLICATE)
    t = np.clip(distance / feather, 0, 1)
    alpha = 1 - t * t * (3 - 2 * t)
    alpha[core] = 1
    return alpha


def _edit_core(core: np.ndarray, allowed: np.ndarray, expansion: float) -> np.ndarray:
    if expansion > 0:
        return (distance_transform_edt(~core) <= expansion) & allowed
    if expansion < 0:
        # The crop exterior is zero, including when the core touches every edge.
        distance = distance_transform_edt(np.pad(core, 1))[1:-1, 1:-1]
        return distance > -expansion
    return core


def _usable_core(core: np.ndarray, face: Face, bounds: tuple, width: int, height: int) -> bool:
    """Reject tiny patches and parses that leave visible measured features exposed."""
    clipped = Box(*bounds).bounds(width, height)
    if clipped is None:
        return False
    a, b, c, d = clipped
    core = core[b - bounds[1] : d - bounds[1], a - bounds[0] : c - bounds[0]]
    bounds = clipped
    visible = face.box.bounds(width, height)
    if visible is None:
        return False
    area = (visible[2] - visible[0]) * (visible[3] - visible[1])
    if np.count_nonzero(core) < max(1, 0.15 * area):
        return False
    y, x = np.nonzero(core)
    # Feature points alone can accept a lower-face patch. Require broad enough
    # extent to reach the expected forehead, chin, and both sides of the face.
    if (
        x.min() + bounds[0] > visible[0] + 0.35 * (visible[2] - visible[0])
        or x.max() + bounds[0] + 1 < visible[0] + 0.65 * (visible[2] - visible[0])
        or y.min() + bounds[1] > visible[1] + 0.35 * (visible[3] - visible[1])
        or y.max() + bounds[1] + 1 < visible[1] + 0.8 * (visible[3] - visible[1])
    ):
        return False
    if face.landmarks is not None:
        for name in (
            "nose_tip",
            "left_eye",
            "right_eye",
            "left_mouth_corner",
            "right_mouth_corner",
        ):
            if name not in face.landmarks.names:
                continue
            x, y = face.landmarks[name]
            if not (0 <= x < width and 0 <= y < height):
                continue
            ix, iy = math.floor(x) - bounds[0], math.floor(y) - bounds[1]
            if not (0 <= ix < core.shape[1] and 0 <= iy < core.shape[0] and core[iy, ix]):
                return False
    return True


def _measurements(frame: FrameAnalysis) -> dict[int, Detection]:
    """Associate cached raw observations with current tracks for coverage checks.

    Tracked landmarks have been remapped into smoothed boxes and can lag real
    pixels. This gated one-to-one association reconstructs measurement links
    from older caches, which do not store those links explicitly.
    """
    faces = [face for face in frame.faces if face.observed]
    detections = frame.detections or ()
    if not faces or not detections:
        return {}
    iou = np.array([[face.box.iou(d.box) for d in detections] for face in faces])
    weights = np.where(iou >= 0.1, min(iou.shape) + 1 + iou, 0)
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return {
        faces[r].track_id: detections[c]
        for r, c in zip(rows, columns, strict=True)
        if weights[r, c] > 0
    }


class ParsedMaskSession:
    """Fresh temporal state and per-frame diagnostics for one parsed mask config."""

    def __init__(self, mask: ParsedFaceMask) -> None:
        self.mask = mask
        self.diagnostics: dict[int, str] = {}
        self._last_index = -1
        self._history: dict[int, tuple[Face, MaskRegion, FaceParsing]] = {}
        self._accepted_at: dict[int, int] = {}
        self._recovering: dict[int, tuple[Box, int]] = {}
        self._fallback = mask.fallback_mask

    def _hold(
        self,
        face: Face,
        previous: tuple[Face, MaskRegion, FaceParsing] | None,
        index: int,
        width: int,
        height: int,
    ) -> MaskRegion:
        if previous is None:
            raise ValueError(
                f"No accepted semantic mask to hold for track {face.track_id} at frame {index}"
            )
        if index - self._accepted_at[face.track_id] > self.mask.max_hold_frames:
            raise ValueError(
                f"Semantic hold limit of {self.mask.max_hold_frames} frames exceeded "
                f"for track {face.track_id} at frame {index}"
            )
        old_face, old_region, _ = previous
        bounds = Box(*_remap_bounds(old_region.bounds, old_face.box, face.box)).bounds(
            width, height
        )
        if bounds is None:
            raise ValueError(f"Held semantic mask is outside the frame for track {face.track_id}")
        return MaskRegion(bounds, _warp_previous(old_region, old_face.box, face.box, bounds))

    def build(self, frame: FrameAnalysis, width: int, height: int) -> dict[int, MaskRegion]:
        return self._build(frame, width, height, None)

    def build_selected(
        self, frame: FrameAnalysis, width: int, height: int, track_ids: frozenset[int]
    ) -> dict[int, MaskRegion]:
        """Validate the original frame, then process only selected track IDs."""
        return self._build(frame, width, height, track_ids)

    def _build(
        self, frame: FrameAnalysis, width: int, height: int, track_ids: frozenset[int] | None
    ) -> dict[int, MaskRegion]:
        if frame.index <= self._last_index:
            raise ValueError("Frame indices must strictly increase in a mask session")
        parsed = {face.track_id: face for face in self.mask.parsing.for_frame(frame, width, height)}
        if frame.index != self._last_index + 1:
            self._history.clear()
            self._accepted_at.clear()
            self._recovering.clear()
        regions, history, diagnostics, recovering = {}, {}, {}, {}
        accepted_at = {}

        def hold(face, previous, status, recovery=0):
            regions[face.track_id] = self._hold(face, previous, frame.index, width, height)
            # Retain the original accepted anchor to avoid accumulating warp blur.
            history[face.track_id] = previous
            accepted_at[face.track_id] = self._accepted_at[face.track_id]
            diagnostics[face.track_id] = status
            recovering[face.track_id] = (face.box, recovery)

        measurements = _measurements(frame)
        ids = tuple(FACE_PARSING_LABELS.index(name) for name in self.mask.included_labels)
        for face in frame.faces:
            if track_ids is not None and face.track_id not in track_ids:
                continue
            previous = self._history.get(face.track_id)
            if previous is not None and previous[0].box.iou(face.box) < 0.1:
                previous = None
            measurement = measurements.get(face.track_id)
            checked_face = replace(face, landmarks=measurement.landmarks) if measurement else face
            fallback_box = measurement.box if measurement else face.box
            record = parsed.get(face.track_id)
            status = "parsed"
            if record is None and self.mask.fallback == "hold":
                hold(face, previous, "held_missing")
                continue
            if record is None and previous is not None:
                old_face, _, old_record = previous
                old_box = old_face.box
                checked_face = replace(
                    face,
                    landmarks=(
                        old_face.landmarks.remap(old_box, face.box) if old_face.landmarks else None
                    ),
                )
                edges = _remap_bounds(old_record.bounds, old_box, face.box)
                record = FaceParsing(face.track_id, edges, old_record.labels)
                status = "held"
            # Reserve the outward margin and the smoothing kernel's footprint.
            padding = math.ceil(max(0, self.mask.expansion) + self.mask.feather)
            padding += math.ceil(3 * self.mask.feather / 8)
            extra = _prior_support(previous, face) if previous is not None else None
            sampled = (
                None
                if record is None
                else _labels_in_frame(
                    record, width, height, padding=padding, extra_bounds=extra, clip=False
                )
            )
            if sampled is None:
                if self.mask.fallback == "hold":
                    hold(face, previous, "held_missing")
                    continue
                if self.mask.fallback == "error":
                    raise ValueError(
                        f"No usable parsing for track {face.track_id} at frame {frame.index}"
                    )
                region = self._fallback.build(
                    fallback_box, width, height, landmarks=checked_face.landmarks
                )
                diagnostics[face.track_id] = "fallback_missing"
                recovering[face.track_id] = (face.box, 0)
                if region is not None:
                    regions[face.track_id] = region
                continue
            bounds, labels = sampled
            core = _component(np.isin(labels, ids), checked_face, bounds)
            # Close enclosed background holes such as unrecognized nostrils,
            # while preserving holes explicitly labelled hair, ears, or neck.
            core |= binary_fill_holes(core) & (labels == 0)
            margin_ids = tuple(
                FACE_PARSING_LABELS.index(name) for name in self.mask.boundary_labels
            )
            allowed = core | np.isin(labels, margin_ids)
            if not _usable_core(core, checked_face, bounds, width, height):
                if self.mask.fallback == "hold":
                    hold(face, previous, "held_empty" if not np.any(core) else "held_incomplete")
                    continue
                if self.mask.fallback == "error":
                    raise ValueError(
                        f"Incomplete parsing for track {face.track_id} at frame {frame.index}"
                    )
                fallback = self._fallback.build(
                    fallback_box, width, height, landmarks=checked_face.landmarks
                )
                if fallback is not None:
                    regions[face.track_id] = fallback
                diagnostics[face.track_id] = (
                    "fallback_empty" if not np.any(core) else "fallback_incomplete"
                )
                recovering[face.track_id] = (face.box, 0)
                continue
            else:
                pending = self._recovering.get(face.track_id)
                if pending is not None and pending[0].iou(face.box) >= 0.1:
                    count = pending[1] + 1 if face.observed else 0
                    if count < self.mask.recovery_frames:
                        if self.mask.fallback == "hold":
                            hold(face, previous, "held_recovery", count)
                            continue
                        region = self._fallback.build(
                            fallback_box, width, height, landmarks=checked_face.landmarks
                        )
                        if region is not None:
                            regions[face.track_id] = region
                        diagnostics[face.track_id] = "fallback_recovery"
                        recovering[face.track_id] = (face.box, count)
                        continue
                core = _edit_core(core, allowed, self.mask.expansion)
                alpha = _feather_core(core, self.mask.feather)
                if self.mask.expansion < 0:
                    # Deliberate shrinkage must not be undone by carried history.
                    allowed &= alpha > 0
            if previous is not None:
                old_face, old_region, _ = previous
                prior = _warp_previous(old_region, old_face.box, face.box, bounds)
                if face.observed:
                    weight = self.mask.stabilization
                    alpha = np.maximum(alpha, weight * alpha + (1 - weight) * prior)
                else:
                    alpha = np.maximum(core, prior)
            # Apply exclusions LAST. Only explicitly allowed boundary classes
            # can receive margin coverage. Disconnected faces remain excluded.
            alpha[~allowed] = 0
            region = MaskRegion(bounds, alpha.astype(np.float32))
            # Clip only the public output. Offscreen semantic support must remain
            # in history so a frame edge cannot move inward as a rectangular cut.
            clipped = Box(*bounds).bounds(width, height)
            if clipped is not None:
                x1, y1, x2, y2 = clipped
                regions[face.track_id] = MaskRegion(
                    clipped,
                    region.alpha[y1 - bounds[1] : y2 - bounds[1], x1 - bounds[0] : x2 - bounds[0]],
                )
            history[face.track_id] = (checked_face, region, record)
            accepted_at[face.track_id] = (
                frame.index if status == "parsed" else self._accepted_at[face.track_id]
            )
            diagnostics[face.track_id] = status
        self._history = history
        self._accepted_at = accepted_at
        self._recovering = recovering
        self.diagnostics = diagnostics
        self._last_index = frame.index
        return regions
