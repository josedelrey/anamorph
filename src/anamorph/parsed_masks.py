"""Semantic mask boundaries with explicit exclusions and per-track state."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
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

    For accepted parses, all selected face pixels stay opaque. feather is an
    OUTWARD distance in full-frame pixels, restricted to predicted background.
    Excluded semantic classes remain zero after feathering and temporal filtering. Thus
    skin/hair boundaries can be hard. Exclusions follow model predictions and
    cannot guarantee the real anatomical boundary.

    included_labels can change coverage without inference. fallback='geometry'
    uses an unsegmented geometric mask for unusable parses, prioritizing face
    coverage. These fallbacks can include hair and ears. fallback='error'
    refuses them. Inspect session.diagnostics to distinguish parsed, held,
    and fallback masks.
    """

    parsing: VideoParsing
    feather: float = 2
    stabilization: float = 0.65
    included_labels: tuple[str, ...] = _FACE_LABELS
    fallback: Literal["geometry", "error"] = "geometry"

    def __post_init__(self) -> None:
        if not isinstance(self.parsing, VideoParsing):
            raise ValueError("parsing must be a VideoParsing cache")
        if not math.isfinite(self.feather) or self.feather < 0:
            raise ValueError("feather must be finite and nonnegative")
        if not math.isfinite(self.stabilization) or not 0 < self.stabilization <= 1:
            raise ValueError("stabilization must be in (0, 1]")
        if self.fallback not in ("geometry", "error"):
            raise ValueError("fallback must be geometry or error")
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

    def validate_analysis(self, analysis: VideoAnalysis) -> None:
        self.parsing.validate_analysis(analysis)

    def create_session(self) -> ParsedMaskSession:
        return ParsedMaskSession(self)


def _labels_in_frame(
    parsed: FaceParsing, width: int, height: int
) -> tuple[tuple, np.ndarray] | None:
    bounds = Box(*parsed.bounds).bounds(width, height)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    a, b, c, d = parsed.bounds
    gh, gw = parsed.labels.shape
    # Pixel-center mapping, including un-clipped crop geometry at image borders.
    ix = np.floor((np.arange(x1, x2) + 0.5 - a) * gw / (c - a)).astype(int)
    iy = np.floor((np.arange(y1, y2) + 0.5 - b) * gh / (d - b)).astype(int)
    labels = parsed.labels[np.ix_(np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1))]
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


def _usable_core(core: np.ndarray, face: Face, bounds: tuple, width: int, height: int) -> bool:
    """Reject tiny patches and parses that leave visible measured features exposed."""
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
        self._fallback = LandmarkFaceMask()

    def build(self, frame: FrameAnalysis, width: int, height: int) -> dict[int, MaskRegion]:
        if frame.index <= self._last_index:
            raise ValueError("Frame indices must strictly increase in a mask session")
        parsed = {face.track_id: face for face in self.mask.parsing.for_frame(frame, width, height)}
        if frame.index != self._last_index + 1:
            self._history.clear()
        regions, history, diagnostics = {}, {}, {}
        measurements = _measurements(frame)
        ids = tuple(FACE_PARSING_LABELS.index(name) for name in self.mask.included_labels)
        for face in frame.faces:
            previous = self._history.get(face.track_id)
            if previous is not None and previous[0].box.iou(face.box) < 0.1:
                previous = None
            measurement = measurements.get(face.track_id)
            checked_face = replace(face, landmarks=measurement.landmarks) if measurement else face
            fallback_box = measurement.box if measurement else face.box
            record = parsed.get(face.track_id)
            status = "parsed"
            if record is None and previous is not None:
                old_face, _, old_record = previous
                old_box = old_face.box
                checked_face = replace(
                    face,
                    landmarks=(
                        old_face.landmarks.remap(old_box, face.box) if old_face.landmarks else None
                    ),
                )
                a, b, c, d = old_record.bounds
                sx, sy = face.box.width / old_box.width, face.box.height / old_box.height
                edges = (
                    math.floor(face.box.x1 + (a - old_box.x1) * sx),
                    math.floor(face.box.y1 + (b - old_box.y1) * sy),
                    math.ceil(face.box.x1 + (c - old_box.x1) * sx),
                    math.ceil(face.box.y1 + (d - old_box.y1) * sy),
                )
                record = FaceParsing(face.track_id, edges, old_record.labels)
                status = "held"
            sampled = None if record is None else _labels_in_frame(record, width, height)
            if sampled is None:
                if self.mask.fallback == "error":
                    raise ValueError(
                        f"No usable parsing for track {face.track_id} at frame {frame.index}"
                    )
                region = self._fallback.build(
                    fallback_box, width, height, landmarks=checked_face.landmarks
                )
                diagnostics[face.track_id] = "fallback_missing"
                if region is not None:
                    regions[face.track_id] = region
                continue
            bounds, labels = sampled
            core = _component(np.isin(labels, ids), checked_face, bounds)
            # Close enclosed background holes such as unrecognized nostrils,
            # while preserving holes explicitly labelled hair, ears, or neck.
            core |= binary_fill_holes(core) & (labels == 0)
            allowed = core | (labels == 0)
            if not _usable_core(core, checked_face, bounds, width, height):
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
                continue
            else:
                alpha = core.astype(np.float32)
                if self.mask.feather > 0:
                    distance = cv2.distanceTransform(
                        (~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                    )
                    t = np.clip(distance / self.mask.feather, 0, 1)
                    alpha = 1 - t * t * (3 - 2 * t)
            if previous is not None:
                old_face, old_region, _ = previous
                prior = _warp_previous(old_region, old_face.box, face.box, bounds)
                if face.observed:
                    weight = self.mask.stabilization
                    alpha = np.maximum(alpha, weight * alpha + (1 - weight) * prior)
                else:
                    alpha = np.maximum(core, prior)
            # Apply class exclusions LAST, so neither feathering nor past masks
            # can bleed onto hair, ears, neck, or another disconnected face.
            alpha[~allowed] = 0
            region = MaskRegion(bounds, alpha.astype(np.float32))
            regions[face.track_id] = region
            history[face.track_id] = (checked_face, region, record)
            diagnostics[face.track_id] = status
        self._history = history
        self.diagnostics = diagnostics
        self._last_index = frame.index
        return regions
