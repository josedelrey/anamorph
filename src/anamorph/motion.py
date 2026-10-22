"""Optional, bounded face-box translation from image evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

import cv2
import numpy as np

from .types import Box, Face, Image, validate_image


@dataclass(frozen=True)
class MotionEstimate:
    predictions: Mapping[int, Box]
    diagnostics: Mapping[int, str]
    scene_cut: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", MappingProxyType(dict(self.predictions)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class SparseOpticalFlow:
    """Estimate translation using face-interior corners and bidirectional LK.

    This predicts existing boxes only. It cannot discover undetected faces,
    infer hidden anatomy, or guarantee identity through crossings. Rejected
    estimates leave the tracker's normal missing-frame policy in control.
    max_missing bounds consecutive predictions without a detector observation.
    Scene cuts use a coarse normalized RGB-change heuristic and clear tracks
    in video analysis. Neither cuts nor occlusions are perfectly classified.
    """

    max_points: int = 80
    min_points: int = 8
    min_inlier_fraction: float = 0.6
    max_missing: int = 3
    forward_backward_threshold: float = 1.5
    max_photometric_error: float = 20
    max_displacement: float = 0.35
    scene_cut_threshold: float = 0.35

    def __post_init__(self) -> None:
        if (
            type(self.max_points) is not int
            or type(self.min_points) is not int
            or not 3 <= self.min_points <= self.max_points
            or type(self.max_missing) is not int
            or self.max_missing < 1
        ):
            raise ValueError("Point counts must satisfy 3 <= min <= max and max_missing >= 1")
        for name in ("min_inlier_fraction", "max_displacement", "scene_cut_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("forward_backward_threshold", "max_photometric_error"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def metadata(self) -> dict:
        return {"name": type(self).__name__, "model": "median_translation", **asdict(self)}

    def estimate(self, previous: Image, current: Image, faces: tuple[Face, ...]) -> MotionEstimate:
        validate_image(previous)
        validate_image(current)
        if previous.shape != current.shape:
            raise ValueError("Motion estimation requires equal frame dimensions")
        old_small = cv2.resize(previous, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
        new_small = cv2.resize(current, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
        if np.mean(np.abs(new_small - old_small)) / 255 >= self.scene_cut_threshold:
            return MotionEstimate({}, {f.track_id: "scene_cut" for f in faces}, True)
        old = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        new = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        height, width = old.shape
        predictions, diagnostics = {}, {}
        for face in faces:
            if face.missing_frames >= self.max_missing:
                diagnostics[face.track_id] = "gap_limit"
                continue
            box = face.box
            inner = Box(
                box.x1 + 0.1 * box.width,
                box.y1 + 0.1 * box.height,
                box.x2 - 0.1 * box.width,
                box.y2 - 0.1 * box.height,
            ).bounds(width, height)
            if inner is None:
                diagnostics[face.track_id] = "offscreen"
                continue
            a, b, c, d = inner
            points = cv2.goodFeaturesToTrack(old[b:d, a:c], self.max_points, 0.01, 3)
            if points is None or len(points) < self.min_points:
                diagnostics[face.track_id] = "insufficient_features"
                continue
            points += np.array([a, b], dtype=np.float32)
            params = dict(
                winSize=(15, 15),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            forward, found, error = cv2.calcOpticalFlowPyrLK(old, new, points, None, **params)
            if forward is None or found is None or error is None:
                diagnostics[face.track_id] = "flow_failed"
                continue
            finite = (found.ravel() == 1) & np.isfinite(forward.reshape(-1, 2)).all(axis=1)
            if finite.sum() < self.min_points:
                diagnostics[face.track_id] = "inconsistent_flow"
                continue
            # Failed LK coordinates are unspecified. Never feed NaNs or garbage
            # from failed points into the backward pass, even though later
            # filtering would reject their measurements.
            safe_forward = forward.copy()
            safe_forward[~finite] = points[~finite]
            backward, returned, _ = cv2.calcOpticalFlowPyrLK(new, old, safe_forward, None, **params)
            if backward is None or returned is None:
                diagnostics[face.track_id] = "flow_failed"
                continue
            p, q, r = (v.reshape(-1, 2) for v in (points, forward, backward))
            valid = (
                (found.ravel() == 1)
                & (returned.ravel() == 1)
                & np.isfinite(q).all(axis=1)
                & np.isfinite(r).all(axis=1)
                & np.isfinite(error.ravel())
                & (np.linalg.norm(p - r, axis=1) <= self.forward_backward_threshold)
                & (error.ravel() <= self.max_photometric_error)
                & (q[:, 0] >= 0)
                & (q[:, 0] < width)
                & (q[:, 1] >= 0)
                & (q[:, 1] < height)
            )
            if valid.sum() < self.min_points:
                diagnostics[face.track_id] = "inconsistent_flow"
                continue
            delta = q - p
            center = np.median(delta[valid], axis=0)
            inliers = valid & (
                np.linalg.norm(delta - center, axis=1)
                <= max(1.5, 0.05 * min(box.width, box.height))
            )
            if inliers.sum() < max(self.min_points, self.min_inlier_fraction * len(points)):
                diagnostics[face.track_id] = "insufficient_inliers"
                continue
            spread = np.ptp(p[inliers], axis=0)
            if spread[0] < 0.2 * box.width or spread[1] < 0.2 * box.height:
                diagnostics[face.track_id] = "insufficient_support"
                continue
            dx, dy = np.median(delta[inliers], axis=0).astype(float)
            if np.hypot(dx / box.width, dy / box.height) > self.max_displacement:
                diagnostics[face.track_id] = "excessive_motion"
                continue
            prediction = Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)
            if prediction.bounds(width, height) is None:
                diagnostics[face.track_id] = "offscreen"
                continue
            predictions[face.track_id] = prediction
            diagnostics[face.track_id] = "accepted"
        return MotionEstimate(predictions, diagnostics)
