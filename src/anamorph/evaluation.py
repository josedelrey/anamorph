"""Mask measurements against explicit reference silhouettes, without inference.

All arrays use full-frame pixels. These measurements describe agreement with
the supplied annotations, whose quality and provenance must be reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def polygon_mask(polygons: Sequence, width: int, height: int) -> np.ndarray:
    """Union simple polygons, sampling at pixel centers in frame coordinates.

    Vertices describe continuous pixel edges, just like Box. Polygons may be
    concave. Holes should be supplied separately as ignore or exclusion masks.
    """
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Frame dimensions must be positive integers")
    result = np.zeros((height, width), bool)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("Polygons need at least three finite (x, y) vertices")
        x, y = points.T
        if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
            raise ValueError("Reference vertices must lie within the frame edges")
        if abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < 1e-8:
            raise ValueError("Reference polygons must have positive area")
        a, b = np.floor(points.min(axis=0)).astype(int)
        c, d = np.ceil(points.max(axis=0)).astype(int)
        px = np.arange(a, c)[None, :] + 0.5
        py = np.arange(b, d)[:, None] + 0.5
        inside = np.zeros((d - b, c - a), bool)
        for (x1, y1), (x2, y2) in zip(points, np.roll(points, -1, axis=0), strict=True):
            if y1 != y2:
                inside ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
        result[b:d, a:c] |= inside
    return result


def _arrays(alpha: np.ndarray, reference: np.ndarray, ignore: np.ndarray | None):
    alpha = np.asarray(alpha)
    reference = np.asarray(reference)
    if (
        alpha.ndim != 2
        or min(alpha.shape, default=0) == 0
        or reference.shape != alpha.shape
        or reference.dtype != bool
        or not np.isfinite(alpha).all()
        or np.any((alpha < 0) | (alpha > 1))
    ):
        raise ValueError("Expected finite alpha in [0, 1] and an equal-size boolean reference")
    ignore = np.zeros_like(reference) if ignore is None else np.asarray(ignore)
    if ignore.shape != alpha.shape or ignore.dtype != bool:
        raise ValueError("ignore must be an equal-size boolean array")
    return alpha, reference, ~ignore


def evaluate_mask(
    alpha: np.ndarray,
    reference: np.ndarray,
    *,
    exclusions: Mapping[str, np.ndarray] | None = None,
    ignore: np.ndarray | None = None,
    opaque_threshold: float = 0.999,
) -> dict:
    """Measure opaque coverage, alpha deficit, silhouette IoU and exclusions.

    Ignored pixels do not affect area measurements. Boundary p95 uses the raw
    outlines, including uncertain borders, to preserve distances to the actual
    annotated contour. It is undefined for an empty predicted outline. Exclusion
    masks describe explicitly annotated hair/ears, not predicted model classes.
    """
    alpha, reference, valid = _arrays(alpha, reference, ignore)
    if not np.isfinite(opaque_threshold) or not 0 < opaque_threshold <= 1:
        raise ValueError("opaque_threshold must be in (0, 1]")
    face = reference & valid
    area = int(np.count_nonzero(face))
    if area == 0:
        raise ValueError("Reference has no evaluable facial area")
    opaque = alpha >= opaque_threshold
    intersection = int(np.count_nonzero(opaque & face))
    union = int(np.count_nonzero((opaque | reference) & valid))
    boundary = reference & ~binary_erosion(reference, border_value=0)
    predicted_boundary = opaque & ~binary_erosion(opaque, border_value=0)
    distances = None
    if np.any(predicted_boundary):
        distances = np.concatenate(
            [
                distance_transform_edt(~boundary)[predicted_boundary],
                distance_transform_edt(~predicted_boundary)[boundary],
            ]
        )
    excluded = {}
    for name, mask in (exclusions or {}).items():
        mask = np.asarray(mask)
        if mask.shape != alpha.shape or mask.dtype != bool:
            raise ValueError("Exclusions must be equal-size boolean arrays")
        if np.any(mask & face & valid):
            raise ValueError("Facial and excluded reference areas must not overlap outside ignore")
        selected = mask & valid
        size = int(np.count_nonzero(selected))
        excluded[name] = {
            "pixels": size,
            "mean_alpha": float(np.mean(alpha[selected])) if size else None,
            "opaque_fraction": float(np.mean(opaque[selected])) if size else None,
        }
    return {
        "face_pixels": area,
        "missed_opaque_face_pixels": area - intersection,
        "missed_opaque_face_fraction": 1 - intersection / area,
        "mean_face_alpha_deficit": float(np.mean(1 - alpha[face])),
        "opaque_iou": intersection / union,
        "extra_alpha_pixels": float(np.sum(alpha[valid & ~reference])),
        "boundary_p95_pixels": float(np.percentile(distances, 95))
        if distances is not None
        else None,
        "exclusions": excluded,
    }


def temporal_error_change(
    previous_alpha: np.ndarray,
    previous_reference: np.ndarray,
    alpha: np.ndarray,
    reference: np.ndarray,
    *,
    previous_ignore: np.ndarray | None = None,
    ignore: np.ndarray | None = None,
    context: float = 0.25,
) -> float | None:
    """Mean change in signed mask error after reference-box motion alignment.

    Compare adjacent frames of the SAME manually identified face only. Affine
    alignment uses reference bounds, never tracker predictions. The measurement
    is a flicker proxy, affected by pose changes and annotation noise. It does
    not establish perceptual temporal stability or optical-flow accuracy.
    context pads reference bounds by a fraction of their width/height,
    including outward boundary flicker.
    """
    old, old_ref, old_valid = _arrays(previous_alpha, previous_reference, previous_ignore)
    current, ref, valid = _arrays(alpha, reference, ignore)
    if not np.isfinite(context) or not 0 <= context <= 1:
        raise ValueError("context must be in [0, 1]")
    if old.shape != current.shape:
        raise ValueError("Temporal measurements require equal frame dimensions")
    if not np.any(old_ref) or not np.any(ref):
        return None
    oy, ox = np.nonzero(old_ref)
    y, x = np.nonzero(ref)
    a, b, c, d = x.min(), y.min(), x.max() + 1, y.max() + 1
    left = max(0, int(np.floor(a - context * (c - a))))
    right = min(ref.shape[1], int(np.ceil(c + context * (c - a))))
    top = max(0, int(np.floor(b - context * (d - b))))
    bottom = min(ref.shape[0], int(np.ceil(d + context * (d - b))))
    px = (np.arange(left, right) + 0.5 - a) * (np.ptp(ox) + 1) / (c - a) + ox.min() - 0.5
    py = (np.arange(top, bottom) + 0.5 - b) * (np.ptp(oy) + 1) / (d - b) + oy.min() - 0.5
    mx, my = (v.astype(np.float32) for v in np.meshgrid(px, py))
    old_error = old.astype(np.float32) - old_ref
    warped = cv2.remap(old_error, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    known = cv2.remap(old_valid.astype(np.float32), mx, my, cv2.INTER_LINEAR) >= 0.999
    selected = known & valid[top:bottom, left:right]
    return (
        float(np.mean(np.abs(warped[selected] - (current - ref)[top:bottom, left:right][selected])))
        if np.any(selected)
        else None
    )
