"""Deterministic one-to-one IoU association and bounded missing-frame holds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Box, Detection, Face, Landmarks


@dataclass
class _Track:
    box: Box
    confidence: float | None
    landmarks: Landmarks | None = None
    missing: int = 0


class IoUTracker:
    """Associate detections globally, smooth boxes, and hold short dropouts.

    max_missing counts decoded frames, including skipped inference frames.
    smoothing is the weight of the current observation (1 disables smoothing).
    External image-derived box predictions can be supplied explicitly. The
    tracker itself estimates no motion or recognition embedding.
    Landmarks are mapped from the latest measured box into the smoothed box.
    Without predictions, held tracks retain both box and landmarks unchanged.
    """

    def __init__(
        self, *, min_iou: float = 0.2, max_missing: int = 5, smoothing: float = 0.7
    ) -> None:
        if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        if type(max_missing) is not int or max_missing < 0:
            raise ValueError("max_missing must be a nonnegative integer")
        if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.min_iou, self.max_missing, self.smoothing = min_iou, max_missing, smoothing
        self.reset()

    def reset(self) -> None:
        self._next_id = 0
        self.clear()

    def clear(self) -> None:
        """Forget active tracks while preserving unused IDs, for example on a cut."""
        self._tracks: dict[int, _Track] = {}

    def metadata(self) -> dict:
        return {
            "name": type(self).__name__,
            "min_iou": self.min_iou,
            "max_missing": self.max_missing,
            "smoothing": self.smoothing,
        }

    def update(
        self, detections: list[Detection], *, predictions: Mapping[int, Box] | None = None
    ) -> tuple[Face, ...]:
        """Associate from optional externally predicted boxes for existing IDs.

        Predictions are image-derived estimates, not detector observations.
        They do not reset missing age. Smoothing blends predictions with new
        observations, avoiding added box lag when motion is accurate.
        """
        predictions = predictions or {}
        if any(
            type(key) is not int or key not in self._tracks or not isinstance(box, Box)
            for key, box in predictions.items()
        ):
            raise ValueError("Predictions must map existing integer track IDs to Box values")
        for track_id, box in predictions.items():
            track = self._tracks[track_id]
            if track.landmarks is not None:
                track.landmarks = track.landmarks.remap(track.box, box)
            track.box = box
        detections = sorted(detections, key=lambda d: d.box.as_tuple())
        ids = sorted(self._tracks)
        matched: dict[int, int] = {}
        if ids and detections:
            scores = np.array([[self._tracks[i].box.iou(d.box) for d in detections] for i in ids])
            # Invalid edges cannot displace valid matches. A cardinality bonus
            # first maximizes the number of valid pairs, then their summed IoU.
            bonus = min(scores.shape) + 1
            weights = np.where(scores >= self.min_iou, bonus + scores, 0.0)
            rows, cols = linear_sum_assignment(weights, maximize=True)
            matched = {ids[r]: int(c) for r, c in zip(rows, cols, strict=True) if weights[r, c] > 0}

        for track_id in ids:
            track = self._tracks[track_id]
            if track_id in matched:
                detection = detections[matched[track_id]]
                track.box = Box(
                    *(
                        (1 - self.smoothing) * a + self.smoothing * b
                        for a, b in zip(track.box.as_tuple(), detection.box.as_tuple(), strict=True)
                    )
                )
                track.missing, track.confidence = 0, detection.confidence
                track.landmarks = (
                    detection.landmarks.remap(detection.box, track.box)
                    if detection.landmarks is not None
                    else None
                )
            else:
                track.missing += 1
                if track.missing > self.max_missing:
                    del self._tracks[track_id]

        used = set(matched.values())
        for index, detection in enumerate(detections):
            if index not in used:
                self._tracks[self._next_id] = _Track(
                    detection.box, detection.confidence, detection.landmarks
                )
                self._next_id += 1

        return tuple(
            Face(i, t.box, t.missing == 0, t.missing, t.confidence, t.landmarks)
            for i, t in sorted(self._tracks.items())
        )
