"""Portable, inspectable face analysis, bound to exact source bytes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .tracking import IoUTracker
from .types import Box, Detection, Face, FrameAnalysis, Landmarks


def _load_landmarks(data: dict | None) -> Landmarks | None:
    return None if data is None else Landmarks(data["names"], data["points"])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.width, self.height)):
            raise ValueError("Video dimensions must be positive integers")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video FPS must be positive and finite")


@dataclass(frozen=True)
class VideoAnalysis:
    """One record per decoded frame. Timestamps are index / fps (CFR only)."""

    source_sha256: str
    video: VideoInfo
    frames: tuple[FrameAnalysis, ...]
    provenance: dict = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Unsupported analysis schema version")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        if not self.frames or any(frame.index != i for i, frame in enumerate(self.frames)):
            raise ValueError("Analysis must contain contiguous frames starting at zero")
        if not isinstance(self.provenance, dict):
            raise ValueError("Analysis provenance must be a JSON object")
        json.dumps(self.provenance, allow_nan=False)

    def save(self, path: str | Path) -> None:
        """Save JSON without image data. Explicitly overwrites the selected cache."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")

    def retrack(self, *, tracker: IoUTracker) -> VideoAnalysis:
        """Reassociate stored detections without video decoding or inference.

        Resets the supplied tracker. Requires raw detections for every original
        inference frame. Skipped frames retain the original inference schedule.
        """
        interval = self.provenance.get("detection_interval", 1)
        if type(interval) is not int or interval < 1:
            raise ValueError("Invalid detection interval in analysis provenance")
        if tracker.max_missing < interval - 1:
            raise ValueError("max_missing must cover the skipped inference frames")
        if any(frame.detections is None and frame.index % interval == 0 for frame in self.frames):
            raise ValueError("Raw detections are unavailable for retracking")
        tracker.reset()
        frames = tuple(
            replace(frame, faces=tracker.update(list(frame.detections or ())))
            for frame in self.frames
        )
        return replace(
            self, frames=frames, provenance=self.provenance | {"tracker": tracker.metadata()}
        )

    @classmethod
    def load(cls, path: str | Path) -> VideoAnalysis:
        try:
            data = json.loads(Path(path).read_text())
            if type(data["schema_version"]) is not int or data["schema_version"] not in (1, 2):
                raise ValueError("Unsupported analysis schema version")
            frames = tuple(
                FrameAnalysis(
                    frame["index"],
                    tuple(
                        Face(
                            track_id=face["track_id"],
                            box=Box(**face["box"]),
                            observed=face["observed"],
                            missing_frames=face["missing_frames"],
                            confidence=face["confidence"],
                            landmarks=_load_landmarks(face.get("landmarks")),
                        )
                        for face in frame["faces"]
                    ),
                    None
                    if frame.get("detections") is None
                    else tuple(
                        Detection(
                            Box(**detection["box"]),
                            detection["confidence"],
                            _load_landmarks(detection.get("landmarks")),
                        )
                        for detection in frame["detections"]
                    ),
                )
                for frame in data["frames"]
            )
            return cls(
                source_sha256=data["source_sha256"],
                video=VideoInfo(**data["video"]),
                frames=frames,
                provenance=data["provenance"],
                schema_version=2,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Anamorph analysis: {exc}") from exc
