"""Optional semantic face analysis with source-bound, non-pickle label caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ._progress import frame_progress
from .analysis import VideoAnalysis, VideoInfo, file_sha256
from .types import Box, FrameAnalysis, Image, validate_image

# CelebAMask-HQ layout used by yakhyo/face-parsing.
FACE_PARSING_LABELS = (
    "background",
    "skin",
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "glasses",
    "left_ear",
    "right_ear",
    "earring",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
    "neck",
    "necklace",
    "clothing",
    "hair",
    "hat",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _check_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected a lowercase SHA-256 digest")


class FaceParser(Protocol):
    """Return a 2D uint8 label map in the explicit 19-class layout, without mutation.

    Input is a single face-centered RGB crop. Output covers the entire crop,
    at any positive grid resolution. metadata must describe model and preprocessing.
    """

    def parse(self, image: Image) -> np.ndarray: ...

    def metadata(self) -> dict: ...


class BiSeNetFaceParser:
    """CPU OpenCV DNN adapter for yakhyo's BiSeNet ONNX exports.

    Takes local weights only. No implicit download or detector inference.
    Instances hold mutable DNN state and must be used serially.
    """

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError("Expected ONNX weights, received a Git LFS pointer")
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise ValueError(f"Cannot load BiSeNet ONNX model: {exc}") from exc
        self._model_sha256 = file_sha256(path)

    def parse(self, image: Image) -> np.ndarray:
        validate_image(image)
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        try:
            self._net.setInput(np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]))
            # The export also contains auxiliary heads. Select the named primary head.
            output = self._net.forward("output")
        except cv2.error as exc:
            raise ValueError(
                f"Cannot run BiSeNet model with the required output head: {exc}"
            ) from exc
        if output.shape != (1, 19, 512, 512) or not np.isfinite(output).all():
            raise ValueError("BiSeNet must return finite 1 x 19 x 512 x 512 logits")
        return output[0].argmax(axis=0).astype(np.uint8)

    def metadata(self) -> dict:
        return {
            "name": type(self).__name__,
            "model_sha256": self._model_sha256,
            "backend": "opencv_dnn_cpu",
            "input": "RGB NCHW 512x512",
            "resize": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "output": "output, argmax of 19 logits",
            "labels": list(FACE_PARSING_LABELS),
            "opencv": cv2.__version__,
        }


@dataclass(frozen=True, eq=False)
class FaceParsing:
    """One face's labels over continuous full-frame integer crop edges.

    bounds may extend outside the video. Each grid cell covers an equal fraction
    of these bounds. Label arrays are copied and made read-only on construction.
    """

    track_id: int
    bounds: tuple[int, int, int, int]
    labels: np.ndarray
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaceParsing):
            return NotImplemented
        return (
            self.track_id == other.track_id
            and self.bounds == other.bounds
            and np.array_equal(self.labels, other.labels)
        )

    def __post_init__(self) -> None:
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        if len(self.bounds) != 4 or any(type(v) is not int for v in self.bounds):
            raise ValueError("Parsing bounds must be four integer edges")
        if self.bounds[2] <= self.bounds[0] or self.bounds[3] <= self.bounds[1]:
            raise ValueError("Parsing bounds must have positive dimensions")
        labels = np.asarray(self.labels)
        if (
            labels.dtype != np.uint8
            or labels.ndim != 2
            or min(labels.shape) < 1
            or max(labels.shape) > 2048
            or np.any(labels >= len(FACE_PARSING_LABELS))
        ):
            raise ValueError("Parsing labels must be a nonempty uint8 grid of class IDs 0..18")
        labels = labels.copy()
        labels.flags.writeable = False
        object.__setattr__(self, "bounds", tuple(self.bounds))
        object.__setattr__(self, "labels", labels)

    def labels_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sample full-frame coordinates, returning background outside crop bounds."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Label sample coordinates must be finite")
        a, b, c, d = self.bounds
        gh, gw = self.labels.shape
        ix = np.floor((x - a) * gw / (c - a)).astype(int)
        iy = np.floor((y - b) * gh / (d - b)).astype(int)
        result = self.labels[np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1)]
        return np.where((x >= a) & (x < c) & (y >= b) & (y < d), result, 0).astype(np.uint8)


@dataclass(frozen=True)
class VideoParsing:
    """Dense sidecar for a complete clip or an explicitly bounded frame interval.

    Detection/track changes invalidate this cache. Frame digests also protect
    direct array rendering from accidental use with different geometry.
    """

    source_sha256: str
    analysis_sha256: str
    video: VideoInfo
    start: int
    frames: tuple[tuple[FaceParsing, ...], ...]
    frame_sha256: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_digest(self.source_sha256)
        _check_digest(self.analysis_sha256)
        if type(self.start) is not int or self.start < 0:
            raise ValueError("Parsing start must be a nonnegative integer")
        if not self.frames or len(self.frames) != len(self.frame_sha256):
            raise ValueError("Parsing must contain matching nonempty frames and frame digests")
        for faces, digest in zip(self.frames, self.frame_sha256, strict=True):
            _check_digest(digest)
            if len({face.track_id for face in faces}) != len(faces):
                raise ValueError("Duplicate parsing track IDs in a frame")
        if not isinstance(self.provenance, dict):
            raise ValueError("Parsing provenance must be a JSON object")
        json.dumps(self.provenance, allow_nan=False)

    def validate_analysis(self, analysis: VideoAnalysis, *, complete: bool = True) -> None:
        if (
            analysis.source_sha256 != self.source_sha256
            or analysis.video != self.video
            or _digest(analysis) != self.analysis_sha256
        ):
            raise ValueError("Parsing cache does not match this source and analysis")
        if complete and (self.start != 0 or len(self.frames) != len(analysis.frames)):
            raise ValueError("Video rendering requires parsing for the complete analysis")

    def for_frame(self, frame: FrameAnalysis, width: int, height: int) -> tuple[FaceParsing, ...]:
        position = frame.index - self.start
        if (width, height) != (self.video.width, self.video.height):
            raise ValueError("Parsing dimensions do not match this frame")
        if not 0 <= position < len(self.frames):
            raise ValueError("Frame is outside the parsed interval")
        if _digest(frame) != self.frame_sha256[position]:
            raise ValueError("Parsing cache does not match this frame's face analysis")
        faces = self.frames[position]
        if {f.track_id for f in faces} != {f.track_id for f in frame.faces if f.observed}:
            raise ValueError("Parsing cache must contain every observed face exactly once")
        return faces

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Atomically publish a compressed NPZ with JSON metadata and no pickle."""
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        arrays = {}
        manifest = []
        for i, faces in enumerate(self.frames):
            entries = []
            for j, face in enumerate(faces):
                key = f"labels_{i}_{j}"
                arrays[key] = face.labels
                entries.append({"track_id": face.track_id, "bounds": face.bounds, "key": key})
            manifest.append(entries)
        metadata = {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "analysis_sha256": self.analysis_sha256,
            "video": asdict(self.video),
            "start": self.start,
            "frames": manifest,
            "frame_sha256": self.frame_sha256,
            "provenance": self.provenance,
            "labels": FACE_PARSING_LABELS,
        }
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                np.savez_compressed(stream, **arrays)
            if overwrite:
                os.replace(name, destination)
            else:
                os.link(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> VideoParsing:
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = json.loads(str(archive["metadata"].item()))
                if (
                    type(data["schema_version"]) is not int
                    or data["schema_version"] != 1
                    or tuple(data["labels"]) != FACE_PARSING_LABELS
                ):
                    raise ValueError("Unsupported parsing schema or label layout")
                frames = tuple(
                    tuple(
                        FaceParsing(face["track_id"], tuple(face["bounds"]), archive[face["key"]])
                        for face in faces
                    )
                    for faces in data["frames"]
                )
            return cls(
                data["source_sha256"],
                data["analysis_sha256"],
                VideoInfo(**data["video"]),
                data["start"],
                frames,
                tuple(data["frame_sha256"]),
                data["provenance"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Anamorph parsing cache: {exc}") from exc


def _parsing_bounds(box: Box, crop_scale: float) -> tuple[int, int, int, int]:
    side = math.ceil(max(box.width, box.height) * crop_scale)
    x1 = math.floor((box.x1 + box.x2 - side) / 2)
    y1 = math.floor((box.y1 + box.y2 - side) / 2)
    return x1, y1, x1 + side, y1 + side


def _parsing_crop(image: Image, bounds: tuple[int, int, int, int]) -> Image:
    x1, y1, x2, y2 = bounds
    sx, sy = (x2 - x1) / 512, (y2 - y1) / 512
    inverse = np.array([[sx, 0, x1 + sx / 2 - 0.5], [0, sy, y1 + sy / 2 - 0.5]])
    return cv2.warpAffine(
        image,
        inverse,
        (512, 512),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def parse_video(
    source: str | Path,
    *,
    analysis: VideoAnalysis,
    parser: FaceParser,
    crop_scale: float = 1.8,
    start: int = 0,
    stop: int | None = None,
    progress: bool = True,
) -> VideoParsing:
    """Parse observed tracks using cached detections, with no repeated detection.

    Square, context-expanded crops preserve geometry across frame-edge padding.
    start/stop select a half-open decoded-frame interval for experiments. Full
    rendering needs a complete cache. Held tracks are handled by mask sessions.
    """
    from .video import VideoError, _frames, _reader

    if not math.isfinite(crop_scale) or crop_scale < 1:
        raise ValueError("crop_scale must be finite and at least one")
    stop = len(analysis.frames) if stop is None else stop
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= len(analysis.frames)
    ):
        raise ValueError("Choose a nonempty start/stop interval inside the analysis")
    if file_sha256(source) != analysis.source_sha256:
        raise VideoError("Analysis source SHA-256 does not match this video")
    frames = []
    with (
        _reader(source) as (capture, info),
        frame_progress("Parsing faces", total=stop - start, enabled=progress) as bar,
    ):
        if info != analysis.video:
            raise VideoError("Video metadata does not match the analysis")
        for index, image in enumerate(_frames(capture, info)):
            if index >= stop:
                break
            if index < start:
                continue
            records = []
            for face in analysis.frames[index].faces:
                if face.observed:
                    bounds = _parsing_bounds(face.box, crop_scale)
                    labels = parser.parse(_parsing_crop(image, bounds))
                    records.append(FaceParsing(face.track_id, bounds, labels))
            frames.append(tuple(records))
            bar.update()
    if len(frames) != stop - start:
        raise VideoError("Video ended before the requested parsing interval")
    if file_sha256(source) != analysis.source_sha256:
        raise VideoError("Source changed during parsing")
    return VideoParsing(
        analysis.source_sha256,
        _digest(analysis),
        analysis.video,
        start,
        tuple(frames),
        tuple(_digest(frame) for frame in analysis.frames[start:stop]),
        {
            "parser": parser.metadata(),
            "crop_scale": crop_scale,
            "crop_sampling": "512 square RGB",
            "padding": "replicate",
            "inference": "observed tracks only",
            "numpy": np.__version__,
        },
    )
