"""Two-pass constant-frame-rate video analysis and verified, video-only export."""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from .analysis import VideoAnalysis, VideoInfo, file_sha256
from .compositing import FrameRenderer
from .detection import FaceDetector, HaarFaceDetector
from .effects import Effect
from .masks import FaceMask, LandmarkFaceMask
from .tracking import IoUTracker
from .types import FrameAnalysis, Image


class VideoError(RuntimeError):
    """A video could not be decoded, encoded, or matched to its analysis."""


class VideoExportWarning(UserWarning):
    """Export is video-only and normalizes timing to constant frame rate."""


@dataclass(frozen=True)
class RenderReport:
    output: Path
    frame_count: int
    face_instances: int
    codec: str
    video: VideoInfo


@contextmanager
def _reader(path: str | Path) -> Iterator[tuple[cv2.VideoCapture, VideoInfo]]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(f"Cannot open video: {path}")
        try:
            info = VideoInfo(
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                capture.get(cv2.CAP_PROP_FPS),
            )
        except (ValueError, OverflowError) as exc:
            raise VideoError(f"Invalid video dimensions or frame rate: {path}") from exc
        yield capture, info
    finally:
        capture.release()


def _frames(capture: cv2.VideoCapture, info: VideoInfo) -> Iterator[Image]:
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if bgr.shape != (info.height, info.width, 3):
            raise VideoError("Video resolution or channel count changed during decoding")
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze_video(
    source: str | Path,
    *,
    detector: FaceDetector | None = None,
    tracker: IoUTracker | None = None,
    detection_interval: int = 1,
) -> VideoAnalysis:
    """Decode once, detect, associate faces, and return reusable geometry.

    detection_interval is measured in decoded frames. Held boxes remain static
    during skipped or missed detections. The supplied tracker is reset on entry.
    Source bytes must remain unchanged throughout analysis and future renders.
    """
    if type(detection_interval) is not int or detection_interval < 1:
        raise ValueError("detection_interval must be a positive integer")
    detector = detector if detector is not None else HaarFaceDetector()
    tracker = tracker if tracker is not None else IoUTracker()
    if tracker.max_missing < detection_interval - 1:
        raise ValueError("max_missing must cover detection_interval - 1 skipped frames")
    source_hash = file_sha256(source)
    tracker.reset()
    records = []
    with _reader(source) as (capture, info):
        for index, image in enumerate(_frames(capture, info)):
            detections = detector.detect(image) if index % detection_interval == 0 else None
            records.append(
                FrameAnalysis(
                    index,
                    tracker.update(detections or []),
                    None if detections is None else tuple(detections),
                )
            )
    if not records:
        raise VideoError("Video contains no decodable frames")
    if file_sha256(source) != source_hash:
        raise VideoError("Source changed during analysis")
    return VideoAnalysis(
        source_hash,
        info,
        tuple(records),
        {
            "anamorph": version("anamorph"),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "detector": detector.metadata(),
            "tracker": tracker.metadata(),
            "detection_interval": detection_interval,
            "timing": "constant_frame_rate",
            "coordinates": "full_frame_pixel_edges",
        },
    )


def _verify_export(path: Path, expected: VideoInfo, count: int) -> None:
    with _reader(path) as (capture, actual):
        if (actual.width, actual.height) != (expected.width, expected.height):
            raise VideoError("Encoder changed video dimensions")
        if not math.isclose(actual.fps, expected.fps, rel_tol=1e-3, abs_tol=1e-3):
            raise VideoError("Encoder changed video frame rate")
        decoded = sum(1 for _ in _frames(capture, actual))
        if decoded != count:
            raise VideoError(f"Encoded frame count mismatch: expected {count}, decoded {decoded}")


def render_video(
    source: str | Path,
    output: str | Path,
    *,
    analysis: VideoAnalysis,
    effect: Effect,
    mask: FaceMask | LandmarkFaceMask | None = None,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> RenderReport:
    """Render without inference, verify encoding, then publish the output.

    Output has constant FPS and no audio, subtitles, or container metadata.
    Existing files require overwrite=True. Errors leave destinations unchanged.
    Odd dimensions are rejected to avoid OpenCV silently truncating pixels.
    Codec/container availability depends on the installed OpenCV backend.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError("Output must differ from the source video")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not output.suffix:
        raise ValueError("Output needs a video container extension, such as .mp4 or .avi")
    if len(codec) != 4 or not codec.isascii() or not codec.isprintable():
        raise ValueError("codec must be a printable four-character ASCII code")
    if analysis.video.width % 2 or analysis.video.height % 2:
        raise ValueError("Video export currently requires even width and height")
    if file_sha256(source) != analysis.source_sha256:
        raise VideoError("Analysis source SHA-256 does not match this video")
    warnings.warn(
        "Export uses constant frame rate and contains video only. Audio, subtitles, "
        "and source metadata are not copied.",
        VideoExportWarning,
        stacklevel=2,
    )
    renderer = FrameRenderer(effect, mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=output.suffix, dir=output.parent
    )
    os.close(handle)
    temporary = Path(name)
    writer = None
    count, face_instances = 0, 0
    try:
        with _reader(source) as (capture, info):
            if info != analysis.video:
                raise VideoError("Video metadata does not match the analysis")
            writer = cv2.VideoWriter(
                str(temporary), cv2.VideoWriter_fourcc(*codec), info.fps, (info.width, info.height)
            )
            if not writer.isOpened():
                raise VideoError(f"Cannot initialize encoder {codec!r} for {output.suffix}")
            for index, image in enumerate(_frames(capture, info)):
                if index >= len(analysis.frames):
                    raise VideoError("Video contains more frames than its analysis")
                record = analysis.frames[index]
                result = renderer.render(image, record, timestamp=index / info.fps)
                writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                count += 1
                face_instances += len(record.faces)
            if count != len(analysis.frames):
                raise VideoError("Video contains fewer frames than its analysis")
        writer.release()
        writer = None
        _verify_export(temporary, analysis.video, count)
        if file_sha256(source) != analysis.source_sha256:
            raise VideoError("Source changed during rendering")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Same-directory hard link publishes without a check-then-replace race.
            os.link(temporary, output)
        return RenderReport(output, count, face_instances, codec, analysis.video)
    finally:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)


def process_video(
    source: str | Path,
    output: str | Path,
    *,
    effect: Effect,
    mask: FaceMask | LandmarkFaceMask | None = None,
    detector: FaceDetector | None = None,
    tracker: IoUTracker | None = None,
    detection_interval: int = 1,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> RenderReport:
    """Convenience two-pass analysis and rendering. Use separate calls for reuse."""
    analysis = analyze_video(
        source, detector=detector, tracker=tracker, detection_interval=detection_interval
    )
    return render_video(
        source,
        output,
        analysis=analysis,
        effect=effect,
        mask=mask,
        codec=codec,
        overwrite=overwrite,
    )
