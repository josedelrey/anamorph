"""Face mattes for editing software, independent of image effects."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ._progress import frame_progress
from .analysis import VideoAnalysis, file_sha256
from .masks import FaceMask, LandmarkFaceMask, MaskProvider, MaskSession
from .video import RenderReport, VideoError, VideoExportWarning, _frames, _reader, _verify_export


def _alpha_frame(session: MaskSession, analysis: VideoAnalysis, index: int) -> np.ndarray:
    info = analysis.video
    alpha = np.zeros((info.height, info.width), dtype=np.float32)
    for region in session.build(analysis.frames[index], info.width, info.height).values():
        x1, y1, x2, y2 = region.bounds
        crop = alpha[y1:y2, x1:x2]
        np.maximum(crop, region.alpha, out=crop)
    return np.rint(alpha * 65535).astype(np.uint16)


def export_masks(
    source: str | Path,
    output: str | Path,
    *,
    analysis: VideoAnalysis,
    mask: FaceMask | LandmarkFaceMask | MaskProvider | None = None,
    format: Literal["mov", "png"] = "mov",
    overwrite: bool = False,
    progress: bool = True,
) -> RenderReport:
    """Export white faces on black, with gray feathering and maximum overlap.

    The default is a video-only CFR ProRes 4444 MOV with 10-bit luminance,
    requiring ffmpeg with prores_ks on PATH. It uses video levels and has no
    alpha channel. PNG writes a lossless 16-bit grayscale sequence and timing
    metadata into a new directory. Existing PNG directories are never replaced.
    Both formats retain every source frame and reuse the existing mask sessions.
    Outputs are staged and verified before publication. No inference is run.
    """
    source, output = Path(source), Path(output)
    if format not in ("mov", "png"):
        raise ValueError("format must be mov or png")
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError("Output must differ from the source video")
    if output.exists() and (format == "png" or not overwrite):
        raise FileExistsError(f"Destination already exists: {output}")
    if format == "mov" and output.suffix.lower() != ".mov":
        raise ValueError("MOV mask export requires a .mov output path")
    executable = shutil.which("ffmpeg") if format == "mov" else None
    if format == "mov" and executable is None:
        raise VideoError(
            "MOV mask export requires ffmpeg with the prores_ks encoder on PATH. "
            "Install FFmpeg or use --format png."
        )
    if file_sha256(source) != analysis.source_sha256:
        raise VideoError("Analysis source SHA-256 does not match this video")
    if mask is not None and not isinstance(mask, (FaceMask, LandmarkFaceMask)):
        mask.validate_analysis(analysis)
    warnings.warn(
        "Mask export uses constant frame rate, starting at source frame zero. "
        "Audio, source timestamps, and container metadata are not copied.",
        VideoExportWarning,
        stacklevel=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    session = MaskSession(mask)
    fps = Fraction(analysis.video.fps).limit_denominator(1_000_000)
    count = 0
    face_instances = 0
    # Staging beside the destination allows publication on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as staging:
        temporary = Path(staging) / ("matte.mov" if format == "mov" else "sequence")
        process = None
        with tempfile.TemporaryFile() as errors:
            try:
                if format == "mov":
                    info = analysis.video
                    process = subprocess.Popen(
                        [
                            executable,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "gray16le",
                            "-video_size",
                            f"{info.width}x{info.height}",
                            "-framerate",
                            str(fps),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "prores_ks",
                            "-profile:v",
                            "4444",
                            "-pix_fmt",
                            "yuv444p10le",
                            "-vf",
                            "scale=in_range=full:out_range=tv",
                            "-color_range",
                            "tv",
                            "-movflags",
                            "+faststart",
                            "-f",
                            "mov",
                            str(temporary),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=errors,
                    )
                else:
                    temporary.mkdir()
                with (
                    _reader(source) as (capture, info),
                    frame_progress(
                        "Exporting masks", total=len(analysis.frames), enabled=progress
                    ) as bar,
                ):
                    if info != analysis.video:
                        raise VideoError("Video metadata does not match the analysis")
                    for index, _ in enumerate(_frames(capture, info)):
                        if index >= len(analysis.frames):
                            raise VideoError("Video contains more frames than its analysis")
                        image = _alpha_frame(session, analysis, index)
                        if process is not None:
                            process.stdin.write(image.astype("<u2", copy=False).tobytes())
                        else:
                            path = temporary / f"mask_{index:08d}.png"
                            if not cv2.imwrite(str(path), image):
                                raise VideoError(f"Cannot encode PNG mask frame {index}")
                            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                            if (
                                decoded is None
                                or decoded.dtype != image.dtype
                                or not np.array_equal(decoded, image)
                            ):
                                raise VideoError(f"PNG mask verification failed at frame {index}")
                        count += 1
                        face_instances += len(analysis.frames[index].faces)
                        bar.update()
                    if count != len(analysis.frames):
                        raise VideoError("Video contains fewer frames than its analysis")
                if process is not None:
                    process.stdin.close()
                    if process.wait() != 0:
                        errors.seek(0)
                        detail = errors.read().decode(errors="replace")
                        raise VideoError(f"FFmpeg mask encoding failed: {detail}")
                    _verify_export(temporary, analysis.video, count, progress=progress)
                else:
                    metadata = {
                        "format": "anamorph_mask_sequence",
                        "schema_version": 1,
                        "pattern": "mask_%08d.png",
                        "start_frame": 0,
                        "frame_count": count,
                        "width": info.width,
                        "height": info.height,
                        "fps": info.fps,
                        "fps_numerator": fps.numerator,
                        "fps_denominator": fps.denominator,
                        "bit_depth": 16,
                        "black": 0,
                        "white": 65535,
                        "overlap": "maximum",
                        "source_sha256": analysis.source_sha256,
                    }
                    (temporary / "sequence.json").write_text(
                        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                    )
                if file_sha256(source) != analysis.source_sha256:
                    raise VideoError("Source changed during mask export")
                if format == "png":
                    if output.exists():
                        raise FileExistsError(output)
                    temporary.rename(output)
                elif overwrite:
                    os.replace(temporary, output)
                else:
                    os.link(temporary, output)
            except BrokenPipeError as exc:
                # FFmpeg can exit early when an encoder is missing or rejects input.
                if process is not None:
                    process.wait()
                errors.seek(0)
                detail = errors.read().decode(errors="replace")
                raise VideoError(f"FFmpeg mask encoding failed: {detail}") from exc
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    if process.stdin is not None and not process.stdin.closed:
                        try:
                            process.stdin.close()
                        except BrokenPipeError:
                            pass
    return RenderReport(
        output, count, face_instances, "prores_ks" if format == "mov" else "png16", analysis.video
    )
