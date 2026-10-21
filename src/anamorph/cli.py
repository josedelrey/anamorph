"""Small argparse interface to the library's analysis and render stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import VideoAnalysis
from .detection import HaarFaceDetector, YuNetFaceDetector
from .effects import GaussianBlur, Pixelate
from .masks import FaceMask
from .models import download_yunet_model
from .tracking import IoUTracker
from .video import VideoError, analyze_video, render_video


def _inference_size(value: str) -> int | None:
    if value == "native":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use an integer pixel limit or 'native'") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experimental facial transformations for video")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Detect and track once, save reusable JSON")
    analyze.add_argument("source")
    analyze.add_argument("analysis")
    analyze.add_argument("--detection-interval", type=int, default=1)
    analyze.add_argument("--max-missing", type=int, default=5)
    analyze.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    analyze.add_argument(
        "--model", type=Path, help="Local YuNet ONNX model, required for --detector yunet"
    )
    analyze.add_argument("--score-threshold", type=float, default=0.9)
    analyze.add_argument("--nms-threshold", type=float, default=0.3)
    analyze.add_argument("--top-k", type=int, default=5000)
    analyze.add_argument(
        "--max-size",
        type=_inference_size,
        default=640,
        help="Longest inference edge in pixels or 'native' (default: 640)",
    )
    download = commands.add_parser(
        "download-yunet", help="Download pinned YuNet weights and license"
    )
    download.add_argument("destination", type=Path, help="Destination ONNX file path")
    render = commands.add_parser("render", help="Render a video-only CFR output from analysis")
    render.add_argument("source")
    render.add_argument("analysis")
    render.add_argument("output")
    render.add_argument("--effect", choices=["blur", "pixelate"], default="blur")
    render.add_argument("--sigma", type=float, default=12)
    render.add_argument("--cells", type=int, default=8)
    render.add_argument("--shape", choices=["ellipse", "rectangle"], default="ellipse")
    render.add_argument("--scale-x", type=float, default=1.25)
    render.add_argument("--scale-y", type=float, default=1.5)
    render.add_argument("--offset-y", type=float, default=-0.05)
    render.add_argument("--feather", type=float, default=0.15)
    render.add_argument("--codec", default="mp4v")
    render.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download-yunet":
            print(f"YuNet model ready: {download_yunet_model(args.destination)}")
        elif args.command == "analyze":
            source, cache = Path(args.source), Path(args.analysis)
            if source.resolve() == cache.resolve() or (
                source.exists() and cache.exists() and source.samefile(cache)
            ):
                raise ValueError("Analysis cache must differ from the source video")
            if args.detector == "yunet":
                if args.model is None:
                    raise ValueError("--detector yunet requires --model with a local ONNX path")
                detector = YuNetFaceDetector(
                    args.model,
                    score_threshold=args.score_threshold,
                    nms_threshold=args.nms_threshold,
                    top_k=args.top_k,
                    max_size=args.max_size,
                )
            else:
                if args.model is not None:
                    raise ValueError("--model requires --detector yunet")
                detector = HaarFaceDetector()
            analysis = analyze_video(
                args.source,
                detector=detector,
                detection_interval=args.detection_interval,
                tracker=IoUTracker(max_missing=args.max_missing),
            )
            analysis.save(args.analysis)
            print(f"Saved {len(analysis.frames)} analyzed frames to {args.analysis}")
        else:
            effect = GaussianBlur(args.sigma) if args.effect == "blur" else Pixelate(args.cells)
            report = render_video(
                args.source,
                args.output,
                analysis=VideoAnalysis.load(args.analysis),
                effect=effect,
                mask=FaceMask(args.shape, args.scale_x, args.scale_y, args.offset_y, args.feather),
                codec=args.codec,
                overwrite=args.overwrite,
            )
            print(f"Wrote {report.frame_count} frames to {report.output} (video only)")
    except (OSError, ValueError, VideoError) as exc:
        parser.exit(2, f"anamorph: {exc}\n")
