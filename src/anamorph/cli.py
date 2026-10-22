"""Small argparse interface to the library's analysis and render stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import VideoAnalysis
from .detection import HaarFaceDetector, YuNetFaceDetector
from .effects import GaussianBlur, Pixelate
from .masks import FaceMask, LandmarkFaceMask
from .models import download_face_parser_model, download_yunet_model
from .parsed_masks import ParsedFaceMask
from .parsing import BiSeNetFaceParser, VideoParsing, parse_video
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
    download_parser = commands.add_parser(
        "download-face-parser", help="Download verified BiSeNet weights and license"
    )
    download_parser.add_argument("destination", type=Path)
    parse = commands.add_parser("parse", help="Cache semantic face labels using existing analysis")
    parse.add_argument("source", type=Path)
    parse.add_argument("analysis", type=Path)
    parse.add_argument("parsing", type=Path)
    parse.add_argument("--model", type=Path, required=True)
    parse.add_argument("--crop-scale", type=float, default=1.8)
    parse.add_argument("--start", type=int, default=0)
    parse.add_argument("--stop", type=int, default=None)
    parse.add_argument("--overwrite", action="store_true")
    render = commands.add_parser("render", help="Render a video-only CFR output from analysis")
    render.add_argument("source")
    render.add_argument("analysis")
    render.add_argument("output")
    render.add_argument("--effect", choices=["blur", "pixelate"], default="blur")
    render.add_argument("--sigma", type=float, default=12)
    render.add_argument("--cells", type=int, default=8)
    render.add_argument(
        "--shape", choices=["face", "parsed", "ellipse", "rectangle"], default="face"
    )
    render.add_argument("--parsing", type=Path, help="Semantic cache required by --shape parsed")
    render.add_argument(
        "--fallback",
        choices=["geometry", "error"],
        default="geometry",
        help="Unusable semantic parse policy (default: geometry)",
    )
    render.add_argument("--scale-x", type=float, default=None)
    render.add_argument("--scale-y", type=float, default=None)
    render.add_argument("--offset-y", type=float, default=None)
    render.add_argument(
        "--expansion",
        type=float,
        default=None,
        help="Parsed core growth in source pixels, negative to shrink (default: 0)",
    )
    render.add_argument(
        "--feather",
        type=float,
        default=None,
        help="Radius fraction for geometric masks, source pixels for parsed masks",
    )
    render.add_argument(
        "--chin-extension",
        type=float,
        default=0.10,
        help="Face-mask lower radius extension in [0, 1] (default: 0.10)",
    )
    render.add_argument(
        "--stabilization",
        type=float,
        default=0.65,
        help="Current mask state weight in (0, 1] (default: 0.65)",
    )
    render.add_argument("--codec", default="mp4v")
    render.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download-yunet":
            print(f"YuNet model ready: {download_yunet_model(args.destination)}")
        elif args.command == "download-face-parser":
            print(f"BiSeNet model ready: {download_face_parser_model(args.destination)}")
        elif args.command == "parse":
            for original in (args.source, args.analysis, args.model):
                if args.parsing.resolve() == original.resolve() or (
                    args.parsing.exists() and original.exists() and args.parsing.samefile(original)
                ):
                    raise ValueError("Parsing output must differ from source, analysis, and model")
            if args.parsing.exists() and not args.overwrite:
                raise FileExistsError(args.parsing)
            parsing = parse_video(
                args.source,
                analysis=VideoAnalysis.load(args.analysis),
                parser=BiSeNetFaceParser(args.model),
                crop_scale=args.crop_scale,
                start=args.start,
                stop=args.stop,
            )
            parsing.save(args.parsing, overwrite=args.overwrite)
            print(f"Saved {len(parsing.frames)} parsed frames to {args.parsing}")
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
            mask_options = {
                key: getattr(args, key)
                for key in ("scale_x", "scale_y", "offset_y", "feather")
                if getattr(args, key) is not None
            }
            if args.shape == "parsed":
                if args.parsing is None:
                    raise ValueError("--shape parsed requires --parsing")
                if any(key != "feather" for key in mask_options):
                    raise ValueError("Parsed masks do not use scale-x, scale-y, or offset-y")
                mask = ParsedFaceMask(
                    VideoParsing.load(args.parsing),
                    feather=args.feather if args.feather is not None else 2,
                    stabilization=args.stabilization,
                    fallback=args.fallback,
                    expansion=args.expansion if args.expansion is not None else 0,
                )
            else:
                if args.parsing is not None:
                    raise ValueError("--parsing requires --shape parsed")
                if args.expansion is not None:
                    raise ValueError("--expansion requires --shape parsed")
                mask = (
                    LandmarkFaceMask(
                        **mask_options,
                        stabilization=args.stabilization,
                        chin_extension=args.chin_extension,
                    )
                    if args.shape == "face"
                    else FaceMask(shape=args.shape, **mask_options)
                )
            report = render_video(
                args.source,
                args.output,
                analysis=VideoAnalysis.load(args.analysis),
                effect=effect,
                mask=mask,
                codec=args.codec,
                overwrite=args.overwrite,
            )
            print(f"Wrote {report.frame_count} frames to {report.output} (video only)")
    except (OSError, ValueError, VideoError) as exc:
        parser.exit(2, f"anamorph: {exc}\n")
