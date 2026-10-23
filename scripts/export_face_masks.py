"""Prepare a video, detect and parse faces, then export masks from YAML settings."""

from __future__ import annotations

import argparse
import inspect
import math
import shutil
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from anamorph import (
    FACE_PARSING_LABELS,
    BiSeNetFaceParser,
    IoUTracker,
    LandmarkFaceMask,
    ParsedFaceMask,
    YuNetFaceDetector,
    analyze_video,
    export_masks,
    parse_video,
)
from anamorph.video import VideoError

_SECTIONS = {"runtime", "prepare", "detector", "tracker", "parsing", "mask", "output"}
_KEYS = {
    "runtime": {"progress", "opencv_threads"},
    "prepare": {
        "enabled",
        "filter",
        "fps",
        "video_codec",
        "profile",
        "pixel_format",
        "audio_codec",
    },
    "detector": (set(inspect.signature(YuNetFaceDetector).parameters) - {"model_path"})
    | {"model", "detection_interval"},
    "tracker": set(inspect.signature(IoUTracker).parameters),
    "parsing": {"model", "crop_scale"},
    "mask": {item.name for item in fields(ParsedFaceMask)} - {"parsing"},
    "output": {"directory", "format", "overwrite"},
}


def _mapping(value: object, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
    return dict(value)


def _path(value: object, base: Path, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return str((base / path).resolve())


def load_config(path: Path) -> dict:
    """Load safe YAML, reject unknown keys, and resolve paths beside the YAML."""
    config = _mapping(yaml.safe_load(path.read_text()), _SECTIONS | {"version"}, "config")
    if type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("config.version must be 1")
    for section in sorted(_SECTIONS):
        config[section] = _mapping(config.get(section), _KEYS[section], section)
    for section in ("detector", "parsing"):
        config[section]["model"] = _path(
            config[section].get("model"), path.resolve().parent, f"{section}.model"
        )
    output = config["output"]
    output["directory"] = _path(output.get("directory"), path.resolve().parent, "output.directory")
    if output.get("format") not in ("mov", "png"):
        raise ValueError("output.format must be mov or png")
    for section, key in (("runtime", "progress"), ("prepare", "enabled"), ("output", "overwrite")):
        if type(config[section].get(key)) is not bool:
            raise ValueError(f"{section}.{key} must be true or false")
    threads = config["runtime"].get("opencv_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("runtime.opencv_threads must be a positive integer")
    if config["prepare"]["enabled"]:
        prepare = config["prepare"]
        missing = _KEYS["prepare"] - prepare.keys()
        if missing:
            raise ValueError(f"Missing prepare settings: {', '.join(sorted(missing))}")
        if (
            type(prepare["fps"]) not in (int, float)
            or not math.isfinite(prepare["fps"])
            or prepare["fps"] <= 0
        ):
            raise ValueError("prepare.fps must be finite and positive")
        for key in ("filter", "video_codec", "profile", "pixel_format", "audio_codec"):
            if not isinstance(prepare[key], (str, int)) or isinstance(prepare[key], bool):
                raise ValueError(f"prepare.{key} must be a string or integer")
    return config


def prepare_video(source: Path, output: Path, options: dict, *, progress: bool) -> None:
    """Stage an FFmpeg conversion, displaying decoded-frame progress."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("Video preparation requires ffmpeg on PATH")
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        temporary = Path(staging) / "footage.mov"
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            str(options["filter"]),
            "-r",
            str(options["fps"]),
            "-fps_mode",
            "cfr",
            "-c:v",
            str(options["video_codec"]),
            "-profile:v",
            str(options["profile"]),
            "-pix_fmt",
            str(options["pixel_format"]),
            "-c:a",
            str(options["audio_codec"]),
            "-progress",
            "pipe:1",
            "-nostats",
            str(temporary),
        ]
        with (
            tempfile.TemporaryFile(mode="w+t") as errors,
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True) as process,
            tqdm(desc="Preparing video", unit="frame", disable=not progress) as bar,
        ):
            try:
                for line in process.stdout:
                    if line.startswith("frame="):
                        count = int(line.partition("=")[2])
                        bar.update(max(0, count - bar.n))
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            if code:
                errors.seek(0)
                raise VideoError(f"FFmpeg preparation failed: {errors.read()[-4000:].strip()}")
        temporary.replace(output)


def run(source: Path, config_path: Path, output_dir: Path | None = None):
    """Run all stages with fresh analysis and parsing, retaining the caches."""
    source, config_path = source.expanduser().resolve(), config_path.expanduser().resolve()
    config = load_config(config_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output = config["output"]
    directory = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else Path(output["directory"]) / source.stem
    )
    prepared = directory / "footage.mov"
    destination = directory / ("face_masks.mov" if output["format"] == "mov" else "face_masks")
    products = [directory / name for name in ("analysis.json", "parsing.npz", "config.yaml")]
    products.append(destination)
    if config["prepare"]["enabled"]:
        products.append(prepared)
    originals = [
        source,
        config_path,
        *(Path(config[key]["model"]) for key in ("detector", "parsing")),
    ]
    for product in products:
        for original in originals:
            if product.resolve() == original or (
                product.exists() and original.exists() and product.samefile(original)
            ):
                raise ValueError(f"Output would replace an input or model: {product}")
        if product.exists() and (
            not output["overwrite"] or (product == destination and output["format"] == "png")
        ):
            raise FileExistsError(
                f"Destination already exists: {product}. Choose another output directory"
            )
    if (config["prepare"]["enabled"] or output["format"] == "mov") and not shutil.which("ffmpeg"):
        raise VideoError("This configuration requires ffmpeg on PATH")
    # Load both models before writing outputs so missing assets fail early.
    detector_options = dict(config["detector"])
    detector_model = detector_options.pop("model")
    detection_interval = detector_options.pop("detection_interval", 1)
    detector = YuNetFaceDetector(detector_model, **detector_options)
    face_parser = BiSeNetFaceParser(config["parsing"]["model"])
    tracker = IoUTracker(**config["tracker"])
    mask_options = dict(config["mask"])
    if "fallback_mask" in mask_options:
        mask_options["fallback_mask"] = LandmarkFaceMask(**mask_options["fallback_mask"])
    if mask_options.get("boundary_labels") == "all":
        included = mask_options.get("included_labels", ParsedFaceMask.included_labels)
        mask_options["boundary_labels"] = tuple(
            label for label in FACE_PARSING_LABELS if label not in included
        )
    progress = config["runtime"]["progress"]
    cv2.setNumThreads(config["runtime"]["opencv_threads"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    footage = source
    if config["prepare"]["enabled"]:
        prepare_video(source, prepared, config["prepare"], progress=progress)
        footage = prepared
    analysis = analyze_video(
        footage,
        detector=detector,
        tracker=tracker,
        detection_interval=detection_interval,
        progress=progress,
    )
    analysis.save(directory / "analysis.json")
    parsing = parse_video(
        footage,
        analysis=analysis,
        parser=face_parser,
        crop_scale=config["parsing"].get("crop_scale", 1.8),
        progress=progress,
    )
    parsing.save(directory / "parsing.npz", overwrite=output["overwrite"])
    return export_masks(
        footage,
        destination,
        analysis=analysis,
        mask=ParsedFaceMask(parsing, **mask_options),
        format=output["format"],
        overwrite=output["overwrite"],
        progress=progress,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Input video, supplied separately from the preset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/face_masks.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the YAML output directory for this run"
    )
    args = parser.parse_args(argv)
    try:
        report = run(args.source, args.config, args.output_dir)
    except (OSError, ValueError, TypeError, yaml.YAMLError, VideoError) as exc:
        parser.exit(2, f"export_face_masks: {exc}\n")
    print(f"Wrote {report.frame_count} mask frames to {report.output}")


if __name__ == "__main__":
    main()
