"""Run: uv run python examples/process_video.py local_inputs/clip.mp4 outputs/demo"""

import argparse
from pathlib import Path

from anamorph import (
    Compose,
    FaceMask,
    GaussianBlur,
    IoUTracker,
    Pixelate,
    TemporalBlend,
    analyze_video,
    render_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    analysis = analyze_video(args.source, tracker=IoUTracker(max_missing=5, smoothing=0.7))
    analysis.save(args.output_directory / "analysis.json")
    for name, effect in {
        "blur": GaussianBlur(sigma=16),
        "pixelate": Pixelate(cells=6),
        "ghost": Compose(Pixelate(cells=6), TemporalBlend(current_weight=0.35)),
    }.items():
        report = render_video(
            args.source,
            args.output_directory / f"{name}.mp4",
            analysis=analysis,
            effect=effect,
            mask=FaceMask(shape="ellipse", scale_x=1.35, scale_y=1.6, feather=0.12),
        )
        print(
            f"{report.output}: {report.frame_count} frames, {report.face_instances} face instances"
        )


if __name__ == "__main__":
    main()
