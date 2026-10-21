"""Evaluate mask alpha at independent annotated facial points, without inference.

Run: uv run python examples/evaluate_mask_coverage.py ANALYSIS LANDMARKS_JSONL OUTPUT_JSON

Labels contain one JSON object per frame with frame_index and landmarks (N x 2).
Evaluate a single annotated subject per clip. Coverage is the union of all masks.
This diagnoses exposed annotated points, not full silhouette coverage or anonymity.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anamorph import FaceMask, LandmarkFaceMask, MaskSession, VideoAnalysis
from anamorph.analysis import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("landmarks_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error(f"Output already exists: {args.output_json}")
    analysis = VideoAnalysis.load(args.analysis)
    labels = [json.loads(line) for line in args.landmarks_jsonl.read_text().splitlines()]
    if len(labels) != len(analysis.frames):
        parser.error("Annotations must contain one record per analyzed frame")
    masks = {"legacy": FaceMask(), "full_face": LandmarkFaceMask()}
    samples = {name: [] for name in masks}
    complete = {name: [] for name in masks}
    sessions = {name: MaskSession(mask) for name, mask in masks.items()}
    for frame, label in zip(analysis.frames, labels, strict=True):
        points = np.asarray(label["landmarks"], dtype=float)
        if (
            label["frame_index"] != frame.index
            or points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            parser.error(f"Invalid annotation at frame {frame.index}")
        for name, session in sessions.items():
            regions = session.build(frame, analysis.video.width, analysis.video.height)
            values = np.zeros(len(points), np.float32)
            pixels = np.floor(points).astype(int)
            for region in regions.values():
                x1, y1, x2, y2 = region.bounds
                inside = (
                    (pixels[:, 0] >= x1)
                    & (pixels[:, 0] < x2)
                    & (pixels[:, 1] >= y1)
                    & (pixels[:, 1] < y2)
                )
                x, y = pixels[inside].T
                values[inside] = np.maximum(values[inside], region.alpha[y - y1, x - x1])
            samples[name].extend(values.tolist())
            complete[name].append(bool(np.all(values >= 0.999)))
    report = {
        "source_sha256": analysis.source_sha256,
        "analysis_sha256": file_sha256(args.analysis),
        "annotations_sha256": file_sha256(args.landmarks_jsonl),
        "frames": len(labels),
        "opaque_threshold": 0.999,
        "limitations": (
            "Exploratory point coverage on one annotated subject. Mask union is evaluated. "
            "Annotations do not define full face area, hairline, or privacy. "
            "Tuning on these clips does not establish generalization."
        ),
        "masks": {},
    }
    for name, values in samples.items():
        values = np.asarray(values)
        report["masks"][name] = {
            "configuration": asdict(masks[name]),
            "annotated_points": len(values),
            "opaque_point_fraction": float(np.mean(values >= 0.999)),
            "supported_point_fraction": float(np.mean(values > 0)),
            "frames_with_all_points_opaque": sum(complete[name]),
            "minimum_alpha": float(values.min()),
            "frames_with_exposed_points": np.flatnonzero(~np.array(complete[name])).tolist(),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Evaluated {len(labels)} frames: {args.output_json}")
    for name, score in report["masks"].items():
        print(
            name,
            f"opaque points: {score['opaque_point_fraction']:.4%}",
            f"complete frames: {score['frames_with_all_points_opaque']}/{len(labels)}",
        )


if __name__ == "__main__":
    main()
