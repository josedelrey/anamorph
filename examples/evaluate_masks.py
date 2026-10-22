"""Evaluate local silhouette annotations against cached masks without inference.

Run: uv run python examples/evaluate_masks.py SOURCE ANALYSIS REFERENCES OUTPUT_DIRECTORY
Use --parsing CACHE [CACHE ...] for semantic masks. Disjoint partial caches are supported.
Reference JSON format and measurement limitations are documented in docs/mask_evaluation.md.
"""

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from anamorph import (
    LandmarkFaceMask,
    ParsedFaceMask,
    PerFaceMask,
    VideoAnalysis,
    VideoParsing,
)
from anamorph.analysis import file_sha256
from anamorph.evaluation import evaluate_mask, polygon_mask, temporal_error_change


def reference_arrays(item, width, height):
    face = polygon_mask(item["face"], width, height)
    excluded = {
        name: polygon_mask(polygons, width, height)
        for name, polygons in item.get("excluded", {}).items()
    }
    ignore = polygon_mask(item.get("ignore", []), width, height)
    uncertainty = item.get("uncertainty_pixels", 0)
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("uncertainty_pixels must be finite and nonnegative")
    boundary = face & ~binary_erosion(face, border_value=0)
    if uncertainty:
        ignore |= distance_transform_edt(~boundary) <= uncertainty
    return face, excluded, ignore


def frame_alpha(region, width, height):
    alpha = np.zeros((height, width), np.float32)
    if region is not None:
        a, b, c, d = region.bounds
        alpha[b:d, a:c] = region.alpha
    return alpha


def configured_mask(configuration, parsing=None):
    """Small experiment-file adapter. Python constructors remain the public API."""
    options = dict(configuration)
    kind = options.pop("kind", "parsed" if parsing is not None else "face")
    if kind == "face":
        return LandmarkFaceMask(**options)
    if kind != "parsed" or parsing is None:
        raise ValueError("Choose kind='face', or kind='parsed' with parsing caches")
    if "fallback_mask" in options:
        options["fallback_mask"] = LandmarkFaceMask(**options["fallback_mask"])
    return ParsedFaceMask(parsing, **options)


def experiment_mask(configuration, parsing=None):
    if set(configuration) - {"default", "overrides"}:
        raise ValueError("Configuration accepts default and overrides only")
    overrides = {}
    for key, value in configuration.get("overrides", {}).items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("Override keys must be canonical nonnegative track IDs")
        overrides[int(key)] = configured_mask(value, parsing)
    return PerFaceMask(configured_mask(configuration.get("default", {}), parsing), overrides)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--parsing", type=Path, nargs="+", default=[])
    parser.add_argument("--config", type=Path, help="JSON default mask and per-track overrides")
    args = parser.parse_args()
    analysis = VideoAnalysis.load(args.analysis)
    refs = json.loads(args.references.read_text())
    if refs["schema_version"] != 1:
        parser.error("Unsupported reference schema")
    if refs["source_sha256"] != file_sha256(args.source) or (
        refs["source_sha256"] != analysis.source_sha256
    ):
        parser.error("References and analysis must match the source bytes")
    if refs["analysis_sha256"] != file_sha256(args.analysis):
        parser.error("Reference track assignments belong to different analysis bytes")
    width, height = analysis.video.width, analysis.video.height
    if (refs["width"], refs["height"]) != (width, height):
        parser.error("Reference dimensions differ from analysis")
    selected = {sample["index"]: sample for sample in refs["frames"]}
    if (
        not selected
        or len(selected) != len(refs["frames"])
        or any(type(i) is not int or not 0 <= i < len(analysis.frames) for i in selected)
    ):
        parser.error("Reference frames must be unique valid indices")
    caches = sorted((VideoParsing.load(p) for p in args.parsing), key=lambda c: c.start)
    for cache in caches:
        cache.validate_analysis(analysis, complete=False)
    if any(a.start + len(a.frames) > b.start for a, b in zip(caches, caches[1:], strict=False)):
        parser.error("Parsing cache intervals must not overlap")
    if caches and any(
        not any(c.start <= i < c.start + len(c.frames) for c in caches) for i in selected
    ):
        parser.error("Every annotated frame needs parsing cache coverage")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if any(args.output_directory.iterdir()):
        parser.error("Choose an empty output directory to preserve existing results")
    configuration = json.loads(args.config.read_text()) if args.config else {}
    masks = (
        [experiment_mask(configuration, c) for c in caches]
        if caches
        else [experiment_mask(configuration)]
    )
    sessions = [mask.create_session() for mask in masks]
    rows, changes, previous = [], [], {}
    capture = cv2.VideoCapture(str(args.source))
    try:
        for frame in analysis.frames[: max(selected) + 1]:
            ok, image = capture.read()
            if not ok or image.shape[:2] != (height, width):
                raise RuntimeError(f"Source decode failed at frame {frame.index}")
            position = (
                next(
                    (
                        j
                        for j, c in enumerate(caches)
                        if c.start <= frame.index < c.start + len(c.frames)
                    ),
                    None,
                )
                if caches
                else 0
            )
            if position is None:
                continue
            session = sessions[position]
            regions = session.build(frame, width, height)
            if frame.index not in selected:
                continue
            sample = selected[frame.index]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != sample["rgb_sha256"]:
                raise ValueError("Decoded reference frame digest differs")
            identifiers = [item["reference_id"] for item in sample["faces"]]
            tracks = [item["track_id"] for item in sample["faces"] if item["track_id"] is not None]
            if len(set(identifiers)) != len(identifiers) or len(set(tracks)) != len(tracks):
                raise ValueError(
                    "Face references and assigned tracks must be unique within a frame"
                )
            target, prediction = image.copy(), image.copy()
            for item in sample["faces"]:
                reference, excluded, ignore = reference_arrays(item, width, height)
                alpha = frame_alpha(regions.get(item["track_id"]), width, height)
                status = session.diagnostics.get(item["track_id"], "missing_track")
                result = evaluate_mask(alpha, reference, exclusions=excluded, ignore=ignore)
                rows.append(
                    {
                        "frame": frame.index,
                        "reference_id": item["reference_id"],
                        "track_id": item["track_id"],
                        "tags": item.get("tags", []),
                        "status": status,
                        **result,
                    }
                )
                old = previous.get(item["reference_id"])
                if old is not None and old[0] == frame.index - 1:
                    changes.append(
                        {
                            "frame": frame.index,
                            "reference_id": item["reference_id"],
                            "error_change": temporal_error_change(
                                old[1],
                                old[2],
                                alpha,
                                reference,
                                previous_ignore=old[3],
                                ignore=ignore,
                            ),
                        }
                    )
                previous[item["reference_id"]] = frame.index, alpha, reference, ignore
                target[reference] = np.rint(target[reference] * 0.5 + np.array([0, 255, 0]) * 0.5)
                for mask in excluded.values():
                    target[mask] = np.rint(target[mask] * 0.5 + np.array([255, 0, 255]) * 0.5)
                target[ignore] = np.rint(target[ignore] * 0.6 + 100)
                weight = alpha[..., None] * 0.45
                prediction = np.rint(
                    prediction * (1 - weight) + np.array([0, 180, 255]) * weight
                ).astype(np.uint8)
            for panel, title in [
                (target, "Reference: face green, exclusions purple, uncertainty grey"),
                (prediction, f"Mask prediction | frame {frame.index}"),
            ]:
                cv2.rectangle(panel, (0, 0), (width, 30), (0, 0, 0), -1)
                cv2.putText(panel, title, (6, 21), 0, 0.45, (255, 255, 255), 1)
            cv2.imwrite(
                str(args.output_directory / f"frame_{frame.index:05d}.jpg"),
                np.concatenate([target, prediction], axis=1),
            )
    finally:
        capture.release()
    if file_sha256(args.source) != analysis.source_sha256:
        raise ValueError("Source changed during evaluation")
    face_area = sum(row["face_pixels"] for row in rows)
    report = {
        "source_sha256": analysis.source_sha256,
        "analysis_sha256": file_sha256(args.analysis),
        "references_sha256": file_sha256(args.references),
        "annotation_provenance": refs["annotation_provenance"],
        "parsing_sha256": [file_sha256(p) for p in args.parsing],
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "mask_implementation_sha256": {
            cls.__name__: file_sha256(inspect.getfile(cls))
            for cls in (LandmarkFaceMask, ParsedFaceMask, PerFaceMask)
        },
        "mask_configuration": configuration,
        "default_kind": "parsed" if caches else "face",
        "face_samples": len(rows),
        "area_weighted_missed_opaque_face_fraction": (
            sum(r["missed_opaque_face_pixels"] for r in rows) / face_area
        ),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "samples": rows,
        "adjacent_frame_error_changes": changes,
        "limitations": "Development annotations are not expert ground truth. "
        "Boundary distances include uncertainty. Error change is a flicker proxy. "
        "Unannotated faces do not contribute. No anonymity guarantee is measured.",
    }
    (args.output_directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("samples", "adjacent_frame_error_changes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
