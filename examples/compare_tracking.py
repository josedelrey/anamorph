"""Compare static holds and optical-flow tracking using cached observations.

Run: uv run python examples/compare_tracking.py SOURCE ANALYSIS OUTPUT_DIRECTORY
Optional --annotations accepts normalized ChokePoint-style JSONL box annotations.
The translucent preview reveals original faces and is not anonymized.
"""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from anamorph import Box, IoUTracker, MaskSession, SparseOpticalFlow, VideoAnalysis, track_video
from anamorph.analysis import file_sha256


def evaluate_boxes(analysis, annotations, threshold=0.5):
    """One-to-one matching against positive reference boxes, never precision."""
    if len(annotations) != len(analysis.frames):
        raise ValueError("Annotations must cover every analysis frame")
    records, previous, missed, longest = [], {}, {}, 0
    switches = pairs = 0
    for frame, truth in zip(analysis.frames, annotations, strict=True):
        if truth["frame_index"] != frame.index:
            raise ValueError("Annotation frame indices differ from analysis")
        labels = [region for region in truth["regions"] if region["face_flag"] == 1]
        if len({r["track_id"] for r in labels}) != len(labels):
            raise ValueError("Duplicate reference identities in one frame")
        matched = {}
        if labels and frame.faces:
            boxes = [Box(x, y, x + w, y + h) for x, y, w, h in (r["box_xywh"] for r in labels)]
            iou = np.array([[box.iou(face.box) for face in frame.faces] for box in boxes])
            weights = np.where(iou >= threshold, min(iou.shape) + 1 + iou, 0)
            rows, cols = linear_sum_assignment(weights, maximize=True)
            matched = {
                int(r): (frame.faces[c], float(iou[r, c]))
                for r, c in zip(rows, cols, strict=True)
                if weights[r, c] > 0
            }
        new_previous, new_missed = {}, {}
        for i, label in enumerate(labels):
            identity = label["track_id"]
            found = matched.get(i)
            old = previous.get(identity)
            if found:
                face, score = found
                if old is not None:
                    pairs += 1
                    switches += face.track_id != old
                new_previous[identity] = face.track_id
                new_missed[identity] = 0
            else:
                new_missed[identity] = missed.get(identity, 0) + 1
                longest = max(longest, new_missed[identity])
            records.append(
                {
                    "frame": frame.index,
                    "reference_id": identity,
                    "track_id": found[0].track_id if found else None,
                    "iou": found[1] if found else None,
                    "observed": found[0].observed if found else None,
                }
            )
        previous, missed = new_previous, new_missed
    matched_count = sum(r["track_id"] is not None for r in records)
    return {
        "reference_boxes": len(records),
        "matched_boxes": matched_count,
        "box_recall": matched_count / len(records) if records else None,
        "adjacent_matched_identity_pairs": pairs,
        "adjacent_id_switches": int(switches),
        "longest_annotated_miss_run": longest,
        "samples": records,
    }


def diagnostics(analysis):
    return {
        "frames": len(analysis.frames),
        "frames_without_detections": sum(not f.detections for f in analysis.frames),
        "frames_without_tracks": sum(not f.faces for f in analysis.frames),
        "distinct_track_ids": len({face.track_id for f in analysis.frames for face in f.faces}),
        "held_face_appearances": sum(
            not face.observed for f in analysis.frames for face in f.faces
        ),
        "motion_status_counts": dict(
            Counter(
                status
                for row in analysis.provenance.get("motion_diagnostics", [])
                for status in row["tracks"].values()
            )
        ),
        "scene_cuts": analysis.provenance.get("scene_cuts", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=None,
        help="Simulate skipping cached observations, without inference",
    )
    parser.add_argument("--preview-start", type=int, default=0)
    parser.add_argument("--preview-frames", type=int, default=750)
    args = parser.parse_args()
    analysis = VideoAnalysis.load(args.analysis)
    if file_sha256(args.source) != analysis.source_sha256:
        parser.error("Source digest differs from cached analysis")
    if args.detection_interval is not None:
        if args.detection_interval < 1:
            parser.error("detection-interval must be positive")
        analysis = replace(
            analysis,
            frames=tuple(
                replace(
                    f, detections=(f.detections if f.index % args.detection_interval == 0 else None)
                )
                for f in analysis.frames
            ),
            provenance=analysis.provenance
            | {
                "detection_interval": args.detection_interval,
                "experiment": "controlled observation skipping",
            },
        )
    if not 0 <= args.preview_start < len(analysis.frames) or args.preview_frames < 1:
        parser.error("Choose a valid preview interval")
    if any(v % 2 for v in (analysis.video.width, analysis.video.height)):
        parser.error("MP4 preview requires even frame dimensions")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if any(args.output_directory.iterdir()):
        parser.error("Choose a new output directory to preserve earlier experiments")
    static = analysis.retrack(tracker=IoUTracker())
    motion = track_video(args.source, analysis=analysis, motion=SparseOpticalFlow())
    static.save(args.output_directory / "static.json")
    motion.save(args.output_directory / "motion.json")
    report = {
        "source_sha256": analysis.source_sha256,
        "input_analysis_sha256": file_sha256(args.analysis),
        "experiment_detection_interval": args.detection_interval,
        "motion": motion.provenance["motion"],
        "static": diagnostics(static),
        "flow": diagnostics(motion),
        "limitations": "Box annotations can be incomplete. No precision, exhaustive face "
        "recall, silhouette accuracy, or anonymity claim is justified. Counts without "
        "annotations are diagnostics. Controlled skips are a separate experiment.",
    }
    if args.annotations:
        annotations = [json.loads(line) for line in args.annotations.read_text().splitlines()]
        report["annotations_sha256"] = file_sha256(args.annotations)
        report["static"]["box_evaluation"] = evaluate_boxes(static, annotations)
        report["flow"]["box_evaluation"] = evaluate_boxes(motion, annotations)
    width, height = analysis.video.width, analysis.video.height
    capture = cv2.VideoCapture(str(args.source))
    destination = args.output_directory / "tracking_comparison.mp4"
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), analysis.video.fps, (2 * width, height)
    )
    sessions = [MaskSession(), MaskSession()]
    stop = min(len(analysis.frames), args.preview_start + args.preview_frames)
    count = 0
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("Cannot open preview input or encoder")
        for i in range(stop):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("Source decode failed")
            panels = []
            for j, variant in enumerate((static, motion)):
                frame = variant.frames[i]
                regions = sessions[j].build(frame, width, height)
                panel = image.copy()
                for face in frame.faces:
                    region = regions.get(face.track_id)
                    if region is None:
                        continue
                    a, b, c, d = region.bounds
                    weight = region.alpha[..., None] * 0.4
                    tint = np.array([0, 180, 255] if face.observed else [255, 200, 0])
                    panel[b:d, a:c] = np.rint(panel[b:d, a:c] * (1 - weight) + tint * weight)
                    cv2.putText(
                        panel,
                        f"{face.track_id} {'obs' if face.observed else 'held'}",
                        (a, max(40, b)),
                        0,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                cv2.rectangle(panel, (0, 0), (width, 30), (0, 0, 0), -1)
                cv2.putText(
                    panel,
                    f"{'Static' if j == 0 else 'Flow'} | frame {i}",
                    (6, 21),
                    0,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                panels.append(panel)
            if i < args.preview_start:
                continue
            result = np.concatenate(panels, axis=1)
            writer.write(result)
            if i in (args.preview_start, (args.preview_start + stop) // 2, stop - 1):
                cv2.imwrite(str(args.output_directory / f"frame_{i:05d}.jpg"), result)
            count += 1
    finally:
        capture.release()
        writer.release()
    check = cv2.VideoCapture(str(destination))
    decoded = 0
    try:
        while True:
            ok, frame = check.read()
            if not ok:
                break
            if frame.shape[:2] != (height, 2 * width):
                raise RuntimeError("Preview dimensions changed")
            decoded += 1
    finally:
        check.release()
    if decoded != count or count != stop - args.preview_start:
        raise RuntimeError("Preview frame count mismatch")
    report["preview_frames_verified"] = decoded
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {decoded} comparison frames in {args.output_directory}")


if __name__ == "__main__":
    main()
