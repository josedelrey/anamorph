"""Compare mask overlays without inference. These previews are NOT anonymized.

Run: uv run python examples/inspect_masks.py SOURCE ANALYSIS OUTPUT_DIRECTORY
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from anamorph import FaceMask, LandmarkFaceMask, MaskSession, VideoAnalysis
from anamorph.analysis import file_sha256


def overlay(image, regions, title):
    result = image.copy()
    for region in regions.values():
        x1, y1, x2, y2 = region.bounds
        roi = result[y1:y2, x1:x2]
        weight = region.alpha[..., None] * 0.45
        roi[:] = np.rint(roi * (1 - weight) + np.array([255, 210, 0]) * weight)
        # Cyan outlines show opaque protection, white outlines show feather support.
        for threshold, color in [(0.01, (255, 255, 255)), (0.999, (255, 255, 0))]:
            contours, _ = cv2.findContours(
                (region.alpha >= threshold).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(roi, contours, -1, color, 1)
    cv2.rectangle(result, (0, 0), (result.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(result, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--scale-x", type=float, default=LandmarkFaceMask.scale_x)
    parser.add_argument("--scale-y", type=float, default=1.10)
    parser.add_argument("--offset-y", type=float, default=0)
    parser.add_argument("--feather", type=float, default=0.08)
    parser.add_argument("--stabilization", type=float, default=0.65)
    parser.add_argument("--chin-extension", type=float, default=0.10)
    args = parser.parse_args()
    analysis = VideoAnalysis.load(args.analysis)
    if file_sha256(args.source) != analysis.source_sha256:
        parser.error("Source digest differs from analysis")
    if not 0 <= args.start < len(analysis.frames) or args.frames <= 0:
        parser.error("Choose a valid start frame and positive frame count")
    width, height = analysis.video.width, analysis.video.height
    if width % 2 or height % 2:
        parser.error("MP4 previews require even dimensions")
    mask = LandmarkFaceMask(
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        offset_y=args.offset_y,
        feather=args.feather,
        stabilization=args.stabilization,
        chin_extension=args.chin_extension,
    )
    sessions = [MaskSession(FaceMask()), MaskSession(mask)]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if any(args.output_directory.iterdir()):
        parser.error("Choose a new or empty output directory to preserve existing files")
    destination = args.output_directory / "mask_comparison.mp4"
    capture = cv2.VideoCapture(str(args.source))
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), analysis.video.fps, (2 * width, height)
    )
    count = 0
    feature_values = []
    guided = fallback = 0
    offscreen = 0
    stop = min(args.start + args.frames, len(analysis.frames))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("Could not open source or preview encoder")
        # Sequential decoding avoids imprecise seeking and warms mask filter history.
        for frame in analysis.frames[:stop]:
            success, image = capture.read()
            if not success or image.shape[:2] != (height, width):
                raise RuntimeError(f"Source decode failed at frame {frame.index}")
            regions = [session.build(frame, width, height) for session in sessions]
            if frame.index < args.start:
                continue
            for face in frame.faces:
                geometry = mask.geometry(face.box, face.landmarks)
                guided += geometry.landmark_guided
                fallback += not geometry.landmark_guided
                region = regions[1].get(face.track_id)
                if face.observed and face.landmarks is not None:
                    for x, y in face.landmarks.points:
                        if not (0 <= x < width and 0 <= y < height):
                            offscreen += 1
                            continue
                        value = 0.0
                        if region is not None:
                            x1, y1, x2, y2 = region.bounds
                            ix, iy = int(np.floor(x)), int(np.floor(y))
                            if x1 <= ix < x2 and y1 <= iy < y2:
                                value = float(region.alpha[iy - y1, ix - x1])
                        feature_values.append(value)
            panels = [
                overlay(image, regions[0], f"Legacy ellipse | frame {frame.index}"),
                overlay(image, regions[1], "Full face | cyan: opaque, white: feather"),
            ]
            preview = np.concatenate(panels, axis=1)
            writer.write(preview)
            if count in (0, (stop - args.start) // 2, stop - args.start - 1):
                cv2.imwrite(str(args.output_directory / f"frame_{frame.index:05d}.jpg"), preview)
            count += 1
    finally:
        capture.release()
        writer.release()
    check = cv2.VideoCapture(str(destination))
    decoded = 0
    try:
        while True:
            success, image = check.read()
            if not success:
                break
            if image.shape[:2] != (height, 2 * width):
                raise RuntimeError("Preview encoder changed dimensions")
            decoded += 1
    finally:
        check.release()
    if decoded != count or count != stop - args.start:
        raise RuntimeError("Preview frame count verification failed")
    report = {
        "source_sha256": analysis.source_sha256,
        "start": args.start,
        "frames": count,
        "fps": analysis.video.fps,
        "mask": asdict(mask),
        "opencv": cv2.__version__,
        "landmark_guided_face_frames": guided,
        "fallback_face_frames": fallback,
        "visible_tracked_landmark_samples": len(feature_values),
        "offscreen_tracked_landmarks": offscreen,
        "opaque_visible_tracked_landmark_fraction": (
            float(np.mean(np.array(feature_values) >= 0.999)) if feature_values else None
        ),
        "limitations": (
            "Tracked landmarks are estimates. This is not a face-area or anonymity score."
        ),
    }
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {count} preview frames: {destination}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
