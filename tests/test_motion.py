from dataclasses import replace

import cv2
import numpy as np
import pytest

from anamorph import (
    Box,
    Detection,
    Face,
    IoUTracker,
    Landmarks,
    SparseOpticalFlow,
    VideoAnalysis,
    analyze_video,
    track_video,
)


def image(dx=0, dy=0):
    frame = np.full((160, 240, 3), 30, np.uint8)
    patch = np.random.default_rng(34).integers(30, 230, (80, 80, 3), dtype=np.uint8)
    frame[40 + dy : 120 + dy, 30 + dx : 110 + dx] = patch
    return frame


def test_bidirectional_flow_estimates_translation_without_changing_inputs():
    a, b = image(), image(7, -3)
    face = Face(0, Box(30, 40, 110, 120))
    estimate = SparseOpticalFlow().estimate(a, b, (face,))
    assert estimate.diagnostics == {0: "accepted"}
    assert estimate.predictions[0].x1 == pytest.approx(37, abs=0.15)
    assert estimate.predictions[0].y1 == pytest.approx(37, abs=0.15)
    np.testing.assert_array_equal(a, image())
    np.testing.assert_array_equal(b, image(7, -3))
    with pytest.raises(TypeError):
        estimate.predictions[1] = face.box


def test_motion_rejection_texture_occlusion_large_displacement_and_gap():
    flow = SparseOpticalFlow(max_displacement=0.05)
    face = Face(0, Box(30, 40, 110, 120))
    assert not flow.estimate(image(), image(12), (face,)).predictions
    blank = np.full_like(image(), 30)
    assert flow.estimate(blank, blank, (face,)).diagnostics == {0: "insufficient_features"}
    assert not flow.estimate(image(), blank, (face,)).predictions
    held = replace(face, observed=False, missing_frames=3)
    assert flow.estimate(image(), image(1), (held,)).diagnostics == {0: "gap_limit"}


def test_multiple_faces_have_independent_motion():
    before, after = image(), image(3, 1)
    patch = np.random.default_rng(58).integers(30, 230, (60, 60, 3), dtype=np.uint8)
    before[70:130, 150:210] = patch
    after[72:132, 146:206] = patch
    faces = (Face(0, Box(30, 40, 110, 120)), Face(1, Box(150, 70, 210, 130)))
    result = SparseOpticalFlow().estimate(before, after, faces)
    assert result.diagnostics == {0: "accepted", 1: "accepted"}
    assert result.predictions[0].x1 == pytest.approx(33, abs=0.2)
    assert result.predictions[1].x1 == pytest.approx(146, abs=0.2)


def test_scene_cut_rejects_all_motion_even_without_tracks():
    black, white = np.zeros_like(image()), np.full_like(image(), 255)
    result = SparseOpticalFlow().estimate(black, white, ())
    assert result.scene_cut and not result.predictions


def test_predictions_preserve_missing_age_and_move_landmarks():
    tracker = IoUTracker(smoothing=1, max_missing=2)
    marks = Landmarks(("nose",), ((70, 80),))
    tracker.update([Detection(Box(30, 40, 110, 120), landmarks=marks)])
    first = tracker.update([], predictions={0: Box(37, 37, 117, 117)})[0]
    assert not first.observed and first.missing_frames == 1
    assert first.landmarks["nose"] == (77, 77)
    assert tracker.update([], predictions={0: Box(40, 40, 120, 120)})[0].missing_frames == 2
    assert tracker.update([]) == ()
    with pytest.raises(ValueError, match="existing"):
        tracker.update([], predictions={0: Box(1, 1, 10, 10)})


def test_prediction_aids_reassociation_and_clear_does_not_reuse_ids():
    tracker = IoUTracker(smoothing=0.5)
    tracker.update([Detection(Box(0, 0, 10, 10))])
    # A raw overlap association would lose the ID after a large displacement.
    faces = tracker.update([Detection(Box(20, 0, 30, 10))], predictions={0: Box(20, 0, 30, 10)})
    assert len(faces) == 1 and faces[0].track_id == 0 and faces[0].box.x1 == 20
    tracker.clear()
    assert tracker.update([Detection(Box(20, 0, 30, 10))])[0].track_id == 1


class MovingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        # The textured patch provides a deterministic observed box.
        self.calls += 1
        y, x = np.nonzero(np.any(frame != 30, axis=2))
        return [
            Detection(Box(float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)))
        ]

    def metadata(self):
        return {"name": "controlled_moving_patch"}


def test_motion_video_replay_reuses_detections_and_roundtrips_cache(tmp_path):
    source = tmp_path / "moving.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 25, (240, 160))
    assert writer.isOpened()
    for i in range(8):
        writer.write(cv2.cvtColor(image(i * 4), cv2.COLOR_RGB2BGR))
    writer.release()
    detector = MovingDetector()
    original = analyze_video(source, detector=detector, detection_interval=3)
    replayed = track_video(source, analysis=original, motion=SparseOpticalFlow())
    assert detector.calls == 3
    assert original.frames[2].faces[0].box.x1 == 30
    for i, frame in enumerate(replayed.frames):
        assert len(frame.faces) == 1 and frame.faces[0].track_id == 0
        assert frame.faces[0].box.x1 == pytest.approx(30 + i * 4, abs=0.3)
        assert frame.detections == original.frames[i].detections
    direct = analyze_video(
        source, detector=MovingDetector(), detection_interval=3, motion=SparseOpticalFlow()
    )
    assert direct.frames == replayed.frames
    cache = tmp_path / "motion.json"
    replayed.save(cache)
    assert VideoAnalysis.load(cache) == replayed
    box_only = replayed.retrack(tracker=IoUTracker())
    assert "motion" not in box_only.provenance and "motion_diagnostics" not in box_only.provenance
    assert box_only.frames == original.frames
    with pytest.raises(ValueError, match="Raw detections"):
        track_video(
            source,
            analysis=replace(
                original, frames=tuple(replace(f, detections=None) for f in original.frames)
            ),
            motion=SparseOpticalFlow(),
        )


def test_cut_forces_detection_and_clears_tracks_with_fresh_ids(tmp_path):
    source = tmp_path / "cut.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 48))
    for color in (0, 0, 255, 255):
        writer.write(np.full((48, 64, 3), color, np.uint8))
    writer.release()

    class Detector:
        def detect(self, image):
            return [Detection(Box(10, 10, 30, 30))]

        def metadata(self):
            return {"name": "controlled"}

    analysis = analyze_video(
        source, detector=Detector(), detection_interval=3, motion=SparseOpticalFlow()
    )
    assert analysis.provenance["scene_cuts"] == [2]
    assert analysis.frames[2].detections is not None
    assert analysis.frames[0].faces[0].track_id == 0
    assert analysis.frames[2].faces[0].track_id == 1
    assert analysis.retrack(tracker=IoUTracker()).frames[2].faces[0].track_id == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_points": 2},
        {"max_points": 4},
        {"max_missing": 0},
        {"min_inlier_fraction": 0},
        {"scene_cut_threshold": 2},
        {"max_photometric_error": float("nan")},
    ],
)
def test_motion_config_validation(kwargs):
    with pytest.raises(ValueError):
        SparseOpticalFlow(**kwargs)


def test_reference_box_evaluation_counts_switches_and_consecutive_misses():
    import runpy
    from pathlib import Path

    from anamorph import FrameAnalysis, VideoInfo

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/compare_tracking.py")
    )
    a, b = Box(0, 0, 10, 10), Box(15, 0, 25, 10)
    analysis = VideoAnalysis(
        "0" * 64,
        VideoInfo(40, 20, 25),
        (
            FrameAnalysis(0, (Face(0, a), Face(1, b))),
            FrameAnalysis(1, (Face(0, a), Face(2, b))),
            FrameAnalysis(2),
            FrameAnalysis(3),
        ),
    )
    annotations = [
        {
            "frame_index": i,
            "regions": [
                {"track_id": 10, "face_flag": 1, "box_xywh": [0, 0, 10, 10]},
                {"track_id": 20, "face_flag": 1, "box_xywh": [15, 0, 10, 10]},
            ],
        }
        for i in range(4)
    ]
    result = module["evaluate_boxes"](analysis, annotations)
    assert result["reference_boxes"] == 8 and result["matched_boxes"] == 4
    assert result["box_recall"] == 0.5
    assert result["adjacent_id_switches"] == 1
    assert result["adjacent_matched_identity_pairs"] == 2
    assert result["longest_annotated_miss_run"] == 2
