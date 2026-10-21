from dataclasses import replace

import numpy as np
import pytest

from anamorph import (
    Box,
    Detection,
    Effect,
    Face,
    FrameAnalysis,
    HaarFaceDetector,
    IoUTracker,
    Pixelate,
    VideoExportWarning,
    VideoInfo,
    analyze_video,
    render_video,
)
from anamorph.cli import main


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_factor": 1},
        {"scale_factor": float("nan")},
        {"min_neighbors": -1},
        {"min_size": 0},
    ],
)
def test_detector_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        HaarFaceDetector(**kwargs)


def test_detector_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        HaarFaceDetector(cascade_path=tmp_path / "absent.xml")


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 3), np.float32),
        np.zeros((0, 4, 3), np.uint8),
    ],
)
def test_image_contract(image):
    with pytest.raises(ValueError, match="uint8 RGB"):
        Pixelate().apply(image)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 2, 2), confidence)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_id": -1},
        {"missing_frames": -1},
        {"observed": False},
        {"missing_frames": 1},
    ],
)
def test_face_observation_contract(kwargs):
    options = {"track_id": 0, "box": Box(0, 0, 2, 2)} | kwargs
    with pytest.raises(ValueError):
        Face(**options)


def test_frame_contract():
    with pytest.raises(ValueError):
        FrameAnalysis(-1)


@pytest.mark.parametrize("args", [(0, 4, 30), (4, 4, 0), (4, 4, float("nan"))])
def test_video_info_contract(args):
    with pytest.raises(ValueError):
        VideoInfo(*args)


def test_assignment_gates_before_optimization():
    tracker = IoUTracker(min_iou=0.4, smoothing=1, max_missing=0)
    tracker.update([Detection(Box(0, 0, 10, 10)), Detection(Box(6, 0, 16, 10))])
    faces = tracker.update([Detection(Box(2, 0, 12, 10)), Detection(Box(-4, 0, 6, 10))])
    assert [(face.track_id, face.box.x1) for face in faces] == [(0, -4), (1, 2)]


def test_odd_dimensions_rejected(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    odd = replace(analysis, video=replace(analysis.video, width=63))
    with pytest.raises(ValueError, match="even"):
        render_video(source, tmp_path / "out.avi", analysis=odd, effect=Pixelate())


def test_metadata_mismatch_rejected(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    changed = replace(analysis, video=replace(analysis.video, fps=30))
    with pytest.warns(VideoExportWarning), pytest.raises(RuntimeError, match="metadata"):
        render_video(
            source, tmp_path / "out.avi", analysis=changed, effect=Pixelate(), codec="FFV1"
        )
    assert not list(tmp_path.glob(".out-*"))


def test_cli_cannot_overwrite_source_with_analysis(video, capsys):
    source, _ = video
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        main(["analyze", str(source), str(source)])
    assert "must differ" in capsys.readouterr().err
    assert source.read_bytes() == original


def test_session_creation_failure_leaves_no_temporary_file(video, detector, tmp_path):
    class BrokenFactory(Effect):
        def create_session(self):
            raise RuntimeError("session creation failed")

    source, _ = video
    analysis = analyze_video(source, detector=detector)
    before = set(tmp_path.iterdir())
    with pytest.warns(VideoExportWarning), pytest.raises(RuntimeError, match="session creation"):
        render_video(
            source, tmp_path / "out.avi", analysis=analysis, effect=BrokenFactory(), codec="FFV1"
        )
    assert set(tmp_path.iterdir()) == before


def test_provenance_must_be_an_object(video, detector):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    with pytest.raises(ValueError, match="JSON object"):
        replace(analysis, provenance=[])
