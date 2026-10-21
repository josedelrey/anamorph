import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from anamorph import (
    Box,
    Face,
    FaceMask,
    FrameAnalysis,
    FrameRenderer,
    GaussianBlur,
    HaarFaceDetector,
    IoUTracker,
    Pixelate,
    SpatialEffect,
    VideoAnalysis,
    VideoError,
    VideoExportWarning,
    analyze_video,
    process_video,
    render_video,
)
from anamorph.cli import main


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def test_analysis_cache_roundtrip(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    assert detector.calls == 6
    assert len(analysis.frames) == 6
    assert analysis.video.fps == 12
    assert {frame.faces[0].track_id for frame in analysis.frames} == {0}
    path = tmp_path / "analysis.json"
    analysis.save(path)
    assert VideoAnalysis.load(path) == analysis
    assert (
        json.loads(path.read_text())["provenance"]["detector"]["name"] == "controlled_test_detector"
    )


def test_reuse_and_exact_lossless_video_pipeline(video, detector, tmp_path):
    source, frames = video
    analysis = analyze_video(source, detector=detector)
    mask = FaceMask("rectangle", 1, 1, 0, 0)
    for effect, name in [(Pixelate(2), "pixels"), (GaussianBlur(3), "blur")]:
        output = tmp_path / f"{name}.avi"
        with pytest.warns(VideoExportWarning):
            report = render_video(
                source, output, analysis=analysis, effect=effect, mask=mask, codec="FFV1"
            )
        assert report.frame_count == 6
        assert report.face_instances == 6
        renderer = FrameRenderer(effect, mask)
        decoded = read_frames(output)
        assert len(decoded) == 6
        for index, image in enumerate(decoded):
            expected = renderer.render(frames[index], analysis.frames[index], timestamp=index / 12)
            np.testing.assert_array_equal(image, expected)
            np.testing.assert_array_equal(image[:12], frames[index][:12])
        assert np.mean(np.abs(decoded[0].astype(float) - frames[0])) > 1
    assert detector.calls == 6


def test_mp4_convenience_pipeline(video, detector, tmp_path):
    source, _ = video
    output = tmp_path / "output.mp4"
    with pytest.warns(VideoExportWarning):
        report = process_video(source, output, detector=detector, effect=Pixelate(3))
    assert report.frame_count == len(read_frames(output)) == 6


def test_detection_interval_and_held_records(video, detector):
    source, _ = video
    analysis = analyze_video(source, detector=detector, detection_interval=3)
    assert detector.calls == 2
    assert [frame.faces[0].missing_frames for frame in analysis.frames] == [0, 1, 2, 0, 1, 2]
    assert analysis.frames[0].detections is not None
    assert analysis.frames[1].detections is None
    retracked = analysis.retrack(tracker=IoUTracker(smoothing=1))
    assert [frame.faces for frame in retracked.frames] == [frame.faces for frame in analysis.frames]
    assert retracked.provenance["tracker"]["smoothing"] == 1
    assert detector.calls == 2
    without_raw = replace(
        analysis, frames=tuple(replace(f, detections=None) for f in analysis.frames)
    )
    with pytest.raises(ValueError, match="unavailable"):
        without_raw.retrack(tracker=IoUTracker())
    with pytest.raises(ValueError, match="max_missing"):
        analyze_video(
            source, detector=detector, tracker=IoUTracker(max_missing=0), detection_interval=2
        )


def test_default_haar_pipeline_without_external_model(video, tmp_path):
    source, _ = video
    detector = HaarFaceDetector()
    assert detector.metadata()["model_sha256"]
    assert detector.detect(np.zeros((48, 64, 3), np.uint8)) == []
    analysis = analyze_video(source)
    assert len(analysis.frames) == 6
    output = tmp_path / "default.avi"
    with pytest.warns(VideoExportWarning):
        render_video(source, output, analysis=analysis, effect=Pixelate(), codec="FFV1")
    assert len(read_frames(output)) == 6


def test_wrong_source_cache_rejected(video, detector, tmp_path):
    source, _ = video
    analysis = replace(analyze_video(source, detector=detector), source_sha256="0" * 64)
    output = tmp_path / "out.avi"
    with pytest.raises(VideoError, match="SHA-256"):
        render_video(source, output, analysis=analysis, effect=Pixelate())
    assert not output.exists()


def test_destination_protection_and_overwrite(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    output = tmp_path / "out.avi"
    output.write_bytes(b"existing destination")
    with pytest.raises(FileExistsError):
        render_video(source, output, analysis=analysis, effect=Pixelate())
    with pytest.raises(ValueError, match="differ"):
        render_video(source, source, analysis=analysis, effect=Pixelate(), overwrite=True)
    assert output.read_bytes() == b"existing destination"
    with pytest.warns(VideoExportWarning):
        render_video(
            source, output, analysis=analysis, effect=Pixelate(), codec="FFV1", overwrite=True
        )
    assert len(read_frames(output)) == 6


def test_effect_failure_preserves_destination_and_cleans_temporary(video, detector, tmp_path):
    class Broken(SpatialEffect):
        def apply(self, image):
            raise RuntimeError("intentional effect failure")

    source, _ = video
    analysis = analyze_video(source, detector=detector)
    output = tmp_path / "out.avi"
    output.write_bytes(b"preserved")
    before = set(tmp_path.iterdir())
    with pytest.warns(VideoExportWarning), pytest.raises(RuntimeError, match="intentional"):
        render_video(
            source, output, analysis=analysis, effect=Broken(), codec="FFV1", overwrite=True
        )
    assert output.read_bytes() == b"preserved"
    assert set(tmp_path.iterdir()) == before


def test_analysis_length_mismatch_does_not_publish(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    output = tmp_path / "out.avi"
    for frames in [analysis.frames[:-1], analysis.frames + (FrameAnalysis(6),)]:
        with pytest.warns(VideoExportWarning), pytest.raises(VideoError, match="frames"):
            render_video(
                source,
                output,
                analysis=replace(analysis, frames=frames),
                effect=Pixelate(),
                codec="FFV1",
            )
        assert not output.exists()


def test_no_faces_is_identity(video, detector, tmp_path):
    source, frames = video
    analysis = analyze_video(source, detector=detector)
    analysis = replace(analysis, frames=tuple(FrameAnalysis(i) for i in range(6)))
    output = tmp_path / "identity.avi"
    with pytest.warns(VideoExportWarning):
        render_video(source, output, analysis=analysis, effect=Pixelate(), codec="FFV1")
    np.testing.assert_array_equal(read_frames(output), frames)


def test_cache_schema_validation(video, detector, tmp_path):
    source, _ = video
    cache = tmp_path / "analysis.json"
    analysis = analyze_video(source, detector=detector)
    analysis.save(cache)
    data = json.loads(cache.read_text())
    data["schema_version"] = 99
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        VideoAnalysis.load(cache)
    cache.write_text('{"frames": []}')
    with pytest.raises(ValueError, match="Invalid"):
        VideoAnalysis.load(cache)
    with pytest.raises(ValueError, match="contiguous"):
        replace(analysis, frames=(FrameAnalysis(1),))
    with pytest.raises(ValueError, match="duplicate"):
        FrameAnalysis(0, (Face(0, Box(0, 0, 1, 1)), Face(0, Box(1, 1, 2, 2))))


def test_invalid_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_video(tmp_path / "missing.mp4")
    source = tmp_path / "invalid.mp4"
    source.write_text("not a video")
    with pytest.raises(VideoError):
        analyze_video(source)


def test_cli_analyze_render(video, tmp_path, capsys):
    source, _ = video
    cache, output = tmp_path / "analysis.json", tmp_path / "output.avi"
    main(["analyze", str(source), str(cache)])
    with pytest.warns(VideoExportWarning):
        main(["render", str(source), str(cache), str(output), "--codec", "FFV1"])
    assert "Wrote 6 frames" in capsys.readouterr().out
    assert len(read_frames(output)) == 6


def test_cli_reports_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(tmp_path / "missing.mp4"), str(tmp_path / "cache.json")])
    assert exc.value.code == 2
    assert "anamorph:" in capsys.readouterr().err
