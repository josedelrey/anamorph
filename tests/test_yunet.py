from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from anamorph import (
    Box,
    Pixelate,
    VideoAnalysis,
    VideoExportWarning,
    YuNetFaceDetector,
    analyze_video,
    render_video,
)
from anamorph.cli import main


@pytest.fixture
def backend(monkeypatch, tmp_path):
    class Backend:
        def __init__(self):
            self.rows = np.array(
                [
                    [10, 20, 30, 20, 12, 23, 28, 23, 20, 30, 14, 35, 26, 35, 0.95],
                ],
                dtype=np.float32,
            )

        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            self.image = image.copy()
            return 1, self.rows

    backend = Backend()
    monkeypatch.setattr(cv2, "FaceDetectorYN", SimpleNamespace(create=lambda *args: backend))
    path = tmp_path / "controlled-model.onnx"
    path.write_bytes(b"controlled backend weights")
    return backend, path


def test_original_coordinate_mapping_color_contract_and_input_preservation(backend):
    native, model = backend
    detector = YuNetFaceDetector(model, max_size=100)
    image = np.full((101, 203, 3), [1, 2, 3], np.uint8)
    before = image.copy()
    result = detector.detect(image)[0]
    assert native.size == (100, 50)
    np.testing.assert_array_equal(native.image[0, 0], [3, 2, 1])
    np.testing.assert_array_equal(image, before)
    np.testing.assert_allclose(result.box.as_tuple(), [20.3, 40.4, 81.2, 80.8])
    assert result.landmarks.names == (
        "right_eye",
        "left_eye",
        "nose_tip",
        "right_mouth_corner",
        "left_mouth_corner",
    )
    np.testing.assert_allclose(result.landmarks["nose_tip"], [40.6, 60.6])
    assert result.confidence == pytest.approx(0.95)
    assert detector.metadata()["max_size"] == 100
    detector.detect(np.zeros((40, 60, 3), np.uint8))
    assert native.size == (60, 40)  # Never upscale, even when the model is reused.


def test_native_resolution_and_empty_output(backend):
    native, model = backend
    detector = YuNetFaceDetector(model, max_size=None)
    assert detector.detect(np.zeros((200, 300, 3), np.uint8))[0].box == Box(10, 20, 40, 40)
    assert native.size == (300, 200)
    native.rows = None
    assert detector.detect(np.zeros((100, 200, 3), np.uint8)) == []


def test_invalid_backend_rows_are_rejected(backend):
    native, model = backend
    detector = YuNetFaceDetector(model)
    for rows in [np.zeros((1, 14), np.float32), np.full((1, 15), np.nan, np.float32)]:
        native.rows = rows
        with pytest.raises(RuntimeError, match="malformed"):
            detector.detect(np.zeros((100, 100, 3), np.uint8))


@pytest.mark.parametrize(
    "settings",
    [
        {"score_threshold": 0},
        {"score_threshold": 2},
        {"score_threshold": float("nan")},
        {"nms_threshold": 0},
        {"top_k": 0},
        {"top_k": True},
        {"max_size": 0},
        {"max_size": 1.5},
    ],
)
def test_parameters_are_validated_before_loading(settings):
    with pytest.raises(ValueError):
        YuNetFaceDetector("not-used.onnx", **settings)


def test_missing_invalid_and_lfs_models(tmp_path):
    path = tmp_path / "invalid.onnx"
    with pytest.raises(FileNotFoundError):
        YuNetFaceDetector(path)
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS"):
        YuNetFaceDetector(path)
    path.write_text("not an ONNX file")
    with pytest.raises(ValueError, match="Cannot load"):
        YuNetFaceDetector(path)


def test_cli_yunet_model_selection(backend, video, tmp_path):
    _, model = backend
    source, _ = video
    cache = tmp_path / "analysis.json"
    main(["analyze", str(source), str(cache), "--detector", "yunet", "--model", str(model)])
    analysis = VideoAnalysis.load(cache)
    assert analysis.provenance["detector"]["name"] == "YuNetFaceDetector"
    assert analysis.frames[0].faces[0].landmarks is not None
    main(
        [
            "analyze",
            str(source),
            str(cache),
            "--detector",
            "yunet",
            "--model",
            str(model),
            "--score-threshold",
            "0.7",
            "--max-size",
            "native",
        ]
    )
    native = VideoAnalysis.load(cache)
    assert native.provenance["detector"]["max_size"] is None
    assert native.provenance["detector"]["score_threshold"] == 0.7
    with pytest.raises(SystemExit):
        main(["analyze", str(source), str(cache), "--detector", "yunet"])
    with pytest.raises(SystemExit):
        main(["analyze", str(source), str(cache), "--model", str(model)])
    with pytest.raises(SystemExit):
        main(["analyze", str(source), str(cache), "--max-size", "invalid"])


def test_real_yunet_inference_and_video_export(yunet_model, video, tmp_path):
    detector = YuNetFaceDetector(yunet_model)
    for shape in [(48, 64, 3), (100, 200, 3)]:
        assert detector.detect(np.zeros(shape, np.uint8)) == []
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    assert len(analysis.frames) == 6
    assert all(d.landmarks is not None for f in analysis.frames for d in f.detections)
    with pytest.warns(VideoExportWarning):
        report = render_video(
            source, tmp_path / "yunet.avi", analysis=analysis, effect=Pixelate(), codec="FFV1"
        )
    assert report.frame_count == 6
