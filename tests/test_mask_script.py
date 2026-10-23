import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from anamorph import VideoAnalysis, VideoExportWarning, VideoParsing
from anamorph.analysis import file_sha256
from anamorph.video import VideoError


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/export_face_masks.py"
    spec = importlib.util.spec_from_file_location("export_face_masks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings(tmp_path):
    directory = tmp_path / "settings"
    directory.mkdir()
    (tmp_path / "detector.onnx").write_bytes(b"controlled detector")
    (tmp_path / "parser.onnx").write_bytes(b"controlled parser")
    return directory / "preset.yaml", {
        "version": 1,
        "runtime": {"progress": False, "opencv_threads": 1},
        "prepare": {
            "enabled": False,
            "filter": "scale=64:48,setsar=1",
            "fps": 12,
            "video_codec": "prores_ks",
            "profile": 3,
            "pixel_format": "yuv422p10le",
            "audio_codec": "pcm_s16le",
        },
        "detector": {
            "model": "../detector.onnx",
            "score_threshold": 0.6,
            "max_size": 320,
            "detection_interval": 1,
        },
        "tracker": {"min_iou": 0.3, "max_missing": 2, "smoothing": 1},
        "parsing": {"model": "../parser.onnx", "crop_scale": 1.8},
        "mask": {
            "expansion": 2,
            "feather": 4,
            "boundary_labels": "all",
            "fallback": "hold",
            "max_hold_frames": 3,
            "island_area_threshold": 0.02,
        },
        "output": {"directory": "../results", "format": "png", "overwrite": False},
    }


class ControlledParser:
    def parse(self, image):
        result = np.zeros((64, 64), np.uint8)
        result[12:52, 16:48] = 1
        return result

    def metadata(self):
        return {"name": "controlled parser"}


def models(script, monkeypatch, detector):
    calls = []

    def make_detector(path, **options):
        calls.append((Path(path), options))
        return detector

    monkeypatch.setattr(script, "YuNetFaceDetector", make_detector)
    monkeypatch.setattr(script, "BiSeNetFaceParser", lambda _: ControlledParser())
    return calls


def test_yaml_paths_resolve_relative_to_config_outside_working_directory(script, settings):
    path, config = settings
    path.write_text(yaml.safe_dump(config))
    loaded = script.load_config(path)
    assert Path(loaded["detector"]["model"]) == path.parent.parent / "detector.onnx"
    assert Path(loaded["parsing"]["model"]) == path.parent.parent / "parser.onnx"
    assert Path(loaded["output"]["directory"]) == path.parent.parent / "results"


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("format", ["png", "mov"])
def test_yaml_pipeline_exports_frames_and_bound_caches(
    script, settings, video, detector, monkeypatch, prepare, format
):
    path, config = settings
    source, _ = video
    config["prepare"]["enabled"] = prepare
    config["output"]["format"] = format
    path.write_text(yaml.safe_dump(config))
    calls = models(script, monkeypatch, detector)
    with pytest.warns(VideoExportWarning):
        report = script.run(source, path)
    directory = path.parent.parent / "results" / source.stem
    footage = directory / "footage.mov" if prepare else source
    assert report.frame_count == 6 and report.face_instances == 6
    assert report.video.fps == 12
    assert calls == [
        (path.parent.parent / "detector.onnx", {"score_threshold": 0.6, "max_size": 320})
    ]
    analysis = VideoAnalysis.load(directory / "analysis.json")
    assert analysis.source_sha256 == file_sha256(footage)
    assert analysis.provenance["tracker"]["smoothing"] == 1
    parsing = VideoParsing.load(directory / "parsing.npz")
    parsing.validate_analysis(analysis)
    snapshot = script.load_config(directory / "config.yaml")
    assert snapshot["mask"] == config["mask"]
    if format == "png":
        images = sorted(report.output.glob("mask_*.png"))
        assert len(images) == 6
        alpha = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
        assert alpha.dtype == np.uint16
        assert np.any(alpha == 65535) and np.any((alpha > 0) & (alpha < 65535))
    else:
        assert report.output == directory / "face_masks.mov"
        assert report.codec == "prores_ks"
    # Destination collisions fail before loading models or repeating inference.
    monkeypatch.setattr(script, "YuNetFaceDetector", lambda *a, **kw: pytest.fail("reloaded model"))
    with pytest.raises(FileExistsError):
        script.run(source, path)


def test_output_override_and_cli_preserve_existing_png_sequence(
    script, settings, video, detector, monkeypatch, tmp_path, capsys
):
    path, config = settings
    source, _ = video
    config["output"]["overwrite"] = True
    path.write_text(yaml.safe_dump(config))
    models(script, monkeypatch, detector)
    output = tmp_path / "override"
    with pytest.warns(VideoExportWarning):
        script.main([str(source), "--config", str(path), "--output-dir", str(output)])
    assert "Wrote 6 mask frames" in capsys.readouterr().out
    first = (output / "face_masks/mask_00000000.png").read_bytes()
    with pytest.raises(SystemExit) as error:
        script.main([str(source), "--config", str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    assert (output / "face_masks/mask_00000000.png").read_bytes() == first


def test_preparation_cannot_overwrite_the_input(script, settings, video):
    path, config = settings
    source, _ = video
    config["output"]["overwrite"] = True
    config["prepare"]["enabled"] = True
    path.write_text(yaml.safe_dump(config))
    prepared = source.parent / "footage.mov"
    prepared.write_bytes(source.read_bytes())
    before = prepared.read_bytes()
    with pytest.raises(ValueError, match="replace an input"):
        script.run(prepared, path, source.parent)
    assert prepared.read_bytes() == before


def test_mov_overwrite_recomputes_using_changed_yaml_settings(
    script, settings, video, detector, monkeypatch
):
    path, config = settings
    source, _ = video
    config["output"].update(format="mov", overwrite=True)
    config["mask"].update(expansion=0, feather=0)
    path.write_text(yaml.safe_dump(config))
    models(script, monkeypatch, detector)

    def first_frame():
        with pytest.warns(VideoExportWarning):
            report = script.run(source, path)
        capture = cv2.VideoCapture(str(report.output))
        ok, image = capture.read()
        capture.release()
        assert ok
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    before = first_frame()
    config["mask"].update(expansion=4, feather=12)
    path.write_text(yaml.safe_dump(config))
    after = first_frame()
    assert np.count_nonzero(after > 20) > np.count_nonzero(before > 20)
    assert np.any((before < 5) & (after > 50) & (after < 240))


def test_failed_preparation_leaves_no_published_video(script, settings, video, tmp_path):
    path, config = settings
    source, _ = video
    options = config["prepare"] | {"filter": "invalid_filter_name"}
    output = tmp_path / "footage.mov"
    with pytest.raises(VideoError, match="preparation failed"):
        script.prepare_video(source, output, options, progress=False)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepare-*"))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("mask", "feathre", 8, "Unknown mask"),
        ("runtime", "progress", "false", "runtime.progress"),
        ("runtime", "opencv_threads", 0, "runtime.opencv_threads"),
        ("output", "format", "mp4", "output.format"),
        ("detector", "model", None, "detector.model"),
        ("prepare", "fps", 0, "prepare.fps"),
    ],
)
def test_invalid_yaml_settings_fail_clearly(script, settings, section, key, value, message):
    path, config = settings
    config["prepare"]["enabled"] = True
    config[section][key] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        script.load_config(path)


def test_yaml_rejects_python_object_constructors(script, tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.print ['should never execute']")
    with pytest.raises(yaml.YAMLError):
        script.load_config(path)
