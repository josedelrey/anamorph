import json
import shutil
import subprocess
from dataclasses import replace

import cv2
import numpy as np
import pytest

from anamorph import (
    Box,
    FaceMask,
    FrameAnalysis,
    MaskRegion,
    ParsedFaceMask,
    VideoError,
    VideoExportWarning,
    analyze_video,
    export_masks,
    parse_video,
)
from anamorph.cli import main


@pytest.fixture
def ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is required for ProRes integration tests")
    encoders = subprocess.run(
        [executable, "-hide_banner", "-encoders"], capture_output=True, check=True
    ).stdout
    if b"prores_ks" not in encoders:
        pytest.skip("FFmpeg prores_ks encoder is unavailable")
    return executable


def test_png_has_exact_16bit_masks_black_frames_and_timing(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    analysis = replace(
        analysis,
        frames=tuple(
            FrameAnalysis(i, () if i in (0, 5) else record.faces)
            for i, record in enumerate(analysis.frames)
        ),
    )
    output = tmp_path / "masks"
    with pytest.warns(VideoExportWarning):
        report = export_masks(
            source, output, analysis=analysis, mask=FaceMask("rectangle", 1, 1, 0, 0), format="png"
        )
    paths = sorted(output.glob("*.png"))
    assert len(paths) == report.frame_count == 6
    assert report.face_instances == 4
    assert report.codec == "png16"
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert path.name == f"mask_{i:08d}.png"
        assert image.dtype == np.uint16
        expected = np.zeros((48, 64), np.uint16)
        if i not in (0, 5):
            expected[12:36, 16:40] = 65535
        np.testing.assert_array_equal(image, expected)
    metadata = json.loads((output / "sequence.json").read_text())
    assert (metadata["fps_numerator"], metadata["fps_denominator"]) == (12, 1)
    assert metadata["start_frame"] == 0
    assert metadata["frame_count"] == 6
    assert metadata["source_sha256"] == analysis.source_sha256
    assert detector.calls == 6


def test_feathered_overlap_uses_maximum_without_weakening_or_accumulation(
    video, detector, tmp_path
):
    class FixedMasks:
        def validate_analysis(self, analysis):
            pass

        def create_session(self):
            return self

        def build(self, frame, width, height):
            return {
                0: MaskRegion((0, 0, 3, 1), np.array([[0, 0.25, 0.5]], np.float32)),
                1: MaskRegion((1, 0, 4, 1), np.array([[0.125, 0.75, 1]], np.float32)),
            }

    source, _ = video
    analysis = analyze_video(source, detector=detector)
    with pytest.warns(VideoExportWarning):
        export_masks(source, tmp_path / "png", analysis=analysis, mask=FixedMasks(), format="png")
    image = cv2.imread(str(tmp_path / "png/mask_00000000.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(image[0, :4], [0, 16384, 49151, 65535])
    assert not np.any(image[1:])


def test_default_mov_codec_timing_levels_and_feather_fidelity(video, detector, tmp_path, ffmpeg):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    analysis = replace(analysis, frames=analysis.frames[:-1] + (FrameAnalysis(5),))
    mask = FaceMask("ellipse", 1, 1, 0, 0.3)
    with pytest.warns(VideoExportWarning):
        report = export_masks(source, tmp_path / "masks.mov", analysis=analysis, mask=mask)
    assert report.codec == "prores_ks"
    assert report.frame_count == 6
    assert report.face_instances == 5
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(report.output),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    decoded = np.frombuffer(raw, dtype="<u2").reshape(6, 48, 64)
    expected = np.zeros((6, 48, 64), np.uint16)
    region = mask.build(Box(16, 12, 40, 36), 64, 48)
    x1, y1, x2, y2 = region.bounds
    expected[:5, y1:y2, x1:x2] = np.rint(region.alpha * 65535).astype(np.uint16)
    # ProRes is a high-quality intermediate, with bounded quantization and ringing.
    assert np.max(np.abs(decoded.astype(int) - expected.astype(int))) < 400
    # Limited-range to full-range decoding can round black to one in 65535.
    assert np.max(decoded[5]) <= 1
    assert detector.calls == 6
    probe = shutil.which("ffprobe")
    if probe:
        metadata = json.loads(
            subprocess.run(
                [probe, "-v", "error", "-show_streams", "-of", "json", str(report.output)],
                capture_output=True,
                check=True,
            ).stdout
        )
        (stream,) = metadata["streams"]
        assert stream["codec_name"] == "prores"
        assert stream["profile"] == "4444"
        assert stream["pix_fmt"].startswith("yuv444p")
        assert stream["avg_frame_rate"] == "12/1"
        assert (stream["width"], stream["height"]) == (64, 48)


def test_missing_ffmpeg_suggests_png_and_png_needs_no_ffmpeg(
    video, detector, tmp_path, monkeypatch
):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    monkeypatch.setattr("anamorph.mask_export.shutil.which", lambda _: None)
    with pytest.raises(VideoError, match="--format png"):
        export_masks(source, tmp_path / "masks.mov", analysis=analysis)
    with pytest.warns(VideoExportWarning):
        export_masks(source, tmp_path / "png", analysis=analysis, format="png")


def test_fractional_frame_rate_is_retained_in_both_formats(tmp_path, detector, ffmpeg):
    source = tmp_path / "fractional.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x48:r=30000/1001",
            "-frames:v",
            "6",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    analysis = analyze_video(source, detector=detector)
    assert analysis.video.fps == pytest.approx(30000 / 1001)
    for format, output in [("mov", tmp_path / "masks.mov"), ("png", tmp_path / "png")]:
        with pytest.warns(VideoExportWarning):
            report = export_masks(source, output, analysis=analysis, format=format)
        assert report.frame_count == 6
        if format == "mov":
            capture = cv2.VideoCapture(str(output))
            try:
                assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30000 / 1001)
            finally:
                capture.release()
        else:
            metadata = json.loads((output / "sequence.json").read_text())
            assert (metadata["fps_numerator"], metadata["fps_denominator"]) == (30000, 1001)


@pytest.mark.parametrize("format", ["mov", "png"])
def test_source_and_analysis_mismatch_rejected(video, detector, tmp_path, format, monkeypatch):
    source, _ = video
    analysis = replace(analyze_video(source, detector=detector), source_sha256="0" * 64)
    monkeypatch.setattr("anamorph.mask_export.shutil.which", lambda _: "unused")
    output = tmp_path / ("masks.mov" if format == "mov" else "png")
    with pytest.raises(VideoError, match="SHA-256"):
        export_masks(source, output, analysis=analysis, format=format)
    assert not output.exists()


@pytest.mark.parametrize("length", [5, 7])
@pytest.mark.parametrize("format", ["mov", "png"])
def test_bad_frame_count_leaves_destinations_unchanged(
    video, detector, tmp_path, format, length, ffmpeg
):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    frames = analysis.frames[:length] if length == 5 else analysis.frames + (FrameAnalysis(6),)
    analysis = replace(analysis, frames=frames)
    output = tmp_path / ("masks.mov" if format == "mov" else "png")
    if format == "mov":
        output.write_bytes(b"preserved")
    before = set(tmp_path.iterdir())
    with pytest.warns(VideoExportWarning), pytest.raises(VideoError, match="frames"):
        export_masks(source, output, analysis=analysis, format=format, overwrite=True)
    assert set(tmp_path.iterdir()) == before
    if format == "mov":
        assert output.read_bytes() == b"preserved"


def test_destination_protection_and_mov_overwrite(video, detector, tmp_path, ffmpeg):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    output = tmp_path / "masks.mov"
    output.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        export_masks(source, output, analysis=analysis)
    with pytest.raises(ValueError, match="differ"):
        export_masks(source, source, analysis=analysis, overwrite=True)
    with pytest.raises(ValueError, match=".mov"):
        export_masks(source, tmp_path / "bad.mp4", analysis=analysis)
    with pytest.warns(VideoExportWarning):
        export_masks(source, output, analysis=analysis, overwrite=True)
    assert output.read_bytes() != b"preserved"
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "user_file.txt").write_text("preserved")
    with pytest.raises(FileExistsError):
        export_masks(source, directory, analysis=analysis, format="png", overwrite=True)
    assert (directory / "user_file.txt").read_text() == "preserved"


def test_encoder_failure_preserves_destination_and_cleans_staging(
    video, detector, tmp_path, ffmpeg, monkeypatch
):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    output = tmp_path / "masks.mov"
    output.write_bytes(b"preserved")
    original = subprocess.Popen

    def missing_encoder(command, **kwargs):
        command[command.index("prores_ks")] = "nonexistent_test_encoder"
        return original(command, **kwargs)

    monkeypatch.setattr("anamorph.mask_export.subprocess.Popen", missing_encoder)
    before = set(tmp_path.iterdir())
    with (
        pytest.warns(VideoExportWarning),
        pytest.raises(VideoError, match="nonexistent_test_encoder"),
    ):
        export_masks(source, output, analysis=analysis, overwrite=True)
    assert output.read_bytes() == b"preserved"
    assert set(tmp_path.iterdir()) == before


def test_png_encode_failure_does_not_publish(video, detector, tmp_path, monkeypatch):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    monkeypatch.setattr("anamorph.mask_export.cv2.imwrite", lambda *args: False)
    before = set(tmp_path.iterdir())
    with pytest.warns(VideoExportWarning), pytest.raises(VideoError, match="PNG"):
        export_masks(source, tmp_path / "png", analysis=analysis, format="png")
    assert set(tmp_path.iterdir()) == before


def test_cli_default_mov_and_optional_png(video, detector, tmp_path, ffmpeg, capsys):
    source, _ = video
    cache = tmp_path / "analysis.json"
    analyze_video(source, detector=detector).save(cache)
    for extra, output in [([], tmp_path / "masks.mov"), (["--format", "png"], tmp_path / "png")]:
        with pytest.warns(VideoExportWarning):
            main(["export-masks", str(source), str(cache), str(output), *extra])
        assert output.exists()
        assert "Wrote 6 mask frames" in capsys.readouterr().out


def test_semantic_export_cli_reuses_cached_parsing(video, detector, tmp_path):
    class SkinParser:
        def parse(self, image):
            return np.ones((32, 32), np.uint8)

        def metadata(self):
            return {"name": "controlled_skin"}

    source, _ = video
    analysis = analyze_video(source, detector=detector)
    analysis.save(tmp_path / "analysis.json")
    parsing = parse_video(source, analysis=analysis, parser=SkinParser())
    parsing.save(tmp_path / "parsing.npz")
    output = tmp_path / "semantic"
    with pytest.warns(VideoExportWarning):
        main(
            [
                "export-masks",
                str(source),
                str(tmp_path / "analysis.json"),
                str(output),
                "--format",
                "png",
                "--shape",
                "parsed",
                "--parsing",
                str(tmp_path / "parsing.npz"),
                "--expansion",
                "-1",
                "--feather",
                "3",
            ]
        )
    session = ParsedFaceMask(parsing, expansion=-1, feather=3).create_session()
    for frame in analysis.frames:
        expected = np.zeros((48, 64), np.uint16)
        region = session.build(frame, 64, 48)[0]
        x1, y1, x2, y2 = region.bounds
        expected[y1:y2, x1:x2] = np.rint(region.alpha * 65535).astype(np.uint16)
        decoded = cv2.imread(str(output / f"mask_{frame.index:08d}.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, expected)


def test_cli_protects_analysis_cache_even_with_overwrite(video, detector, tmp_path, capsys):
    source, _ = video
    cache = tmp_path / "analysis.json"
    analyze_video(source, detector=detector).save(cache)
    original = cache.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["export-masks", str(source), str(cache), str(cache), "--overwrite"])
    assert error.value.code == 2
    assert "must differ" in capsys.readouterr().err
    assert cache.read_bytes() == original
