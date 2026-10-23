import math
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from anamorph import (
    Pixelate,
    SparseOpticalFlow,
    VideoExportWarning,
    analyze_video,
    process_video,
    track_video,
)
from anamorph.cli import main


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_frame_progress_defaults_and_opt_out(video, tmp_path, monkeypatch, capsys, quiet):
    class Parser:
        def parse(self, image):
            return np.ones((32, 32), np.uint8)

        def metadata(self):
            return {"name": "controlled_parser"}

    monkeypatch.setattr("anamorph.cli.BiSeNetFaceParser", lambda _: Parser())
    source, _ = video
    cache = tmp_path / "analysis.json"
    flags = ["--no-progress"] if quiet else []
    main(["analyze", str(source), str(cache), *flags])
    main(
        [
            "parse",
            str(source),
            str(cache),
            str(tmp_path / "parsing.npz"),
            "--model",
            str(tmp_path / "unused.onnx"),
            "--start",
            "1",
            "--stop",
            "4",
            *flags,
        ]
    )
    with pytest.warns(VideoExportWarning):
        main(
            [
                "render",
                str(source),
                str(cache),
                str(tmp_path / "render.avi"),
                "--codec",
                "FFV1",
                *flags,
            ]
        )
    with pytest.warns(VideoExportWarning):
        main(
            [
                "export-masks",
                str(source),
                str(cache),
                str(tmp_path / "png"),
                "--format",
                "png",
                *flags,
            ]
        )
    output = capsys.readouterr()
    assert "Wrote 6 mask frames" in output.out
    if quiet:
        assert output.err == ""
    else:
        for stage in (
            "Analyzing",
            "Parsing faces",
            "Rendering",
            "Verifying export",
            "Exporting masks",
        ):
            assert stage in output.err
            assert stage not in output.out
        assert "3/3" in output.err
        assert "6/6" in output.err


def test_api_opt_out_reaches_both_processing_passes_and_motion(video, detector, tmp_path, capsys):
    source, _ = video
    with pytest.warns(VideoExportWarning):
        report = process_video(
            source,
            tmp_path / "output.avi",
            effect=Pixelate(),
            detector=detector,
            codec="FFV1",
            progress=False,
        )
    assert report.frame_count == 6
    analysis = analyze_video(source, detector=detector, progress=False)
    replay = track_video(source, analysis=analysis, motion=SparseOpticalFlow(), progress=False)
    assert len(replay.frames) == 6
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("estimate", [0, -1, math.nan, math.inf])
def test_unknown_frame_estimate_still_processes_every_frame(
    video, detector, monkeypatch, capsys, estimate
):
    from anamorph import video as module

    original = module._reader

    class Capture:
        def __init__(self, inner):
            self.inner = inner

        def get(self, key):
            return estimate if key == cv2.CAP_PROP_FRAME_COUNT else self.inner.get(key)

        def read(self):
            return self.inner.read()

    @contextmanager
    def reader(path):
        with original(path) as (capture, info):
            yield Capture(capture), info

    monkeypatch.setattr(module, "_reader", reader)
    source, _ = video
    assert len(analyze_video(source, detector=detector).frames) == 6
    assert "6frame" in capsys.readouterr().err
