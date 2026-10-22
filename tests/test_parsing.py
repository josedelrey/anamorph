import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from anamorph import (
    FACE_PARSING_LABELS,
    BiSeNetFaceParser,
    Box,
    Detection,
    Face,
    FaceParsing,
    FrameAnalysis,
    FrameRenderer,
    LandmarkFaceMask,
    Landmarks,
    MaskSession,
    ParsedFaceMask,
    PerFaceMask,
    Pixelate,
    VideoAnalysis,
    VideoExportWarning,
    VideoInfo,
    VideoParsing,
    analyze_video,
    parse_video,
    render_video,
)
from anamorph.cli import main
from anamorph.parsing import _digest, _parsing_crop

NAMES = ("right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner")
MARKS = Landmarks(NAMES, ((12, 12), (20, 12), (16, 16), (13, 20), (19, 20)))
FACE = Face(0, Box(6, 6, 26, 26), landmarks=MARKS)


def labels():
    result = np.zeros((48, 64), np.uint8)
    result[8:24, 8:24] = 1
    result[4:8, 8:24] = 17
    result[12:20, 4:8] = 7
    result[12:20, 24:28] = 8
    result[24:30, 8:24] = 14
    return result


def cache(frames=None, maps=None):
    frames = tuple(frames or (FrameAnalysis(0, (FACE,)),))
    maps = maps or [(labels(),)] * len(frames)
    analysis = VideoAnalysis("0" * 64, VideoInfo(64, 48, 12), frames)
    parsed_frames = []
    for frame, frame_maps in zip(frames, maps, strict=True):
        observed = [f for f in frame.faces if f.observed]
        parsed_frames.append(
            tuple(
                FaceParsing(f.track_id, (0, 0, 64, 48), data)
                for f, data in zip(observed, frame_maps, strict=True)
            )
        )
    parsing = VideoParsing(
        analysis.source_sha256,
        _digest(analysis),
        analysis.video,
        0,
        tuple(parsed_frames),
        tuple(_digest(f) for f in frames),
    )
    return analysis, parsing


def full(region):
    result = np.zeros((48, 64), np.float32)
    if region:
        a, b, c, d = region.bounds
        result[b:d, a:c] = region.alpha
    return result


def test_semantic_core_opaque_with_explicit_hair_ear_neck_exclusions():
    analysis, parsing = cache()
    result = full(
        MaskSession(ParsedFaceMask(parsing, feather=4)).build(analysis.frames[0], 64, 48)[0]
    )
    data = labels()
    assert np.all(result[data == 1] == 1)
    assert np.all(result[np.isin(data, [7, 8, 14, 17])] == 0)
    assert np.any((result > 0) & (result < 1) & (data == 0))


def test_expansion_grows_core_in_pixels_without_crossing_excluded_classes():
    analysis, parsing = cache()
    base = full(
        ParsedFaceMask(parsing, feather=0).create_session().build(analysis.frames[0], 64, 48)[0]
    )
    grown = full(
        ParsedFaceMask(parsing, feather=0, expansion=2)
        .create_session()
        .build(analysis.frames[0], 64, 48)[0]
    )
    assert base[10, 25] == 0 and grown[10, 25] == 1
    assert np.all(grown[base == 1] == 1)
    assert np.all(grown[np.isin(labels(), [7, 8, 14, 17])] == 0)
    assert grown.dtype == np.float32


def test_negative_expansion_erodes_core_before_feathering():
    analysis, parsing = cache()
    region = (
        ParsedFaceMask(parsing, feather=0, expansion=-2)
        .create_session()
        .build(analysis.frames[0], 64, 48)[0]
    )
    expected = np.zeros((48, 64), np.float32)
    expected[10:22, 10:22] = 1
    np.testing.assert_array_equal(full(region), expected)


def test_boundary_class_tolerance_is_explicit_and_limited_to_the_margin():
    analysis, parsing = cache()
    mask = ParsedFaceMask(
        parsing, expansion=2, feather=0, boundary_labels=("background", "hair", "neck")
    )
    alpha = full(mask.create_session().build(analysis.frames[0], 64, 48)[0])
    assert alpha[6, 16] == 1 and alpha[5, 16] == 0
    assert alpha[25, 16] == 1 and alpha[26, 16] == 0
    assert np.all(alpha[np.isin(labels(), [7, 8])] == 0)


def test_total_erosion_does_not_revive_temporal_history():
    big, small = labels(), labels()
    small[8:24, 8:24] = 0
    small[12:20, 12:20] = 1
    # Missing landmarks allow a smaller usable core, still checked against box extent.
    face = replace(FACE, landmarks=None, box=Box(10, 10, 22, 22))
    frames = (FrameAnalysis(0, (face,)), FrameAnalysis(1, (face,)))
    analysis, parsing = cache(frames, [(big,), (small,)])
    session = ParsedFaceMask(parsing, expansion=-5, feather=3, stabilization=0.1).create_session()
    assert np.any(full(session.build(analysis.frames[0], 64, 48)[0]))
    assert not np.any(full(session.build(analysis.frames[1], 64, 48)[0]))
    assert session.diagnostics == {0: "parsed"}


def test_custom_fallback_geometry_is_independent_of_semantic_boundary_edits():
    analysis, parsing = cache(maps=[(np.zeros((48, 64), np.uint8),)])
    geometry = LandmarkFaceMask(scale_x=1.5, scale_y=1.3, feather=0)
    session = ParsedFaceMask(parsing, fallback_mask=geometry, expansion=-20).create_session()
    actual = full(session.build(analysis.frames[0], 64, 48)[0])
    expected = full(geometry.build(FACE.box, 64, 48, landmarks=FACE.landmarks))
    np.testing.assert_array_equal(actual, expected)
    assert session.diagnostics == {0: "fallback_empty"}


def test_semantic_fallback_protects_a_measured_profile_nose_outside_its_box():
    marks = Landmarks(NAMES, ((12, 12), (20, 12), (4, 16), (13, 20), (19, 20)))
    frame = FrameAnalysis(0, (replace(FACE, landmarks=marks),))
    analysis, parsing = cache((frame,))
    session = ParsedFaceMask(parsing).create_session()
    alpha = full(session.build(analysis.frames[0], 64, 48)[0])
    assert session.diagnostics == {0: "fallback_incomplete"}
    assert alpha[16, 4] == 1


def test_per_face_overrides_keep_cache_binding_and_scope_strict_errors():
    other = replace(FACE, track_id=1, box=Box(36, 6, 56, 26), landmarks=None)
    frame = FrameAnalysis(0, (FACE, other))
    analysis, parsing = cache((frame,), [(labels(), np.zeros((48, 64), np.uint8))])
    strict = ParsedFaceMask(parsing, fallback="error", feather=0, expansion=2)
    geometric = LandmarkFaceMask()
    # The unusable parse for ID 1 must not fail a strict semantic config for ID 0.
    mask = PerFaceMask(strict, {1: geometric})
    mask.validate_analysis(analysis)
    session = mask.create_session()
    regions = session.build(frame, 64, 48)
    assert full(regions[0])[10, 25] == 1
    np.testing.assert_array_equal(full(regions[1]), full(geometric.build(other.box, 64, 48)))
    assert session.diagnostics == {0: "parsed", 1: "geometry"}
    with pytest.raises(ValueError, match="face analysis"):
        mask.create_session().build(replace(frame, faces=(FACE,)), 64, 48)
    with pytest.raises(ValueError, match="source and analysis"):
        mask.validate_analysis(replace(analysis, source_sha256="1" * 64))
    # An unused strict default has no authority over the geometric override.
    all_geometry = PerFaceMask(strict, {0: geometric, 1: geometric})
    assert all_geometry.create_session().build(frame, 64, 48).keys() == {0, 1}


def test_semantic_per_track_parameters_and_sessions_are_independent():
    other = replace(FACE, track_id=1, box=Box(36, 6, 56, 26), landmarks=None)
    data = labels()
    data[8:24, 38:54] = 1
    frames = (FrameAnalysis(0, (FACE, other)), FrameAnalysis(1, (FACE, other)))
    analysis, parsing = cache(frames, [(data, data), (data, data)])
    mask = PerFaceMask(
        ParsedFaceMask(parsing, feather=0), {1: ParsedFaceMask(parsing, feather=0, expansion=-2)}
    )
    a, b = mask.create_session(), mask.create_session()
    regions = a.build(analysis.frames[0], 64, 48)
    assert full(regions[0])[8, 10] == 1
    assert full(regions[1])[8, 40] == 0
    assert full(regions[1])[10, 40] == 1
    a.build(analysis.frames[1], 64, 48)
    again = b.build(analysis.frames[0], 64, 48)
    np.testing.assert_array_equal(full(regions[1]), full(again[1]))


def test_recovery_requires_consecutive_observations_and_preserves_fallback_coverage():
    frames = tuple(FrameAnalysis(i, (FACE,)) for i in range(7))
    empty = np.zeros((48, 64), np.uint8)
    analysis, parsing = cache(
        frames,
        [(labels(),), (empty,), (labels(),), (empty,), (labels(),), (labels(),), (labels(),)],
    )
    session = ParsedFaceMask(parsing, recovery_frames=3).create_session()
    statuses = []
    for frame in analysis.frames:
        alpha = full(session.build(frame, 64, 48)[0])
        statuses.append(session.diagnostics[0])
        for x, y in MARKS.points:
            assert alpha[int(y), int(x)] == 1
    assert statuses == [
        "parsed",
        "fallback_empty",
        "fallback_recovery",
        "fallback_empty",
        "fallback_recovery",
        "fallback_recovery",
        "parsed",
    ]
    # Stable observations eventually restore the strict semantic exclusions.
    assert np.all(alpha[np.isin(labels(), [7, 8, 14, 17])] == 0)


def test_held_observations_cannot_complete_recovery():
    held = replace(FACE, observed=False, missing_frames=1)
    frames = (
        FrameAnalysis(0, (FACE,)),
        FrameAnalysis(1, (FACE,)),
        FrameAnalysis(2, (held,)),
        FrameAnalysis(3, (FACE,)),
        FrameAnalysis(4, (FACE,)),
    )
    analysis, parsing = cache(
        frames, [(np.zeros((48, 64), np.uint8),), (labels(),), (), (labels(),), (labels(),)]
    )
    session = ParsedFaceMask(parsing, recovery_frames=2).create_session()
    statuses = []
    for frame in analysis.frames:
        session.build(frame, 64, 48)
        statuses.append(session.diagnostics[0])
    assert statuses == [
        "fallback_empty",
        "fallback_recovery",
        "fallback_missing",
        "fallback_recovery",
        "parsed",
    ]


def test_recovery_state_resets_across_absence_or_frame_gap():
    frames = (FrameAnalysis(0, (FACE,)), FrameAnalysis(1), FrameAnalysis(2, (FACE,)))
    analysis, parsing = cache(frames, [(np.zeros((48, 64), np.uint8),), (), (labels(),)])
    for skip_absent in [False, True]:
        session = ParsedFaceMask(parsing, recovery_frames=3).create_session()
        session.build(analysis.frames[0], 64, 48)
        if not skip_absent:
            session.build(analysis.frames[1], 64, 48)
        session.build(analysis.frames[2], 64, 48)
        assert session.diagnostics == {0: "parsed"}


def test_current_semantics_veto_previous_mask_after_temporal_filtering():
    first, second = labels(), labels()
    second[10:12, 17:19] = 17
    analysis, parsing = cache(
        (FrameAnalysis(0, (FACE,)), FrameAnalysis(1, (FACE,))), [(first,), (second,)]
    )
    session = ParsedFaceMask(parsing, stabilization=0.1).create_session()
    before = full(session.build(analysis.frames[0], 64, 48)[0])
    after = full(session.build(analysis.frames[1], 64, 48)[0])
    assert np.all(before[10:12, 17:19] == 1)
    assert np.all(after[10:12, 17:19] == 0)
    assert session.diagnostics == {0: "parsed"}


def test_included_labels_change_without_inference():
    analysis, parsing = cache()
    mask = ParsedFaceMask(
        parsing, included_labels=ParsedFaceMask.included_labels + ("hair", "left_ear", "right_ear")
    )
    result = full(mask.create_session().build(analysis.frames[0], 64, 48)[0])
    assert np.all(result[labels() == 17] == 1)
    assert np.all(result[labels() == 7] == 1)
    assert np.all(result[labels() == 14] == 0)


def test_disconnected_neighbor_does_not_enter_own_face_mask():
    data = labels()
    data[8:24, 38:54] = 1
    other = Face(
        1,
        Box(36, 6, 56, 26),
        landmarks=Landmarks(NAMES, tuple((x + 30, y) for x, y in MARKS.points)),
    )
    analysis, parsing = cache((FrameAnalysis(0, (FACE, other)),), [(data, data)])
    session = ParsedFaceMask(parsing).create_session()
    result = session.build(analysis.frames[0], 64, 48)
    assert np.all(full(result[0])[8:24, 8:24] == 1)
    assert np.all(full(result[0])[8:24, 38:54] == 0)
    assert np.all(full(result[1])[8:24, 38:54] == 1)
    assert np.all(full(result[1])[8:24, 8:24] == 0)
    assert session.diagnostics == {0: "parsed", 1: "parsed"}


def test_no_landmarks_selects_component_in_own_box():
    data = labels()
    data[2:40, 36:62] = 1
    face = replace(FACE, landmarks=None)
    analysis, parsing = cache((FrameAnalysis(0, (face,)),), [(data,)])
    region = ParsedFaceMask(parsing).create_session().build(analysis.frames[0], 64, 48)[0]
    assert np.all(full(region)[8:24, 38:54] == 0)
    assert full(region)[16, 16] == 1


def test_raw_landmarks_prevent_accepting_mask_that_only_covers_smoothed_points():
    measured = Detection(
        Box(6, 0, 26, 20), landmarks=Landmarks(NAMES, tuple((x, y - 6) for x, y in MARKS.points))
    )
    frame = FrameAnalysis(0, (FACE,), (measured,))
    analysis, parsing = cache((frame,))
    session = ParsedFaceMask(parsing).create_session()
    result = full(session.build(analysis.frames[0], 64, 48)[0])
    assert session.diagnostics == {0: "fallback_incomplete"}
    assert result[6, 12] == 1


def test_face_extent_rejects_patch_even_when_all_feature_points_are_covered():
    data = labels()
    data[8:11, 8:24] = 0
    data[21:24, 8:24] = 0
    analysis, parsing = cache(maps=[(data,)])
    session = ParsedFaceMask(parsing).create_session()
    session.build(analysis.frames[0], 64, 48)
    assert session.diagnostics == {0: "fallback_incomplete"}


def test_shuffled_raw_detections_associate_with_separate_faces():
    data = labels()
    data[8:24, 38:54] = 1
    other = Face(
        1,
        Box(36, 6, 56, 26),
        landmarks=Landmarks(NAMES, tuple((x + 30, y) for x, y in MARKS.points)),
    )
    detections = (
        Detection(other.box, landmarks=other.landmarks),
        Detection(FACE.box, landmarks=FACE.landmarks),
    )
    analysis, parsing = cache((FrameAnalysis(0, (FACE, other), detections),), [(data, data)])
    session = ParsedFaceMask(parsing).create_session()
    regions = session.build(analysis.frames[0], 64, 48)
    assert session.diagnostics == {0: "parsed", 1: "parsed"}
    assert full(regions[0])[16, 16] == 1 and full(regions[0])[16, 46] == 0
    assert full(regions[1])[16, 46] == 1 and full(regions[1])[16, 16] == 0


@pytest.mark.parametrize("empty", [True, False])
def test_unusable_parsing_falls_back_explicitly_or_refuses(empty):
    data = np.zeros((48, 64), np.uint8) if empty else labels()
    if not empty:
        data[16, 16] = 17
    analysis, parsing = cache(maps=[(data,)])
    session = ParsedFaceMask(parsing).create_session()
    region = session.build(analysis.frames[0], 64, 48)[0]
    assert session.diagnostics[0] == ("fallback_empty" if empty else "fallback_incomplete")
    assert full(region)[16, 16] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        ParsedFaceMask(parsing, fallback="error").create_session().build(analysis.frames[0], 64, 48)


def test_held_parse_freezes_and_absent_id_clears_history():
    held = replace(FACE, observed=False, missing_frames=1)
    frames = (
        FrameAnalysis(0, (FACE,)),
        FrameAnalysis(1, (held,)),
        FrameAnalysis(2),
        FrameAnalysis(3, (held,)),
    )
    analysis, parsing = cache(frames, [(labels(),), (), (), ()])
    session = ParsedFaceMask(parsing).create_session()
    initial = full(session.build(analysis.frames[0], 64, 48)[0])
    np.testing.assert_array_equal(full(session.build(analysis.frames[1], 64, 48)[0]), initial)
    assert session.diagnostics == {0: "held"}
    assert session.build(analysis.frames[2], 64, 48) == {}
    session.build(analysis.frames[3], 64, 48)
    assert session.diagnostics == {0: "fallback_missing"}


def test_frame_gap_clears_held_history():
    held = replace(FACE, observed=False, missing_frames=1)
    analysis, parsing = cache(
        (FrameAnalysis(0, (FACE,)), FrameAnalysis(1), FrameAnalysis(2, (held,))),
        [(labels(),), (), ()],
    )
    session = ParsedFaceMask(parsing).create_session()
    session.build(analysis.frames[0], 64, 48)
    session.build(analysis.frames[2], 64, 48)
    assert session.diagnostics == {0: "fallback_missing"}
    with pytest.raises(ValueError, match="increase"):
        session.build(analysis.frames[2], 64, 48)


def test_cache_roundtrip_nonpickle_and_existing_file_preservation(tmp_path):
    analysis, parsing = cache()
    path = tmp_path / "parsing.npz"
    parsing.save(path)
    loaded = VideoParsing.load(path)
    assert loaded == parsing
    loaded.validate_analysis(analysis)
    np.testing.assert_array_equal(loaded.frames[0][0].labels, labels())
    assert not loaded.frames[0][0].labels.flags.writeable
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        parsing.save(path)
    assert path.read_bytes() == before
    parsing.save(path, overwrite=True)
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["metadata"].item()))["schema_version"] == 1


def test_cache_binds_geometry_source_frame_and_dimensions():
    analysis, parsing = cache()
    with pytest.raises(ValueError, match="source and analysis"):
        parsing.validate_analysis(replace(analysis, source_sha256="1" * 64))
    with pytest.raises(ValueError, match="face analysis"):
        parsing.for_frame(FrameAnalysis(0), 64, 48)
    with pytest.raises(ValueError, match="dimensions"):
        parsing.for_frame(analysis.frames[0], 65, 48)
    with pytest.raises(ValueError, match="interval"):
        parsing.for_frame(FrameAnalysis(1), 64, 48)
    corrupt = replace(parsing, frames=((),))
    with pytest.raises(ValueError, match="every observed"):
        corrupt.for_frame(analysis.frames[0], 64, 48)


def test_label_grid_is_immutable_and_validated():
    data = labels()
    record = FaceParsing(0, (-2, -3, 62, 45), data)
    data[:] = 0
    assert record.labels[16, 16] == 1
    with pytest.raises(ValueError):
        record.labels[0, 0] = 1
    for invalid in (
        np.zeros((2, 2), float),
        np.full((2, 2), 19, np.uint8),
        np.zeros((0, 2), np.uint8),
    ):
        with pytest.raises(ValueError, match="labels"):
            FaceParsing(0, (0, 0, 2, 2), invalid)


def test_label_sampling_uses_full_frame_edges():
    record = FaceParsing(0, (10, 20, 14, 24), np.array([[1, 17], [7, 8]], np.uint8))
    result = record.labels_at(
        np.array([10.5, 12.5, 10.5, 12.5, 14]), np.array([20.5, 20.5, 22.5, 22.5, 22.5])
    )
    np.testing.assert_array_equal(result, [1, 17, 7, 8, 0])
    with pytest.raises(ValueError, match="finite"):
        record.labels_at(np.array([np.nan]), np.array([0]))


def test_enclosed_background_holes_are_filled_but_hair_holes_remain_excluded():
    data = labels()
    data[16, 16] = 0
    data[17, 17] = 17
    analysis, parsing = cache(maps=[(data,)])
    session = ParsedFaceMask(parsing).create_session()
    result = full(session.build(analysis.frames[0], 64, 48)[0])
    assert session.diagnostics == {0: "parsed"}
    assert result[16, 16] == 1
    assert result[17, 17] == 0


def test_frame_edge_mapping_preserves_unclipped_coordinates():
    analysis, parsing = cache()
    data = labels()
    # Shift grid and all face geometry left by eight pixels, then clip at frame edge.
    shifted = replace(
        FACE,
        box=Box(-2, 6, 18, 26),
        landmarks=Landmarks(NAMES, tuple((x - 8, y) for x, y in MARKS.points)),
    )
    original = full(ParsedFaceMask(parsing).create_session().build(analysis.frames[0], 64, 48)[0])
    a, p = cache((FrameAnalysis(0, (shifted,)),), [(data,)])
    p = replace(p, frames=((FaceParsing(0, (-8, 0, 56, 48), data),),))
    clipped = full(ParsedFaceMask(p).create_session().build(a.frames[0], 64, 48)[0])
    np.testing.assert_array_equal(clipped[:, :56], original[:, 8:])


def test_crop_sampling_preserves_border_geometry_and_rgb():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = np.arange(8)[None, :] * 20
    crop = _parsing_crop(rgb, (-4, 0, 4, 8))
    assert crop.shape == (512, 512, 3)
    assert np.all(crop[:, :200, 0] == 0)
    assert np.all(crop[..., 1:] == 0)
    assert crop[256, -1, 0] > 50


class ControlledParser:
    def __init__(self):
        self.calls = 0

    def parse(self, image):
        self.calls += 1
        result = np.zeros((64, 64), np.uint8)
        result[12:52, 16:48] = 1
        result[6:12, 16:48] = 17
        return result

    def metadata(self):
        return {"name": "controlled_parser", "labels": list(FACE_PARSING_LABELS)}


def test_parse_and_lossless_render_reuse_inference(video, detector, tmp_path):
    source, inputs = video
    analysis = analyze_video(source, detector=detector)
    parser = ControlledParser()
    parsing = parse_video(source, analysis=analysis, parser=parser)
    assert parser.calls == 6 and detector.calls == 6
    parsing.save(tmp_path / "labels.npz")
    loaded = VideoParsing.load(tmp_path / "labels.npz")
    mask = PerFaceMask(ParsedFaceMask(loaded), {0: ParsedFaceMask(loaded, expansion=2)})
    output = tmp_path / "rendered.avi"
    with pytest.warns(VideoExportWarning):
        render_video(source, output, analysis=analysis, effect=Pixelate(2), mask=mask, codec="FFV1")
    renderer = FrameRenderer(Pixelate(2), mask)
    capture = cv2.VideoCapture(str(output))
    for frame, image in zip(analysis.frames, inputs, strict=True):
        ok, result = capture.read()
        assert ok
        expected = renderer.render(image, frame, timestamp=frame.index / 12)
        np.testing.assert_array_equal(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), expected)
    assert not capture.read()[0]
    capture.release()
    assert parser.calls == 6 and detector.calls == 6


def test_partial_cache_and_held_inference_schedule(video, detector):
    source, _ = video
    analysis = analyze_video(source, detector=detector, detection_interval=2)
    parser = ControlledParser()
    parsing = parse_video(source, analysis=analysis, parser=parser, start=2, stop=5)
    assert parser.calls == 2
    assert parsing.start == 2 and len(parsing.frames) == 3
    assert not parsing.frames[1]
    parsing.validate_analysis(analysis, complete=False)
    with pytest.raises(ValueError, match="complete"):
        parsing.validate_analysis(analysis)


def test_parse_and_render_cli(video, detector, tmp_path, monkeypatch):
    from anamorph import cli

    source, _ = video
    analysis = analyze_video(source, detector=detector)
    analysis_path = tmp_path / "analysis.json"
    analysis.save(analysis_path)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"controlled model")
    parser = ControlledParser()
    monkeypatch.setattr(cli, "BiSeNetFaceParser", lambda _: parser)
    parsing_path = tmp_path / "parsing.npz"
    main(["parse", str(source), str(analysis_path), str(parsing_path), "--model", str(model)])
    assert parser.calls == 6
    output = tmp_path / "parsed.avi"
    with pytest.warns(VideoExportWarning):
        main(
            [
                "render",
                str(source),
                str(analysis_path),
                str(output),
                "--shape",
                "parsed",
                "--parsing",
                str(parsing_path),
                "--codec",
                "FFV1",
                "--expansion",
                "2",
                "--recovery-frames",
                "3",
            ]
        )
    assert output.exists() and parser.calls == 6
    with pytest.raises(SystemExit):
        main(
            [
                "render",
                str(source),
                str(analysis_path),
                str(tmp_path / "invalid.avi"),
                "--expansion",
                "2",
            ]
        )
    assert not (tmp_path / "invalid.avi").exists()
    with pytest.raises(SystemExit):
        main(["render", str(source), str(analysis_path), str(output), "--shape", "parsed"])
    with pytest.raises(SystemExit):
        main(
            [
                "parse",
                str(source),
                str(analysis_path),
                str(source),
                "--model",
                str(model),
                "--overwrite",
            ]
        )


def test_render_rejects_different_track_analysis_before_export(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    parsing = parse_video(source, analysis=analysis, parser=ControlledParser())
    altered = replace(
        analysis,
        frames=tuple(
            replace(
                frame,
                faces=tuple(replace(face, track_id=face.track_id + 1) for face in frame.faces),
            )
            for frame in analysis.frames
        ),
    )
    output = tmp_path / "invalid.avi"
    with pytest.raises(ValueError, match="source and analysis"):
        render_video(
            source,
            output,
            analysis=altered,
            effect=Pixelate(),
            mask=ParsedFaceMask(parsing),
            codec="FFV1",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feather": -1},
        {"feather": float("nan")},
        {"stabilization": 0},
        {"stabilization": 1.1},
        {"included_labels": ()},
        {"included_labels": ("skin", "skin")},
        {"included_labels": ("unknown",)},
        {"included_labels": ("background",)},
        {"included_labels": ("skin",)},
        {"fallback": "empty"},
        {"expansion": float("nan")},
        {"expansion": float("inf")},
        {"fallback_mask": None},
        {"boundary_labels": ("skin",)},
        {"boundary_labels": ("unknown",)},
        {"boundary_labels": ("hair", "hair")},
        {"boundary_labels": "hair"},
        {"recovery_frames": 0},
        {"recovery_frames": True},
        {"recovery_frames": 1.5},
    ],
)
def test_mask_parameter_validation(kwargs):
    _, parsing = cache()
    with pytest.raises(ValueError):
        ParsedFaceMask(parsing, **kwargs)


class FakeNet:
    def setPreferableBackend(self, backend):
        pass

    def setPreferableTarget(self, target):
        pass

    def setInput(self, tensor):
        self.tensor = tensor

    def forward(self, name):
        assert name == "output"
        result = np.zeros((1, 19, 512, 512), np.float32)
        result[:, 10] = 1
        return result


def test_bisenet_rgb_normalization_and_primary_head(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"controlled model")
    net = FakeNet()
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _: net)
    parser = BiSeNetFaceParser(path)
    rgb = np.full((4, 8, 3), (255, 0, 128), np.uint8)
    result = parser.parse(rgb)
    assert np.all(result == 10)
    assert net.tensor.shape == (1, 3, 512, 512)
    np.testing.assert_allclose(
        net.tensor[0, :, 0, 0],
        (np.array([1, 0, 128 / 255]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
        rtol=1e-6,
    )
    assert parser.metadata()["backend"] == "opencv_dnn_cpu"


@pytest.mark.parametrize("invalid", ["shape", "nan", "head_error"])
def test_bisenet_refuses_wrong_export_outputs(tmp_path, monkeypatch, invalid):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"controlled model")
    net = FakeNet()

    def forward(_):
        if invalid == "head_error":
            raise cv2.error("Missing primary head")
        return (
            np.zeros((1, 3, 512, 512), np.float32)
            if invalid == "shape"
            else np.full((1, 19, 512, 512), np.nan, np.float32)
        )

    monkeypatch.setattr(net, "forward", forward)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _: net)
    with pytest.raises(ValueError, match="BiSeNet"):
        BiSeNetFaceParser(path).parse(np.zeros((8, 8, 3), np.uint8))


def test_bisenet_rejects_missing_file_and_lfs_pointer(tmp_path):
    with pytest.raises(FileNotFoundError):
        BiSeNetFaceParser(tmp_path / "missing.onnx")
    pointer = tmp_path / "pointer.onnx"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS"):
        BiSeNetFaceParser(pointer)


def test_unsupported_cache_layout_is_refused(tmp_path):
    _, parsing = cache()
    path = tmp_path / "labels.npz"
    parsing.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["labels"][17] = "unknown_class"
    arrays["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="label layout"):
        VideoParsing.load(path)


def test_cache_write_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    _, parsing = cache()
    path = tmp_path / "existing.npz"
    path.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("Controlled write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="Controlled"):
        parsing.save(path, overwrite=True)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_real_bisenet_cpu_inference(face_parser_model):
    parser = BiSeNetFaceParser(face_parser_model)
    result = parser.parse(np.full((64, 64, 3), 127, np.uint8))
    assert result.shape == (512, 512) and result.dtype == np.uint8
    assert np.all(result < 19)
