import math
from dataclasses import replace

import numpy as np
import pytest

from anamorph import (
    Box,
    Face,
    FaceMaskGeometry,
    FrameAnalysis,
    FrameRenderer,
    LandmarkFaceMask,
    Landmarks,
    MaskSession,
    PerFaceMask,
    SpatialEffect,
)

BOX = Box(20, 20, 100, 120)
NAMES = ("right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner")


def test_per_face_mask_copies_overrides_and_preserves_defaults():
    default = LandmarkFaceMask(feather=0)
    override = LandmarkFaceMask(scale_x=0.5, feather=0)
    overrides = {1: override}
    mask = PerFaceMask(default, overrides)
    overrides.clear()
    with pytest.raises(TypeError):
        mask.overrides[0] = default
    faces = FrameAnalysis(0, (Face(0, BOX), Face(1, BOX)))
    regions = MaskSession(mask).build(faces, 160, 160)
    np.testing.assert_array_equal(full(regions[0]), full(default.build(BOX, 160, 160)))
    np.testing.assert_array_equal(full(regions[1]), full(override.build(BOX, 160, 160)))


@pytest.mark.parametrize(
    "overrides",
    [
        {-1: LandmarkFaceMask()},
        {True: LandmarkFaceMask()},
        {"1": LandmarkFaceMask()},
        {1: object()},
    ],
)
def test_per_face_mask_rejects_invalid_overrides(overrides):
    with pytest.raises(ValueError):
        PerFaceMask(overrides=overrides)


def test_nested_per_face_masks_select_only_requested_tracks():
    default = LandmarkFaceMask(feather=0)
    narrow = LandmarkFaceMask(scale_x=0.5, feather=0)
    nested = PerFaceMask(default, {1: narrow})
    session = PerFaceMask(nested, {0: narrow}).create_session()
    frame = FrameAnalysis(0, (Face(0, BOX), Face(1, BOX), Face(2, BOX)))
    regions = session.build(frame, 160, 160)
    assert regions.keys() == {0, 1, 2}
    assert session.diagnostics == {0: "geometry", 1: "geometry", 2: "geometry"}
    np.testing.assert_array_equal(full(regions[0]), full(regions[1]))
    assert full(regions[2]).sum() > full(regions[1]).sum()
    with pytest.raises(ValueError, match="increase"):
        session.build(frame, 160, 160)


def points(angle=0, dx=0, dy=0):
    raw = np.array([[44, 57], [76, 57], [60, 75], [48, 92], [72, 92]], dtype=float)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return Landmarks(NAMES, (raw - (60, 57)) @ rotation.T + (60 + dx, 57 + dy))


def full(region, width=160, height=160):
    alpha = np.zeros((height, width), np.float32)
    if region is not None:
        x1, y1, x2, y2 = region.bounds
        alpha[y1:y2, x1:x2] = region.alpha
    return alpha


def frame(index, angle=0, track=0, box=BOX, landmarks=True):
    return FrameAnalysis(index, (Face(track, box, landmarks=points(angle) if landmarks else None),))


def test_eye_line_guides_roll_and_features_have_opaque_coverage():
    mask = LandmarkFaceMask()
    for angle in (0, -0.4, 0.4):
        marks = points(angle)
        geometry = mask.geometry(BOX, marks)
        assert geometry.landmark_guided
        assert geometry.angle == pytest.approx(angle)
        alpha = full(mask.build(BOX, 160, 160, landmarks=marks))
        for x, y in marks.points:
            assert alpha[int(y), int(x)] == 1


def test_protruding_profile_nose_gets_local_opaque_protection():
    box = Box(6, 6, 26, 26)
    marks = Landmarks(NAMES, ((12, 12), (20, 12), (4, 16), (13, 20), (19, 20)))
    plain = LandmarkFaceMask(protect_features=False)
    protected = LandmarkFaceMask()
    before = full(plain.build(box, 160, 160, landmarks=marks))
    after = full(protected.build(box, 160, 160, landmarks=marks))
    assert before[16, 4] < 1 and after[16, 4] == 1
    assert np.all(after >= before)
    assert np.count_nonzero(after != before) < 100
    session = MaskSession(protected)
    observed = FrameAnalysis(0, (Face(0, box, landmarks=marks),))
    np.testing.assert_array_equal(full(session.build(observed, 160, 160)[0]), after)


def test_feature_guard_ignores_far_outliers_and_incomplete_layouts():
    box = Box(6, 6, 26, 26)
    for marks in [
        Landmarks(NAMES, ((12, 12), (20, 12), (-100, 16), (13, 20), (19, 20))),
        Landmarks(("nose_tip",), ((4, 16),)),
    ]:
        before = full(
            LandmarkFaceMask(protect_features=False).build(box, 160, 160, landmarks=marks)
        )
        after = full(LandmarkFaceMask().build(box, 160, 160, landmarks=marks))
        np.testing.assert_array_equal(after, before)


def test_feature_guard_option_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        LandmarkFaceMask(protect_features=1)


def test_forehead_and_chin_belong_to_opaque_core():
    mask = LandmarkFaceMask()
    geometry = mask.geometry(BOX, points())
    alpha = full(mask.build(BOX, 160, 160, landmarks=points()))
    x, y = geometry.center
    assert alpha[int(y - 0.9 * geometry.radii[1]), int(x)] == 1
    assert alpha[int(y + 0.9 * geometry.radii[1]), int(x)] == 1
    assert alpha[int(y), int(x + 1.2 * geometry.radii[0])] == 0


def test_downward_pose_retains_forehead_coverage():
    marks = Landmarks(NAMES, ((44, 90), (76, 90), (60, 100), (48, 105), (72, 105)))
    mask = LandmarkFaceMask()
    assert mask.geometry(BOX, marks).landmark_guided
    alpha = full(mask.build(BOX, 160, 160, landmarks=marks))
    assert alpha[25, 60] == 1
    assert alpha[110, 60] == 1


def test_chin_extension_preserves_upper_boundary():
    geometry = FaceMaskGeometry((80, 80), (30, 40))
    baseline = full(LandmarkFaceMask(chin_extension=0).rasterize(geometry, 160, 160))
    extended = full(LandmarkFaceMask(chin_extension=0.3).rasterize(geometry, 160, 160))
    np.testing.assert_array_equal(baseline[:80], extended[:80])
    assert np.all(extended >= baseline)
    assert np.any(extended[80:] > baseline[80:])


def test_continuous_alpha_matches_pixel_center_rasterization():
    mask = LandmarkFaceMask()
    geometry = mask.geometry(BOX, points(0.4))
    region = mask.rasterize(geometry, 160, 160)
    x1, y1, x2, y2 = region.bounds
    y, x = np.mgrid[y1:y2, x1:x2]
    np.testing.assert_array_equal(mask.alpha_at(geometry, x + 0.5, y + 0.5), region.alpha)


def test_outward_feather_preserves_every_opaque_core_pixel():
    mask = LandmarkFaceMask(feather=0)
    hard = full(mask.build(BOX, 160, 160, landmarks=points(0.3)))
    soft = full(replace(mask, feather=0.3).build(BOX, 160, 160, landmarks=points(0.3)))
    assert soft.dtype == np.float32
    assert np.isfinite(soft).all()
    np.testing.assert_array_equal(soft[hard == 1], 1)
    assert np.any((soft > 0) & (soft < 1) & (hard == 0))
    assert soft.min() == 0 and soft.max() == 1


def test_rotation_is_in_image_coordinates():
    mask = LandmarkFaceMask()
    geometry = FaceMaskGeometry((80, 80), (25, 40))
    original = full(mask.rasterize(geometry, 160, 160))
    rotated = full(mask.rasterize(replace(geometry, angle=math.pi / 2), 160, 160))
    np.testing.assert_allclose(rotated, np.rot90(original, -1), atol=1e-6)


def test_clipping_preserves_rotated_geometry():
    mask = LandmarkFaceMask()
    original = full(mask.rasterize(FaceMaskGeometry((40, 70), (35, 50), 0.4), 160, 160))
    clipped = full(mask.rasterize(FaceMaskGeometry((0, 70), (35, 50), 0.4), 160, 160))
    np.testing.assert_array_equal(clipped[:, :120], original[:, 40:])
    assert mask.build(Box(500, 500, 600, 600), 160, 160) is None


@pytest.mark.parametrize(
    "marks",
    [
        None,
        Landmarks(("nose_tip",), ((60, 70),)),
        Landmarks(NAMES, ((60, 60),) * 5),
        Landmarks(NAMES, ((76, 57), (44, 57), (60, 75), (48, 92), (72, 92))),
        Landmarks(NAMES, ((44, 57), (76, 57), (60, 50), (48, 92), (72, 92))),
        Landmarks(NAMES, ((44, 57), (76, 57), (60, 75), (48, 30), (72, 30))),
        Landmarks(NAMES, ((-50, 57), (76, 57), (60, 75), (48, 92), (72, 92))),
    ],
)
def test_missing_or_implausible_landmarks_use_upright_fallback(marks):
    mask = LandmarkFaceMask()
    assert mask.geometry(BOX, marks) == mask.geometry(BOX)
    assert not mask.geometry(BOX, marks).landmark_guided


def test_temporal_filter_preserves_current_coverage_and_softens_changes():
    mask = LandmarkFaceMask(stabilization=0.3)
    session = MaskSession(mask)
    session.build(frame(0, -0.4), 160, 160)
    current = full(mask.build(BOX, 160, 160, landmarks=points(0.4)))
    stable = full(session.build(frame(1, 0.4), 160, 160)[0])
    assert np.all(stable >= current)
    assert np.any(stable > current)


def test_translation_does_not_add_mask_lag():
    session = MaskSession()
    before = full(session.build(frame(0), 160, 160)[0])
    moved = FrameAnalysis(1, (Face(0, Box(30, 30, 110, 130), landmarks=points(dx=10, dy=10)),))
    after = full(session.build(moved, 160, 160)[0])
    np.testing.assert_array_equal(after[10:, 10:], before[:-10, :-10])


def test_temporal_state_is_independent_for_each_track_and_session():
    session = MaskSession()
    session.build(frame(0, -0.4), 160, 160)
    result = session.build(
        FrameAnalysis(
            1,
            (
                Face(0, BOX, landmarks=points(0.4)),
                Face(1, BOX, landmarks=points(0.4)),
            ),
        ),
        160,
        160,
    )
    fresh = MaskSession().build(frame(1, 0.4, track=1), 160, 160)[1]
    np.testing.assert_array_equal(full(result[1]), full(fresh))
    assert np.any(full(result[0]) > full(result[1]))


@pytest.mark.parametrize("reset", ["gap", "absent", "fallback", "jump", "dimensions"])
def test_history_resets(reset):
    session = MaskSession()
    session.build(frame(0, -0.4), 160, 160)
    index, width = 1, 160
    if reset == "gap":
        index = 3
    elif reset == "absent":
        session.build(FrameAnalysis(1), 160, 160)
        index = 2
    elif reset == "fallback":
        session.build(frame(1, landmarks=False), 160, 160)
        index = 2
    elif reset == "jump":
        session.build(frame(1, box=Box(300, 300, 380, 400)), 160, 160)
        index = 2
    else:
        width = 170
    result = session.build(frame(index, 0.4), width, 160)[0]
    fresh = MaskSession().build(frame(index, 0.4), width, 160)[0]
    np.testing.assert_array_equal(full(result, width), full(fresh, width))


def test_held_face_keeps_mask_and_absent_face_removes_it():
    session = MaskSession()
    first = frame(0)
    original = full(session.build(first, 160, 160)[0])
    held = replace(first.faces[0], observed=False, missing_frames=1)
    np.testing.assert_array_equal(
        full(session.build(FrameAnalysis(1, (held,)), 160, 160)[0]), original
    )
    assert session.build(FrameAnalysis(2), 160, 160) == {}
    with pytest.raises(ValueError, match="increase"):
        session.build(FrameAnalysis(2), 160, 160)


def test_held_detection_does_not_advance_geometry_filter():
    session = MaskSession()
    session.build(frame(0, -0.4), 160, 160)
    current = frame(1, 0.4)
    observed = full(session.build(current, 160, 160)[0])
    held = replace(current.faces[0], observed=False, missing_frames=1)
    np.testing.assert_array_equal(
        full(session.build(FrameAnalysis(2, (held,)), 160, 160)[0]), observed
    )


def test_renderer_uses_landmarks_and_default_mask_session():
    class White(SpatialEffect):
        def apply(self, image):
            return np.full_like(image, 255)

    renderer = FrameRenderer(White())
    session = MaskSession()
    for analysis in (frame(0, -0.4), frame(1, 0.4)):
        output = renderer.render(np.zeros((160, 160, 3), np.uint8), analysis, timestamp=0)
        alpha = full(session.build(analysis, 160, 160)[0])
        np.testing.assert_array_equal(output[..., 0], np.rint(alpha * 255).astype(np.uint8))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_x": 0},
        {"scale_y": -1},
        {"offset_y": float("nan")},
        {"feather": -0.1},
        {"feather": 1.1},
        {"stabilization": 0},
        {"stabilization": 1.1},
        {"chin_extension": -0.1},
        {"chin_extension": 1.1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LandmarkFaceMask(**kwargs)


@pytest.mark.parametrize(
    "geometry",
    [
        {"center": (0,)},
        {"radii": (0, 2)},
        {"angle": float("inf")},
    ],
)
def test_invalid_geometry(geometry):
    options = {"center": (0, 0), "radii": (2, 2)} | geometry
    with pytest.raises(ValueError):
        FaceMaskGeometry(**options)


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        LandmarkFaceMask().build(BOX, 0, 10)
    with pytest.raises(ValueError, match="dimensions"):
        MaskSession().build(FrameAnalysis(0), 10, 0)
