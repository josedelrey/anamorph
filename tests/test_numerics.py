import numpy as np
import pytest

from anamorph import (
    Box,
    Compose,
    EffectContext,
    Face,
    FaceMask,
    FrameAnalysis,
    FrameRenderer,
    GaussianBlur,
    Pixelate,
    SpatialEffect,
    TemporalBlend,
    alpha_composite,
)


class Solid(SpatialEffect):
    def __init__(self, value):
        self.value = value

    def apply(self, image):
        return np.full_like(image, self.value)


def test_box_coordinates():
    box = Box(-0.2, 1.2, 4.1, 5.2)
    assert box.bounds(4, 8) == (0, 1, 4, 6)
    assert Box(10, 10, 12, 12).bounds(4, 8) is None
    assert Box(0, 0, 2, 2).iou(Box(1, 0, 3, 2)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("values", [(1, 1, 1, 2), (0, 0, -1, 1), (0, 0, float("nan"), 1)])
def test_invalid_box(values):
    with pytest.raises(ValueError):
        Box(*values)


def test_rectangular_mask_and_clipping():
    mask = FaceMask("rectangle", scale_x=1, scale_y=1, offset_y=0, feather=0)
    region = mask.build(Box(-2, 1, 4, 5), 10, 10)
    assert region.bounds == (0, 1, 4, 5)
    np.testing.assert_array_equal(region.alpha, np.ones((4, 4), dtype=np.float32))
    assert mask.build(Box(20, 20, 30, 30), 10, 10) is None


def test_clipped_mask_keeps_original_center():
    mask = FaceMask(scale_x=1, scale_y=1, offset_y=0, feather=0.2)
    full = mask.build(Box(2, 2, 10, 10), 12, 12)
    clipped = mask.build(Box(-2, 2, 6, 10), 12, 12)
    np.testing.assert_array_equal(clipped.alpha, full.alpha[:, 2:])


def test_ellipse_feather_is_bounded_and_symmetric():
    region = FaceMask(scale_x=1, scale_y=1, offset_y=0, feather=0.5).build(
        Box(0, 0, 10, 10), 10, 10
    )
    assert region.alpha.dtype == np.float32
    assert region.alpha[0, 0] == 0
    assert region.alpha[5, 5] == 1
    assert np.any((region.alpha > 0) & (region.alpha < 1))
    np.testing.assert_array_equal(region.alpha, region.alpha[::-1, ::-1])


def test_alpha_endpoints_and_midpoint():
    original = np.zeros((1, 3, 3), dtype=np.uint8)
    changed = np.full_like(original, 100)
    result = alpha_composite(original, changed, np.array([[0, 0.5, 1]], dtype=np.float32))
    np.testing.assert_array_equal(result[0, :, 0], [0, 50, 100])


@pytest.mark.parametrize("alpha", [np.array([[np.nan]]), np.array([[1.1]]), np.array([[1]])])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        alpha_composite(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8), alpha)


def test_gaussian_reduces_variance_and_preserves_dc(texture):
    result = GaussianBlur(3).apply(texture)
    assert result.shape == texture.shape
    assert result.dtype == np.uint8
    assert result.var() < texture.var() / 20
    constant = np.full((3, 5, 3), 79, dtype=np.uint8)
    np.testing.assert_array_equal(GaussianBlur(8).apply(constant), constant)


def test_pixelation_known_area_average():
    image = np.repeat(np.array([[0, 20], [100, 120]], dtype=np.uint8)[..., None], 3, axis=2)
    np.testing.assert_array_equal(Pixelate(1).apply(image), np.full_like(image, 60))
    np.testing.assert_array_equal(Pixelate(8).apply(image), image)


def test_pixelation_blocks(texture):
    result = Pixelate(4).apply(texture)
    assert np.unique(result.reshape(-1, 3), axis=0).shape[0] <= 16


def test_renderer_preserves_input_and_outside_mask(texture):
    before = texture.copy()
    mask = FaceMask("rectangle", 1, 1, 0, 0)
    renderer = FrameRenderer(Solid(0), mask)
    result = renderer.render(texture, FrameAnalysis(0, (Face(0, Box(4, 6, 12, 14)),)), timestamp=0)
    np.testing.assert_array_equal(texture, before)
    expected = before.copy()
    expected[6:14, 4:12] = 0
    np.testing.assert_array_equal(result, expected)


def test_compose_applies_in_order(texture):
    renderer = FrameRenderer(Compose(Solid(10), Solid(50)), FaceMask("rectangle", 1, 1, 0, 0))
    result = renderer.render(texture, FrameAnalysis(0, (Face(0, Box(0, 0, 64, 48)),)), timestamp=0)
    np.testing.assert_array_equal(result, np.full_like(texture, 50))


def test_overlap_reads_original_and_is_deterministic(texture):
    class TrackSolid(SpatialEffect):
        def apply(self, image):
            return 255 - image

    mask = FaceMask("rectangle", 1, 1, 0, 0)
    faces = (Face(9, Box(0, 0, 20, 20)), Face(2, Box(0, 0, 20, 20)))
    result = FrameRenderer(TrackSolid(), mask).render(texture, FrameAnalysis(0, faces), timestamp=0)
    np.testing.assert_array_equal(result[:20, :20], 255 - texture[:20, :20])


def context(index, track=0):
    return EffectContext(index, index / 10, Face(track, Box(0, 0, 2, 2)), (0, 0, 2, 2))


def test_temporal_blend_state_identity_resize_and_gap():
    effect = TemporalBlend(0.25)
    session = effect.create_session()
    black = np.zeros((2, 2, 3), np.uint8)
    white = np.full_like(black, 100)
    session.apply(black, context(0))
    np.testing.assert_array_equal(session.apply(white, context(1)), np.full_like(white, 25))
    np.testing.assert_array_equal(session.apply(white, context(1, 1)), white)
    resized = session.apply(np.full((4, 4, 3), 100, np.uint8), context(2))
    assert resized[0, 0, 0] == 44
    np.testing.assert_array_equal(session.apply(white, context(4)), white)
    np.testing.assert_array_equal(effect.create_session().apply(white, context(1)), white)
    session.end_frame(set())
    np.testing.assert_array_equal(session.apply(black, context(5)), black)


def test_renderer_empty_frame_clears_temporal_state():
    renderer = FrameRenderer(TemporalBlend(0.5), FaceMask("rectangle", 1, 1, 0, 0))
    face = Face(0, Box(0, 0, 2, 2))
    renderer.render(np.zeros((2, 2, 3), np.uint8), FrameAnalysis(0, (face,)), timestamp=0)
    renderer.render(np.zeros((2, 2, 3), np.uint8), FrameAnalysis(1), timestamp=1)
    white = np.full((2, 2, 3), 100, np.uint8)
    np.testing.assert_array_equal(
        renderer.render(white, FrameAnalysis(2, (face,)), timestamp=2), white
    )
    with pytest.raises(ValueError, match="increase"):
        renderer.render(white, FrameAnalysis(2), timestamp=2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianBlur(0),
        lambda: GaussianBlur(float("nan")),
        lambda: Pixelate(0),
        lambda: Pixelate(1.5),
        lambda: TemporalBlend(0),
        lambda: Compose(),
        lambda: FaceMask(feather=1.1),
        lambda: FaceMask(scale_x=0),
        lambda: FaceMask(shape="wrong"),
    ],
)
def test_parameter_validation(factory):
    with pytest.raises(ValueError):
        factory()


def test_bad_effect_contract(texture):
    class Bad(SpatialEffect):
        def apply(self, image):
            return image.astype(np.float32)

    renderer = FrameRenderer(Compose(Bad(), Solid(0)))
    with pytest.raises(ValueError, match="uint8"):
        renderer.render(texture, FrameAnalysis(0, (Face(0, Box(4, 4, 20, 20)),)), timestamp=0)
