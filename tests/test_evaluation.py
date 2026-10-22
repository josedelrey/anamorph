import numpy as np
import pytest

from anamorph.evaluation import evaluate_mask, polygon_mask, temporal_error_change


def rectangle(a=2, b=2, c=6, d=6):
    return polygon_mask([[(a, b), (c, b), (c, d), (a, d)]], 12, 10)


def test_polygon_edges_use_pixel_centers_and_union_concave_shapes():
    actual = polygon_mask([[(1, 1), (5, 1), (5, 3), (3, 3), (3, 5), (1, 5)]], 8, 8)
    expected = np.zeros((8, 8), bool)
    expected[1:3, 1:5] = True
    expected[3:5, 1:3] = True
    np.testing.assert_array_equal(actual, expected)
    assert polygon_mask([[(0, 0), (1, 0), (1, 1), (0, 1)]], 2, 2).sum() == 1


@pytest.mark.parametrize(
    "points",
    [
        [],
        [(1, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(-1, 0), (2, 0), (2, 2)],
        [(0, 0), (np.nan, 0), (2, 2)],
    ],
)
def test_invalid_polygons_refused(points):
    with pytest.raises(ValueError):
        polygon_mask([points], 8, 8)


def test_exact_mask_has_zero_error_and_unit_iou():
    reference = rectangle()
    result = evaluate_mask(reference.astype(float), reference)
    assert result["face_pixels"] == 16
    assert result["missed_opaque_face_pixels"] == 0
    assert result["mean_face_alpha_deficit"] == 0
    assert result["opaque_iou"] == 1
    assert result["boundary_p95_pixels"] == 0
    assert result["extra_alpha_pixels"] == 0


def test_soft_alpha_and_excluded_area_measured_separately():
    reference = rectangle()
    ears = rectangle(6, 2, 7, 6)
    alpha = reference.astype(float)
    alpha[2:4, 2:6] = 0.5
    alpha[ears] = 0.25
    result = evaluate_mask(alpha, reference, exclusions={"ears": ears})
    assert result["missed_opaque_face_fraction"] == 0.5
    assert result["mean_face_alpha_deficit"] == 0.25
    assert result["opaque_iou"] == 0.5
    assert result["extra_alpha_pixels"] == 1
    assert result["exclusions"]["ears"] == {"pixels": 4, "mean_alpha": 0.25, "opaque_fraction": 0}


def test_ignored_uncertain_area_does_not_hide_interior_failures():
    reference = rectangle()
    ignore = rectangle(2, 2, 6, 3)
    alpha = reference.astype(float)
    alpha[2:4, 2:6] = 0
    result = evaluate_mask(alpha, reference, ignore=ignore)
    assert result["face_pixels"] == 12
    assert result["missed_opaque_face_pixels"] == 4
    assert result["opaque_iou"] == pytest.approx(2 / 3)


def test_missing_mask_is_complete_miss_with_undefined_boundary():
    ref = rectangle()
    result = evaluate_mask(np.zeros_like(ref, float), ref)
    assert result["missed_opaque_face_fraction"] == 1
    assert result["opaque_iou"] == 0
    assert result["boundary_p95_pixels"] is None


def test_exclusions_cannot_overlap_known_face():
    ref = rectangle()
    with pytest.raises(ValueError, match="overlap"):
        evaluate_mask(ref.astype(float), ref, exclusions={"hair": ref})


@pytest.mark.parametrize("value", [np.nan, -0.1, 1.1])
def test_invalid_alpha_refused(value):
    ref = rectangle()
    with pytest.raises(ValueError, match="finite alpha"):
        evaluate_mask(np.full(ref.shape, value), ref)


def test_reference_translation_does_not_count_as_flicker():
    old = rectangle()
    new = rectangle(5, 3, 9, 7)
    # The same one-column undercoverage translated with the subject has zero change.
    a, b = old.astype(float), new.astype(float)
    a[2:6, 2] = 0
    b[3:7, 5] = 0
    assert temporal_error_change(a, old, b, new) == 0
    b[3:7, 5] = 1
    assert temporal_error_change(a, old, b, new) == 0.25


def test_temporal_invalid_shapes_and_ignored_pairs():
    ref = rectangle()
    alpha = ref.astype(float)
    assert temporal_error_change(alpha, ref, alpha, ref, ignore=np.ones_like(ref)) is None
    assert temporal_error_change(alpha, np.zeros_like(ref), alpha, ref) is None
    with pytest.raises(ValueError, match="dimensions"):
        temporal_error_change(alpha, ref, alpha[:8], ref[:8])


@pytest.mark.parametrize("use_config", [False, True])
def test_evaluation_example_replays_video_and_counts_missing_tracks(
    video, detector, tmp_path, monkeypatch, use_config
):
    import hashlib
    import json
    import runpy
    import sys
    from pathlib import Path

    from anamorph import analyze_video
    from anamorph.analysis import file_sha256

    source, images = video
    analysis = analyze_video(source, detector=detector)
    cache = tmp_path / "analysis.json"
    analysis.save(cache)
    references = tmp_path / "references.json"
    references.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": file_sha256(source),
                "analysis_sha256": file_sha256(cache),
                "width": 64,
                "height": 48,
                "annotation_provenance": "Controlled test rectangles, not human annotations",
                "frames": [
                    {
                        "index": i,
                        "rgb_sha256": hashlib.sha256(images[i].tobytes()).hexdigest(),
                        "faces": [
                            {
                                "reference_id": "a",
                                "track_id": 0 if i == 0 else None,
                                "face": [[[18, 14], [38, 14], [38, 34], [18, 34]]],
                            }
                        ],
                    }
                    for i in range(2)
                ],
            }
        )
    )
    output = tmp_path / "review"
    argv = ["evaluate_masks", str(source), str(cache), str(references), str(output)]
    if use_config:
        configuration = tmp_path / "settings.json"
        configuration.write_text(
            json.dumps(
                {"default": {"kind": "face"}, "overrides": {"0": {"kind": "face", "scale_x": 0.7}}}
            )
        )
        argv += ["--config", str(configuration)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/evaluate_masks.py"), run_name="__main__"
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status_counts"] == {"geometry": 1, "missing_track": 1}
    assert report["samples"][1]["missed_opaque_face_fraction"] == 1
    assert len(report["adjacent_frame_error_changes"]) == 1
    assert detector.calls == 6
    assert bool(report["mask_configuration"]) == use_config
    before = (output / "report.json").read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/evaluate_masks.py"),
            run_name="__main__",
        )
    assert (output / "report.json").read_bytes() == before
