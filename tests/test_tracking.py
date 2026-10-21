import pytest

from anamorph import Box, Detection, IoUTracker


def detection(x):
    return Detection(Box(x, 0, x + 10, 10))


def test_multiple_faces_and_detection_order():
    tracker = IoUTracker(smoothing=1)
    first = tracker.update([detection(30), detection(0)])
    second = tracker.update([detection(1), detection(31)])
    assert [face.track_id for face in first] == [0, 1]
    assert [(face.track_id, face.box.x1) for face in second] == [(0, 1), (1, 31)]


def test_missing_expiration_reacquisition_and_reset():
    tracker = IoUTracker(max_missing=2)
    tracker.update([detection(0)])
    held = tracker.update([])[0]
    assert not held.observed and held.missing_frames == 1
    assert tracker.update([detection(1)])[0].track_id == 0
    assert tracker.update([])[0].missing_frames == 1
    assert tracker.update([])[0].missing_frames == 2
    assert tracker.update([]) == ()
    assert tracker.update([detection(0)])[0].track_id == 1
    tracker.reset()
    assert tracker.update([detection(0)])[0].track_id == 0


def test_smoothing():
    tracker = IoUTracker(smoothing=0.25)
    tracker.update([detection(0)])
    assert tracker.update([detection(4)])[0].box.x1 == 1


def test_one_detection_cannot_update_two_tracks():
    tracker = IoUTracker(smoothing=1, max_missing=1)
    tracker.update([detection(0), detection(5)])
    faces = tracker.update([detection(1)])
    assert sum(face.observed for face in faces) == 1
    assert faces[0].box.x1 == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"min_iou": 0},
        {"min_iou": 2},
        {"max_missing": -1},
        {"max_missing": 1.1},
        {"smoothing": 0},
        {"smoothing": float("nan")},
    ],
)
def test_tracker_parameters(settings):
    with pytest.raises(ValueError):
        IoUTracker(**settings)
