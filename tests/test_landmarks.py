import json
from dataclasses import replace

import numpy as np
import pytest

from anamorph import Box, Detection, Face, IoUTracker, Landmarks, VideoAnalysis, analyze_video


def landmarks(x=0):
    return Landmarks(("nose_tip", "left_eye"), ((x + 5, 4), (x + 7, 2)))


def test_named_landmarks_and_affine_remapping():
    points = landmarks()
    assert points["nose_tip"] == (5, 4)
    with pytest.raises(KeyError):
        points["unknown"]
    mapped = points.remap(Box(0, 0, 10, 10), Box(10, 20, 30, 50))
    assert mapped["nose_tip"] == (20, 32)
    assert mapped["left_eye"] == (24, 26)
    assert points["nose_tip"] == (5, 4)


def test_landmarks_are_immutable_and_json_safe():
    raw = np.array([[2, 3]], dtype=np.float32)
    points = Landmarks(["nose_tip"], raw)
    raw[0] = 100
    assert points.points == ((2.0, 3.0),)
    assert type(points.points[0][0]) is float
    assert points.names == ("nose_tip",)


@pytest.mark.parametrize(
    "names,points",
    [
        ((), ()),
        (("eye", "eye"), ((0, 0), (1, 1))),
        (("",), ((0, 0),)),
        (("nose",), ()),
        (("nose",), ((1, 2, 3),)),
        (("nose",), ((float("nan"), 1),)),
        ("nose", ((0, 0),)),
    ],
)
def test_invalid_landmark_layout(names, points):
    with pytest.raises(ValueError):
        Landmarks(names, points)


def test_detection_and_face_reject_untyped_landmarks():
    for factory in [
        lambda: Detection(Box(0, 0, 10, 10), landmarks={}),
        lambda: Face(0, Box(0, 0, 10, 10), landmarks={}),
    ]:
        with pytest.raises(ValueError, match="Landmarks"):
            factory()


def test_tracking_remaps_points_and_holds_them_during_misses():
    tracker = IoUTracker(smoothing=0.5, max_missing=1)
    first = Detection(Box(0, 0, 10, 10), 0.9, landmarks())
    assert tracker.update([first])[0].landmarks == first.landmarks
    current = Detection(Box(2, 0, 12, 10), 0.95, landmarks(2))
    tracked = tracker.update([current])[0]
    assert tracked.box == Box(1, 0, 11, 10)
    assert tracked.landmarks["nose_tip"] == (6, 4)
    assert current.landmarks["nose_tip"] == (7, 4)
    held = tracker.update([])[0]
    assert not held.observed and held.landmarks == tracked.landmarks
    assert tracker.update([]) == ()
    assert tracker.update([current])[0].track_id == 1


def test_tracking_clears_points_when_latest_observation_has_none():
    tracker = IoUTracker()
    tracker.update([Detection(Box(0, 0, 10, 10), landmarks=landmarks())])
    assert tracker.update([Detection(Box(0, 0, 10, 10))])[0].landmarks is None


def test_landmarks_survive_analysis_save_load_and_retrack(video, tmp_path):
    class Detector:
        def detect(self, image):
            return [Detection(Box(0, 0, 10, 10), 0.9, landmarks())]

        def metadata(self):
            return {"name": "controlled_landmark_detector"}

    source, _ = video
    analysis = analyze_video(source, detector=Detector(), detection_interval=2)
    cache = tmp_path / "landmarks.json"
    analysis.save(cache)
    restored = VideoAnalysis.load(cache)
    assert restored == analysis
    assert restored.schema_version == 2
    assert restored.frames[0].detections[0].landmarks == landmarks()
    assert restored.frames[1].faces[0].landmarks == landmarks()
    assert restored.retrack(tracker=IoUTracker()).frames == analysis.frames


def test_schema_one_caches_upgrade_without_landmarks(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    cache = tmp_path / "legacy.json"
    analysis.save(cache)
    data = json.loads(cache.read_text())
    data["schema_version"] = 1
    for frame in data["frames"]:
        for face in frame["faces"]:
            del face["landmarks"]
        for observation in frame["detections"] or []:
            del observation["landmarks"]
    cache.write_text(json.dumps(data))
    upgraded = VideoAnalysis.load(cache)
    assert upgraded.schema_version == 2
    assert upgraded.frames == analysis.frames
    upgraded.save(cache)
    assert json.loads(cache.read_text())["schema_version"] == 2


def test_malformed_cached_landmarks_rejected(video, detector, tmp_path):
    source, _ = video
    analysis = analyze_video(source, detector=detector)
    cache = tmp_path / "invalid.json"
    analysis.save(cache)
    data = json.loads(cache.read_text())
    data["frames"][0]["faces"][0]["landmarks"] = {"names": ["eye"], "points": [[1, 2, 3]]}
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finite"):
        VideoAnalysis.load(cache)
    with pytest.raises(ValueError, match="schema"):
        replace(analysis, schema_version=1)
