from pathlib import Path

import cv2
import numpy as np
import pytest

from anamorph import Box, Detection


@pytest.fixture
def texture():
    return np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)


@pytest.fixture
def video(tmp_path: Path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 12, (64, 48))
    if not writer.isOpened():
        pytest.skip("FFV1 encoder unavailable in this OpenCV build")
    frames = []
    rng = np.random.default_rng(23)
    try:
        for _ in range(6):
            image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            frames.append(image)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path, frames


class FixedDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(Box(16, 12, 40, 36))]

    def metadata(self):
        return {"name": "controlled_test_detector"}


@pytest.fixture
def detector():
    return FixedDetector()
