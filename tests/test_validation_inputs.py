import io
import runpy
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_vidtimit_conversion_preserves_source_decodes_and_existing_output(tmp_path):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/prepare_vidtimit.py")
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(1, 4):
            image = np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)
            ok, data = cv2.imencode(".jpg", image)
            assert ok
            archive.writestr(f"subject/video/head/{i:03d}", data.tobytes())
        archive.writestr("../unexpected.txt", "This entry must never be extracted")
    output = tmp_path / "head.avi"
    with zipfile.ZipFile(buffer) as archive:
        result = module["prepare_rotation"](archive, "subject", "head", output)
        assert result["frames"] == 3 and result["fps"] == 25
        assert result["first_source_frame"] == "subject/video/head/001"
        assert result["width"] == 64 and result["height"] == 48
        original = output.read_bytes()
        with pytest.raises(FileExistsError):
            module["prepare_rotation"](archive, "subject", "head", output)
        assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_vidtimit_gaps_are_rejected_without_inventing_frames(tmp_path):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/prepare_vidtimit.py")
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("subject/video/head/001", b"unused")
        archive.writestr("subject/video/head/003", b"unused")
    output = tmp_path / "head.avi"
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError, match="contiguous"):
        module["prepare_rotation"](archive, "subject", "head", output)
    assert not output.exists()
