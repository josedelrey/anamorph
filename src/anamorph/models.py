"""Explicit retrieval of the pinned OpenCV 4.x YuNet model and its license."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

YUNET_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_MODEL_URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{YUNET_REVISION}/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_LICENSE_URL = (
    f"https://raw.githubusercontent.com/opencv/opencv_zoo/{YUNET_REVISION}/"
    "models/face_detection_yunet/LICENSE"
)
# URL, byte count, SHA-256, destination suffix. Both assets are immutable pins.
_ASSETS = (
    (
        _LICENSE_URL,
        1085,
        "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958",
        ".LICENSE.txt",
    ),
    (_MODEL_URL, 232589, YUNET_MODEL_SHA256, ""),
)


def download_yunet_model(destination: str | Path, *, timeout: float = 30) -> Path:
    """Download verified weights and a LICENSE.txt sidecar to an explicit path.

    Existing matching files are reused without network requests. Existing files
    with different content are rejected. All downloads are verified before
    publication. No model is included in the Anamorph distribution.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    destination = Path(destination)
    assets = [
        (Path(str(destination) + suffix), url, size, sha) for url, size, sha, suffix in _ASSETS
    ]
    for path, _, size, sha in assets:
        if path.exists() and (
            path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != sha
        ):
            raise FileExistsError(f"Existing file does not match the pinned YuNet asset: {path}")
    pending = []
    for path, url, size, sha in assets:
        if path.exists():
            continue
        with urlopen(url, timeout=timeout) as response:
            data = response.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"YuNet asset failed size/SHA-256 verification: {url}")
        pending.append((path, data))
    for path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            # Do not overwrite a different file created while downloading.
            if path.exists() and path.read_bytes() != data:
                raise FileExistsError(path)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    return destination
