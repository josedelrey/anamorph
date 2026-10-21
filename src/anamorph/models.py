"""Explicit retrieval of pinned external models and their license sidecars."""

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

FACE_PARSER_REVISION = "8a4729d95118d0e97c44185f9bdef3d6bfeaaf99"
FACE_PARSER_MODEL_SHA256 = "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e"
_PARSING_ASSETS = (
    (
        f"https://raw.githubusercontent.com/yakhyo/face-parsing/{FACE_PARSER_REVISION}/LICENSE",
        1080,
        "a4f7998feb5c90555daae4329db61caf7049e6b2706991dc1d9eb2a0923847ae",
        ".LICENSE.txt",
    ),
    (
        "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx",
        93632554,
        FACE_PARSER_MODEL_SHA256,
        "",
    ),
)


def download_yunet_model(destination: str | Path, *, timeout: float = 30) -> Path:
    """Download verified weights and a LICENSE.txt sidecar to an explicit path.

    Existing matching files are reused without network requests. Existing files
    with different content are rejected. All downloads are verified before
    publication. No model is included in the Anamorph distribution.
    """
    return _download_assets(destination, _ASSETS, timeout, "YuNet")


def download_face_parser_model(destination: str | Path, *, timeout: float = 120) -> Path:
    """Download verified ResNet34 BiSeNet weights and the upstream MIT license.

    The mutable upstream release URL is guarded by a fixed byte count and hash.
    Model and license are external assets. CelebAMask-HQ training-data terms
    are separate from the upstream code license. See the README before use.
    Existing differing files are refused. Matching files are reused offline.
    """
    return _download_assets(destination, _PARSING_ASSETS, timeout, "BiSeNet")


def _download_assets(
    destination: str | Path, asset_specs: tuple, timeout: float, label: str
) -> Path:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    destination = Path(destination)
    assets = [
        (Path(str(destination) + suffix), url, size, sha) for url, size, sha, suffix in asset_specs
    ]
    for path, _, size, sha in assets:
        if path.exists() and (
            path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != sha
        ):
            raise FileExistsError(f"Existing file does not match the pinned {label} asset: {path}")
    pending = []
    for path, url, size, sha in assets:
        if path.exists():
            continue
        with urlopen(url, timeout=timeout) as response:
            data = response.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{label} asset failed size/SHA-256 verification: {url}")
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
