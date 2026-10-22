"""Prepare two VidTIMIT subjects' head rotations for local validation.

Run: uv run python examples/prepare_vidtimit.py --download
Inputs and lossless video derivatives stay under ignored local_inputs/.
Upstream terms: https://conradsanderson.id.au/vidtimit/
"""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from anamorph.analysis import file_sha256

ASSETS = {
    "fadg0.zip": (
        "https://zenodo.org/records/158963/files/fadg0.zip?download=1",
        81593138,
        "0525bf2c82a84e2d13bc51a50450713e642958588fbf59169d05648b4639c784",
    ),
    "mabw0.zip": (
        "https://zenodo.org/records/158963/files/mabw0.zip?download=1",
        86958066,
        "3898adee3ad5f4b4751250115d452ae01dc67739c7cd976e712bcef97826ea01",
    ),
    "vidtimit_documentation.pdf": (
        "https://conradsanderson.id.au/vidtimit/zips/vidtimit_documentation.pdf",
        68648,
        "7bf2ae60be1e19c27cf76e7b7aa09f03949eb95d1b9a355bc42ec6fdf1d6c0df",
    ),
}


def acquire(root, name, download=False):
    url, size, digest = ASSETS[name]
    target = root / name
    if not target.exists():
        if not download:
            raise FileNotFoundError(f"Missing {target}. Use --download for explicit acquisition")
        handle, temporary = tempfile.mkstemp(prefix=".download-", dir=root)
        try:
            with os.fdopen(handle, "wb") as stream, urlopen(url, timeout=120) as response:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > size:
                        raise ValueError(f"Unexpected download size for {name}")
                    stream.write(block)
            if Path(temporary).stat().st_size != size or file_sha256(temporary) != digest:
                raise ValueError(f"Download integrity check failed for {name}")
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if target.stat().st_size != size or file_sha256(target) != digest:
        raise ValueError(f"Existing asset differs from the pinned release: {target}")
    return target


def rotation_frames(archive, subject, sequence):
    prefix = f"{subject}/video/{sequence}/"
    names = sorted(
        (
            name
            for name in archive.namelist()
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ),
        key=lambda name: int(name[len(prefix) :]),
    )
    if not names:
        raise ValueError(f"No head-rotation frames for {subject}/{sequence}")
    numbers = [int(name[len(prefix) :]) for name in names]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("Source frame numbering is not contiguous")
    for name in names:
        if archive.getinfo(name).file_size > 2 * 1024 * 1024:
            raise ValueError("Unexpected JPEG size")
        image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {name}")
        yield name, image


def prepare_rotation(archive, subject, sequence, output):
    """Encode FFV1 and compare every decoded pixel to the source JPEG decode."""
    if output.exists():
        raise FileExistsError(output)
    writer = None
    names, digests, dimensions = [], [], None
    try:
        for name, image in rotation_frames(archive, subject, sequence):
            height, width = image.shape[:2]
            if dimensions is None:
                dimensions = width, height
                if width % 2 or height % 2:
                    raise ValueError("Lossless video preparation requires even dimensions")
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"FFV1"), 25, dimensions
                )
                if not writer.isOpened():
                    raise RuntimeError("FFV1 encoder unavailable")
            if (width, height) != dimensions:
                raise ValueError("Source dimensions changed")
            names.append(name)
            digests.append(hashlib.sha256(image.tobytes()).hexdigest())
            writer.write(image)
        writer.release()
        writer = None
        capture = cv2.VideoCapture(str(output))
        try:
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 25) > 0.001:
                raise RuntimeError("Prepared frame rate differs")
            for expected in digests:
                ok, image = capture.read()
                if not ok or hashlib.sha256(image.tobytes()).hexdigest() != expected:
                    raise RuntimeError("Prepared video differs from source JPEG pixels")
            if capture.read()[0]:
                raise RuntimeError("Prepared video contains extra frames")
        finally:
            capture.release()
        return {
            "file": output.name,
            "sha256": file_sha256(output),
            "fps": 25,
            "width": dimensions[0],
            "height": dimensions[1],
            "frames": len(names),
            "first_source_frame": names[0],
            "last_source_frame": names[-1],
        }
    except BaseException:
        if writer is not None:
            writer.release()
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local_inputs/face_evaluation/vidtimit"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    prepared = args.root / "prepared"
    if prepared.exists():
        parser.error("Prepared directory exists. Choose a fresh root to preserve existing data")
    paths = {name: acquire(args.root, name, args.download) for name in ASSETS}
    prepared.mkdir()
    # Download subjects serially above, as the upstream documentation requests.
    records = []
    for subject in ("fadg0", "mabw0"):
        with zipfile.ZipFile(paths[f"{subject}.zip"]) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC check failed")
            for sequence in ("head", "head2", "head3"):
                records.append(
                    prepare_rotation(
                        archive, subject, sequence, prepared / f"{subject}_{sequence}.avi"
                    )
                )
    manifest = {
        "source": "https://conradsanderson.id.au/vidtimit/",
        "terms": "Preserved without modification in ../vidtimit_documentation.pdf",
        "citation": "C. Sanderson and B.C. Lovell, Multi-Region Probabilistic Histograms "
        "for Robust and Scalable Identity Inference, LNCS 5558, 199-208, 2009",
        "purpose": "Pose and temporal stress testing. No ground-truth face masks are supplied",
        "selection": "First listed female and male subject IDs, all three head rotations",
        "conversion": "Ordered JPEG decodes to pixel-verified FFV1, 25 fps, without audio",
        "assets": {
            name: {"url": value[0], "bytes": value[1], "sha256": value[2]}
            for name, value in ASSETS.items()
        },
        "videos": records,
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
