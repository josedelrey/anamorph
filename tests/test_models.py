import hashlib
import io

import pytest

from anamorph import download_face_parser_model, download_yunet_model, models
from anamorph.cli import main


@pytest.fixture
def assets(monkeypatch):
    data = {
        "https://example.test/license": b"controlled license",
        "https://example.test/model": b"controlled weights",
    }
    monkeypatch.setattr(
        models,
        "_ASSETS",
        tuple(
            (
                url,
                len(value),
                hashlib.sha256(value).hexdigest(),
                ".LICENSE.txt" if "license" in url else "",
            )
            for url, value in data.items()
        ),
    )
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(data[url])

    monkeypatch.setattr(models, "urlopen", open_url)
    return data, calls


def test_download_verifies_both_assets_and_reuses_cached_files(assets, tmp_path):
    data, calls = assets
    path = tmp_path / "models/face.onnx"
    assert download_yunet_model(path) == path
    assert path.read_bytes() == data["https://example.test/model"]
    assert (
        path.with_name(path.name + ".LICENSE.txt").read_bytes()
        == data["https://example.test/license"]
    )
    assert len(calls) == 2
    download_yunet_model(path)
    assert len(calls) == 2
    assert not list(path.parent.glob(".*"))


@pytest.mark.parametrize("corruption", ["same_size", "truncated", "extra_byte"])
def test_download_failure_publishes_no_unverified_files(assets, tmp_path, corruption):
    data, _ = assets
    original = data["https://example.test/model"]
    data["https://example.test/model"] = {
        "same_size": b"x" * len(original),
        "truncated": original[:-1],
        "extra_byte": original + b"x",
    }[corruption]
    path = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="verification"):
        download_yunet_model(path)
    assert list(tmp_path.iterdir()) == []


def test_existing_different_model_is_preserved(assets, tmp_path):
    data, calls = assets
    path = tmp_path / "model.onnx"
    original = b"x" * len(data["https://example.test/model"])
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        download_yunet_model(path)
    assert path.read_bytes() == original
    assert calls == []


def test_download_cli(assets, tmp_path, capsys):
    path = tmp_path / "model.onnx"
    main(["download-yunet", str(path)])
    assert path.exists()
    assert "YuNet model ready" in capsys.readouterr().out


def test_face_parser_downloader_reuses_verification_contract(assets, monkeypatch, tmp_path, capsys):
    _, calls = assets
    monkeypatch.setattr(models, "_PARSING_ASSETS", models._ASSETS)
    path = tmp_path / "parser.onnx"
    assert download_face_parser_model(path) == path
    assert len(calls) == 2
    main(["download-face-parser", str(path)])
    assert "BiSeNet model ready" in capsys.readouterr().out
    assert len(calls) == 2


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_invalid_download_timeout(tmp_path, timeout):
    with pytest.raises(ValueError):
        download_yunet_model(tmp_path / "not-downloaded.onnx", timeout=timeout)
