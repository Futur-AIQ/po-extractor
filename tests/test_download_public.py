"""Tests for the public-data helpers (Step 1.6). No network: a tiny fake FATURA archive."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import download_public
from scripts.download_public import FATURA_ROOT, copy_sample, md5_of, select_sample, unzip_fatura


def _fake_fatura(dest: Path, templates: int = 50, instances: int = 3) -> Path:
    """A zip with FATURA's layout: white and coloured images plus original annotations."""
    archive = dest / "FATURA2.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for t in range(1, templates + 1):
            for i in range(instances):
                stem = f"Template{t}_Instance{i}"
                z.writestr(f"{FATURA_ROOT}/images/{stem}.jpg", b"white" + stem.encode())
                z.writestr(f"{FATURA_ROOT}/colored_images/{stem}.jpg", b"colour")
                z.writestr(
                    f"{FATURA_ROOT}/Annotations/Original_Format/{stem}.json",
                    json.dumps({"template": t, "instance": i}),
                )
    return archive


def test_select_sample_one_per_template_and_deterministic() -> None:
    names = [f"Template{t}_Instance{i}.jpg" for t in range(1, 51) for i in range(200)]
    names += ["readme.txt", "Template3_Instance1.png"]  # ignored
    chosen = select_sample(names)
    assert len(chosen) == 50
    assert sorted(int(n.split("_")[0][8:]) for n in chosen) == list(range(1, 51))
    assert select_sample(list(reversed(names))) == chosen  # order of the listing irrelevant


def test_md5_of(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert md5_of(path) == hashlib.md5(b"x" * 3_000_000).hexdigest()


def test_unzip_once_and_copy_white_sample_with_annotations(tmp_path: Path) -> None:
    archive = _fake_fatura(tmp_path)
    root = unzip_fatura(archive, tmp_path)
    assert (root / "images").is_dir() and (tmp_path / ".unzipped").exists()
    (root / "images" / "Template1_Instance0.jpg").write_bytes(b"edited")
    assert unzip_fatura(archive, tmp_path) == root  # second call does not re-extract
    assert (root / "images" / "Template1_Instance0.jpg").read_bytes() == b"edited"

    out = tmp_path / "sample_50"
    copied = copy_sample(root, out)
    images = sorted(out.glob("*.jpg"))
    assert len(images) == 50 and len(list(out.glob("*.json"))) == 50
    for image in images:  # white-background copy, with its own annotation next to it
        assert image.read_bytes().startswith((b"white", b"edited"))
        assert (out / f"{image.stem}.json").exists()
    assert copy_sample(root, out) == copied  # idempotent


def test_copy_sample_requires_all_templates(tmp_path: Path) -> None:
    root = unzip_fatura(_fake_fatura(tmp_path, templates=10), tmp_path)
    with pytest.raises(RuntimeError, match="expected 50 templates, found 10"):
        copy_sample(root, tmp_path / "sample_50")


def test_verified_zip_is_not_downloaded_again(tmp_path: Path, monkeypatch) -> None:
    archive = _fake_fatura(tmp_path)
    monkeypatch.setattr(download_public, "FATURA_MD5", md5_of(archive))

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(download_public.httpx, "stream", no_network)
    assert download_public.fetch_fatura_zip(tmp_path) == archive
