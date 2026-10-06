"""Download the public datasets used besides the synthetic POs (PRD §11.1). Idempotent.

Run from the repo root:
    uv run python -m scripts.download_public            # both
    uv run python -m scripts.download_public --only fatura

- Northwind purchase orders (Hugging Face, AyoubChLin/northwind_PurchaseOrders, Apache-2.0)
  -> data/northwind/   (smoke test)
- FATURA invoices (Zenodo record 10371464, CC BY 4.0) -> data/fatura/
  The 690 MB zip is streamed with progress (resumable), MD5-verified and unzipped; then one
  white-background image per template (50) is copied with its original annotation to
  data/fatura/sample_50/ (scanned-robustness sample).

This is a one-off developer download, not part of the app (which makes no external calls).
"""

import argparse
import hashlib
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import httpx

from app.core.settings import get_settings

NORTHWIND_REPO = "AyoubChLin/northwind_PurchaseOrders"
FATURA_URL = "https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"
FATURA_MD5 = "4c9404462f22c5241eb1a290a02eb2a2"
FATURA_ROOT = "invoices_dataset_final"  # top folder inside the zip
FATURA_TEMPLATES = 50
CHUNK = 1 << 20


# =========================================================================================
# Northwind
# =========================================================================================


def download_northwind(dest: Path) -> Path:
    """Snapshot the Northwind purchase-order PDFs (already-present files are skipped)."""
    from huggingface_hub import snapshot_download  # dev dependency, only needed here

    path = snapshot_download(repo_id=NORTHWIND_REPO, repo_type="dataset", local_dir=dest)
    count = len(list(Path(path).glob("*.pdf")))
    print(f"Northwind: {count} PDFs in {dest}")
    return Path(path)


# =========================================================================================
# FATURA
# =========================================================================================


def md5_of(path: Path) -> str:
    """MD5 of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_fatura_zip(dest: Path) -> Path:
    """Download FATURA2.zip unless a verified copy exists; resume a partial download."""
    target = dest / "FATURA2.zip"
    if target.exists() and md5_of(target) == FATURA_MD5:
        print(f"FATURA: {target} present, MD5 verified")
        return target
    partial = dest / "FATURA2.zip.download"
    done = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with httpx.stream("GET", FATURA_URL, headers=headers, follow_redirects=True, timeout=60) as r:
        if r.status_code == 200 and done:  # server ignored the range: start over
            done = 0
        r.raise_for_status()
        total = done + int(r.headers.get("content-length", 0))
        with partial.open("ab" if done else "wb") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stderr.write(f"\rFATURA: {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
    sys.stderr.write("\n")
    actual = md5_of(partial)
    if actual != FATURA_MD5:
        partial.unlink()
        raise RuntimeError(f"FATURA MD5 mismatch: expected {FATURA_MD5}, got {actual}")
    partial.rename(target)
    print(f"FATURA: downloaded and verified {target}")
    return target


def unzip_fatura(archive: Path, dest: Path) -> Path:
    """Extract the archive once (a marker file records a complete extraction)."""
    root = dest / FATURA_ROOT
    marker = dest / ".unzipped"
    if marker.exists() and root.is_dir():
        print(f"FATURA: already extracted to {root}")
        return root
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    marker.write_text(FATURA_MD5 + "\n")
    print(f"FATURA: extracted to {root}")
    return root


def select_sample(
    image_names: list[str], per_template: int = 1, seed: str = "fatura-50"
) -> list[str]:
    """One instance per template (deterministic), from white-background image names.

    Names look like 'Template17_Instance112.jpg'.
    """
    by_template: dict[int, list[str]] = {}
    for name in sorted(image_names):
        match = re.fullmatch(r"Template(\d+)_Instance(\d+)\.jpg", name)
        if match:
            by_template.setdefault(int(match[1]), []).append(name)
    rng = random.Random(seed)
    return [
        name
        for template in sorted(by_template)
        for name in rng.sample(by_template[template], per_template)
    ]


def copy_sample(root: Path, out: Path) -> list[Path]:
    """Copy one white-background image per template plus its annotation to `out`."""
    images = root / "images"  # white background (colored_images/ has coloured backgrounds)
    annotations = root / "Annotations" / "Original_Format"
    chosen = select_sample([p.name for p in images.glob("*.jpg")])
    if len(chosen) != FATURA_TEMPLATES:
        raise RuntimeError(f"expected {FATURA_TEMPLATES} templates, found {len(chosen)}")
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in chosen:
        stem = Path(name).stem
        for source in (images / name, annotations / f"{stem}.json"):
            target = out / source.name
            if not target.exists():
                shutil.copy2(source, target)
            copied.append(target)
    print(f"FATURA: {len(chosen)} images + annotations in {out}")
    return copied


def download_fatura(dest: Path) -> Path:
    """Download, verify, extract and sample FATURA into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = fetch_fatura_zip(dest)
    root = unzip_fatura(archive, dest)
    copy_sample(root, dest / "sample_50")
    return dest / "sample_50"


def main() -> None:
    """CLI entry point."""
    data_dir = Path(get_settings().data_dir)
    parser = argparse.ArgumentParser(description="Download Northwind and FATURA (idempotent).")
    parser.add_argument("--only", choices=["northwind", "fatura"], default=None)
    args = parser.parse_args()
    if args.only in (None, "northwind"):
        download_northwind(data_dir / "northwind")
    if args.only in (None, "fatura"):
        download_fatura(data_dir / "fatura")


if __name__ == "__main__":
    main()
