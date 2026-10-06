"""Scanned twins and mixed POs: native PDF pages turned into realistic scans (PRD §5, §11.1).

    uv run python -m datagen.scanify --src data/synthetic

A scanned page is the native page rasterised at 200 dpi (typical office scanner), rotated
slightly (+-1 degree), given a paper tone, mild blur and sensor noise, JPEG-compressed, and
placed as the only content of a new PDF page: no text layer, exactly like a scanner output.

For every PO in the dataset:
    pdfs/PO_0001_S.pdf   scanned twin: every page scanned (same truth)
    pdfs/PO_00xx_M.pdf   mixed (10 POs from the test split): 1-2 pages scanned, rest native

The truth file of each PO lists its variants and their page kinds; manifest.csv gets one row
per variant (variant N / S / M) and splits.json lists each variant per split. Everything is
seeded from the PO id, so re-running gives identical files (idempotent).
"""

import argparse
import csv
import io
import json
import os
import random
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from datagen.build import MANIFEST_FIELDS

log = get_logger(__name__)

SCAN_DPI = 200
MIXED_COUNT = 10
NATIVE, SCANNED = "native", "scanned"


# =========================================================================================
# One page -> a scanned image
# =========================================================================================


@dataclass(frozen=True)
class ScanParams:
    """Random scanner/paper characteristics of one scanned page."""

    color: bool  # most office scans are grayscale
    angle: float  # degrees; the sheet is never perfectly straight
    blur: float  # Gaussian radius in pixels (optics, slight defocus)
    noise: float  # sensor noise standard deviation (0-255 scale)
    paper: float  # paper tone: white maps to this brightness
    ink: float  # darkest ink maps to this brightness
    jpeg_quality: int

    @classmethod
    def draw(cls, rng: random.Random) -> "ScanParams":
        return cls(
            color=rng.random() < 0.25,
            angle=rng.uniform(-1.0, 1.0),
            blur=rng.uniform(0.3, 0.8),
            noise=rng.uniform(2.0, 4.5),
            paper=rng.uniform(236, 250),
            ink=rng.uniform(8, 30),
            jpeg_quality=rng.randint(62, 80),
        )


def scan_page_image(page: pymupdf.Page, rng: random.Random, dpi: int = SCAN_DPI) -> bytes:
    """Rasterise a PDF page and degrade it like a scanner would; returns JPEG bytes."""
    params = ScanParams.draw(rng)
    noise_rng = np.random.default_rng(rng.getrandbits(64))
    colorspace = pymupdf.csRGB if params.color else pymupdf.csGRAY
    pix = page.get_pixmap(dpi=dpi, colorspace=colorspace)
    mode = "RGB" if params.color else "L"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

    white = (255, 255, 255) if params.color else 255
    image = image.rotate(params.angle, resample=Image.Resampling.BICUBIC, fillcolor=white)
    image = image.filter(ImageFilter.GaussianBlur(params.blur))

    pixels = np.asarray(image, dtype=np.float32)
    pixels = params.ink + pixels * (params.paper - params.ink) / 255  # paper tone, ink grey
    pixels += noise_rng.normal(0.0, params.noise, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode)

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=params.jpeg_quality, optimize=True)
    return buffer.getvalue()


def scan_pdf(pdf: bytes, scanned_pages: set[int], seed_key: str) -> bytes:
    """Copy of `pdf` where the 0-based `scanned_pages` are image-only scans; others stay native.

    Each scanned page is seeded from (seed_key, page index), so the output is reproducible.
    """
    with pymupdf.open(stream=pdf, filetype="pdf") as source, pymupdf.open() as out:
        for index, page in enumerate(source):
            if index in scanned_pages:
                image = scan_page_image(page, random.Random(f"{seed_key}:{index}"))
                new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(new_page.rect, stream=image)
            else:
                out.insert_pdf(source, from_page=index, to_page=index)
        # No random trailer /ID, so re-running produces byte-identical files.
        return out.tobytes(garbage=3, deflate=True, no_new_id=True)


def page_kinds(page_count: int, scanned_pages: set[int]) -> list[str]:
    """Per-page kind list, e.g. ['native', 'scanned', 'native']."""
    return [SCANNED if i in scanned_pages else NATIVE for i in range(page_count)]


# =========================================================================================
# Dataset variants
# =========================================================================================


@dataclass(frozen=True)
class VariantJob:
    """One PDF to produce: which source, which pages to scan, where to write it."""

    base_id: str
    variant: str  # "S" or "M"
    source: Path
    target: Path
    scanned_pages: frozenset[int]
    page_count: int

    @property
    def kinds(self) -> list[str]:
        return page_kinds(self.page_count, set(self.scanned_pages))


def run_job(job: VariantJob) -> str:
    """Write one variant PDF (runs in a worker process)."""
    pdf = scan_pdf(job.source.read_bytes(), set(job.scanned_pages), f"{job.base_id}:{job.variant}")
    job.target.write_bytes(pdf)
    return job.target.name


def choose_mixed(
    base_rows: Sequence[dict[str, str]], test_ids: set[str], count: int
) -> dict[str, frozenset[int]]:
    """Pick `count` test POs spread across layouts, and 1-2 pages of each to scan.

    At least one page always stays native and one is scanned, so the PO is really mixed.
    """
    rng = random.Random("mixed-v1")
    by_layout: dict[str, list[dict[str, str]]] = {}
    for row in sorted(base_rows, key=lambda r: r["id"]):
        if row["id"] in test_ids:
            by_layout.setdefault(row["layout"], []).append(row)
    for rows in by_layout.values():
        rng.shuffle(rows)

    chosen: dict[str, frozenset[int]] = {}
    while len(chosen) < count and any(by_layout.values()):
        for layout in sorted(by_layout):
            if by_layout[layout] and len(chosen) < count:
                row = by_layout[layout].pop()
                pages = int(row["page_count"])
                k = min(rng.randint(1, 2), pages - 1)
                chosen[row["id"]] = frozenset(rng.sample(range(pages), k))
    return chosen


def plan_jobs(src: Path, base_rows: Sequence[dict[str, str]], mixed: dict[str, frozenset[int]]):
    """Scanned twin for every PO, plus the mixed variants."""
    jobs = []
    for row in base_rows:
        base_id, pages = row["id"], int(row["page_count"])
        source = src / row["pdf"]
        jobs.append(
            VariantJob(base_id, "S", source, src / "pdfs" / f"{base_id}_S.pdf",
                       frozenset(range(pages)), pages)
        )  # fmt: skip
        if base_id in mixed:
            jobs.append(
                VariantJob(base_id, "M", source, src / "pdfs" / f"{base_id}_M.pdf",
                           mixed[base_id], pages)
            )  # fmt: skip
    return jobs


def build_variants(src: Path, mixed_count: int = MIXED_COUNT, workers: int | None = None) -> int:
    """Create all scanned twins and mixed POs under `src`; update truth, manifest and splits.

    Idempotent: the base (native) rows are the input; variant files and rows are rebuilt.
    Returns the number of variant PDFs written.
    """
    with (src / "manifest.csv").open(encoding="utf-8") as f:
        base_rows = [row for row in csv.DictReader(f) if row["variant"] == "N"]
    splits = json.loads((src / "splits.json").read_text(encoding="utf-8"))
    test_ids = set(splits["test"]["N"])

    mixed = choose_mixed(base_rows, test_ids, mixed_count)
    jobs = plan_jobs(src, base_rows, mixed)
    _remove_stale_variants(src, {job.target.name for job in jobs})
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for name in pool.map(run_job, jobs):
            log.debug("variant written", extra={"file": name})

    jobs_by_base: dict[str, list[VariantJob]] = {}
    for job in jobs:
        jobs_by_base.setdefault(job.base_id, []).append(job)
    _update_truth(src, base_rows, jobs_by_base)
    _write_manifest(src, base_rows, jobs_by_base)
    _write_splits(src, splits, jobs_by_base)
    return len(jobs)


def _remove_stale_variants(src: Path, keep: set[str]) -> None:
    for path in (src / "pdfs").glob("*_[SM].pdf"):
        if path.name not in keep:
            path.unlink()


def _update_truth(src: Path, base_rows, jobs_by_base: dict[str, list["VariantJob"]]) -> None:
    for row in base_rows:
        path = src / row["truth"]
        truth = json.loads(path.read_text(encoding="utf-8"))
        native = truth["meta"]["variants"]["N"]
        truth["meta"]["variants"] = {"N": native} | {
            job.variant: {"pdf": f"pdfs/{job.target.name}", "page_kinds": job.kinds}
            for job in jobs_by_base[row["id"]]
        }
        path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")


def _write_manifest(src: Path, base_rows, jobs_by_base: dict[str, list["VariantJob"]]) -> None:
    rows = []
    for row in base_rows:
        rows.append(row)
        for job in jobs_by_base[row["id"]]:
            rows.append(
                row
                | {
                    "id": f"{row['id']}_{job.variant}",
                    "variant": job.variant,
                    "page_kinds": " ".join(job.kinds),
                    "pdf": f"pdfs/{job.target.name}",
                }
            )
    with (src / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_splits(src: Path, splits, jobs_by_base: dict[str, list["VariantJob"]]) -> None:
    out = {}
    for name, variants in splits.items():
        base_ids = variants["N"]
        out[name] = {"N": base_ids}
        for variant in ("S", "M"):
            out[name][variant] = [
                f"{base_id}_{variant}"
                for base_id in base_ids
                if any(job.variant == variant for job in jobs_by_base[base_id])
            ]
    (src / "splits.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create scanned twins and mixed POs.")
    parser.add_argument("--src", type=Path, default=Path(settings.data_dir) / "synthetic")
    parser.add_argument("--mixed", type=int, default=MIXED_COUNT, help="number of mixed POs")
    parser.add_argument("--workers", type=int, default=None, help="worker processes")
    args = parser.parse_args()
    configure_logging(settings.log_level)

    start = time.perf_counter()
    written = build_variants(args.src, args.mixed, args.workers)
    log.info(
        "scanned variants built",
        extra={"src": str(args.src), "pdfs": written,
               "seconds": round(time.perf_counter() - start, 1)},
    )  # fmt: skip


if __name__ == "__main__":
    main()
