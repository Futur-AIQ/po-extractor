"""Write an "oracle" run whose outputs are the ground truth (for testing scoring and reports).

    uv run python scripts/make_oracle_run.py --split dev          # -> runs/oracle_dev/
    uv run python scripts/make_oracle_run.py --split test --variants N,S

Every document of the split gets outputs/<doc_id>.json = its truth PO, an empty raw file and a
'completed' record with zero timings and tokens. An existing oracle run is replaced. The run is
then scored as a self-check: every metric must be 100%.
"""

import argparse
import sys
from pathlib import Path

if not __package__:  # run as a file: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.settings import get_settings
from eval.dataset import VARIANTS, load_dataset, read_truth_po
from eval.run_format import DocRecord, RunConfig, RunWriter, current_git_commit
from eval.score import score_run


def make_oracle_run(
    dataset_dir: Path,
    runs_dir: Path,
    split: str | None,
    variants: tuple[str, ...] = VARIANTS,
    run_id: str | None = None,
) -> Path:
    """Write the oracle run and return its directory."""
    docs = load_dataset(dataset_dir, split, variants)
    if not docs:
        raise ValueError(f"no documents for split={split} variants={variants} in {dataset_dir}")
    config = RunConfig(
        pipeline="oracle",
        git_commit=current_git_commit(),
        extra={"dataset": str(dataset_dir), "split": split or "all", "variants": list(variants)},
    )
    writer = RunWriter.create(
        runs_dir, run_id or f"oracle_{split or 'all'}", config, overwrite=True
    )
    for doc in docs:
        writer.write_output(doc.doc_id, read_truth_po(doc))
        writer.write_raw(doc.doc_id, {"responses": [], "note": "oracle: output copied from truth"})
        writer.append_record(
            DocRecord(doc_id=doc.doc_id, variant=doc.variant, status="completed",
                      timings_ms={"total": 0.0})
        )  # fmt: skip
    return writer.run_dir


def main() -> None:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Write a run whose outputs equal the truth.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="e.g. N,S")
    parser.add_argument("--dataset", type=Path, default=Path(settings.data_dir) / "synthetic")
    parser.add_argument("--runs-dir", type=Path, default=Path(settings.runs_dir))
    parser.add_argument("--run-id", default=None, help="default: oracle_<split>")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    run_dir = make_oracle_run(args.dataset, args.runs_dir, split, variants, args.run_id)

    score = score_run(run_dir, args.dataset)
    overall = score.overall
    counts = {variant: summary.docs for variant, summary in score.by_variant.items()}
    print(f"Oracle run written to {run_dir}: {overall.docs} documents {counts}")
    print(
        f"Self-check: critical-doc accuracy {overall.critical_doc_accuracy:.1%}, "
        f"field accuracy {overall.field_accuracy:.1%}, row recall {overall.row_recall:.1%}, "
        f"errors {len(score.errors)}"
    )
    if score.errors or overall.critical_doc_accuracy != 1:
        sys.exit("oracle run does not score 100%: scoring or truth files are inconsistent")


if __name__ == "__main__":
    main()
