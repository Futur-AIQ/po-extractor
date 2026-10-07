"""Compare a candidate run with a reference run and apply the PRD §11.3 acceptance rule.

    uv run python -m eval.compare --ref runs/<A> --cand runs/<B>
    (or: make compare REF=runs/<A> CAND=runs/<B>)

Only documents present in both runs are compared, so both sides are scored on the same set.
A lossy change is accepted only if, versus the reference, critical-field document accuracy
drops by at most 0.5 percentage points AND field accuracy by at most 1 pp.

Prints the side-by-side table and the decision, and writes compare.md into the candidate run.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from eval.report import dataset_dir_for, number, pct, seconds, table
from eval.run_format import read_config, read_records
from eval.score import GroupSummary, RunScore, score_run

MAX_CRITICAL_DOC_DROP_PP = 0.5
MAX_FIELD_DROP_PP = 1.0
_EPSILON = 1e-9  # float noise when comparing a drop with its limit


@dataclass(frozen=True)
class MetricRow:
    """One line of the side-by-side table."""

    name: str
    ref: float | None
    cand: float | None
    kind: str  # pct (delta in pp), ms (delta in s) or count

    @property
    def delta(self) -> float | None:
        if self.ref is None or self.cand is None:
            return None
        return self.cand - self.ref

    def formatted(self) -> list[str]:
        fmt = {"pct": pct, "ms": seconds, "count": number}[self.kind]
        return [self.name, fmt(self.ref), fmt(self.cand), format_delta(self.delta, self.kind)]


def format_delta(delta: float | None, kind: str) -> str:
    if delta is None:
        return "n/a"
    if kind == "pct":
        return f"{delta * 100:+.2f} pp"
    if kind == "ms":
        return f"{delta / 1000:+.1f} s"
    return f"{delta:+,.0f}"


@dataclass(frozen=True)
class Decision:
    """Outcome of the PRD §11.3 acceptance rule."""

    accept: bool
    critical_doc_drop_pp: float
    field_drop_pp: float
    reasons: list[str]

    @property
    def verdict(self) -> str:
        return "ACCEPT" if self.accept else "REJECT"


def acceptance(ref: GroupSummary, cand: GroupSummary) -> Decision:
    """PRD §11.3: critical-doc accuracy drop <= 0.5 pp AND field accuracy drop <= 1 pp."""
    critical_drop = ((ref.critical_doc_accuracy or 0) - (cand.critical_doc_accuracy or 0)) * 100
    field_drop = ((ref.field_accuracy or 0) - (cand.field_accuracy or 0)) * 100
    critical_ok = critical_drop <= MAX_CRITICAL_DOC_DROP_PP + _EPSILON
    field_ok = field_drop <= MAX_FIELD_DROP_PP + _EPSILON
    reasons = [
        f"critical-doc accuracy {_change(critical_drop)} "
        f"({'within' if critical_ok else 'exceeds'} the {MAX_CRITICAL_DOC_DROP_PP} pp limit)",
        f"field accuracy {_change(field_drop)} "
        f"({'within' if field_ok else 'exceeds'} the {MAX_FIELD_DROP_PP:g} pp limit)",
    ]
    if not (critical_ok and field_ok):  # only the reasons that broke the rule
        reasons = [r for r, ok in zip(reasons, (critical_ok, field_ok), strict=True) if not ok]
    return Decision(critical_ok and field_ok, critical_drop, field_drop, reasons)


def _change(drop_pp: float) -> str:
    if abs(drop_pp) < _EPSILON:
        return "unchanged"
    return f"drops {drop_pp:.2f} pp" if drop_pp > 0 else f"rises {-drop_pp:.2f} pp"


@dataclass
class Comparison:
    ref_dir: Path
    cand_dir: Path
    common_docs: int
    ref_only: int
    cand_only: int
    ref: RunScore
    cand: RunScore
    decision: Decision

    def overall_rows(self) -> list[MetricRow]:
        r, c = self.ref.overall, self.cand.overall
        rl, cl = self.ref.latency, self.cand.latency
        return [
            MetricRow(
                "Critical-doc accuracy", r.critical_doc_accuracy, c.critical_doc_accuracy, "pct"
            ),
            MetricRow("Field accuracy", r.field_accuracy, c.field_accuracy, "pct"),
            MetricRow("Header accuracy", r.header.accuracy, c.header.accuracy, "pct"),
            MetricRow("Cell accuracy", r.cells.accuracy, c.cells.accuracy, "pct"),
            MetricRow("Row recall", r.row_recall, c.row_recall, "pct"),
            MetricRow("Row precision", r.row_precision, c.row_precision, "pct"),
            MetricRow("Latency p50", rl.p50_ms, cl.p50_ms, "ms"),
            MetricRow("Latency p95", rl.p95_ms, cl.p95_ms, "ms"),
            MetricRow(
                "Mean completion tokens",
                rl.mean_completion_tokens,
                cl.mean_completion_tokens,
                "count",
            ),
            MetricRow("Parse errors", rl.parse_errors, cl.parse_errors, "count"),
        ]

    def variant_rows(self) -> list[MetricRow]:
        rows = []
        for variant in sorted(set(self.ref.by_variant) | set(self.cand.by_variant)):
            r = self.ref.by_variant.get(variant, GroupSummary())
            c = self.cand.by_variant.get(variant, GroupSummary())
            rows += [
                MetricRow(
                    f"{variant}: critical-doc accuracy",
                    r.critical_doc_accuracy,
                    c.critical_doc_accuracy,
                    "pct",
                ),
                MetricRow(f"{variant}: field accuracy", r.field_accuracy, c.field_accuracy, "pct"),
            ]
        return rows


def compare_runs(
    ref_dir: Path, cand_dir: Path, dataset_dir: Path | None = None, include_computed: bool = True
) -> Comparison:
    """Score both runs on the documents they share and apply the acceptance rule."""
    ref_ids = {r.doc_id for r in read_records(ref_dir)}
    cand_ids = {r.doc_id for r in read_records(cand_dir)}
    common = ref_ids & cand_ids
    if not common:
        raise ValueError(f"{ref_dir} and {cand_dir} have no documents in common")
    dataset = dataset_dir or dataset_dir_for(read_config(cand_dir))
    ref = score_run(ref_dir, dataset, include_computed, doc_ids=common)
    cand = score_run(cand_dir, dataset, include_computed, doc_ids=common)
    return Comparison(
        ref_dir=ref_dir,
        cand_dir=cand_dir,
        common_docs=len(common),
        ref_only=len(ref_ids - common),
        cand_only=len(cand_ids - common),
        ref=ref,
        cand=cand,
        decision=acceptance(ref.overall, cand.overall),
    )


def render_markdown(comparison: Comparison) -> str:
    c, d = comparison, comparison.decision
    header = ["Metric", f"Reference `{c.ref_dir.name}`", f"Candidate `{c.cand_dir.name}`", "Delta"]
    lines = [
        f"# Comparison: {c.cand_dir.name} vs {c.ref_dir.name}",
        "",
        f"{c.common_docs} documents in both runs (reference only: {c.ref_only}, "
        f"candidate only: {c.cand_only}; those are left out).",
        "",
    ]
    lines += table(header, [row.formatted() for row in c.overall_rows()])
    lines += ["", "### By variant", ""]
    lines += table(header, [row.formatted() for row in c.variant_rows()])
    lines += [
        "",
        "## Acceptance rule (PRD §11.3)",
        "",
        f"Critical-doc accuracy drop ≤ {MAX_CRITICAL_DOC_DROP_PP} pp AND field accuracy drop "
        f"≤ {MAX_FIELD_DROP_PP:g} pp versus the reference.",
        "",
        f"**{d.verdict}**: " + "; ".join(d.reasons) + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare two runs (PRD §11.3 acceptance rule).")
    parser.add_argument("--ref", type=Path, required=True, help="reference run folder")
    parser.add_argument("--cand", type=Path, required=True, help="candidate run folder")
    parser.add_argument("--dataset", type=Path, default=None, help="default: from config.json")
    parser.add_argument("--out", type=Path, default=None, help="default: <cand>/compare.md")
    parser.add_argument(
        "--exclude-computed", action="store_true", help="do not score COMPUTED_FIELDS"
    )
    args = parser.parse_args()

    comparison = compare_runs(args.ref, args.cand, args.dataset, not args.exclude_computed)
    markdown = render_markdown(comparison)
    out = args.out or args.cand / "compare.md"
    out.write_text(markdown, "utf-8")
    header = ["Metric", "Reference", "Candidate", "Delta"]
    print(f"{args.cand.name} vs {args.ref.name}: {comparison.common_docs} common documents")
    widths = [24, 12, 12, 12]
    for row in [header] + [r.formatted() for r in comparison.overall_rows()]:
        print("  " + "".join(text.ljust(width) for text, width in zip(row, widths, strict=True)))
    decision = comparison.decision
    print(f"{decision.verdict}: " + "; ".join(decision.reasons))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
