"""Evaluation report for one run (PRD §3, §11.2).

    uv run python -m eval.report --run runs/<run_id>          (or: make eval RUN=runs/<run_id>)

Writes into the run folder:
    report.md    accuracy (overall, by variant / layout / issuer type), line items,
                 latency and tokens, top 10 problem fields with examples
    report.json  the same numbers, machine-readable
    errors.csv   every wrong / missed / hallucinated value, most frequent fields first
and prints a compact summary.
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.settings import get_settings
from eval.run_format import RunConfig, read_config
from eval.score import (
    ErrorRecord,
    GroupSummary,
    LatencySummary,
    Outcome,
    RunScore,
    score_run,
)

TOP_FIELDS = 10
EXAMPLES_PER_FIELD = 2
# PRD §3 accuracy targets (latency targets depend on load; see the benchmark, Phase 7).
TARGETS = {"field_accuracy": 0.97, "critical_doc_accuracy": 0.98, "row_recall": 0.995}
ERROR_CSV_FIELDS = ["doc_id", "variant", "field", "outcome", "row", "truth", "pred", "fuzzy"]


def dataset_dir_for(config: RunConfig) -> Path:
    """The dataset a run was made from (config.extra['dataset']), else the default one."""
    dataset = config.extra.get("dataset")
    return Path(dataset) if dataset else Path(get_settings().data_dir) / "synthetic"


# =========================================================================================
# Problem fields and errors
# =========================================================================================


@dataclass(frozen=True)
class ProblemField:
    """A field with errors: how many of each kind, and a few examples."""

    field: str
    errors: int
    outcomes: dict[str, int]
    examples: list[ErrorRecord]


def errors_by_frequency(errors: list[ErrorRecord]) -> list[ErrorRecord]:
    """Errors of the most frequent field first; within a field by document and row."""
    counts = Counter(e.field for e in errors)
    return sorted(errors, key=lambda e: (-counts[e.field], e.field, e.doc_id, e.row or 0))


def top_problem_fields(
    errors: list[ErrorRecord], n: int = TOP_FIELDS, examples: int = EXAMPLES_PER_FIELD
) -> list[ProblemField]:
    """The `n` fields with the most errors. Examples come from different documents when
    possible, so one bad document does not fill the section."""
    by_field: dict[str, list[ErrorRecord]] = {}
    for error in errors_by_frequency(errors):
        by_field.setdefault(error.field, []).append(error)
    problems = []
    for name, field_errors in list(by_field.items())[:n]:
        picked: list[ErrorRecord] = []
        for error in field_errors:  # one per document first
            if len(picked) < examples and error.doc_id not in {p.doc_id for p in picked}:
                picked.append(error)
        for error in field_errors:  # then fill up from any document
            if len(picked) < examples and error not in picked:
                picked.append(error)
        outcomes = Counter(e.outcome.value for e in field_errors)
        problems.append(ProblemField(name, len(field_errors), dict(outcomes), picked))
    return problems


def write_errors_csv(errors: list[ErrorRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        for e in errors_by_frequency(errors):
            writer.writerow({
                "doc_id": e.doc_id, "variant": e.variant, "field": e.field,
                "outcome": e.outcome.value, "row": "" if e.row is None else e.row,
                "truth": "" if e.truth is None else e.truth,
                "pred": "" if e.pred is None else e.pred,
                "fuzzy": "" if e.fuzzy is None else round(e.fuzzy, 1),
            })  # fmt: skip


# =========================================================================================
# Formatting helpers
# =========================================================================================


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f} s"


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def cell(value: Any, width: int = 60) -> str:
    """A value made safe for one Markdown table cell."""
    text = "∅" if value is None else " ".join(str(value).split())
    text = text if len(text) <= width else text[: width - 1] + "…"
    return text.replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


ACCURACY_HEADER = [
    "Group", "Docs", "Critical-doc acc", "Field acc", "Header acc", "Cell acc",
    "Text exact", "Text fuzzy≥90", "Row recall", "Row precision", "Parse errors",
]  # fmt: skip


def accuracy_row(name: str, s: GroupSummary) -> list[str]:
    text_total = s.header.text_total + s.cells.text_total
    exact = (s.header.text_exact + s.cells.text_exact) / text_total if text_total else None
    fuzzy = (s.header.text_fuzzy + s.cells.text_fuzzy) / text_total if text_total else None
    return [
        name, str(s.docs), pct(s.critical_doc_accuracy), pct(s.field_accuracy),
        pct(s.header.accuracy), pct(s.cells.accuracy), pct(exact), pct(fuzzy),
        pct(s.row_recall), pct(s.row_precision), str(s.statuses.get("parse_error", 0)),
    ]  # fmt: skip


LATENCY_HEADER = [
    "Group", "Docs", "p50", "p95", "max", "Prompt tok", "Completion tok", "Reasoning tok",
    "Calls", "Parse errors",
]  # fmt: skip


def latency_row(name: str, lat: LatencySummary) -> list[str]:
    reasoning = number(lat.mean_reasoning_tokens)
    if lat.mean_reasoning_tokens is not None and lat.reasoning_estimated:
        reasoning = "~" + reasoning
    calls = "n/a" if lat.mean_calls is None else f"{lat.mean_calls:.1f}"
    return [
        name, str(lat.docs), seconds(lat.p50_ms), seconds(lat.p95_ms), seconds(lat.max_ms),
        number(lat.mean_prompt_tokens), number(lat.mean_completion_tokens), reasoning,
        calls, str(lat.parse_errors),
    ]  # fmt: skip


# =========================================================================================
# report.md / report.json
# =========================================================================================


def render_markdown(run_dir: Path, config: RunConfig, score: RunScore) -> str:
    """The full report as Markdown."""
    o = score.overall
    lines = [
        f"# Evaluation report: {run_dir.name}",
        "",
        f"Pipeline **{config.pipeline}**, model `{config.model_alias}`, strategy "
        f"`{config.strategy}`, dpi {config.dpi}, thinking `{config.thinking}`, git "
        f"`{config.git_commit}`, run at {config.timestamp}.",
        f"{o.docs} documents scored; computed fields "
        f"{'included' if score.include_computed else 'excluded'}. Field accuracy = header "
        "fields + cells of matched rows (correct-absent counts as correct).",
        "",
        "## Targets (PRD §3)",
        "",
    ]
    values = {
        "field_accuracy": o.field_accuracy,
        "critical_doc_accuracy": o.critical_doc_accuracy,
        "row_recall": o.row_recall,
    }
    lines += table(
        ["Metric", "Value", "Target", "Met"],
        [
            [name.replace("_", " "), pct(values[name]), f"≥ {pct(target)}",
             "yes" if values[name] is not None and values[name] >= target else "**no**"]
            for name, target in TARGETS.items()
        ],
    )  # fmt: skip
    lines += ["", "## Accuracy", ""]
    lines += table(ACCURACY_HEADER, [accuracy_row("overall", o)])
    for title, groups in [
        ("By variant (N native, S scanned, M mixed)", score.by_variant),
        ("By layout", score.by_layout),
        ("By issuer type", score.by_issuer),
    ]:
        lines += ["", f"### {title}", ""]
        lines += table(ACCURACY_HEADER, [accuracy_row(k, v) for k, v in groups.items()])

    lines += ["", "## Line items", ""]
    lines += table(
        ["Truth rows", "Predicted rows", "Matched", "Recall", "Precision", "F1", "Cell acc"],
        [[str(o.truth_rows), str(o.pred_rows), str(o.matched_rows), pct(o.row_recall),
          pct(o.row_precision), pct(o.row_f1), pct(o.cells.accuracy)]],
    )  # fmt: skip
    columns = {k: v for k, v in o.by_field.items() if k.startswith("line_items.")}
    if columns:
        lines += ["", "Per column, over matched rows:", ""]
        lines += table(
            ["Column", "Accuracy", "Wrong", "Missed", "Hallucinated"],
            [[name.removeprefix("line_items."), pct(t.accuracy),
              str(t.outcomes[Outcome.WRONG]), str(t.outcomes[Outcome.MISSED]),
              str(t.outcomes[Outcome.HALLUCINATED])]
             for name, t in columns.items()],
        )  # fmt: skip

    lines += ["", "## Latency and tokens (per document, means)", ""]
    lines += table(
        LATENCY_HEADER,
        [latency_row("overall", score.latency)]
        + [latency_row(k, v) for k, v in score.latency_by_variant.items()],
    )
    if score.latency.reasoning_estimated:
        lines += ["", "~ reasoning tokens estimated from the reasoning text (server reports none)."]

    lines += ["", f"## Top {TOP_FIELDS} problem fields"]
    problems = top_problem_fields(score.errors)
    if not problems:
        lines += ["", "No errors."]
    for rank, p in enumerate(problems, start=1):
        kinds = ", ".join(f"{count} {kind}" for kind, count in sorted(p.outcomes.items()))
        lines += ["", f"{rank}. **{p.field}**: {p.errors} errors ({kinds})"]
        for e in p.examples:
            where = e.doc_id + ("" if e.row is None else f" row {e.row}")
            lines.append(f"   - {where}: truth `{cell(e.truth)}` → pred `{cell(e.pred)}`")
    lines += ["", f"All {len(score.errors)} error records: `errors.csv`.", ""]
    return "\n".join(lines)


def report_dict(run_dir: Path, config: RunConfig, score: RunScore) -> dict[str, Any]:
    """report.json content."""
    problems = [
        {"field": p.field, "errors": p.errors, "outcomes": p.outcomes,
         "examples": [{"doc_id": e.doc_id, "row": e.row, "truth": e.truth, "pred": e.pred}
                      for e in p.examples]}
        for p in top_problem_fields(score.errors)
    ]  # fmt: skip
    return {
        "run_id": run_dir.name,
        "config": config.model_dump(mode="json"),
        "targets": TARGETS,
        **score.as_dict(),
        "top_problem_fields": problems,
    }


def write_report(
    run_dir: Path, dataset_dir: Path | None = None, include_computed: bool = True
) -> RunScore:
    """Score the run and write report.md, report.json and errors.csv into it."""
    config = read_config(run_dir)
    score = score_run(run_dir, dataset_dir or dataset_dir_for(config), include_computed)
    (run_dir / "report.md").write_text(render_markdown(run_dir, config, score), "utf-8")
    report = report_dict(run_dir, config, score)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8"
    )
    write_errors_csv(score.errors, run_dir / "errors.csv")
    return score


def terminal_summary(run_dir: Path, score: RunScore) -> str:
    """A few lines for the terminal."""
    o, lat = score.overall, score.latency
    lines = [
        f"{run_dir.name}: {o.docs} docs | critical-doc {pct(o.critical_doc_accuracy)} | "
        f"field {pct(o.field_accuracy)} | rows R {pct(o.row_recall)} P {pct(o.row_precision)} | "
        f"p50 {seconds(lat.p50_ms)} p95 {seconds(lat.p95_ms)} | "
        f"completion {number(lat.mean_completion_tokens)} tok | parse errors {lat.parse_errors}",
    ]
    for variant, s in score.by_variant.items():
        lines.append(
            f"  {variant}: {s.docs} docs | critical-doc {pct(s.critical_doc_accuracy)} | "
            f"field {pct(s.field_accuracy)} | row recall {pct(s.row_recall)}"
        )
    problems = top_problem_fields(score.errors, n=3)
    if problems:
        lines.append("  top problems: " + ", ".join(f"{p.field} ({p.errors})" for p in problems))
    lines.append(f"  wrote {run_dir}/report.md, report.json, errors.csv")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score a run and write its report.")
    parser.add_argument("--run", type=Path, required=True, help="run folder, e.g. runs/b0_dev")
    parser.add_argument("--dataset", type=Path, default=None, help="default: from config.json")
    parser.add_argument(
        "--exclude-computed", action="store_true", help="do not score COMPUTED_FIELDS"
    )
    args = parser.parse_args()
    score = write_report(args.run, args.dataset, include_computed=not args.exclude_computed)
    print(terminal_summary(args.run, score))


if __name__ == "__main__":
    main()
