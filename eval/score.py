"""Scoring engine: compare a run's outputs with the ground truth (PRD §3, §11.2).

Header fields
    Each field is correct, wrong, missed (truth present, prediction absent), hallucinated
    (truth absent, prediction present) or correct-absent. Free-text fields also get a fuzzy
    score (rapidfuzz token_set_ratio); exact and fuzzy (>= 90) accuracy are reported apart.
Line items
    Rows are matched by line_no; rows without a usable line_no fall back to greedy matching on
    item code, quantity and fuzzy description. Row recall / precision / F1, and per-column cell
    accuracy over matched rows (cells use the same five outcomes).
Field accuracy
    Header fields plus matched-row cells; correct-absent counts as correct. Dropped and extra
    rows are measured by row recall and precision, not here.
Critical-field document accuracy
    A document is correct only if every critical header field is correct, every matched row's
    critical columns are correct, and row recall is 100%.

COMPUTED_FIELDS (total_tax, per-line tax amounts) are scored by default, because the final
output is what matters; pass include_computed=False to leave them out.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.common.stats import percentile
from eval.dataset import DatasetDoc, load_dataset, read_truth_po
from eval.normalize import FIELD_TYPES, FieldType, normalize
from eval.run_format import DocRecord, read_output, read_records
from schema.fields import (
    COMPUTED_HEADER_FIELDS,
    COMPUTED_LINE_ITEM_FIELDS,
    CRITICAL_HEADER_FIELDS,
    CRITICAL_LINE_ITEM_FIELDS,
    HEADER_FIELDS,
    LINE_ITEM_COLUMNS,
)

FUZZY_THRESHOLD = 90  # free text counts as fuzzy-correct at token_set_ratio >= 90
ROW_MATCH_MIN_DESCRIPTION = 80  # fallback row match: description similarity needed alone


class Outcome(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"
    MISSED = "missed"  # truth present, prediction absent
    HALLUCINATED = "hallucinated"  # truth absent, prediction present
    CORRECT_ABSENT = "correct_absent"  # both absent


ERROR_OUTCOMES = (Outcome.WRONG, Outcome.MISSED, Outcome.HALLUCINATED)


# =========================================================================================
# One value
# =========================================================================================


@dataclass(frozen=True)
class FieldResult:
    """Outcome of one header field or one line-item cell."""

    field: str
    outcome: Outcome
    truth: Any  # raw values, for the error list
    pred: Any
    fuzzy: float | None = None  # free-text fields with both values present
    row: int | None = None  # truth line_no, for line-item cells

    @property
    def correct(self) -> bool:
        return self.outcome in (Outcome.CORRECT, Outcome.CORRECT_ABSENT)

    @property
    def fuzzy_correct(self) -> bool:
        return self.correct or (self.fuzzy is not None and self.fuzzy >= FUZZY_THRESHOLD)


def score_value(name: str, truth: Any, pred: Any, row: int | None = None) -> FieldResult:
    """Classify one predicted value against the truth, after normalisation."""
    t, p = normalize(name, truth), normalize(name, pred)
    if t is None and p is None:
        outcome = Outcome.CORRECT_ABSENT
    elif p is None:
        outcome = Outcome.MISSED
    elif t is None:
        outcome = Outcome.HALLUCINATED
    else:
        outcome = Outcome.CORRECT if t == p else Outcome.WRONG
    fuzzy = None
    if FIELD_TYPES[name] is FieldType.TEXT and t is not None and p is not None:
        fuzzy = fuzz.token_set_ratio(str(t), str(p))
    return FieldResult(name, outcome, truth, pred, fuzzy, row)


# =========================================================================================
# Line-item row matching
# =========================================================================================


@dataclass(frozen=True)
class RowMatch:
    """Which predicted row (index) was matched to which truth row (index), and how."""

    truth_index: int
    pred_index: int
    by_line_no: bool


def _line_no(row: dict[str, Any]) -> int | None:
    value = normalize("line_no", row.get("line_no"))
    if value is None or isinstance(value, str) or value != value.to_integral_value():
        return None
    return int(value)


def _fallback_score(truth_row: dict[str, Any], pred_row: dict[str, Any]) -> float | None:
    """Similarity of two rows without a usable line_no, or None if they are not a plausible
    match: the description must be similar, or item code and quantity must both agree."""
    code = score_value("item_code", truth_row.get("item_code"), pred_row.get("item_code"))
    quantity = score_value("quantity", truth_row.get("quantity"), pred_row.get("quantity"))
    same_code = code.outcome is Outcome.CORRECT
    same_quantity = quantity.outcome is Outcome.CORRECT
    t = normalize("description", truth_row.get("description"))
    p = normalize("description", pred_row.get("description"))
    description = fuzz.token_set_ratio(str(t), str(p)) if t and p else 0.0
    if description < ROW_MATCH_MIN_DESCRIPTION and not (same_code and same_quantity):
        return None
    return same_code + same_quantity + description / 100


def match_rows(truth_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[RowMatch]:
    """Match predicted rows to truth rows: by line_no first, then greedily by content.

    A predicted line_no is usable if it is an integer, unique among predicted rows and
    present in the truth. Remaining rows are paired best-score-first (see _fallback_score).
    """
    truth_by_no = {_line_no(row): i for i, row in enumerate(truth_rows)}
    pred_nos = [_line_no(row) for row in pred_rows]
    counts = Counter(pred_nos)

    matches: list[RowMatch] = []
    for pred_index, number in enumerate(pred_nos):
        if number is not None and counts[number] == 1 and number in truth_by_no:
            matches.append(RowMatch(truth_by_no[number], pred_index, by_line_no=True))

    used_truth = {m.truth_index for m in matches}
    used_pred = {m.pred_index for m in matches}
    candidates = []
    for ti, truth_row in enumerate(truth_rows):
        if ti in used_truth:
            continue
        for pi, pred_row in enumerate(pred_rows):
            if pi not in used_pred and (score := _fallback_score(truth_row, pred_row)):
                candidates.append((-score, ti, pi))
    for _, ti, pi in sorted(candidates):
        if ti not in used_truth and pi not in used_pred:
            matches.append(RowMatch(ti, pi, by_line_no=False))
            used_truth.add(ti)
            used_pred.add(pi)
    return sorted(matches, key=lambda m: m.truth_index)


# =========================================================================================
# One document
# =========================================================================================


@dataclass(frozen=True)
class ErrorRecord:
    """One wrong, missed or hallucinated value (or row) in one document."""

    doc_id: str
    variant: str
    field: str  # header field, 'line_items.<column>' for a cell, 'line_items' for a row
    outcome: Outcome
    truth: Any
    pred: Any
    row: int | None = None  # truth line_no (cells, missed rows) or printed line_no (extra rows)
    fuzzy: float | None = None


@dataclass
class DocScore:
    """All results for one document variant."""

    doc: DatasetDoc
    status: str  # from records.jsonl: completed, needs_review, parse_error, failed
    header: list[FieldResult]
    cells: list[FieldResult]  # matched rows only
    truth_rows: int
    pred_rows: int
    matches: list[RowMatch]
    missed_rows: list[dict[str, Any]]  # truth rows without a match
    extra_rows: list[dict[str, Any]]  # predicted rows without a match

    @property
    def critical_correct(self) -> bool:
        header_ok = all(r.correct for r in self.header if r.field in CRITICAL_HEADER_FIELDS)
        cells_ok = all(r.correct for r in self.cells if r.field in CRITICAL_LINE_ITEM_FIELDS)
        return header_ok and cells_ok and not self.missed_rows

    def errors(self) -> list[ErrorRecord]:
        """Error records: header fields, then cells, then missed and extra rows."""
        doc_id, variant = self.doc.doc_id, self.doc.variant
        out = [
            ErrorRecord(doc_id, variant, r.field, r.outcome, r.truth, r.pred, fuzzy=r.fuzzy)
            for r in self.header
            if r.outcome in ERROR_OUTCOMES
        ]
        out += [
            ErrorRecord(
                doc_id, variant, f"line_items.{r.field}", r.outcome, r.truth, r.pred, r.row, r.fuzzy
            )
            for r in self.cells
            if r.outcome in ERROR_OUTCOMES
        ]
        out += [
            ErrorRecord(doc_id, variant, "line_items", Outcome.MISSED,
                        row.get("description"), None, row.get("line_no"))
            for row in self.missed_rows
        ]  # fmt: skip
        out += [
            ErrorRecord(doc_id, variant, "line_items", Outcome.HALLUCINATED,
                        None, row.get("description"), row.get("line_no"))
            for row in self.extra_rows
        ]  # fmt: skip
        return out


def scored_header_fields(include_computed: bool = True) -> list[str]:
    return [f for f in HEADER_FIELDS if include_computed or f not in COMPUTED_HEADER_FIELDS]


def scored_line_item_columns(include_computed: bool = True) -> list[str]:
    return [c for c in LINE_ITEM_COLUMNS if include_computed or c not in COMPUTED_LINE_ITEM_FIELDS]


def _rows(po: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = (po or {}).get("line_items")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def score_document(
    doc: DatasetDoc,
    truth: dict[str, Any],
    pred: dict[str, Any] | None,
    status: str = "completed",
    include_computed: bool = True,
) -> DocScore:
    """Score one predicted PO (JSON, full field names; None = no output) against its truth."""
    pred = pred or {}
    header = [
        score_value(name, truth.get(name), pred.get(name))
        for name in scored_header_fields(include_computed)
    ]
    truth_rows, pred_rows = _rows(truth), _rows(pred)
    matches = match_rows(truth_rows, pred_rows)
    columns = scored_line_item_columns(include_computed)
    cells = [
        score_value(
            column,
            truth_rows[m.truth_index].get(column),
            pred_rows[m.pred_index].get(column),
            row=truth_rows[m.truth_index].get("line_no"),
        )
        for m in matches
        for column in columns
    ]
    matched_truth = {m.truth_index for m in matches}
    matched_pred = {m.pred_index for m in matches}
    return DocScore(
        doc=doc,
        status=status,
        header=header,
        cells=cells,
        truth_rows=len(truth_rows),
        pred_rows=len(pred_rows),
        matches=matches,
        missed_rows=[r for i, r in enumerate(truth_rows) if i not in matched_truth],
        extra_rows=[r for i, r in enumerate(pred_rows) if i not in matched_pred],
    )


# =========================================================================================
# Aggregation
# =========================================================================================


def _ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator, or None when there is nothing to divide (shown as n/a)."""
    return numerator / denominator if denominator else None


@dataclass
class Tally:
    """Outcome counts over a set of field results (header fields or cells)."""

    outcomes: Counter = field(default_factory=Counter)
    text_total: int = 0  # free-text values, for exact vs fuzzy accuracy
    text_exact: int = 0
    text_fuzzy: int = 0

    def add(self, result: FieldResult) -> None:
        self.outcomes[result.outcome] += 1
        if FIELD_TYPES[result.field] is FieldType.TEXT:
            self.text_total += 1
            self.text_exact += result.correct
            self.text_fuzzy += result.fuzzy_correct

    @property
    def total(self) -> int:
        return sum(self.outcomes.values())

    @property
    def correct(self) -> int:
        return self.outcomes[Outcome.CORRECT] + self.outcomes[Outcome.CORRECT_ABSENT]

    @property
    def accuracy(self) -> float | None:
        return _ratio(self.correct, self.total)

    def as_dict(self) -> dict[str, Any]:
        return {
            **{outcome.value: self.outcomes[outcome] for outcome in Outcome},
            "total": self.total,
            "accuracy": self.accuracy,
            "text_exact_accuracy": _ratio(self.text_exact, self.text_total),
            "text_fuzzy_accuracy": _ratio(self.text_fuzzy, self.text_total),
        }


@dataclass
class GroupSummary:
    """Metrics over a group of documents (overall, one variant, one layout, ...)."""

    docs: int = 0
    critical_correct_docs: int = 0
    statuses: Counter = field(default_factory=Counter)
    header: Tally = field(default_factory=Tally)
    cells: Tally = field(default_factory=Tally)
    by_field: dict[str, Tally] = field(default_factory=dict)  # header field / cell column
    truth_rows: int = 0
    pred_rows: int = 0
    matched_rows: int = 0

    def add(self, score: DocScore) -> None:
        self.docs += 1
        self.critical_correct_docs += score.critical_correct
        self.statuses[score.status] += 1
        for result in score.header:
            self.header.add(result)
            self.by_field.setdefault(result.field, Tally()).add(result)
        for result in score.cells:
            self.cells.add(result)
            self.by_field.setdefault(f"line_items.{result.field}", Tally()).add(result)
        self.truth_rows += score.truth_rows
        self.pred_rows += score.pred_rows
        self.matched_rows += len(score.matches)

    @property
    def critical_doc_accuracy(self) -> float | None:
        return _ratio(self.critical_correct_docs, self.docs)

    @property
    def field_accuracy(self) -> float | None:
        return _ratio(
            self.header.correct + self.cells.correct, self.header.total + self.cells.total
        )

    @property
    def row_recall(self) -> float | None:
        return _ratio(self.matched_rows, self.truth_rows)

    @property
    def row_precision(self) -> float | None:
        return _ratio(self.matched_rows, self.pred_rows)

    @property
    def row_f1(self) -> float | None:
        recall, precision = self.row_recall, self.row_precision
        if recall is None or precision is None or recall + precision == 0:
            return None
        return 2 * recall * precision / (recall + precision)

    def as_dict(self) -> dict[str, Any]:
        return {
            "docs": self.docs,
            "statuses": dict(self.statuses),
            "critical_doc_accuracy": self.critical_doc_accuracy,
            "field_accuracy": self.field_accuracy,
            "header": self.header.as_dict(),
            "line_items": {
                "truth_rows": self.truth_rows,
                "pred_rows": self.pred_rows,
                "matched_rows": self.matched_rows,
                "recall": self.row_recall,
                "precision": self.row_precision,
                "f1": self.row_f1,
                "cells": self.cells.as_dict(),
            },
            "by_field": {name: tally.as_dict() for name, tally in self.by_field.items()},
        }


def summarize(scores: Iterable[DocScore]) -> GroupSummary:
    summary = GroupSummary()
    for score in scores:
        summary.add(score)
    return summary


def group_by(scores: list[DocScore], key: Callable[[DocScore], str]) -> dict[str, GroupSummary]:
    """One summary per key value, keys sorted."""
    groups: dict[str, list[DocScore]] = {}
    for score in scores:
        groups.setdefault(key(score), []).append(score)
    return {name: summarize(groups[name]) for name in sorted(groups)}


# =========================================================================================
# Latency and tokens (records.jsonl)
# =========================================================================================


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class LatencySummary:
    """End-to-end latency and token use over a set of documents."""

    docs: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    mean_reasoning_tokens: float | None  # over docs where it is known
    reasoning_estimated: bool  # some reasoning counts are estimates (see TokenUsage)
    mean_calls: float | None
    parse_errors: int


def summarize_latency(records: list[DocRecord]) -> LatencySummary:
    totals = [r.total_ms for r in records]
    reasoning = [r.tokens.reasoning for r in records if r.tokens.reasoning is not None]
    return LatencySummary(
        docs=len(records),
        p50_ms=percentile(totals, 50),
        p95_ms=percentile(totals, 95),
        max_ms=max(totals, default=None),
        mean_prompt_tokens=_mean([r.tokens.prompt for r in records]),
        mean_completion_tokens=_mean([r.tokens.completion for r in records]),
        mean_reasoning_tokens=_mean(reasoning),
        reasoning_estimated=any(r.tokens.reasoning_estimated for r in records),
        mean_calls=_mean([r.calls for r in records]),
        parse_errors=sum(r.status == "parse_error" for r in records),
    )


# =========================================================================================
# A whole run
# =========================================================================================


@dataclass
class RunScore:
    """Everything the evaluation report needs for one run."""

    include_computed: bool
    docs: list[DocScore]
    overall: GroupSummary
    by_variant: dict[str, GroupSummary]
    by_layout: dict[str, GroupSummary]
    by_issuer: dict[str, GroupSummary]  # frequent / one_off
    errors: list[ErrorRecord]
    latency: LatencySummary
    latency_by_variant: dict[str, LatencySummary]

    def as_dict(self) -> dict[str, Any]:
        """JSON-able form (the error list is left to the report)."""

        def groups(summaries: dict[str, GroupSummary]) -> dict[str, Any]:
            return {name: summary.as_dict() for name, summary in summaries.items()}

        return {
            "include_computed": self.include_computed,
            "overall": self.overall.as_dict(),
            "by_variant": groups(self.by_variant),
            "by_layout": groups(self.by_layout),
            "by_issuer": groups(self.by_issuer),
            "latency": vars(self.latency),
            "latency_by_variant": {k: vars(v) for k, v in self.latency_by_variant.items()},
            "errors": len(self.errors),
        }


def aggregate_run(
    scores: list[DocScore], records: list[DocRecord], include_computed: bool = True
) -> RunScore:
    """Aggregate document scores and run records into a RunScore."""
    variants = sorted({r.variant for r in records})
    return RunScore(
        include_computed=include_computed,
        docs=scores,
        overall=summarize(scores),
        by_variant=group_by(scores, lambda s: s.doc.variant),
        by_layout=group_by(scores, lambda s: s.doc.layout),
        by_issuer=group_by(scores, lambda s: "frequent" if s.doc.issuer_frequent else "one_off"),
        errors=[error for score in scores for error in score.errors()],
        latency=summarize_latency(records),
        latency_by_variant={
            v: summarize_latency([r for r in records if r.variant == v]) for v in variants
        },
    )


def score_run(
    run_dir: Path,
    dataset_dir: Path,
    include_computed: bool = True,
    doc_ids: set[str] | None = None,
) -> RunScore:
    """Score every document listed in the run's records.jsonl against the dataset truth.

    A document without a PO in outputs/ (parse_error, failed) scores as all-missed.
    `doc_ids` limits scoring to those documents (e.g. the ones two compared runs share).
    """
    docs = {doc.doc_id: doc for doc in load_dataset(dataset_dir)}
    records = read_records(run_dir)
    if doc_ids is not None:
        records = [r for r in records if r.doc_id in doc_ids]
    unknown = [r.doc_id for r in records if r.doc_id not in docs]
    if unknown:
        raise ValueError(f"documents not in {dataset_dir / 'manifest.csv'}: {unknown[:5]}")

    scores = []
    for record in records:
        doc = docs[record.doc_id]
        output = read_output(run_dir, record.doc_id)
        pred = output.po if output else None
        scores.append(
            score_document(doc, read_truth_po(doc), pred, record.status, include_computed)
        )
    return aggregate_run(scores, records, include_computed)
