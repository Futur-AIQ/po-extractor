"""The on-disk run format every pipeline writes (baseline, optimised, benchmark).

    runs/<run_id>/config.json            how the run was made (pipeline, model alias, ...)
    runs/<run_id>/outputs/<doc_id>.json  final PO in FULL field names, or an error record
    runs/<run_id>/raw/<doc_id>.json      raw model responses, for debugging
    runs/<run_id>/records.jsonl          one line per document: status, timings, tokens, calls

doc_id carries the variant suffix: PO_0001 (native), PO_0001_S (scanned), PO_0001_M (mixed).

The helpers do small synchronous file writes; async pipelines call them via
`asyncio.to_thread` so the event loop never blocks.
"""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schema.po_schema import PurchaseOrder

# completed / needs_review: the pipeline produced a PO (needs_review = failed validation).
# parse_error: the model output could not be parsed into a PO. failed: anything else broke.
Status = Literal["completed", "needs_review", "parse_error", "failed"]
STATUSES_WITH_OUTPUT = ("completed", "needs_review")


# =========================================================================================
# doc_id
# =========================================================================================


def doc_id_for(base_id: str, variant: str) -> str:
    """'PO_0001' + 'N' -> 'PO_0001'; + 'S' -> 'PO_0001_S'."""
    return base_id if variant == "N" else f"{base_id}_{variant}"


def split_doc_id(doc_id: str) -> tuple[str, str]:
    """'PO_0001_S' -> ('PO_0001', 'S'); 'PO_0001' -> ('PO_0001', 'N')."""
    base, _, suffix = doc_id.rpartition("_")
    if suffix in ("S", "M"):
        return base, suffix
    return doc_id, "N"


# =========================================================================================
# Models
# =========================================================================================


class RunConfig(BaseModel):
    """config.json: everything needed to tell runs apart and reproduce one."""

    model_config = ConfigDict(extra="forbid")

    pipeline: str  # baseline, optimised, oracle, ...
    model_alias: str | None = None  # LiteLLM alias, e.g. po-fast
    strategy: str | None = None  # two_call / per_page / adaptive
    dpi: int | None = None
    thinking: str | None = None  # off / low / default
    git_commit: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    extra: dict[str, Any] = Field(default_factory=dict)  # split, variants, limits, ...


class TokenUsage(BaseModel):
    """Token counts summed over all LLM calls of one document."""

    prompt: int = 0
    completion: int = 0  # all generated tokens, reasoning included
    reasoning: int | None = None  # None when unknown
    reasoning_estimated: bool = False  # True: derived from the reasoning text, not reported


class DocRecord(BaseModel):
    """One line of records.jsonl."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    variant: str
    status: Status
    timings_ms: dict[str, Any]  # must contain "total"; other keys are pipeline stages
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    calls: int = 0  # number of LLM calls
    error: str | None = None

    @field_validator("timings_ms")
    @classmethod
    def _has_total(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "total" not in value:
            raise ValueError("timings_ms must contain 'total'")
        return value

    @property
    def total_ms(self) -> float:
        return float(self.timings_ms["total"])


class RunOutput(BaseModel):
    """outputs/<doc_id>.json: the final PO (full field names) or an error record."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    status: Status
    po: dict[str, Any] | None = None  # JSON of the PO; may be partial or schema-invalid
    error: str | None = None


# =========================================================================================
# Writer
# =========================================================================================


def current_git_commit() -> str | None:
    """Short HEAD commit, with '-dirty' if tracked files have changes; None outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if changes else commit


def new_run_id(pipeline: str, now: datetime | None = None) -> str:
    """Default run id, e.g. 'baseline_20260314-101500'."""
    return f"{pipeline}_{(now or datetime.now()):%Y%m%d-%H%M%S}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8")


class RunWriter:
    """Writes one run directory. Create it with `RunWriter.create`."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @classmethod
    def create(
        cls, runs_dir: Path, run_id: str, config: RunConfig, overwrite: bool = False
    ) -> "RunWriter":
        """Make runs_dir/run_id with config.json. An existing run is replaced only if
        `overwrite` is set (and only if it really is a run directory)."""
        run_dir = runs_dir / run_id
        if run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"run {run_dir} already exists")
            if not (run_dir / "config.json").exists():
                raise FileExistsError(f"{run_dir} exists but is not a run directory")
            shutil.rmtree(run_dir)
        (run_dir / "outputs").mkdir(parents=True)
        (run_dir / "raw").mkdir()
        (run_dir / "records.jsonl").touch()
        _write_json(run_dir / "config.json", config.model_dump(mode="json"))
        return cls(run_dir)

    def write_output(
        self,
        doc_id: str,
        po: PurchaseOrder | dict[str, Any],
        status: Status = "completed",
        error: str | None = None,
    ) -> None:
        """Save the final PO in full field names. A dict that is not a valid PurchaseOrder
        (e.g. status parse_error after schema validation) is saved as-is with its error, so
        the values the model did read can still be scored."""
        data = po.model_dump(mode="json") if isinstance(po, PurchaseOrder) else po
        output = RunOutput(doc_id=doc_id, status=status, po=data, error=error)
        path = self.run_dir / "outputs" / f"{doc_id}.json"
        _write_json(path, output.model_dump(mode="json"))

    def write_error(self, doc_id: str, status: Status, error: str) -> None:
        """Save an error record in place of a PO (e.g. parse_error)."""
        output = RunOutput(doc_id=doc_id, status=status, error=error)
        _write_json(self.run_dir / "outputs" / f"{doc_id}.json", output.model_dump(mode="json"))

    def write_raw(self, doc_id: str, responses: Any) -> None:
        """Save the raw model responses of one document (any JSON-able value)."""
        _write_json(self.run_dir / "raw" / f"{doc_id}.json", responses)

    def append_record(self, record: DocRecord) -> None:
        """Append one line to records.jsonl."""
        with (self.run_dir / "records.jsonl").open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")


# =========================================================================================
# Readers
# =========================================================================================


def read_config(run_dir: Path) -> RunConfig:
    """Load config.json of a run."""
    return RunConfig.model_validate_json((run_dir / "config.json").read_text("utf-8"))


def read_records(run_dir: Path) -> list[DocRecord]:
    """Load records.jsonl of a run, in write order."""
    lines = (run_dir / "records.jsonl").read_text("utf-8").splitlines()
    return [DocRecord.model_validate_json(line) for line in lines if line.strip()]


def read_output(run_dir: Path, doc_id: str) -> RunOutput | None:
    """Load outputs/<doc_id>.json, or None if the pipeline wrote nothing for this doc."""
    path = run_dir / "outputs" / f"{doc_id}.json"
    if not path.exists():
        return None
    return RunOutput.model_validate_json(path.read_text("utf-8"))


def read_raw(run_dir: Path, doc_id: str) -> Any:
    """Load raw/<doc_id>.json, or None if absent."""
    path = run_dir / "raw" / f"{doc_id}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None
