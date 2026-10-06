"""Tests for the `make doctor` summary and exit-code logic (Step 0.5). Probes are mocked."""

from pydantic import SecretStr

from app.core.settings import Settings
from scripts.doctor import (
    Check,
    CheckFailed,
    CheckResult,
    build_checks,
    exit_code,
    render_report,
    run_checks,
)


def passing(detail: str = "ok"):
    async def probe() -> str:
        return detail

    return probe


def failing(exc: Exception):
    async def probe() -> str:
        raise exc

    return probe


async def test_all_passing_checks_exit_zero() -> None:
    checks = [Check("A", "fix A", passing("a fine")), Check("B", "fix B", passing())]

    results = await run_checks(checks)

    assert [r.ok for r in results] == [True, True]
    assert results[0].detail == "a fine"
    assert exit_code(results) == 0


async def test_any_failure_exits_non_zero() -> None:
    checks = [Check("A", "fix A", passing()), Check("B", "fix B", failing(CheckFailed("down")))]

    results = await run_checks(checks)

    assert results[1] == CheckResult("B", False, "down", "fix B")
    assert exit_code(results) == 1


async def test_unexpected_exception_becomes_fail_not_crash() -> None:
    results = await run_checks([Check("A", "fix A", failing(ConnectionRefusedError("refused")))])

    assert not results[0].ok
    assert results[0].detail == "ConnectionRefusedError: refused"


async def test_results_keep_check_order() -> None:
    names = ["first", "second", "third"]

    results = await run_checks([Check(n, "", passing()) for n in names])

    assert [r.name for r in results] == names


def test_report_lists_status_and_hints_for_failures_only() -> None:
    results = [
        CheckResult("Docker", True, "Docker 29.8.2", "start Docker Desktop"),
        CheckResult("App Redis :6380", False, "ConnectionError: refused", "make up"),
    ]

    report = render_report(results)

    assert "Docker           PASS    Docker 29.8.2" in report
    assert "App Redis :6380  FAIL    ConnectionError: refused" in report
    assert "1 of 2 checks failed. Fixes:" in report
    assert "  - App Redis :6380: make up" in report
    assert "start Docker Desktop" not in report


def test_report_all_passed_summary() -> None:
    report = render_report([CheckResult("A", True, "ok", "fix")])

    assert "All 1 checks passed." in report


def test_report_truncates_long_details() -> None:
    report = render_report([CheckResult("A", False, "x" * 500, "fix")])

    assert "x" * 500 not in report
    assert "..." in report


def test_build_checks_covers_every_service_and_alias() -> None:
    settings = Settings(_env_file=None, litellm_api_key=SecretStr("sk-test"))

    names = [c.name for c in build_checks(settings)]

    assert names == [
        ".env",
        "Docker",
        "App Redis :6380",
        "Langfuse UI :3000",
        "Langfuse keys",
        "llama.cpp :8081",
        "LiteLLM :4000",
        "alias po-fast",
        "alias po-accurate",
        "alias po-baseline",
    ]
