"""Health check of every local dev service: `make doctor`.

Run from the repo root:  uv run python -m scripts.doctor

Each check is a small async probe that returns a detail string on success and raises on
failure. All probes run concurrently; the script prints a PASS/FAIL table, a fix hint for
every failure, and exits 1 if anything failed.
"""

import asyncio
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from dotenv import dotenv_values

from app.core.settings import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
REQUIRED_ENV_KEYS = ("LITELLM_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LOCAL_GGUF")
HEALTH_TIMEOUT_S = 5.0
COMPLETION_TIMEOUT_S = 60.0  # a 1-token call still pays prompt processing on a cold model
DETAIL_WIDTH = 60


class CheckFailed(Exception):
    """Raised by a probe with a short, human-readable reason."""


@dataclass(frozen=True)
class Check:
    """One named health check. `probe` returns a detail string or raises on failure."""

    name: str
    hint: str
    probe: Callable[[], Awaitable[str]]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of running one check."""

    name: str
    ok: bool
    detail: str
    hint: str


# --- probes ------------------------------------------------------------------------------


async def probe_env_file() -> str:
    """`.env` exists and every required key is non-empty (in `.env` or the environment)."""
    if not ENV_FILE.is_file():
        raise CheckFailed(f"{ENV_FILE.name} not found in repo root")
    values = await asyncio.to_thread(dotenv_values, ENV_FILE)
    missing = [k for k in REQUIRED_ENV_KEYS if not (os.environ.get(k) or values.get(k))]
    if missing:
        raise CheckFailed(f"empty or missing: {', '.join(missing)}")
    return f"{len(REQUIRED_ENV_KEYS)} required keys set"


async def probe_docker() -> str:
    """Docker daemon answers on its Unix socket (Engine API `/_ping` and `/version`)."""
    socket_path = find_docker_socket()
    if socket_path is None:
        raise CheckFailed("no Docker socket found (is Docker Desktop running?)")
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    async with httpx.AsyncClient(transport=transport, timeout=HEALTH_TIMEOUT_S) as client:
        ping = await client.get("http://docker/_ping")
        if ping.status_code != 200:
            raise CheckFailed(f"/_ping returned HTTP {ping.status_code}")
        version = (await client.get("http://docker/version")).json().get("Version", "?")
    return f"Docker {version}"


def find_docker_socket() -> str | None:
    """Return the Docker socket path from DOCKER_HOST or the usual macOS/Linux locations."""
    docker_host = os.environ.get("DOCKER_HOST", "")
    if docker_host.startswith("unix://"):
        return docker_host.removeprefix("unix://")
    for candidate in (Path.home() / ".docker/run/docker.sock", Path("/var/run/docker.sock")):
        if candidate.exists():
            return str(candidate)
    return None


async def probe_redis(settings: Settings) -> str:
    """App Redis answers PING."""
    client = aioredis.from_url(settings.redis_url, socket_connect_timeout=HEALTH_TIMEOUT_S)
    try:
        await client.ping()
    finally:
        await client.aclose()
    return f"PONG from {settings.redis_url}"


async def probe_langfuse_ui(settings: Settings) -> str:
    """Langfuse web is up (public health endpoint)."""
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        response = await client.get(f"{settings.langfuse_host}/api/public/health")
    if response.status_code != 200:
        raise CheckFailed(f"/api/public/health returned HTTP {response.status_code}")
    return f"Langfuse {response.json().get('version', '?')} at {settings.langfuse_host}"


async def probe_langfuse_keys(settings: Settings) -> str:
    """Langfuse accepts the project API keys (same endpoint the SDK's auth_check uses)."""
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key.get_secret_value()
    if not (public_key and secret_key):
        raise CheckFailed("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        response = await client.get(
            f"{settings.langfuse_host}/api/public/projects", auth=(public_key, secret_key)
        )
    if response.status_code != 200:
        raise CheckFailed(f"keys rejected (HTTP {response.status_code})")
    projects = [p.get("name", "?") for p in response.json().get("data", [])]
    return f"keys valid for project {', '.join(projects) or '?'}"


async def probe_llamacpp(settings: Settings) -> str:
    """llama-server is up and has finished loading its model."""
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        response = await client.get(f"{settings.llamacpp_base_url}/health")
    if response.status_code == 503:
        raise CheckFailed("server up but still loading the model")
    if response.status_code != 200:
        raise CheckFailed(f"/health returned HTTP {response.status_code}")
    return f"ready at {settings.llamacpp_base_url}"


async def probe_litellm(settings: Settings) -> str:
    """LiteLLM proxy process is alive."""
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        response = await client.get(f"{settings.litellm_base_url}/health/liveliness")
    if response.status_code != 200:
        raise CheckFailed(f"/health/liveliness returned HTTP {response.status_code}")
    return f"alive at {settings.litellm_base_url}"


async def probe_alias(settings: Settings, alias: str) -> str:
    """`alias` answers a 1-token chat completion through LiteLLM."""
    api_key = settings.litellm_api_key.get_secret_value()
    if not api_key:
        raise CheckFailed("LITELLM_API_KEY not set")
    body = {
        "model": alias,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=COMPLETION_TIMEOUT_S) as client:
        response = await client.post(
            f"{settings.litellm_base_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    latency_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        raise CheckFailed(f"HTTP {response.status_code}: {response.text[:120]}")
    return f"1-token reply in {latency_ms:.0f} ms"


# --- running and reporting ----------------------------------------------------------------


def build_checks(settings: Settings) -> list[Check]:
    """All checks, in the order they are reported."""
    alias_hint = "start `make model` and `make litellm`; alias must exist in the LiteLLM config"
    return [
        Check(".env", "cp .env.example .env and fill in the empty keys", probe_env_file),
        Check("Docker", "start Docker Desktop", probe_docker),
        Check(
            "App Redis :6380",
            "make up (then: make ping)",
            lambda: probe_redis(settings),
        ),
        Check("Langfuse UI :3000", "make langfuse-up", lambda: probe_langfuse_ui(settings)),
        Check(
            "Langfuse keys",
            "create API keys in the Langfuse UI, set them in .env (infra/langfuse/README.md)",
            lambda: probe_langfuse_keys(settings),
        ),
        Check("llama.cpp :8081", "make model", lambda: probe_llamacpp(settings)),
        Check("LiteLLM :4000", "make litellm", lambda: probe_litellm(settings)),
        *[
            Check(f"alias {alias}", alias_hint, lambda alias=alias: probe_alias(settings, alias))
            for alias in (settings.po_fast, settings.po_accurate, settings.po_baseline)
        ],
    ]


async def run_check(check: Check) -> CheckResult:
    """Run one check, turning any exception into a FAIL result."""
    try:
        detail = await check.probe()
    except CheckFailed as exc:
        return CheckResult(check.name, False, str(exc), check.hint)
    except Exception as exc:  # connection refused, timeouts, bad JSON, ...
        reason = str(exc) or "no details"
        return CheckResult(check.name, False, f"{type(exc).__name__}: {reason}", check.hint)
    return CheckResult(check.name, True, detail, check.hint)


async def run_checks(checks: list[Check]) -> list[CheckResult]:
    """Run all checks concurrently; results keep the order of `checks`."""
    return list(await asyncio.gather(*(run_check(c) for c in checks)))


def render_report(results: list[CheckResult]) -> str:
    """Format results as a PASS/FAIL table followed by fix hints for failures."""
    name_width = max([len("CHECK"), *(len(r.name) for r in results)])
    lines = [f"{'CHECK':<{name_width}}  STATUS  DETAIL", "-" * (name_width + 16 + DETAIL_WIDTH)]
    for r in results:
        detail = r.detail if len(r.detail) <= DETAIL_WIDTH else r.detail[: DETAIL_WIDTH - 3] + "..."
        lines.append(f"{r.name:<{name_width}}  {'PASS' if r.ok else 'FAIL':<6}  {detail}")

    failed = [r for r in results if not r.ok]
    lines.append("")
    if not failed:
        lines.append(f"All {len(results)} checks passed.")
    else:
        lines.append(f"{len(failed)} of {len(results)} checks failed. Fixes:")
        lines.extend(f"  - {r.name}: {r.hint}" for r in failed)
    return "\n".join(lines)


def exit_code(results: list[CheckResult]) -> int:
    """0 if every check passed, otherwise 1."""
    return 0 if all(r.ok for r in results) else 1


async def main() -> int:
    """Run every check, print the report, and return the process exit code."""
    results = await run_checks(build_checks(get_settings()))
    print(render_report(results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
