from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

from coint2.ops.heavy_guardrails import (
    HeavyGuardrailConfig,
    evaluate_heavy_guardrails,
    parse_host_allowlist,
)


def test_parse_host_allowlist_when_empty_then_returns_defaults() -> None:
    hosts = parse_host_allowlist("")
    assert "85.198.90.128" in hosts
    assert "coint" in hosts


def test_guardrails_pass_with_explicit_allow_and_relaxed_limits() -> None:
    host = socket.gethostname().strip().lower()
    config = HeavyGuardrailConfig(
        entrypoint="tests",
        host_allowlist=(host,),
        min_cpu=1,
        min_ram_gb=0.1,
    )
    result = evaluate_heavy_guardrails(config, environ={"ALLOW_HEAVY_RUN": "1"})
    assert result.passed
    assert result.failures == ()


def test_guardrails_fail_when_allow_flag_missing() -> None:
    host = socket.gethostname().strip().lower()
    config = HeavyGuardrailConfig(
        entrypoint="tests",
        host_allowlist=(host,),
        min_cpu=1,
        min_ram_gb=0.1,
    )
    result = evaluate_heavy_guardrails(config, environ={})
    assert not result.passed
    assert any("ALLOW_HEAVY_RUN" in failure for failure in result.failures)


def test_guardrails_fail_when_host_not_allowlisted() -> None:
    config = HeavyGuardrailConfig(
        entrypoint="tests",
        host_allowlist=("host-that-does-not-exist",),
        min_cpu=1,
        min_ram_gb=0.1,
    )
    result = evaluate_heavy_guardrails(config, environ={"ALLOW_HEAVY_RUN": "1"})
    assert not result.passed
    assert any("outside allowlist" in failure for failure in result.failures)


def test_guardrails_cli_returns_nonzero_on_block() -> None:
    app_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    env.pop("ALLOW_HEAVY_RUN", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "coint2.ops.heavy_guardrails",
            "--entrypoint",
            "tests/scripts/test_heavy_guardrails.py",
            "--allowlist",
            socket.gethostname(),
            "--min-cpu",
            "1",
            "--min-ram-gb",
            "0.1",
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "BLOCKED" in proc.stderr
