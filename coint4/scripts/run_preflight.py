#!/usr/bin/env python3
"""Preflight checks for live trading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def run_preflight_checks() -> tuple[bool, list[dict]]:
    """Run lightweight preflight checks."""
    checks: list[dict] = []

    risk_config = Path("configs/risk.yaml")
    checks.append(
        {
            "name": "risk_config_exists",
            "passed": risk_config.exists(),
            "details": str(risk_config),
        }
    )

    overall_ok = all(check["passed"] for check in checks)
    return overall_ok, checks


def generate_preflight_report(checks: list[dict], output_path: Path, success: bool) -> None:
    """Generate markdown preflight report."""
    lines = [
        "# Preflight Report",
        "",
        f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        "",
        "## Overall Status",
        "",
        f"- Status: {'PASS' if success else 'FAIL'}",
        "",
        "## Checks",
        "",
        "| Check | Passed | Details |",
        "|-------|--------|---------|",
    ]
    for check in checks:
        lines.append(
            f"| {check['name']} | {check['passed']} | {check.get('details', '')} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run preflight checks and emit artifacts."""
    artifacts_dir = Path("artifacts/live")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    success, checks = run_preflight_checks()

    log_path = artifacts_dir / "PREFLIGHT.log"
    with open(log_path, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} preflight success={success}\n")

    report_path = artifacts_dir / "PREFLIGHT_REPORT.md"
    generate_preflight_report(checks, report_path, success)

    summary_path = artifacts_dir / "PREFLIGHT.json"
    summary_path.write_text(json.dumps({"success": success, "checks": checks}, indent=2))

    print(f"Preflight {'PASS' if success else 'FAIL'}: {len(checks)} checks")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
