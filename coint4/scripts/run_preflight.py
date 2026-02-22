#!/usr/bin/env python3
"""Preflight checks for live trading."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TRADEABILITY_MIN_LIQUIDITY_USD_DAILY = 300_000.0
TRADEABILITY_MAX_BID_ASK_PCT = 0.60
TRADEABILITY_MAX_AVG_FUNDING_PCT = 0.07
PAIR_STABILITY_MIN_WINDOW_STEPS = 2
PAIR_STABILITY_MIN_STEPS = 2


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_preflight_checks(config_path: Path) -> tuple[bool, list[dict]]:
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

    checks.append(
        {
            "name": "strategy_config_exists",
            "passed": config_path.exists(),
            "details": str(config_path),
        }
    )
    config_payload = _load_yaml(config_path) if config_path.exists() else {}
    checks.append(
        {
            "name": "strategy_config_parseable",
            "passed": bool(config_payload),
            "details": "ok" if config_payload else "invalid_or_empty_yaml",
        }
    )

    pair_selection = config_payload.get("pair_selection", {}) if isinstance(config_payload, dict) else {}
    if not isinstance(pair_selection, dict):
        pair_selection = {}

    tradeability_enabled = bool(pair_selection.get("enable_pair_tradeability_filter", False))
    liquidity = _to_float(pair_selection.get("liquidity_usd_daily"), 0.0)
    max_bid_ask = _to_float(pair_selection.get("max_bid_ask_pct"), 1.0)
    max_funding = _to_float(pair_selection.get("max_avg_funding_pct"), 1.0)
    tradeability_violations: list[str] = []
    if not tradeability_enabled:
        tradeability_violations.append("enable_pair_tradeability_filter=false")
    if liquidity < TRADEABILITY_MIN_LIQUIDITY_USD_DAILY:
        tradeability_violations.append(
            f"liquidity_usd_daily<{int(TRADEABILITY_MIN_LIQUIDITY_USD_DAILY)} ({liquidity:.0f})"
        )
    if max_bid_ask > TRADEABILITY_MAX_BID_ASK_PCT:
        tradeability_violations.append(
            f"max_bid_ask_pct>{TRADEABILITY_MAX_BID_ASK_PCT:.2f} ({max_bid_ask:.4f})"
        )
    if max_funding > TRADEABILITY_MAX_AVG_FUNDING_PCT:
        tradeability_violations.append(
            f"max_avg_funding_pct>{TRADEABILITY_MAX_AVG_FUNDING_PCT:.2f} ({max_funding:.4f})"
        )
    checks.append(
        {
            "name": "tradeability_thresholds_guardrail",
            "passed": not tradeability_violations,
            "details": (
                "ok"
                if not tradeability_violations
                else "; ".join(tradeability_violations)
            ),
        }
    )

    stability_window = int(pair_selection.get("pair_stability_window_steps") or 0)
    stability_min = int(pair_selection.get("pair_stability_min_steps") or 0)
    stability_violations: list[str] = []
    if stability_window <= 0 or stability_min <= 0:
        stability_violations.append("pair_stability_window_steps/min_steps must be set")
    if stability_window < PAIR_STABILITY_MIN_WINDOW_STEPS:
        stability_violations.append(
            f"pair_stability_window_steps<{PAIR_STABILITY_MIN_WINDOW_STEPS} ({stability_window})"
        )
    if stability_min < PAIR_STABILITY_MIN_STEPS:
        stability_violations.append(
            f"pair_stability_min_steps<{PAIR_STABILITY_MIN_STEPS} ({stability_min})"
        )
    if stability_window > 0 and stability_min > stability_window:
        stability_violations.append(
            f"pair_stability_min_steps>{stability_window} ({stability_min})"
        )
    checks.append(
        {
            "name": "pair_stability_guardrail",
            "passed": not stability_violations,
            "details": (
                "ok"
                if not stability_violations
                else "; ".join(stability_violations)
            ),
        }
    )

    overall_ok = all(check["passed"] for check in checks)
    return overall_ok, checks


def generate_preflight_report(checks: list[dict], output_path: Path, success: bool) -> None:
    """Generate markdown preflight report."""
    lines = [
        "# Preflight Report",
        "",
        "*Generated: see PREFLIGHT.log*",
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
    parser = argparse.ArgumentParser(description="Preflight checks for live trading")
    parser.add_argument(
        "--config",
        default="configs/prod_final_budget1000.yaml",
        help="Strategy config path for preflight guardrails",
    )
    args = parser.parse_args()

    artifacts_dir = Path("artifacts/live")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    success, checks = run_preflight_checks(Path(args.config))

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
