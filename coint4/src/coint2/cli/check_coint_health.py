#!/usr/bin/env python3
"""
Check health of cointegration for current universe pairs.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller

from coint2.core.data_loader import DataHandler
from coint2.utils.config import load_config
from coint2.pipeline.pair_scanner import test_cointegration, estimate_half_life


def check_coint_health(pairs_file: str = "bench/pairs_universe.yaml") -> List[Dict]:
    """Check health of cointegration for universe pairs."""
    if not Path(pairs_file).exists():
        print(f"Pairs file not found: {pairs_file}")
        print("Run: python -m coint2.cli.build_universe first")
        return []

    with open(pairs_file, "r") as f:
        pairs_data = yaml.safe_load(f)

    pairs = pairs_data.get("pairs", [])
    print(f"Checking {len(pairs)} pairs")

    app_cfg = load_config("configs/main_2024.yaml")
    handler = DataHandler(app_cfg)

    end_date = pd.Timestamp.now()
    lookback_days = 30
    df = handler.load_all_data_for_period(
        lookback_days=lookback_days,
        end_date=end_date,
    )

    health_results: List[Dict] = []
    for pair_info in pairs:
        if isinstance(pair_info, dict):
            sym1 = pair_info.get("symbol1")
            sym2 = pair_info.get("symbol2")
            original_metrics = pair_info
        else:
            sym1, sym2 = pair_info.split("/")
            original_metrics = {}

        if sym1 not in df.columns or sym2 not in df.columns:
            health_results.append(
                {
                    "pair": f"{sym1}/{sym2}",
                    "status": "MISSING",
                    "issue": "Symbol not found in data",
                }
            )
            continue

        aligned = pd.DataFrame({"y": df[sym1], "x": df[sym2]}).dropna()
        if len(aligned) < 50:
            health_results.append(
                {
                    "pair": f"{sym1}/{sym2}",
                    "status": "INSUFFICIENT",
                    "issue": f"Only {len(aligned)} data points",
                }
            )
            continue

        health = check_pair_health(
            aligned["y"].values, aligned["x"].values, original_metrics
        )
        health["pair"] = f"{sym1}/{sym2}"
        health_results.append(health)

    report = generate_health_report(health_results, pairs_data)
    report_path = Path("artifacts/universe/COINT_HEALTH.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    return health_results


def check_pair_health(y: np.ndarray, x: np.ndarray, original_metrics: dict) -> dict:
    """Check health of a single pair."""
    result = {"status": "OK", "issue": None}

    X = np.column_stack([np.ones(len(x)), x])
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = betas[0], betas[1]
    spread = y - beta * x - alpha

    try:
        adf_result = adfuller(spread, maxlag=10, autolag="BIC")
        pvalue = adf_result[1]
        result["pvalue_current"] = pvalue
        result["pvalue_original"] = original_metrics.get("pvalue", 0)

        if pvalue > 0.10:
            result["status"] = "FAIL"
            result["issue"] = f"P-value degraded to {pvalue:.3f}"
        elif pvalue > 0.05:
            result["status"] = "WARN"
            result["issue"] = f"P-value weakened to {pvalue:.3f}"
    except Exception:
        result["status"] = "FAIL"
        result["issue"] = "ADF test failed"
        result["pvalue_current"] = 1.0

    half_life = estimate_half_life(spread)
    result["half_life_current"] = half_life
    result["half_life_original"] = original_metrics.get("half_life", 0)
    if half_life > 500 and result["status"] == "OK":
        result["status"] = "WARN"
        result["issue"] = f"Half-life increased to {half_life:.0f}"

    mid = len(y) // 2
    beta1 = np.linalg.lstsq(
        np.column_stack([np.ones(mid), x[:mid]]),
        y[:mid],
        rcond=None,
    )[0][1]
    beta2 = np.linalg.lstsq(
        np.column_stack([np.ones(len(y) - mid), x[mid:]]),
        y[mid:],
        rcond=None,
    )[0][1]
    beta_drift = abs(beta2 - beta1) / abs(beta1) if beta1 != 0 else 0
    result["beta_drift_current"] = beta_drift
    if beta_drift > 0.30 and result["status"] != "FAIL":
        result["status"] = "WARN"
        if result["issue"] is None:
            result["issue"] = f"Beta drift {beta_drift:.2%}"

    return result


def generate_health_report(health_results: list, pairs_data: dict) -> str:
    """Generate health report markdown."""
    generated = pairs_data.get("metadata", {}).get("generated", "unknown")
    report = [
        "# Cointegration Health Report",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
        f"*Universe Generated: {generated}*",
        "",
        "## Summary",
        "",
    ]

    ok_count = sum(1 for h in health_results if h["status"] == "OK")
    warn_count = sum(1 for h in health_results if h["status"] == "WARN")
    fail_count = sum(1 for h in health_results if h["status"] == "FAIL")
    report.append(f"- OK: {ok_count}")
    report.append(f"- WARN: {warn_count}")
    report.append(f"- FAIL: {fail_count}")
    report.append("")
    report.append("## Detailed Results")
    report.append("")
    report.append("| Pair | Status | Issue |")
    report.append("|------|--------|-------|")
    for h in health_results:
        issue = h.get("issue") or ""
        report.append(f"| {h['pair']} | {h['status']} | {issue} |")

    return "\n".join(report) + "\n"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check cointegration health")
    parser.add_argument(
        "--pairs-file",
        default="bench/pairs_universe.yaml",
        help="Path to pairs file",
    )
    args = parser.parse_args()
    check_coint_health(args.pairs_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
