#!/usr/bin/env python3
"""CI gate checks for portfolio artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


class CIGateChecker:
    """Validate portfolio artifacts against CI gates."""

    def __init__(
        self,
        config: Dict | None = None,
        config_path: str | None = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_path)

    def _load_config(self, config_path: str | None) -> Dict:
        default_config = {
            "performance": {
                "source": "wfa",
                "wfa": {
                    "path": "artifacts/wfa/results_per_fold.csv",
                    "sharpe_col": "sharpe",
                    "trades_col": "trades",
                    "pnl_col": "pnl",
                    "min_trades": 10,
                    "aggregation": "mean",
                },
            },
            "thresholds": {"min_sharpe": 0.0, "max_drawdown_pct": 50.0, "min_trades": 0},
            "fallbacks": {"on_missing_file": "warn", "on_insufficient_trades": "warn"},
        }
        if not config_path:
            return default_config
        path = Path(config_path)
        if not path.exists():
            return default_config
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        merged = default_config
        merged.update(loaded)
        return merged

    def calc_drawdown_pct(self, equity: pd.Series | pd.DataFrame | list | pd.Index | pd.Series | pd.Series | pd.Series) -> float:
        """Calculate maximum drawdown percentage from equity curve."""
        values = pd.Series(equity, dtype=float)
        if values.empty or values.isna().any():
            return 0.0

        positive_idx = values[values > 0].index
        if positive_idx.empty:
            return 0.0

        start_idx = positive_idx.min()
        values = values.loc[start_idx:]
        running_max = values.cummax()
        drawdowns = (running_max - values) / running_max
        return float(drawdowns.max() if not drawdowns.empty else 0.0)

    def check_performance_metrics(self) -> Tuple[bool, Dict]:
        """Check performance metrics based on configured source."""
        perf_cfg = self.config.get("performance", {})
        source = perf_cfg.get("source", "wfa")
        if self.verbose:
            print(f"Reading performance from source: {source}")

        if source != "wfa":
            return False, {"error": f"Unsupported source: {source}"}

        wfa_cfg = perf_cfg.get("wfa", {})
        path = Path(wfa_cfg.get("path", ""))
        if not path.exists():
            if self.config.get("fallbacks", {}).get("on_missing_file") == "fail_explicit":
                return False, {"error": f"File not found: {path}"}
            return True, {"error": f"File not found: {path}"}

        df = pd.read_csv(path)
        if self.verbose:
            print(f"Loaded {len(df)} rows from {path}")

        if df.empty:
            return False, {"error": "Empty performance data", "total_trades": 0}

        sharpe_col = wfa_cfg.get("sharpe_col", "sharpe")
        trades_col = wfa_cfg.get("trades_col", "trades")
        pnl_col = wfa_cfg.get("pnl_col", "pnl")
        aggregation = wfa_cfg.get("aggregation", "mean")

        sharpe_series = df[sharpe_col].dropna()
        if sharpe_series.empty:
            sharpe_value = 0.0
        elif aggregation == "median":
            sharpe_value = float(sharpe_series.median())
        else:
            sharpe_value = float(sharpe_series.mean())

        total_trades = int(df[trades_col].sum()) if trades_col in df.columns else 0
        min_trades = wfa_cfg.get("min_trades", 0)
        if total_trades < min_trades:
            if self.config.get("fallbacks", {}).get("on_insufficient_trades") == "fail_explicit":
                return False, {"error": "Insufficient trades", "total_trades": total_trades}

        metrics = {
            "sharpe_ratio": sharpe_value,
            "total_trades": total_trades,
            "drawdown_pct": 0.0,
        }

        thresholds = self.config.get("thresholds", {})
        passed = True
        if sharpe_value < thresholds.get("min_sharpe", float("-inf")):
            passed = False
        if total_trades < thresholds.get("min_trades", 0):
            passed = False

        return passed, metrics

    def check_portfolio_gates(self) -> Tuple[bool, Dict]:
        gates = self.config.get("portfolio_gates", {})
        if not gates.get("enabled", False):
            return True, {"message": "Portfolio gates disabled"}

        failures = []
        pairs_file = Path(gates.get("require_pairs_file", ""))
        weights_file = Path(gates.get("weights_file", ""))
        report_file = Path(gates.get("portfolio_report", ""))

        for path, label in [
            (pairs_file, "pairs_file"),
            (weights_file, "weights_file"),
            (report_file, "portfolio_report"),
        ]:
            if not path.exists():
                failures.append(f"Missing {label}: {path}")

        if failures:
            return False, {"failures": failures}

        with open(pairs_file, "r") as f:
            pairs_data = yaml.safe_load(f)
        pairs = pairs_data.get("pairs", [])

        weights_df = pd.read_csv(weights_file)
        max_weight = float(weights_df["weight"].abs().max()) if not weights_df.empty else 0.0
        gross = float(weights_df["weight"].abs().sum()) if not weights_df.empty else 0.0

        metrics = {
            "selected_pairs": len(pairs),
            "max_weight": max_weight,
            "gross_exposure": gross,
        }

        if len(pairs) < gates.get("min_pairs", 0):
            failures.append("Too few pairs selected")
        if max_weight > gates.get("max_weight_per_pair", 1.0):
            failures.append("Max weight exceeds limit")
        if gross > gates.get("max_gross", 1.0):
            failures.append("Gross exposure exceeds limit")

        passed = len(failures) == 0
        result = {"metrics": metrics, "failures": failures}
        if passed:
            result["message"] = "Portfolio gates passed"
        return passed, result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run CI gates")
    parser.add_argument("--config", default="configs/ci_gates.yaml")
    args = parser.parse_args()

    if not Path(args.config).exists():
        return 0

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    checker = CIGateChecker(config, verbose=True)
    passed, result = checker.check_portfolio_gates()
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
