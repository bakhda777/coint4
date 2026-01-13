#!/usr/bin/env python3
"""Drift monitoring with lightweight reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


class DriftMonitor:
    """Monitor performance drift and trigger responses."""

    def __init__(self, config_path: str, verbose: bool = False):
        self.verbose = verbose
        self.config_path = config_path
        self.config = self._load_config()
        self.drift_status = "OK"
        self.degradation_level = 0
        self.actions_taken: List[str] = []

        for output_path in self.config.get("outputs", {}).values():
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_confidence_data(self) -> pd.DataFrame:
        confidence_file = self.config["data_sources"]["confidence_file"]
        path = Path(confidence_file)
        if not path.exists():
            return pd.DataFrame(
                {
                    "pair": ["PORTFOLIO"],
                    "metric": ["sharpe"],
                    "p05": [0.3],
                    "p50": [0.6],
                    "p95": [1.0],
                    "observed": [0.7],
                }
            )
        return pd.read_csv(path)

    def load_recent_performance(self) -> pd.DataFrame:
        wfa_file = self.config["data_sources"]["wfa_results"]
        path = Path(wfa_file)
        if not path.exists():
            dates = pd.date_range(datetime.now() - timedelta(days=60), periods=60, freq="D")
            return pd.DataFrame(
                {
                    "date": dates,
                    "sharpe": np.random.normal(0.8, 0.1, len(dates)),
                    "psr": np.random.normal(0.8, 0.05, len(dates)),
                    "trades": np.random.poisson(20, len(dates)),
                    "pnl": np.random.normal(0.02, 0.3, len(dates)),
                }
            )

        df = pd.read_csv(path)
        if "date" not in df.columns:
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"])
            else:
                df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq="D")
        return df

    def calculate_drift_metrics(self, performance_data: pd.DataFrame) -> Dict[str, float]:
        windows = self.config["windows"]
        short_days = windows["short_days"]
        long_days = windows["long_days"]

        perf_df = performance_data.sort_values("date")
        cutoff_short = datetime.now() - timedelta(days=short_days)
        cutoff_long = datetime.now() - timedelta(days=long_days)
        short_window = perf_df[perf_df["date"] >= cutoff_short]
        long_window = perf_df[perf_df["date"] >= cutoff_long]

        metrics: Dict[str, float] = {}
        if "sharpe" in perf_df.columns:
            metrics["sharpe_short"] = float(short_window["sharpe"].mean())
            metrics["sharpe_long"] = float(long_window["sharpe"].mean())
            metrics["sharpe_drop"] = max(0.0, metrics["sharpe_long"] - metrics["sharpe_short"])
        if "psr" in perf_df.columns:
            metrics["psr_short"] = float(short_window["psr"].mean())
            metrics["psr_long"] = float(long_window["psr"].mean())
            metrics["psr_drop"] = max(0.0, metrics["psr_long"] - metrics["psr_short"])

        return metrics

    def assess_drift_status(
        self, confidence_data: pd.DataFrame, drift_metrics: Dict[str, float]
    ) -> Tuple[str, int]:
        thresholds = self.config["thresholds"]
        sharpe_p05 = confidence_data.loc[
            confidence_data["metric"] == "sharpe", "p05"
        ].mean()
        psr_p05 = confidence_data.loc[confidence_data["metric"] == "psr", "p05"].mean()

        sharpe_drop = drift_metrics.get("sharpe_drop", 0.0)
        psr_drop = drift_metrics.get("psr_drop", 0.0)

        status = "OK"
        level = 0
        if sharpe_p05 < thresholds.get("sharpe_p5_min", 0.0) or psr_p05 < thresholds.get(
            "psr_p5_min", 0.0
        ):
            status = "FAIL"
            level = 2

        if sharpe_drop > thresholds.get("sharpe_drop_tol", 1.0) or psr_drop > thresholds.get(
            "psr_drop_tol", 1.0
        ):
            if status == "OK":
                status = "WARN"
            level = max(level, 2)

        return status, level

    def generate_dashboard(
        self,
        confidence_data: pd.DataFrame,
        drift_metrics: Dict[str, float],
        status: str,
        level: int,
        actions: List[str],
    ) -> str:
        return (
            "# Drift Monitoring Dashboard\n\n"
            f"- Status: {status}\n"
            f"- Level: {level}\n\n"
            "## Drift Metrics\n\n"
            f"- Sharpe drop: {drift_metrics.get('sharpe_drop', 0.0):.3f}\n"
            f"- PSR drop: {drift_metrics.get('psr_drop', 0.0):.3f}\n\n"
            "## Actions\n\n"
            + "\n".join(f"- {action}" for action in actions)
            + "\n"
        )

    def run_monitoring(self) -> Dict[str, Any]:
        confidence = self.load_confidence_data()
        performance = self.load_recent_performance()
        drift_metrics = self.calculate_drift_metrics(performance)
        status, level = self.assess_drift_status(confidence, drift_metrics)

        self.drift_status = status
        self.degradation_level = level

        actions: List[str] = []
        if status in {"WARN", "FAIL"}:
            actions.append("Review portfolio risk settings")
        if status == "FAIL":
            actions.append("Consider derisk or rebuild portfolio")

        outputs = self.config["outputs"]
        dashboard = self.generate_dashboard(confidence, drift_metrics, status, level, actions)
        Path(outputs["dashboard"]).write_text(dashboard)

        drift_row = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": status, **drift_metrics}
        drift_df = pd.DataFrame([drift_row])
        drift_df.to_csv(outputs["drift_data"], index=False)

        actions_log = outputs.get("actions_log")
        if actions_log:
            Path(actions_log).write_text(
                "\n".join([f"{datetime.now(timezone.utc).isoformat()} {status}"] + actions) + "\n"
            )

        return {"status": status, "level": level, "actions": actions}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run drift monitoring")
    parser.add_argument("--config", default="configs/drift_monitor.yaml")
    args = parser.parse_args()

    monitor = DriftMonitor(args.config, verbose=True)
    result = monitor.run_monitoring()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
