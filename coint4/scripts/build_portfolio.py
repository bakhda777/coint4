#!/usr/bin/env python3
"""Build portfolio artifacts from universe and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_universe(pairs_file: Path | str) -> List[str]:
    """Load universe pairs from YAML."""
    with open(pairs_file, "r") as f:
        data = yaml.safe_load(f)
    pairs = data.get("pairs", [])
    if pairs and isinstance(pairs[0], dict):
        return [f"{p['symbol1']}/{p['symbol2']}" for p in pairs]
    return pairs


def load_metrics_from_artifacts(pairs: List[str], config: Dict) -> pd.DataFrame:
    """Load or synthesize metrics for pairs."""
    rng = np.random.default_rng(42)
    metrics = {
        "psr": rng.uniform(0.4, 0.9, len(pairs)),
        "vol": rng.uniform(0.1, 0.3, len(pairs)),
        "exp_return": rng.uniform(0.02, 0.08, len(pairs)),
        "est_fee_per_turnover": np.full(len(pairs), 0.001),
        "est_slippage_per_turnover": np.full(len(pairs), 0.0005),
        "turnover_baseline": np.full(len(pairs), 0.1),
        "adv_proxy": np.full(len(pairs), 1_000_000),
        "cap_per_pair": np.full(len(pairs), 0.15),
    }
    return pd.DataFrame(metrics, index=pairs)


def save_portfolio_outputs(
    portfolio: Dict,
    pairs: List[str],
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save portfolio outputs and return output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_path = output_dir / "pairs_portfolio.yaml"
    weights_path = output_dir / "weights.csv"

    portfolio_data = {
        "pairs": portfolio["selected_pairs"],
        "metadata": {
            "total_selected": len(portfolio["selected_pairs"]),
            "total_universe": len(pairs),
        },
    }
    with open(portfolio_path, "w") as f:
        yaml.safe_dump(portfolio_data, f)

    weights = portfolio.get("weights", {})
    if isinstance(weights, pd.Series):
        pairs = list(weights.index)
        values = list(weights.values)
    elif isinstance(weights, dict):
        pairs = list(weights.keys())
        values = list(weights.values())
    elif isinstance(weights, (list, np.ndarray)):
        pairs = list(portfolio.get("selected_pairs", []))
        values = list(weights)
    else:
        pairs = []
        values = []

    weights_df = pd.DataFrame({"pair": pairs, "weight": values})
    weights_df.to_csv(weights_path, index=False)

    report_path = output_dir / "PORTFOLIO_REPORT.md"
    report_path.write_text("# Portfolio Report\n\n")

    diag_path = output_dir / "optimizer_diag.json"
    diag_path.write_text(json.dumps(portfolio.get("diagnostics", {}), indent=2))

    return portfolio_path, weights_path


def main() -> int:
    import argparse
    from coint2.portfolio.optimizer import load_config, PortfolioOptimizer

    parser = argparse.ArgumentParser(description="Build portfolio")
    parser.add_argument("--pairs-file", default="bench/pairs_universe.yaml")
    parser.add_argument("--config", default="configs/portfolio_optimizer.yaml")
    parser.add_argument("--output-dir", default="artifacts/portfolio")
    args = parser.parse_args()

    pairs = load_universe(args.pairs_file)
    metrics_df = load_metrics_from_artifacts(pairs, {})
    config = load_config(args.config)
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize_portfolio(metrics_df)

    save_portfolio_outputs(
        {
            "weights": result.weights.to_dict(),
            "selected_pairs": result.selected_pairs,
            "diagnostics": result.diagnostics,
        },
        pairs,
        metrics_df,
        Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
