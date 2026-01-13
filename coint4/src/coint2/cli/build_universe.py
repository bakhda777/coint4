#!/usr/bin/env python3
"""
Build universe of cointegrated pairs from available data.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from coint2.core.data_loader import DataHandler
from coint2.utils.config import load_config
from coint2.pipeline.pair_scanner import scan_universe, calculate_pair_score


def load_universe_config(config_path: str) -> dict:
    """Load universe configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_universe(config_path: str = "configs/universe.yaml") -> pd.DataFrame:
    """Build universe of pairs based on configuration."""
    universe_cfg = load_universe_config(config_path)
    app_cfg = load_config("configs/main_2024.yaml")

    handler = DataHandler(app_cfg)
    end_date = pd.Timestamp.now()
    train_days = universe_cfg["universe"]["train_days"]
    valid_days = universe_cfg["universe"]["valid_days"]
    start_date = end_date - pd.Timedelta(days=train_days + valid_days)

    df_results = scan_universe(
        handler,
        universe_cfg["universe"]["symbols"],
        start_date,
        end_date,
        universe_cfg,
    )

    if df_results.empty:
        print("No pairs found in universe")
        return df_results

    df_results["score"] = df_results.apply(
        lambda row: calculate_pair_score(row.to_dict(), universe_cfg), axis=1
    )
    df_results = df_results.sort_values("score", ascending=False)

    passing = df_results[df_results["verdict"] == "PASS"]
    selected = apply_selection_rules(passing, universe_cfg.get("selection", {}))

    save_outputs(selected, df_results, universe_cfg, config_path)
    return selected


def apply_selection_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply selection rules to filter pairs."""
    selected = df.copy()
    top_n = rules.get("top_n")
    if top_n is not None:
        selected = selected.head(top_n)
    return selected


def generate_report(
    selected_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: Dict,
    config_path: str,
) -> str:
    """Generate universe report markdown."""
    report = [
        "# Universe Selection Report",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
        f"*Config: {config_path}*",
        f"*Git: {get_git_hash()}*",
        "",
        "## Summary",
        "",
        f"- Total tested pairs: {len(full_df)}",
        f"- Passing pairs: {len(full_df[full_df['verdict'] == 'PASS'])}",
        f"- Selected pairs: {len(selected_df)}",
        "",
        "## Top Selected Pairs",
        "",
        "| Pair | Score | P-Value | Half-Life |",
        "|------|-------|---------|-----------|",
    ]

    for _, row in selected_df.iterrows():
        report.append(
            f"| {row['pair']} | {row['score']:.4f} | {row['pvalue']:.4f} | {row['half_life']:.1f} |"
        )

    return "\n".join(report) + "\n"


def save_outputs(
    selected_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: dict,
    config_path: str,
) -> None:
    """Save universe outputs."""
    output_dir = Path(cfg.get("output_dir", "bench"))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_pairs = []
    for _, row in selected_df.iterrows():
        selected_pairs.append(
            {
                "symbol1": row["symbol1"],
                "symbol2": row["symbol2"],
                "pvalue": row["pvalue"],
                "half_life": row["half_life"],
                "score": row["score"],
            }
        )

    universe_output = {
        "pairs": selected_pairs,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "config_path": config_path,
            "git_hash": get_git_hash(),
            "total_pairs": len(selected_pairs),
        },
    }

    pairs_path = output_dir / "pairs_universe.yaml"
    with open(pairs_path, "w") as f:
        yaml.safe_dump(universe_output, f)

    full_df.to_csv(output_dir / "universe_full.csv", index=False)

    report = generate_report(selected_df, full_df, cfg, config_path)
    report_path = output_dir / "UNIVERSE_REPORT.md"
    report_path.write_text(report)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build universe of cointegrated pairs")
    parser.add_argument(
        "--config",
        default="configs/universe.yaml",
        help="Path to universe config",
    )
    args = parser.parse_args()
    build_universe(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
