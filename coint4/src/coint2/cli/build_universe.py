#!/usr/bin/env python3
"""
Build universe of cointegrated pairs from available data.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from coint2.core.data_loader import DataHandler
from coint2.utils.config import load_config
from coint2.pipeline.pair_scanner import scan_universe, calculate_pair_score


CRITERIA_KEYS = {
    "coint_pvalue_max",
    "hl_min",
    "hl_max",
    "hurst_min",
    "hurst_max",
    "min_cross",
    "beta_drift_max",
}


def load_universe_config(config_path: str) -> dict:
    """Load universe configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_criteria(raw_cfg: dict) -> dict:
    if not raw_cfg:
        return {}
    if "criteria" in raw_cfg:
        return raw_cfg.get("criteria") or {}
    universe_cfg = raw_cfg.get("universe", {}) if isinstance(raw_cfg.get("universe"), dict) else {}
    if "criteria" in universe_cfg:
        return universe_cfg.get("criteria") or {}
    return {key: raw_cfg[key] for key in CRITERIA_KEYS if key in raw_cfg}


def _load_symbols_file(symbols_path: str) -> List[str]:
    path = Path(symbols_path)
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            symbols = data.get("symbols", [])
            return [str(sym).strip() for sym in symbols if str(sym).strip()]
        if isinstance(data, list):
            return [str(sym).strip() for sym in data if str(sym).strip()]
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_symbols(
    *,
    symbols: Optional[List[str]],
    symbols_file: Optional[str],
    raw_cfg: dict,
    handler: DataHandler,
) -> List[str]:
    if symbols:
        if len(symbols) == 1 and symbols[0].upper() == "ALL":
            return handler.get_all_symbols()
        return symbols
    if symbols_file:
        return _load_symbols_file(symbols_file)
    universe_cfg = raw_cfg.get("universe", {}) if isinstance(raw_cfg.get("universe"), dict) else {}
    cfg_symbols = raw_cfg.get("symbols") or universe_cfg.get("symbols")
    if cfg_symbols:
        return [str(sym).strip() for sym in cfg_symbols if str(sym).strip()]
    return handler.get_all_symbols()


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


def build_universe(
    config_path: str = "configs/criteria_relaxed.yaml",
    base_config_path: str = "configs/main_2024.yaml",
    data_root: str | None = None,
    symbols: Optional[List[str]] = None,
    symbols_file: str | None = None,
    train_days: int | None = None,
    valid_days: int | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Build universe of pairs based on configuration."""
    raw_cfg = load_universe_config(config_path)
    universe_cfg = raw_cfg.get("universe", {}) if isinstance(raw_cfg.get("universe"), dict) else {}
    criteria = _extract_criteria(raw_cfg)
    selection = raw_cfg.get("selection", {}) if isinstance(raw_cfg.get("selection"), dict) else {}

    resolved_train_days = train_days or universe_cfg.get("train_days") or raw_cfg.get("train_days") or 60
    resolved_valid_days = valid_days or universe_cfg.get("valid_days") or raw_cfg.get("valid_days") or 30
    resolved_output_dir = output_dir or raw_cfg.get("output_dir") or universe_cfg.get("output_dir") or "bench"

    if top_n is not None:
        selection = dict(selection)
        selection["top_n"] = top_n

    scan_cfg = {
        "train_days": resolved_train_days,
        "valid_days": resolved_valid_days,
        "criteria": criteria,
        "output_dir": resolved_output_dir,
    }

    app_cfg = load_config(base_config_path)
    handler = DataHandler(app_cfg, root=data_root)

    resolved_symbols = _resolve_symbols(
        symbols=symbols,
        symbols_file=symbols_file,
        raw_cfg=raw_cfg,
        handler=handler,
    )

    if not resolved_symbols:
        print("No symbols provided or found. Check data root and symbols config.")
        return pd.DataFrame()

    end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    start_date = end_dt - pd.Timedelta(days=resolved_train_days + resolved_valid_days)

    df_results = scan_universe(
        handler,
        resolved_symbols,
        start_date,
        end_dt,
        scan_cfg,
    )

    if df_results.empty:
        print("No pairs found in universe")
        return df_results

    df_results["score"] = df_results.apply(
        lambda row: calculate_pair_score(row.to_dict(), scan_cfg), axis=1
    )
    df_results = df_results.sort_values("score", ascending=False)

    passing = df_results[df_results["verdict"] == "PASS"]
    selected = apply_selection_rules(passing, selection)

    save_outputs(selected, df_results, scan_cfg, config_path)
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
        default="configs/criteria_relaxed.yaml",
        help="Path to criteria or universe config",
    )
    parser.add_argument(
        "--base-config",
        default="configs/main_2024.yaml",
        help="Path to base app config",
    )
    parser.add_argument(
        "--data-root",
        help="Override data root directory (defaults to config data_dir)",
    )
    parser.add_argument(
        "--symbols-file",
        help="Path to symbols list (yaml with 'symbols' or text file)",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols list (use ALL for auto-detect)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        help="Override training window (days)",
    )
    parser.add_argument(
        "--valid-days",
        type=int,
        help="Override validation window (days)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to now.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: bench)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Select top N pairs after scoring",
    )
    args = parser.parse_args()
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    build_universe(
        config_path=args.config,
        base_config_path=args.base_config,
        data_root=args.data_root,
        symbols=symbols,
        symbols_file=args.symbols_file,
        train_days=args.train_days,
        valid_days=args.valid_days,
        end_date=args.end_date,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
