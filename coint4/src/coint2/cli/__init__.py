"""Command-line interface entry points for coint2."""

from __future__ import annotations

import argparse
from pathlib import Path

from coint2.pipeline import run_walk_forward
from coint2.utils.config import load_config
from coint2.utils.logging_config import setup_logging_from_config

from . import backtest_fixed
from . import build_universe
from . import check_coint_health

__all__ = [
    "check_coint_health",
    "build_universe",
    "backtest_fixed",
]


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [sym.strip() for sym in value.split(",") if sym.strip()]


def _handle_scan(args: argparse.Namespace) -> int:
    _setup_logging(args.base_config)
    build_universe.build_universe(
        config_path=args.config,
        base_config_path=args.base_config,
        data_root=args.data_root,
        symbols=_parse_symbols(args.symbols),
        symbols_file=args.symbols_file,
        train_days=args.train_days,
        valid_days=args.valid_days,
        end_date=args.end_date,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    return 0


def _handle_backtest(args: argparse.Namespace) -> int:
    _setup_logging(args.config)
    backtest_fixed.run_fixed_backtest(
        base_config=args.config,
        pairs_file=args.pairs_file,
        out_dir=args.out_dir,
        period_start=args.period_start,
        period_end=args.period_end,
        config_delta=args.config_delta,
        data_root=args.data_root,
        max_bars=args.max_bars,
        lookback_days=args.lookback_days,
    )
    return 0


def _handle_walk_forward(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    setup_logging_from_config(cfg)
    if args.data_root:
        cfg.data_dir = Path(args.data_root)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
    run_walk_forward(cfg, use_memory_map=not args.no_memory_map)
    return 0


def _setup_logging(config_path: str | None) -> None:
    if not config_path:
        return
    try:
        cfg = load_config(config_path)
    except Exception:
        return
    setup_logging_from_config(cfg)


def main() -> int:
    """CLI dispatcher for coint2."""
    parser = argparse.ArgumentParser(description="coint2 command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan universe of cointegrated pairs",
    )
    scan_parser.add_argument(
        "--config",
        default="configs/criteria_relaxed.yaml",
        help="Path to criteria or universe config",
    )
    scan_parser.add_argument(
        "--base-config",
        default="configs/main_2024.yaml",
        help="Path to base app config",
    )
    scan_parser.add_argument(
        "--data-root",
        help="Override data root directory (defaults to config data_dir)",
    )
    scan_parser.add_argument(
        "--symbols-file",
        help="Path to symbols list (yaml with 'symbols' or text file)",
    )
    scan_parser.add_argument(
        "--symbols",
        help="Comma-separated symbols list (use ALL for auto-detect)",
    )
    scan_parser.add_argument(
        "--train-days",
        type=int,
        help="Override training window (days)",
    )
    scan_parser.add_argument(
        "--valid-days",
        type=int,
        help="Override validation window (days)",
    )
    scan_parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to now.",
    )
    scan_parser.add_argument(
        "--output-dir",
        help="Output directory (default: bench)",
    )
    scan_parser.add_argument(
        "--top-n",
        type=int,
        help="Select top N pairs after scoring",
    )
    scan_parser.set_defaults(func=_handle_scan)

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run fixed-parameter backtests for a universe",
    )
    backtest_parser.add_argument("--pairs-file", required=True, help="Pairs universe YAML file")
    backtest_parser.add_argument("--config", required=True, help="Base config path")
    backtest_parser.add_argument("--config-delta", help="YAML overlay for config overrides")
    backtest_parser.add_argument("--data-root", help="Override data root directory")
    backtest_parser.add_argument("--period-start", help="Period start (YYYY-MM-DD)")
    backtest_parser.add_argument("--period-end", help="Period end (YYYY-MM-DD)")
    backtest_parser.add_argument("--lookback-days", type=int, help="Override lookback days")
    backtest_parser.add_argument(
        "--out-dir",
        default="outputs/fixed_run",
        help="Output directory",
    )
    backtest_parser.add_argument(
        "--max-bars",
        type=int,
        default=0,
        help="Max bars to use (0=all)",
    )
    backtest_parser.set_defaults(func=_handle_backtest)

    wf_parser = subparsers.add_parser(
        "walk-forward",
        help="Run walk-forward pipeline",
    )
    wf_parser.add_argument(
        "--config",
        default="configs/main_2024.yaml",
        help="Path to base app config",
    )
    wf_parser.add_argument(
        "--data-root",
        help="Override data root directory (defaults to config data_dir)",
    )
    wf_parser.add_argument(
        "--no-memory-map",
        action="store_true",
        help="Disable memory-mapped data loading",
    )
    wf_parser.set_defaults(func=_handle_walk_forward)

    args = parser.parse_args()
    return args.func(args)
