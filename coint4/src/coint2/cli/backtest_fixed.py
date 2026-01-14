#!/usr/bin/env python3
"""Fixed-parameter backtest runner for quick CLI checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from coint2.core import performance
import pandas as pd
import yaml

from coint2.core.data_loader import DataHandler
from coint2.utils.config import AppConfig


def _deep_merge(base: Any, delta: Any) -> Any:
    """Deep merge delta config into base config."""
    if delta is None:
        return base
    if not isinstance(base, dict) or not isinstance(delta, dict):
        return delta
    merged = dict(base)
    for key, value in delta.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_with_overrides(
    base_config_path: str,
    delta_path: str | None = None,
    data_root: str | None = None,
) -> tuple[AppConfig, Dict[str, Any]]:
    """Load config YAML and apply optional overrides."""
    raw_cfg = _load_yaml(base_config_path)
    if delta_path:
        delta_cfg = _load_yaml(delta_path)
        raw_cfg = _deep_merge(raw_cfg, delta_cfg)
    if data_root:
        raw_cfg["data_dir"] = data_root
    data_dir = raw_cfg.get("data_dir")
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    app_cfg = AppConfig(**raw_cfg)
    return app_cfg, raw_cfg


def _parse_pair_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, str):
        if "/" not in entry:
            return None
        sym1, sym2 = entry.split("/", 1)
        return {"symbol1": sym1, "symbol2": sym2, "beta": 1.0, "alpha": 0.0}

    if not isinstance(entry, dict):
        return None

    sym1 = entry.get("symbol1")
    sym2 = entry.get("symbol2")
    if not sym1 or not sym2:
        pair_str = entry.get("pair")
        if isinstance(pair_str, str) and "/" in pair_str:
            sym1, sym2 = pair_str.split("/", 1)

    if not sym1 or not sym2:
        return None

    metrics = entry.get("metrics") or {}
    beta = entry.get("beta", metrics.get("beta", 1.0))
    alpha = entry.get("alpha", metrics.get("alpha", 0.0))

    return {"symbol1": sym1, "symbol2": sym2, "beta": beta, "alpha": alpha}


def load_pairs(pairs_file: str) -> List[Dict[str, Any]]:
    """Load pairs from universe YAML file (supports multiple formats)."""
    data = _load_yaml(pairs_file)
    pairs_data = data.get("pairs", data) if isinstance(data, dict) else data
    if not isinstance(pairs_data, Iterable):
        return []

    pairs: List[Dict[str, Any]] = []
    for entry in pairs_data:
        parsed = _parse_pair_entry(entry)
        if parsed:
            pairs.append(parsed)
    return pairs


def run_backtest(
    prices_df: pd.DataFrame,
    pair_info: Dict[str, Any],
    backtest_cfg: Dict[str, Any],
    max_bars: int = 0,
) -> Optional[Dict[str, Any]]:
    """Run a lightweight backtest for a single pair."""
    sym1, sym2 = pair_info["symbol1"], pair_info["symbol2"]
    beta = float(pair_info.get("beta", 1.0))
    alpha = float(pair_info.get("alpha", 0.0))

    if sym1 not in prices_df.columns or sym2 not in prices_df.columns:
        return None

    y = prices_df[sym1].to_numpy()
    x = prices_df[sym2].to_numpy()

    if max_bars > 0 and len(y) > max_bars:
        y = y[-max_bars:]
        x = x[-max_bars:]

    spread = y - beta * x - alpha

    rolling_window = int(backtest_cfg.get("rolling_window", 30))
    rolling_window = max(1, rolling_window)
    z_scores = np.zeros(len(spread))

    for idx in range(rolling_window, len(spread)):
        window = spread[idx - rolling_window : idx]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            z_scores[idx] = (spread[idx] - mean) / std

    z_enter = float(backtest_cfg.get("zscore_threshold", 1.0))
    z_exit = float(backtest_cfg.get("zscore_exit", 0.0))
    z_stop = float(backtest_cfg.get("stop_loss_multiplier", 3.0))
    time_stop = int(float(backtest_cfg.get("time_stop_multiplier", 2.0)) * rolling_window)

    position = 0
    entry_bar = 0
    entry_price = 0.0
    trades = []
    pnl = []

    for idx in range(rolling_window, len(spread)):
        z_val = z_scores[idx]

        if position == 0:
            if abs(z_val) >= z_enter:
                position = -1 if z_val > 0 else 1
                entry_bar = idx
                entry_price = spread[idx]
        else:
            bars_held = idx - entry_bar
            should_exit = (
                abs(z_val) <= z_exit
                or abs(z_val) >= z_stop
                or bars_held >= time_stop
            )

            if should_exit:
                exit_price = spread[idx]
                trade_pnl = (exit_price - entry_price) * position

                commission_pct = float(backtest_cfg.get("commission_pct", 0.0004))
                slippage_pct = float(backtest_cfg.get("slippage_pct", 0.0005))
                cost_pct = 2 * (commission_pct + slippage_pct)
                trade_pnl *= (1 - cost_pct)

                trades.append(
                    {
                        "entry_bar": entry_bar,
                        "exit_bar": idx,
                        "bars_held": bars_held,
                        "entry_z": float(z_scores[entry_bar]),
                        "exit_z": float(z_val),
                        "pnl": float(trade_pnl),
                    }
                )
                pnl.append(float(trade_pnl))
                position = 0

    return {"trades": trades, "pnl": pnl, "pair": f"{sym1}/{sym2}"}


def calculate_metrics(
    all_trades: List[Dict[str, Any]],
    all_pnl: List[float],
    initial_capital: float,
    annualizing_factor: int,
) -> Dict[str, Any]:
    """Calculate performance metrics."""
    if not all_pnl:
        return {"error": "No trades executed"}

    pnl_array = np.array(all_pnl, dtype=float)
    capital = initial_capital if initial_capital > 0 else 1.0
    returns = pnl_array / capital
    cumulative_pnl = pd.Series(pnl_array).cumsum()

    return {
        "total_pnl": float(np.sum(pnl_array)),
        "num_trades": len(all_trades),
        "win_rate": float(np.mean([1 if t["pnl"] > 0 else 0 for t in all_trades])) if all_trades else 0,
        "sharpe_ratio": float(performance.sharpe_ratio(pd.Series(returns), annualizing_factor)),
        "max_drawdown": float(performance.max_drawdown(cumulative_pnl)) if pnl_array.size else 0,
        "avg_bars_held": float(np.mean([t["bars_held"] for t in all_trades])) if all_trades else 0,
    }


def run_fixed_backtest(
    *,
    base_config: str,
    pairs_file: str,
    out_dir: str,
    period_start: str | None = None,
    period_end: str | None = None,
    config_delta: str | None = None,
    data_root: str | None = None,
    max_bars: int = 0,
    lookback_days: int | None = None,
) -> Dict[str, Any]:
    """Run fixed-parameter backtests for a universe."""
    app_cfg, raw_cfg = load_config_with_overrides(base_config, config_delta, data_root)

    data_handler = DataHandler(app_cfg, root=data_root)

    end_dt = pd.Timestamp(period_end) if period_end else pd.Timestamp.now()
    start_dt = pd.Timestamp(period_start) if period_start else None

    if lookback_days is None:
        if start_dt is not None:
            lookback_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
        else:
            lookback_days = int(raw_cfg.get("pair_selection", {}).get("lookback_days", 60))

    prices_df = data_handler.load_all_data_for_period(
        lookback_days=lookback_days,
        end_date=end_dt,
    )

    if start_dt is not None:
        prices_df = prices_df.loc[start_dt:end_dt]

    pairs = load_pairs(pairs_file)

    all_trades: List[Dict[str, Any]] = []
    all_pnl: List[float] = []
    cumulative_pnl: List[float] = []
    running_sum = 0.0

    backtest_cfg = raw_cfg.get("backtest", {})

    for idx, pair_info in enumerate(pairs, 1):
        result = run_backtest(prices_df, pair_info, backtest_cfg, max_bars=max_bars)
        if not result:
            continue

        for trade in result["trades"]:
            trade["pair"] = result["pair"]
        all_trades.extend(result["trades"])

        for trade_pnl in result["pnl"]:
            running_sum += trade_pnl
            cumulative_pnl.append(running_sum)
        all_pnl.extend(result["pnl"])

    initial_capital = float(raw_cfg.get("portfolio", {}).get("initial_capital", 1.0))
    annualizing_factor = int(raw_cfg.get("backtest", {}).get("annualizing_factor", 252))
    metrics = calculate_metrics(all_trades, all_pnl, initial_capital, annualizing_factor)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(out_path / "trades.csv", index=False)

    if cumulative_pnl:
        equity_df = pd.DataFrame(
            {"bar": range(len(cumulative_pnl)), "cumulative_pnl": cumulative_pnl}
        )
        equity_df.to_csv(out_path / "equity.csv", index=False)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed parameter OOS backtest runner")
    parser.add_argument("--pairs-file", required=True, help="Pairs universe YAML file")
    parser.add_argument("--config", required=True, help="Base config path")
    parser.add_argument("--config-delta", help="YAML overlay to apply on top of base config")
    parser.add_argument("--data-root", help="Override data root")
    parser.add_argument("--period-start", help="Period start (YYYY-MM-DD)")
    parser.add_argument("--period-end", help="Period end (YYYY-MM-DD)")
    parser.add_argument("--lookback-days", type=int, help="Override lookback days")
    parser.add_argument("--out-dir", default="outputs/fixed_run", help="Output directory")
    parser.add_argument("--max-bars", type=int, default=0, help="Max bars to use (0=all)")
    args = parser.parse_args()

    run_fixed_backtest(
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


if __name__ == "__main__":
    raise SystemExit(main())
