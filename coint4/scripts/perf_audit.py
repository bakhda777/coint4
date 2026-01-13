#!/usr/bin/env python3
"""Lightweight performance audit for data loading and cache setup."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.core.fast_coint import fast_coint
from coint2.core.global_rolling_cache import get_global_rolling_manager, initialize_global_rolling_cache
from coint2.core.memory_optimization import setup_optimized_threading
from coint2.utils.config import load_config


def _parse_symbols(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight performance audit")
    parser.add_argument("--config", default="configs/main_2024.yaml", help="Config path")
    parser.add_argument("--data-root", help="Override data root directory")
    parser.add_argument("--lookback-days", type=int, help="Lookback days for data load")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", help="Comma-separated symbols for fast_coint timing")
    parser.add_argument("--skip-cache", action="store_true", help="Skip global cache init")
    parser.add_argument("--skip-coint", action="store_true", help="Skip fast_coint timing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data_dir = Path(args.data_root)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)

    threading_info = setup_optimized_threading(n_jobs=cfg.backtest.n_jobs, verbose=True)
    print(f"[perf] threading: {threading_info.get('optimization_mode')}")

    if not args.skip_cache:
        cfg_dict = cfg.model_dump()
        cache_ok = initialize_global_rolling_cache(cfg_dict)
        cache_info = get_global_rolling_manager().get_cache_info()
        print(f"[perf] global cache initialized: {cache_ok}")
        print(f"[perf] cache windows: {cache_info.get('available_windows')}")
        print(f"[perf] cache arrays: {cache_info.get('num_cached_arrays')}")
        print(f"[perf] cache memory: {cache_info.get('total_memory_mb'):.1f} MB")

    handler = DataHandler(cfg, root=args.data_root)
    lookback_days = args.lookback_days or cfg.pair_selection.lookback_days
    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.now()

    start = time.perf_counter()
    data = handler.load_all_data_for_period(lookback_days=lookback_days, end_date=end_date)
    elapsed = time.perf_counter() - start
    print(f"[perf] data load: {elapsed:.2f}s, shape={data.shape}")

    if not args.skip_coint:
        symbols = _parse_symbols(args.symbols) or list(data.columns[:2])
        if len(symbols) >= 2 and all(sym in data.columns for sym in symbols[:2]):
            x = data[symbols[0]].dropna().to_numpy()
            y = data[symbols[1]].dropna().to_numpy()
            if len(x) and len(y):
                start = time.perf_counter()
                tau, pval, _ = fast_coint(x, y, trend="n")
                elapsed = time.perf_counter() - start
                print(f"[perf] fast_coint: {elapsed:.4f}s, tau={tau:.4f}, pval={pval:.4f}")
            else:
                print("[perf] fast_coint skipped: empty series")
        else:
            print("[perf] fast_coint skipped: need >=2 symbols")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
