"""CLI runner for Bybit demo paper trading."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from coint2.live.bybit_client import BybitRestClient, BybitSettings, resolve_base_url
from coint2.live.paper_engine import PaperTradingEngine, load_pairs
from coint2.utils.config import load_config
from coint2.utils.logging_config import TradingLogger


def build_engine(args: argparse.Namespace) -> PaperTradingEngine:
    cfg = load_config(args.config)

    pairs_path = Path(args.pairs_file) if args.pairs_file else None
    if pairs_path is None:
        wf_pairs = getattr(cfg.walk_forward, "pairs_file", None)
        if wf_pairs:
            pairs_path = Path(str(wf_pairs))
    if pairs_path is None:
        raise RuntimeError("pairs_file is required (arg or walk_forward.pairs_file)")

    metrics_path = Path(args.pairs_metrics) if args.pairs_metrics else None
    if metrics_path is None:
        candidate = pairs_path.parent / "universe_metrics.csv"
        metrics_path = candidate if candidate.exists() else None

    max_pairs = args.max_pairs or cfg.pair_selection.max_pairs
    pairs = load_pairs(pairs_path, metrics_path, max_pairs)

    env = args.env or os.getenv("BYBIT_ENV", "demo")
    base_url = resolve_base_url(env, os.getenv("BYBIT_BASE_URL"))
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET are required")

    settings = BybitSettings(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        recv_window=int(os.getenv("BYBIT_RECV_WINDOW", "5000")),
        category=os.getenv("BYBIT_CATEGORY", "linear"),
        account_type=os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED"),
        settle_coin=os.getenv("BYBIT_SETTLE_COIN", "USDT"),
    )
    max_retries = int(os.getenv("BYBIT_MAX_RETRIES", "3"))
    backoff_seconds = float(os.getenv("BYBIT_RETRY_BACKOFF", "1.0"))
    client = BybitRestClient(settings, max_retries=max_retries, backoff_seconds=backoff_seconds)

    logger = TradingLogger(
        log_dir="artifacts/live/logs",
        level=cfg.logging.debug_level,
        trade_details=cfg.logging.trade_details,
    )

    position_mode = os.getenv("BYBIT_POSITION_MODE", "hedge")
    if args.position_mode:
        position_mode = args.position_mode

    kline_cache_seconds = int(os.getenv("BYBIT_KLINE_CACHE_SECONDS", "60"))
    state_path = Path(os.getenv("BYBIT_STATE_PATH", "artifacts/live/state.json"))
    sync_on_start = os.getenv("BYBIT_SYNC_ON_START", "true").strip().lower() not in {"0", "false", "no"}
    min_notional_buffer_pct = float(os.getenv("BYBIT_MIN_NOTIONAL_BUFFER_PCT", "0.05"))
    if min_notional_buffer_pct < 0:
        min_notional_buffer_pct = 0.0
    return PaperTradingEngine(
        cfg,
        pairs,
        client,
        logger,
        position_mode=position_mode,
        kline_cache_seconds=kline_cache_seconds,
        state_path=state_path,
        sync_on_start=sync_on_start,
        min_notional_buffer_pct=min_notional_buffer_pct,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bybit demo paper trader")
    parser.add_argument("--config", default="configs/prod_candidate_relaxed8_nokpss_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml")
    parser.add_argument("--pairs-file", help="Path to pairs_universe.yaml")
    parser.add_argument("--pairs-metrics", help="Path to universe_metrics.csv")
    parser.add_argument("--max-pairs", type=int, help="Override max pairs")
    parser.add_argument("--env", default="demo", help="Bybit environment: demo/testnet/live")
    parser.add_argument("--position-mode", default="hedge", help="Position mode: hedge/oneway")
    parser.add_argument("--once", action="store_true", help="Run a single iteration")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval")
    args = parser.parse_args()

    load_dotenv()
    engine = build_engine(args)

    if args.once:
        engine.run_once()
        return 0

    while True:
        engine.run_once()
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
