#!/usr/bin/env python3
"""Smoke-test tradeability thresholds on a small list of pairs.

This prints per-leg market snapshot fields and the first failing reason, mirroring
the selection filter logic (market stage) to ensure missing metrics fail-closed.

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/market/smoke_tradeability_filters.py \
    --pairs CELOUSDT-ENJUSDT AVAUSDT-FITFIUSDT CHZUSDC-JSTUSDT ACSUSDT-HOOKUSDT \
    --pairs BTCUSDC-BTCUSDT ETHUSDC-ETHUSDT \
    --liquidity-usd-daily 300000 \
    --max-bid-ask-pct 0.002 \
    --max-avg-funding-pct 0.07 \
    --require-market-metrics \
    --require-same-quote \
    --min-days-live 180 \
    --max-funding-rate-abs 0.001 \
    --max-tick-size-pct 0.0005
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from coint2.pipeline import filters as pf


def _split_pair(raw: str) -> tuple[str, str]:
    s = str(raw or "").strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return a.strip(), b.strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return a.strip(), b.strip()
    raise ValueError(f"Unexpected pair format: {raw!r}")


@dataclass(frozen=True)
class Leg:
    symbol: str
    in_snapshot: bool
    quote_suffix: str
    quote_snapshot: str
    turnover24h_usd: float
    bid_ask_pct: float
    funding_rate: float
    tick_size_pct: float
    days_live: float


def _leg(symbol: str) -> Leg:
    pf._load_market_metrics_once()
    in_snapshot = symbol in pf._MARKET_METRICS
    vol, ba, fund = pf._get_market_metrics(symbol)
    tick, days, q = pf._get_market_extras(symbol)
    return Leg(
        symbol=symbol,
        in_snapshot=bool(in_snapshot),
        quote_suffix=pf._quote_ccy(symbol),
        quote_snapshot=str(q),
        turnover24h_usd=float(vol),
        bid_ask_pct=float(ba),
        funding_rate=float(fund),
        tick_size_pct=float(tick),
        days_live=float(days),
    )


def _first_fail_reason(
    a: Leg,
    b: Leg,
    *,
    liquidity_usd_daily: float,
    max_bid_ask_pct: float,
    max_avg_funding_pct: float,
    require_market_metrics: bool,
    require_same_quote: bool,
    min_volume_usd_24h: float,
    min_days_live: int,
    max_funding_rate_abs: float,
    max_tick_size_pct: float,
) -> str | None:
    if require_market_metrics and (not a.in_snapshot or not b.in_snapshot):
        return "missing_metrics"
    if require_same_quote and a.quote_suffix != b.quote_suffix:
        return "quote_mismatch"
    if min_volume_usd_24h > 0 and min(a.turnover24h_usd, b.turnover24h_usd) < float(min_volume_usd_24h):
        return "volume_24h"
    if min_days_live > 0 and min(a.days_live, b.days_live) < float(min_days_live):
        return "days_live"
    if max_tick_size_pct > 0 and max(a.tick_size_pct, b.tick_size_pct) > float(max_tick_size_pct):
        return "tick_size"
    if max_funding_rate_abs > 0 and max(abs(a.funding_rate), abs(b.funding_rate)) > float(max_funding_rate_abs):
        return "funding_abs"
    if (
        liquidity_usd_daily > 0
        and max_bid_ask_pct < 1.0
        and (
            min(a.turnover24h_usd, b.turnover24h_usd) < float(liquidity_usd_daily)
            or max(a.bid_ask_pct, b.bid_ask_pct) > float(max_bid_ask_pct)
            or max(abs(a.funding_rate), abs(b.funding_rate)) > float(max_avg_funding_pct)
        )
    ):
        return "liquidity_gate"
    return None


def _iter_pairs(raw_pairs: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in raw_pairs:
        if not str(raw or "").strip():
            continue
        out.append(_split_pair(raw))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", action="append", default=[], help="Pair like A-B or A/B (repeatable).")
    ap.add_argument("--liquidity-usd-daily", type=float, default=300_000.0)
    ap.add_argument("--max-bid-ask-pct", type=float, default=0.002)
    ap.add_argument("--max-avg-funding-pct", type=float, default=0.07)
    ap.add_argument("--require-market-metrics", action="store_true")
    ap.add_argument("--require-same-quote", action="store_true")
    ap.add_argument("--min-volume-usd-24h", type=float, default=0.0)
    ap.add_argument("--min-days-live", type=int, default=0)
    ap.add_argument("--max-funding-rate-abs", type=float, default=0.0)
    ap.add_argument("--max-tick-size-pct", type=float, default=0.0)
    args = ap.parse_args()

    pairs = _iter_pairs(args.pairs)
    if not pairs:
        raise SystemExit("No --pairs provided")

    print("tradeability_smoke:")
    print(f"- snapshot: {pf._resolve_market_metrics_path()}")
    print(f"- liquidity_usd_daily={args.liquidity_usd_daily:g} max_bid_ask_pct={args.max_bid_ask_pct:g} max_avg_funding_pct={args.max_avg_funding_pct:g}")
    print(f"- require_market_metrics={bool(args.require_market_metrics)} require_same_quote={bool(args.require_same_quote)}")
    print(f"- min_volume_usd_24h={args.min_volume_usd_24h:g} min_days_live={int(args.min_days_live)} max_funding_rate_abs={args.max_funding_rate_abs:g} max_tick_size_pct={args.max_tick_size_pct:g}")
    print()

    for s1, s2 in pairs:
        a = _leg(s1)
        b = _leg(s2)
        reason = _first_fail_reason(
            a,
            b,
            liquidity_usd_daily=float(args.liquidity_usd_daily),
            max_bid_ask_pct=float(args.max_bid_ask_pct),
            max_avg_funding_pct=float(args.max_avg_funding_pct),
            require_market_metrics=bool(args.require_market_metrics),
            require_same_quote=bool(args.require_same_quote),
            min_volume_usd_24h=float(args.min_volume_usd_24h),
            min_days_live=int(args.min_days_live),
            max_funding_rate_abs=float(args.max_funding_rate_abs),
            max_tick_size_pct=float(args.max_tick_size_pct),
        )
        verdict = "PASS" if reason is None else f"FAIL:{reason}"
        print(f"{s1}-{s2}: {verdict}")
        for leg in (a, b):
            print(
                "  "
                f"{leg.symbol}: in_snapshot={int(leg.in_snapshot)} "
                f"quote_suffix={leg.quote_suffix} quote_snapshot={leg.quote_snapshot} "
                f"turnover24h_usd={leg.turnover24h_usd:g} "
                f"bid_ask_pct={leg.bid_ask_pct:g} "
                f"funding_rate={leg.funding_rate:g} "
                f"tick_size_pct={leg.tick_size_pct:g} "
                f"days_live={leg.days_live:g}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

