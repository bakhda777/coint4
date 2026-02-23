#!/usr/bin/env python3
"""Fetch Bybit linear market metrics snapshot for the current symbol universe.

This script produces a *deterministic* CSV snapshot used by the pair-selection
tradeability filter (volume/bid-ask/funding/tick size/listing age).

No API keys are required (public endpoints).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import urllib.request

import yaml


BYBIT_BASE = "https://api.bybit.com"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_json(url: str, *, timeout_s: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "coint4-metrics-snapshot/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}") from exc


def _quote_ccy(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "BUSD", "FDUSD"):
        if symbol.endswith(suffix):
            return suffix
    return "OTHER"


@dataclass(frozen=True)
class _TickerRow:
    symbol: str
    last_price: Optional[float]
    turnover24h_usd: Optional[float]
    volume24h: Optional[float]
    bid1: Optional[float]
    ask1: Optional[float]
    funding_rate: Optional[float]

    @property
    def bid_ask_pct(self) -> Optional[float]:
        if self.bid1 is None or self.ask1 is None:
            return None
        if self.bid1 <= 0 or self.ask1 <= 0:
            return None
        mid = 0.5 * (self.bid1 + self.ask1)
        if mid <= 0:
            return None
        return float((self.ask1 - self.bid1) / mid)


@dataclass(frozen=True)
class _InstrumentRow:
    symbol: str
    tick_size: Optional[float]
    launch_time_ms: Optional[int]


def _to_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _extract_symbols_from_pairs_file(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise RuntimeError(f"unexpected pairs_file format (missing 'pairs'): {path}")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise RuntimeError(f"unexpected pairs_file format ('pairs' not a list): {path}")
    symbols: set[str] = set()
    for item in pairs:
        if not isinstance(item, dict):
            continue
        # Support both legacy and current schema keys.
        a = str(item.get("symbol1") or item.get("symbol_1") or "").strip()
        b = str(item.get("symbol2") or item.get("symbol_2") or "").strip()
        if a:
            symbols.add(a)
        if b:
            symbols.add(b)
    return sorted(symbols)


def _load_tickers() -> dict[str, _TickerRow]:
    url = f"{BYBIT_BASE}/v5/market/tickers?category=linear"
    payload = _fetch_json(url)
    result = payload.get("result") or {}
    rows = result.get("list") or []
    out: dict[str, _TickerRow] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        out[symbol] = _TickerRow(
            symbol=symbol,
            last_price=_to_float(row.get("lastPrice")),
            turnover24h_usd=_to_float(row.get("turnover24h")),
            volume24h=_to_float(row.get("volume24h")),
            bid1=_to_float(row.get("bid1Price")),
            ask1=_to_float(row.get("ask1Price")),
            funding_rate=_to_float(row.get("fundingRate")),
        )
    return out


def _load_instruments() -> dict[str, _InstrumentRow]:
    # Fetch full list once; 647 instruments as of 2026-02-23.
    url = f"{BYBIT_BASE}/v5/market/instruments-info?category=linear&limit=1000"
    payload = _fetch_json(url)
    result = payload.get("result") or {}
    rows = result.get("list") or []
    out: dict[str, _InstrumentRow] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        price_filter = row.get("priceFilter") or {}
        if not isinstance(price_filter, dict):
            price_filter = {}
        out[symbol] = _InstrumentRow(
            symbol=symbol,
            tick_size=_to_float(price_filter.get("tickSize")),
            launch_time_ms=_to_int(row.get("launchTime")),
        )
    return out


def _days_live(now_ms: int, launch_ms: Optional[int]) -> Optional[float]:
    if launch_ms is None:
        return None
    if launch_ms <= 0:
        return None
    delta_ms = now_ms - int(launch_ms)
    return float(delta_ms / 86_400_000.0)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.12g}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def _write_csv(
    *,
    out_path: Path,
    symbols: Iterable[str],
    tickers: dict[str, _TickerRow],
    instruments: dict[str, _InstrumentRow],
    retrieved_at_utc: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fieldnames = [
        "symbol",
        "quote",
        "retrieved_at_utc",
        "last_price",
        "turnover24h_usd",
        "volume24h",
        "bid1",
        "ask1",
        "bid_ask_pct",
        "funding_rate",
        "funding_rate_abs",
        "tick_size",
        "tick_size_pct",
        "launch_time_ms",
        "days_live",
    ]

    missing_ticker = 0
    missing_instrument = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for symbol in symbols:
            t = tickers.get(symbol)
            i = instruments.get(symbol)
            if t is None:
                missing_ticker += 1
            if i is None:
                missing_instrument += 1

            bid_ask_pct = t.bid_ask_pct if t else None
            funding = t.funding_rate if t else None
            funding_abs = abs(funding) if funding is not None else None
            tick_pct = None
            if t and i and t.last_price and i.tick_size and t.last_price > 0:
                tick_pct = float(i.tick_size / t.last_price)
            writer.writerow(
                {
                    "symbol": symbol,
                    "quote": _quote_ccy(symbol),
                    "retrieved_at_utc": retrieved_at_utc,
                    "last_price": _fmt(t.last_price if t else None),
                    "turnover24h_usd": _fmt(t.turnover24h_usd if t else None),
                    "volume24h": _fmt(t.volume24h if t else None),
                    "bid1": _fmt(t.bid1 if t else None),
                    "ask1": _fmt(t.ask1 if t else None),
                    "bid_ask_pct": _fmt(bid_ask_pct),
                    "funding_rate": _fmt(funding),
                    "funding_rate_abs": _fmt(funding_abs),
                    "tick_size": _fmt(i.tick_size if i else None),
                    "tick_size_pct": _fmt(tick_pct),
                    "launch_time_ms": _fmt_int(i.launch_time_ms if i else None),
                    "days_live": _fmt(_days_live(now_ms, i.launch_time_ms if i else None)),
                }
            )

    if missing_ticker or missing_instrument:
        print(
            f"WARNING: missing_ticker={missing_ticker}, missing_instrument={missing_instrument}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-file",
        default="configs/universe/pruned_v2_pairs_universe.yaml",
        help="Pairs universe YAML (relative to app root).",
    )
    parser.add_argument(
        "--output",
        default="configs/market/bybit_linear_metrics_latest.csv",
        help="Output CSV path (relative to app root).",
    )
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]
    pairs_path = (app_root / args.pairs_file).resolve()
    out_path = (app_root / args.output).resolve()

    symbols = _extract_symbols_from_pairs_file(pairs_path)
    if not symbols:
        raise SystemExit(f"no symbols found in {pairs_path}")

    retrieved_at_utc = _utc_now_iso()
    tickers = _load_tickers()
    instruments = _load_instruments()
    _write_csv(
        out_path=out_path,
        symbols=symbols,
        tickers=tickers,
        instruments=instruments,
        retrieved_at_utc=retrieved_at_utc,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
