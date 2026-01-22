"""Compute Bybit minimum order notional per symbol (v5, linear/inverse/spot)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import requests

from coint2.live.bybit_client import resolve_base_url


def _round_up_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step - 1e-12) * step


def _fetch_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}")
    return data


def _fetch_instruments(base_url: str, category: str) -> Dict[str, Dict[str, Any]]:
    url = f"{base_url}/v5/market/instruments-info"
    data = _fetch_json(url, params={"category": category, "limit": 1000})
    items = data.get("result", {}).get("list", [])
    return {item.get("symbol"): item for item in items if item.get("symbol")}


def _fetch_tickers(base_url: str, category: str) -> Dict[str, float]:
    url = f"{base_url}/v5/market/tickers"
    data = _fetch_json(url, params={"category": category})
    items = data.get("result", {}).get("list", [])
    prices: Dict[str, float] = {}
    for item in items:
        symbol = item.get("symbol")
        if not symbol:
            continue
        last = item.get("lastPrice")
        if last is None:
            continue
        try:
            prices[symbol] = float(last)
        except ValueError:
            continue
    return prices


def _load_symbols_from_pairs(pairs_file: Path) -> set[str]:
    import yaml

    payload = yaml.safe_load(pairs_file.read_text())
    symbols: set[str] = set()
    for row in payload.get("pairs", []):
        symbol1 = str(row.get("symbol1") or "").strip()
        symbol2 = str(row.get("symbol2") or "").strip()
        if symbol1:
            symbols.add(symbol1)
        if symbol2:
            symbols.add(symbol2)
    return symbols


def compute_min_orders(
    instruments: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    symbols: Iterable[str],
) -> tuple[list[Dict[str, Any]], list[str]]:
    rows: list[Dict[str, Any]] = []
    missing: list[str] = []

    for symbol in sorted(set(symbols)):
        info = instruments.get(symbol)
        if not info:
            missing.append(symbol)
            continue
        price = float(prices.get(symbol, 0.0) or 0.0)

        lot = info.get("lotSizeFilter", {}) or {}
        min_order_qty = float(lot.get("minOrderQty", "0") or 0.0)
        qty_step = float(lot.get("qtyStep", "0") or 0.0)
        min_notional_value = float(lot.get("minNotionalValue", "0") or 0.0)

        min_qty_step = _round_up_to_step(min_order_qty, qty_step)
        qty_for_min_notional = 0.0
        if price > 0 and min_notional_value > 0:
            qty_for_min_notional = _round_up_to_step(min_notional_value / price, qty_step)

        effective_qty = max(min_qty_step, qty_for_min_notional)
        effective_notional = effective_qty * price

        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "min_order_qty": min_order_qty,
                "qty_step": qty_step,
                "min_notional_value": min_notional_value,
                "effective_qty": effective_qty,
                "effective_notional": effective_notional,
            }
        )

    return rows, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute minimum Bybit order notional per symbol")
    parser.add_argument("--env", default="live", help="Bybit environment: demo/testnet/live")
    parser.add_argument("--base-url", default=None, help="Override Bybit base URL")
    parser.add_argument("--category", default="linear", help="Bybit v5 category (linear/inverse/spot)")
    parser.add_argument("--pairs-file", type=Path, help="Path to pairs_universe.yaml (optional)")
    parser.add_argument("--symbol", action="append", default=[], help="Symbol to include (repeatable)")
    parser.add_argument("--output-csv", type=Path, help="Write results to CSV (optional)")
    args = parser.parse_args()

    base_url = resolve_base_url(args.env, args.base_url)
    instruments = _fetch_instruments(base_url, args.category)
    prices = _fetch_tickers(base_url, args.category)

    symbols: set[str] = set(args.symbol or [])
    if args.pairs_file:
        symbols |= _load_symbols_from_pairs(args.pairs_file)
    if not symbols:
        symbols = set(instruments.keys())

    rows, missing = compute_min_orders(instruments, prices, symbols)
    rows_sorted = sorted(rows, key=lambda item: float(item.get("effective_notional", 0.0)), reverse=True)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_sorted[0].keys()) if rows_sorted else ["symbol"])
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow(row)

    if missing:
        print(f"Missing symbols on {args.env}/{args.category}: {len(missing)} (sample: {sorted(missing)[:20]})")
    if rows_sorted:
        max_row = rows_sorted[0]
        min_row = rows_sorted[-1]
        print(
            "Min order notional (effective): "
            f"min={min_row['effective_notional']:.4f} "
            f"max={max_row['effective_notional']:.4f}"
        )
        print("Top-10 by min notional:")
        for row in rows_sorted[:10]:
            print(f"  {row['symbol']}: {row['effective_notional']:.2f} (qty {row['effective_qty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

