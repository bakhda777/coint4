"""Stateful paper trading engine for Bybit demo."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from coint2.live.bybit_client import BybitRestClient, clamp_to_step, map_timeframe_to_interval
from coint2.utils.config import AppConfig
from coint2.utils.logging_config import TradingLogger


@dataclass
class PairSpec:
    symbol1: str
    symbol2: str
    score: float
    beta: Optional[float] = None

    @property
    def key(self) -> str:
        return f"{self.symbol1}/{self.symbol2}"


@dataclass
class InstrumentSpec:
    min_qty: float
    qty_step: float


@dataclass
class PairState:
    position: int = 0
    entry_time: Optional[datetime] = None
    entry_price_y: float = 0.0
    entry_price_x: float = 0.0
    qty_y: float = 0.0
    qty_x: float = 0.0
    beta: float = 1.0
    last_exit_time: Optional[datetime] = None
    last_flat_spread: Optional[float] = None
    realized_pnl: float = 0.0
    last_bar_ts: Optional[int] = None


def load_pairs(pairs_path: Path, metrics_path: Optional[Path], max_pairs: Optional[int]) -> List[PairSpec]:
    raw = pairs_path.read_text()
    data = None
    try:
        import yaml
        data = yaml.safe_load(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse pairs file: {pairs_path}") from exc

    pairs = []
    for row in data.get("pairs", []):
        pair_raw = row.get("pair", "")
        pair_parts = pair_raw.split("/") if pair_raw else []
        symbol1 = row.get("symbol1") or (pair_parts[0] if len(pair_parts) > 0 else "")
        symbol2 = row.get("symbol2") or (pair_parts[1] if len(pair_parts) > 1 else "")
        score = float(row.get("score", 0.0))
        if not symbol1 or not symbol2:
            continue
        pairs.append(PairSpec(symbol1=symbol1, symbol2=symbol2, score=score))

    beta_map = {}
    if metrics_path and metrics_path.exists():
        with metrics_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = f"{row.get('symbol1')}/{row.get('symbol2')}"
                try:
                    beta_map[key] = float(row.get("beta", "nan"))
                except ValueError:
                    continue

    for pair in pairs:
        beta = beta_map.get(pair.key)
        if beta and math.isfinite(beta):
            pair.beta = beta

    pairs.sort(key=lambda item: item.score, reverse=True)
    if max_pairs:
        pairs = pairs[: max_pairs]
    return pairs


def align_series(
    series_y: List[Dict[str, float]],
    series_x: List[Dict[str, float]],
    bar_ms: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    y_map = {row["start_time_ms"]: row["close"] for row in series_y}
    x_map = {row["start_time_ms"]: row["close"] for row in series_x}
    common = sorted(set(y_map.keys()) & set(x_map.keys()))
    if not common:
        return np.array([]), np.array([]), 0

    now_ms = int(time.time() * 1000)
    last_ts = common[-1]
    if last_ts + bar_ms > now_ms:
        common = common[:-1]

    y_vals = np.array([y_map[ts] for ts in common], dtype=np.float64)
    x_vals = np.array([x_map[ts] for ts in common], dtype=np.float64)
    return y_vals, x_vals, common[-1] if common else 0


class PaperTradingEngine:
    """Stateful paper engine that drives Bybit demo orders."""

    def __init__(
        self,
        cfg: AppConfig,
        pairs: List[PairSpec],
        client: BybitRestClient,
        logger: TradingLogger,
        position_mode: str = "hedge",
        kline_cache_seconds: int = 60,
        state_path: Optional[Path] = None,
        sync_on_start: bool = True,
    ):
        self.cfg = cfg
        self.pairs = pairs
        self.client = client
        self.logger = logger
        self.position_mode = position_mode
        self.states: Dict[str, PairState] = {pair.key: PairState(beta=pair.beta or 1.0) for pair in pairs}
        self.instrument_specs: Dict[str, InstrumentSpec] = {}
        self.kline_cache: Dict[Tuple[str, str, int], Tuple[int, List[Dict[str, float]]]] = {}
        self.kline_cache_ttl_ms = max(0, kline_cache_seconds) * 1000
        self.blocked_symbols: set[str] = set()
        self.state_path = state_path or Path("artifacts/live/state.json")
        self.sync_on_start = sync_on_start
        self._synced = False
        self.daily_pnl: float = 0.0
        self.daily_pnl_day: Optional[datetime] = None
        self._load_state()

    def _position_idx(self, side: str) -> int:
        if self.position_mode != "hedge":
            return 0
        return 1 if side.lower() == "buy" else 2

    def _load_instruments(self) -> None:
        symbols = sorted({p.symbol1 for p in self.pairs} | {p.symbol2 for p in self.pairs})
        info_map, missing = self.client.batch_get_instruments(symbols, allow_missing=True)
        if not info_map:
            raise RuntimeError(
                "No valid instruments found on Bybit. Check BYBIT_ENV/BYBIT_CATEGORY or pairs file."
            )
        if missing:
            missing_set = set(missing)
            original_pairs = len(self.pairs)
            self.pairs = [
                pair for pair in self.pairs if pair.symbol1 not in missing_set and pair.symbol2 not in missing_set
            ]
            self.states = {
                pair.key: self.states.get(pair.key, PairState(beta=pair.beta or 1.0)) for pair in self.pairs
            }
            sample_missing = sorted(missing_set)[:10]
            self.logger.log_system(
                "Filtered pairs with missing instruments",
                level="warning",
                missing_count=len(missing_set),
                missing_sample=sample_missing,
                removed_pairs=original_pairs - len(self.pairs),
                remaining_pairs=len(self.pairs),
            )
            if not self.pairs:
                raise RuntimeError(
                    "No valid pairs left after filtering missing instruments. Update pairs file or BYBIT_CATEGORY."
                )
        for symbol, info in info_map.items():
            lot = info.get("lotSizeFilter", {})
            min_qty = float(lot.get("minOrderQty", "0") or 0.0)
            qty_step = float(lot.get("qtyStep", "0") or 0.0)
            self.instrument_specs[symbol] = InstrumentSpec(min_qty=min_qty, qty_step=qty_step)

    def _reset_daily_pnl_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        day = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
        if self.daily_pnl_day is None or day > self.daily_pnl_day:
            self.daily_pnl_day = day
            self.daily_pnl = 0.0

    def _get_klines_cached(self, symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
        if self.kline_cache_ttl_ms <= 0:
            return self.client.get_klines(symbol, interval, limit=limit)
        key = (symbol, interval, limit)
        now_ms = int(time.time() * 1000)
        cached = self.kline_cache.get(key)
        if cached and now_ms - cached[0] <= self.kline_cache_ttl_ms:
            return cached[1]
        data = self.client.get_klines(symbol, interval, limit=limit)
        self.kline_cache[key] = (now_ms, data)
        return data

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            self.logger.log_system("Failed to parse state file", level="warning", path=str(self.state_path))
            return

        self.daily_pnl = float(payload.get("daily_pnl", 0.0))
        day_str = payload.get("daily_pnl_day")
        if day_str:
            try:
                self.daily_pnl_day = datetime.fromisoformat(day_str)
            except ValueError:
                self.daily_pnl_day = None

        pair_states = payload.get("pairs", {})
        for key, data in pair_states.items():
            state = self.states.get(key)
            if state is None:
                continue
            state.position = int(data.get("position", 0))
            state.entry_price_y = float(data.get("entry_price_y", 0.0))
            state.entry_price_x = float(data.get("entry_price_x", 0.0))
            state.qty_y = float(data.get("qty_y", 0.0))
            state.qty_x = float(data.get("qty_x", 0.0))
            state.beta = float(data.get("beta", state.beta))
            state.realized_pnl = float(data.get("realized_pnl", 0.0))
            state.last_bar_ts = data.get("last_bar_ts")
            entry_time = data.get("entry_time")
            if entry_time:
                try:
                    state.entry_time = datetime.fromisoformat(entry_time)
                except ValueError:
                    state.entry_time = None
            last_exit = data.get("last_exit_time")
            if last_exit:
                try:
                    state.last_exit_time = datetime.fromisoformat(last_exit)
                except ValueError:
                    state.last_exit_time = None
            last_flat = data.get("last_flat_spread")
            state.last_flat_spread = float(last_flat) if last_flat is not None else None

    def _save_state(self) -> None:
        snapshot = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_day": self.daily_pnl_day.isoformat() if self.daily_pnl_day else None,
            "pairs": {},
        }
        for key, state in self.states.items():
            snapshot["pairs"][key] = {
                "position": state.position,
                "entry_time": state.entry_time.isoformat() if state.entry_time else None,
                "entry_price_y": state.entry_price_y,
                "entry_price_x": state.entry_price_x,
                "qty_y": state.qty_y,
                "qty_x": state.qty_x,
                "beta": state.beta,
                "last_exit_time": state.last_exit_time.isoformat() if state.last_exit_time else None,
                "last_flat_spread": state.last_flat_spread,
                "realized_pnl": state.realized_pnl,
                "last_bar_ts": state.last_bar_ts,
            }

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True))

    def _sync_positions(self) -> None:
        self.blocked_symbols.clear()
        positions = self.client.get_positions()
        symbol_positions: Dict[str, Dict[str, float | str]] = {}
        for pos in positions:
            try:
                size = float(pos.get("size", 0) or 0.0)
            except ValueError:
                continue
            if size == 0:
                continue
            symbol = pos.get("symbol")
            if not symbol:
                continue
            symbol_positions[symbol] = {
                "side": str(pos.get("side", "")).lower(),
                "size": size,
                "avg_price": float(pos.get("avgPrice") or pos.get("entryPrice") or 0.0),
            }

        if not symbol_positions:
            return

        now = datetime.now(timezone.utc)
        for pair in self.pairs:
            state = self.states[pair.key]
            pos_y = symbol_positions.get(pair.symbol1)
            pos_x = symbol_positions.get(pair.symbol2)
            if pos_y and pos_x:
                side_y = pos_y["side"]
                side_x = pos_x["side"]
                if side_y == "buy" and side_x == "sell":
                    state.position = 1
                elif side_y == "sell" and side_x == "buy":
                    state.position = -1
                else:
                    self.blocked_symbols.update({pair.symbol1, pair.symbol2})
                    continue
                state.entry_time = now
                state.entry_price_y = float(pos_y["avg_price"])
                state.entry_price_x = float(pos_x["avg_price"])
                state.qty_y = float(pos_y["size"])
                state.qty_x = float(pos_x["size"])
            else:
                if pos_y:
                    self.blocked_symbols.add(pair.symbol1)
                if pos_x:
                    self.blocked_symbols.add(pair.symbol2)
        if self.blocked_symbols:
            self.logger.log_system(
                "Blocking symbols with existing positions",
                level="warning",
                symbols=sorted(self.blocked_symbols),
            )

    def _compute_zscore(
        self,
        y_vals: np.ndarray,
        x_vals: np.ndarray,
        fallback_beta: float,
        rolling_window: int,
        min_sigma: float,
        min_beta: float,
        max_beta: float,
    ) -> Tuple[float, float, float, float]:
        if len(y_vals) < rolling_window or len(x_vals) < rolling_window:
            return 0.0, 0.0, 0.0, fallback_beta

        y_window = y_vals[-rolling_window:]
        x_window = x_vals[-rolling_window:]
        x_var = np.var(x_window)
        if x_var < 1e-12:
            beta = fallback_beta
        else:
            beta = np.cov(y_window, x_window)[0, 1] / x_var
        if not math.isfinite(beta) or abs(beta) < min_beta or abs(beta) > max_beta:
            beta = fallback_beta

        spread = y_window - beta * x_window
        mu = float(np.mean(spread))
        sigma = float(np.std(spread))
        if sigma < min_sigma:
            return 0.0, mu, sigma, beta
        current_spread = float(y_vals[-1] - beta * x_vals[-1])
        zscore = (current_spread - mu) / sigma
        return zscore, mu, sigma, beta

    def _compute_quantities(
        self,
        symbol_y: str,
        symbol_x: str,
        price_y: float,
        price_x: float,
        beta: float,
    ) -> Tuple[float, float]:
        portfolio = self.cfg.portfolio
        base_notional = portfolio.initial_capital * portfolio.risk_per_position_pct
        min_notional = portfolio.min_notional_per_trade
        max_notional = portfolio.max_notional_per_trade
        if base_notional < min_notional:
            base_notional = min_notional
        if base_notional > max_notional:
            base_notional = max_notional

        beta_abs = abs(beta) if abs(beta) > 1e-6 else 1.0
        notional_y = base_notional / (1.0 + beta_abs)
        notional_x = notional_y * beta_abs

        qty_y = notional_y / price_y
        qty_x = notional_x / price_x

        spec_y = self.instrument_specs.get(symbol_y)
        spec_x = self.instrument_specs.get(symbol_x)
        if spec_y:
            qty_y = clamp_to_step(qty_y, spec_y.qty_step)
            if qty_y < spec_y.min_qty:
                qty_y = 0.0
        if spec_x:
            qty_x = clamp_to_step(qty_x, spec_x.qty_step)
            if qty_x < spec_x.min_qty:
                qty_x = 0.0

        return qty_y, qty_x

    def _place_pair_orders(
        self,
        pair: PairSpec,
        side_y: str,
        side_x: str,
        qty_y: float,
        qty_x: float,
        reduce_only: bool,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        order_id_y = f"{pair.symbol1}-{pair.symbol2}-{ts}-y"
        order_id_x = f"{pair.symbol1}-{pair.symbol2}-{ts}-x"
        self.client.create_order(
            symbol=pair.symbol1,
            side=side_y,
            qty=qty_y,
            position_idx=self._position_idx(side_y),
            reduce_only=reduce_only,
            order_link_id=order_id_y,
        )
        self.client.create_order(
            symbol=pair.symbol2,
            side=side_x,
            qty=qty_x,
            position_idx=self._position_idx(side_x),
            reduce_only=reduce_only,
            order_link_id=order_id_x,
        )

    def run_once(self) -> None:
        if not self.instrument_specs:
            self._load_instruments()
        if self.sync_on_start and not self._synced:
            self._sync_positions()
            self._synced = True

        interval = map_timeframe_to_interval(self.cfg.backtest.timeframe)
        bar_ms = int(self.cfg.time.gap_minutes * 60 * 1000)
        rolling_window = int(self.cfg.backtest.rolling_window)
        min_sigma = float(self.cfg.backtest.min_volatility)
        entry_threshold = float(getattr(self.cfg.backtest, "zscore_entry_threshold", self.cfg.backtest.zscore_threshold))
        exit_threshold = float(getattr(self.cfg.backtest, "zscore_exit", 0.0) or 0.0)
        min_hold_bars = int(self.cfg.backtest.min_position_hold_minutes / self.cfg.time.gap_minutes)
        cooldown_bars = int(self.cfg.backtest.anti_churn_cooldown_minutes / self.cfg.time.gap_minutes)
        max_hold_bars = 0
        if self.cfg.backtest.time_stop_multiplier:
            max_hold_bars = int(self.cfg.backtest.time_stop_multiplier * rolling_window)
        stop_loss_z = float(self.cfg.backtest.pair_stop_loss_zscore)
        pair_stop_loss_usd = float(self.cfg.backtest.pair_stop_loss_usd)
        min_spread_move = float(self.cfg.backtest.min_spread_move_sigma)
        min_beta = float(self.cfg.filter_params.min_beta)
        max_beta = float(self.cfg.filter_params.max_beta)

        history_bars = max(rolling_window + 5, min_hold_bars + 5, cooldown_bars + 5)

        for pair in self.pairs:
            series_y = self._get_klines_cached(pair.symbol1, interval, history_bars)
            series_x = self._get_klines_cached(pair.symbol2, interval, history_bars)
            y_vals, x_vals, last_ts = align_series(series_y, series_x, bar_ms)
            if len(y_vals) < rolling_window or len(x_vals) < rolling_window:
                self.logger.log_system("Not enough data for pair", level="warning", pair=pair.key)
                continue
            state = self.states[pair.key]
            if state.last_bar_ts is not None and last_ts == state.last_bar_ts:
                continue

            self._reset_daily_pnl_if_needed()
            zscore, mu, sigma, beta = self._compute_zscore(
                y_vals, x_vals, state.beta or 1.0, rolling_window, min_sigma, min_beta, max_beta
            )
            state.beta = beta
            current_spread = float(y_vals[-1] - beta * x_vals[-1])
            now = datetime.now(timezone.utc)

            if self.daily_pnl <= -self.cfg.backtest.portfolio_daily_stop_pct * self.cfg.portfolio.initial_capital:
                if state.position != 0:
                    self._exit_position(pair, state, y_vals[-1], x_vals[-1], now)
                continue

            if state.position == 0:
                if pair.symbol1 in self.blocked_symbols or pair.symbol2 in self.blocked_symbols:
                    continue
                if self._active_positions_count() >= self.cfg.portfolio.max_active_positions:
                    continue
                if self._current_notional_usage() > self.cfg.portfolio.max_margin_usage:
                    continue
                if state.last_exit_time and cooldown_bars > 0:
                    cooldown_until = state.last_exit_time + timedelta(minutes=cooldown_bars * self.cfg.time.gap_minutes)
                    if now < cooldown_until:
                        continue

                if min_spread_move > 0.0 and state.last_flat_spread is not None:
                    if abs(current_spread - state.last_flat_spread) < min_spread_move * sigma:
                        continue

                if zscore >= entry_threshold:
                    self._enter_position(pair, state, -1, y_vals[-1], x_vals[-1], now)
                elif zscore <= -entry_threshold:
                    self._enter_position(pair, state, 1, y_vals[-1], x_vals[-1], now)
            else:
                bars_held = 0
                if state.entry_time:
                    bars_held = int((now - state.entry_time).total_seconds() / (self.cfg.time.gap_minutes * 60))

                should_exit = False
                if max_hold_bars and bars_held >= max_hold_bars:
                    should_exit = True
                elif stop_loss_z > 0:
                    if state.position > 0 and zscore <= -stop_loss_z:
                        should_exit = True
                    if state.position < 0 and zscore >= stop_loss_z:
                        should_exit = True

                if not should_exit and bars_held >= min_hold_bars and abs(zscore) <= abs(exit_threshold):
                    should_exit = True

                if not should_exit and pair_stop_loss_usd > 0:
                    unrealized = self._estimate_unrealized_pnl(state, y_vals[-1], x_vals[-1])
                    if unrealized <= -pair_stop_loss_usd:
                        should_exit = True

                if should_exit:
                    self._exit_position(pair, state, y_vals[-1], x_vals[-1], now)

            state.last_bar_ts = last_ts

        self._save_state()

    def _enter_position(
        self,
        pair: PairSpec,
        state: PairState,
        position: int,
        price_y: float,
        price_x: float,
        now: datetime,
    ) -> None:
        qty_y, qty_x = self._compute_quantities(pair.symbol1, pair.symbol2, price_y, price_x, state.beta)
        if qty_y <= 0 or qty_x <= 0:
            self.logger.log_system("Skip entry due to size constraints", level="warning", pair=pair.key)
            return

        side_y = "Buy" if position > 0 else "Sell"
        side_x = "Sell" if position > 0 else "Buy"

        self._place_pair_orders(pair, side_y, side_x, qty_y, qty_x, reduce_only=False)
        state.position = position
        state.entry_time = now
        state.entry_price_y = price_y
        state.entry_price_x = price_x
        state.qty_y = qty_y
        state.qty_x = qty_x
        self.logger.log_trade(
            {
                "pair": pair.key,
                "action": "ENTER",
                "position": position,
                "qty_y": qty_y,
                "qty_x": qty_x,
                "price_y": price_y,
                "price_x": price_x,
                "beta": state.beta,
            }
        )

    def _exit_position(
        self,
        pair: PairSpec,
        state: PairState,
        price_y: float,
        price_x: float,
        now: datetime,
    ) -> None:
        if state.position == 0:
            return

        side_y = "Sell" if state.position > 0 else "Buy"
        side_x = "Buy" if state.position > 0 else "Sell"
        self._place_pair_orders(pair, side_y, side_x, state.qty_y, state.qty_x, reduce_only=True)

        pnl = self._estimate_realized_pnl(state, price_y, price_x)
        state.realized_pnl += pnl
        self.daily_pnl += pnl
        state.last_exit_time = now
        state.last_flat_spread = price_y - state.beta * price_x
        self.logger.log_trade(
            {
                "pair": pair.key,
                "action": "EXIT",
                "position": state.position,
                "qty_y": state.qty_y,
                "qty_x": state.qty_x,
                "exit_price_y": price_y,
                "exit_price_x": price_x,
                "pnl": pnl,
            }
        )

        state.position = 0
        state.entry_time = None
        state.entry_price_y = 0.0
        state.entry_price_x = 0.0
        state.qty_y = 0.0
        state.qty_x = 0.0

    def _estimate_realized_pnl(self, state: PairState, price_y: float, price_x: float) -> float:
        if state.position == 0:
            return 0.0
        if state.position > 0:
            pnl_y = (price_y - state.entry_price_y) * state.qty_y
            pnl_x = (state.entry_price_x - price_x) * state.qty_x
        else:
            pnl_y = (state.entry_price_y - price_y) * state.qty_y
            pnl_x = (price_x - state.entry_price_x) * state.qty_x
        return pnl_y + pnl_x

    def _estimate_unrealized_pnl(self, state: PairState, price_y: float, price_x: float) -> float:
        return self._estimate_realized_pnl(state, price_y, price_x)

    def _active_positions_count(self) -> int:
        return sum(1 for state in self.states.values() if state.position != 0)

    def _current_notional_usage(self) -> float:
        total_notional = 0.0
        for state in self.states.values():
            if state.position == 0:
                continue
            total_notional += abs(state.entry_price_y * state.qty_y) + abs(state.entry_price_x * state.qty_x)
        if self.cfg.portfolio.initial_capital <= 0:
            return 0.0
        return total_notional / self.cfg.portfolio.initial_capital
