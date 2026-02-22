"""Fully-costed overlay for walk-forward backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SCENARIO_MULTIPLIERS: dict[str, float] = {
    "low": 0.6,
    "baseline": 1.0,
    "mid": 1.0,
    "high": 1.5,
    "stress": 2.0,
}


@dataclass(frozen=True)
class CostModelConfig:
    costs_enabled: bool
    fee_bps: float
    slippage_bps: float
    funding_bps_per_8h: float
    funding_series_path: str | None
    cost_scenarios: tuple[str, ...]
    active_scenario: str
    scenario_multiplier: float


@dataclass(frozen=True)
class CostComputation:
    config: CostModelConfig
    net_pnl: pd.Series
    daily_costs: pd.Series
    fee_cost: float
    slippage_cost: float
    funding_cost: float
    total_cost: float
    turnover_notional: float
    turnover_ratio: float


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_scenarios(raw_value: Any) -> tuple[str, ...]:
    if isinstance(raw_value, (list, tuple)):
        values = [str(item).strip().lower() for item in raw_value if str(item).strip()]
        return tuple(values) if values else ("baseline", "low", "mid", "high")

    if isinstance(raw_value, str):
        values = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
        return tuple(values) if values else ("baseline", "low", "mid", "high")

    return ("baseline", "low", "mid", "high")


def resolve_cost_model_config(backtest_cfg: Any, environ: Mapping[str, str] | None = None) -> CostModelConfig:
    env = environ or {}

    default_enabled = bool(getattr(backtest_cfg, "costs_enabled", True))

    fee_bps_default = _parse_float(getattr(backtest_cfg, "fee_bps", None), float(getattr(backtest_cfg, "commission_pct", 0.0) or 0.0) * 10000.0)
    slippage_bps_default = _parse_float(
        getattr(backtest_cfg, "slippage_bps", None),
        float(getattr(backtest_cfg, "slippage_pct", 0.0) or 0.0) * 10000.0,
    )
    funding_bps_default = _parse_float(getattr(backtest_cfg, "funding_bps_per_8h", 0.0), 0.0)

    scenarios_default = _parse_scenarios(getattr(backtest_cfg, "cost_scenarios", ("baseline", "low", "mid", "high")))
    active_scenario_default = str(getattr(backtest_cfg, "cost_scenario", "baseline") or "baseline").strip().lower()

    costs_enabled = _parse_bool(env.get("COSTS_ENABLED"), default_enabled)
    fee_bps = _parse_float(env.get("FEE_BPS"), fee_bps_default)
    slippage_bps = _parse_float(env.get("SLIPPAGE_BPS"), slippage_bps_default)
    funding_bps_per_8h = _parse_float(env.get("FUNDING_BPS_PER_8H"), funding_bps_default)

    funding_series_path = str(
        env.get("FUNDING_SERIES_PATH")
        or getattr(backtest_cfg, "funding_series_path", "")
        or ""
    ).strip()
    funding_series_path = funding_series_path or None

    cost_scenarios = _parse_scenarios(env.get("COST_SCENARIOS", scenarios_default))
    active_scenario = str(env.get("COST_SCENARIO", active_scenario_default) or "baseline").strip().lower()

    scenario_multiplier = SCENARIO_MULTIPLIERS.get(active_scenario, 1.0)

    return CostModelConfig(
        costs_enabled=costs_enabled,
        fee_bps=max(0.0, float(fee_bps)),
        slippage_bps=max(0.0, float(slippage_bps)),
        funding_bps_per_8h=float(funding_bps_per_8h),
        funding_series_path=funding_series_path,
        cost_scenarios=cost_scenarios,
        active_scenario=active_scenario,
        scenario_multiplier=float(scenario_multiplier),
    )


def _safe_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(0.0, index=index, dtype=float)
    out = series.astype(float).reindex(index).fillna(0.0)
    return out


def _allocate_cost(total_cost: float, weights: pd.Series) -> pd.Series:
    if total_cost <= 0.0:
        return pd.Series(0.0, index=weights.index, dtype=float)

    positive = weights.clip(lower=0.0)
    weight_sum = float(positive.sum())
    if weight_sum <= 0.0:
        return pd.Series(total_cost / max(len(weights), 1), index=weights.index, dtype=float)
    return positive / weight_sum * float(total_cost)


def _load_funding_series(path: str, index: pd.Index) -> pd.Series:
    source_path = Path(path)
    if not source_path.exists():
        return pd.Series(0.0, index=index, dtype=float)

    try:
        frame = pd.read_csv(source_path)
    except Exception:
        return pd.Series(0.0, index=index, dtype=float)

    if frame.empty:
        return pd.Series(0.0, index=index, dtype=float)

    candidate_columns = ["funding_bps_per_8h", "funding_bps", "funding_rate", "rate", "value"]
    value_col = next((col for col in candidate_columns if col in frame.columns), frame.columns[-1])
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)

    # Interpret magnitude: if absolute values are likely bps (> 1), convert to decimal rate per 8h.
    if float(values.abs().median()) > 1.0:
        values = values / 10000.0

    timestamp_col = None
    for col in ("timestamp", "time", "datetime", "date"):
        if col in frame.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        # No explicit timestamps: fallback to constant mean.
        return pd.Series(float(values.mean() if not values.empty else 0.0), index=index, dtype=float)

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    valid = timestamps.notna()
    if not valid.any():
        return pd.Series(float(values.mean() if not values.empty else 0.0), index=index, dtype=float)

    series = pd.Series(values[valid].values, index=timestamps[valid]).sort_index()
    if series.empty:
        return pd.Series(0.0, index=index, dtype=float)

    target_index = pd.to_datetime(index, utc=True, errors="coerce")
    aligned = series.reindex(target_index, method="ffill").fillna(method="bfill").fillna(0.0)
    aligned.index = index
    return aligned.astype(float)


def _estimate_turnover_notional(
    turnover_units: pd.Series,
    trade_stats: list[dict[str, Any]] | None,
    initial_capital: float,
) -> float:
    from_series = float(turnover_units.clip(lower=0.0).sum()) * float(initial_capital)

    from_trade_stats = 0.0
    if trade_stats:
        for stat in trade_stats:
            if not isinstance(stat, dict):
                continue
            entry_count = _parse_float(stat.get("entry_notional_count"), 0.0)
            entry_avg = _parse_float(stat.get("entry_notional_avg"), 0.0)
            trade_count = _parse_float(stat.get("trade_count"), 0.0)

            if entry_count > 0.0 and entry_avg > 0.0:
                from_trade_stats += entry_count * entry_avg * 2.0
            elif trade_count > 0.0:
                # Conservative fallback when notional diagnostics were unavailable.
                from_trade_stats += trade_count * float(initial_capital) * 0.10

    turnover_notional = max(from_series, from_trade_stats)
    return max(0.0, turnover_notional)


def apply_fully_costed_model(
    gross_pnl: pd.Series,
    *,
    turnover_units: pd.Series | None,
    exposure_units: pd.Series | None,
    initial_capital: float,
    bar_minutes: int,
    backtest_cfg: Any,
    trade_stats: list[dict[str, Any]] | None = None,
    environ: Mapping[str, str] | None = None,
) -> CostComputation:
    if gross_pnl is None or gross_pnl.empty:
        empty = pd.Series(dtype=float)
        cfg = resolve_cost_model_config(backtest_cfg, environ=environ)
        return CostComputation(
            config=cfg,
            net_pnl=empty,
            daily_costs=empty,
            fee_cost=0.0,
            slippage_cost=0.0,
            funding_cost=0.0,
            total_cost=0.0,
            turnover_notional=0.0,
            turnover_ratio=0.0,
        )

    cfg = resolve_cost_model_config(backtest_cfg, environ=environ)
    index = gross_pnl.index

    turnover = _safe_series(turnover_units, index=index)
    exposure = _safe_series(exposure_units, index=index)

    if not cfg.costs_enabled:
        zero = pd.Series(0.0, index=index, dtype=float)
        return CostComputation(
            config=cfg,
            net_pnl=gross_pnl.astype(float).copy(),
            daily_costs=zero,
            fee_cost=0.0,
            slippage_cost=0.0,
            funding_cost=0.0,
            total_cost=0.0,
            turnover_notional=0.0,
            turnover_ratio=0.0,
        )

    turnover_notional = _estimate_turnover_notional(turnover, trade_stats, initial_capital)
    turnover_ratio = turnover_notional / float(initial_capital) if initial_capital > 0 else 0.0

    multiplier = float(cfg.scenario_multiplier)
    fee_rate = max(0.0, cfg.fee_bps) / 10000.0 * multiplier
    slippage_rate = max(0.0, cfg.slippage_bps) / 10000.0 * multiplier

    fee_cost = turnover_notional * fee_rate
    slippage_cost = turnover_notional * slippage_rate

    funding_series = None
    if cfg.funding_series_path:
        funding_series = _load_funding_series(cfg.funding_series_path, index)

    per_bar_8h = float(max(bar_minutes, 1)) / (8.0 * 60.0)
    exposure_notional = exposure.clip(lower=0.0) * float(initial_capital)

    if float(exposure_notional.sum()) <= 0.0 and turnover_notional > 0.0:
        exposure_notional = pd.Series(
            turnover_notional / max(len(index), 1),
            index=index,
            dtype=float,
        )

    if funding_series is not None:
        funding_rate_per_8h = funding_series.astype(float) * multiplier
    else:
        funding_rate_per_8h = pd.Series(
            cfg.funding_bps_per_8h / 10000.0 * multiplier,
            index=index,
            dtype=float,
        )

    funding_cost_series = (exposure_notional * funding_rate_per_8h.abs() * per_bar_8h).fillna(0.0)
    funding_cost = float(funding_cost_series.sum())

    trading_cost_series = _allocate_cost(fee_cost + slippage_cost, turnover)
    daily_costs = (trading_cost_series + funding_cost_series).astype(float)

    # Numerical safety: exact matching with scalar totals.
    expected_total = fee_cost + slippage_cost + funding_cost
    observed_total = float(daily_costs.sum())
    if abs(expected_total - observed_total) > 1e-8 and len(daily_costs) > 0:
        daily_costs.iloc[-1] += expected_total - observed_total

    net_pnl = gross_pnl.astype(float) - daily_costs

    return CostComputation(
        config=cfg,
        net_pnl=net_pnl,
        daily_costs=daily_costs,
        fee_cost=float(fee_cost),
        slippage_cost=float(slippage_cost),
        funding_cost=float(funding_cost),
        total_cost=float(fee_cost + slippage_cost + funding_cost),
        turnover_notional=float(turnover_notional),
        turnover_ratio=float(turnover_ratio),
    )
