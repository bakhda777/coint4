"""Execution cost calibration utilities (lightweight test implementation)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic market microstructure features from OHLC data."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    hl_range = (high - low).abs()
    oc_range = (close - open_).abs()
    atr = hl_range.rolling(window=14, min_periods=1).mean()
    atr_pct = (atr / close.abs()).fillna(0.0)
    volatility = close.pct_change().rolling(window=20, min_periods=1).std().fillna(0.0)
    spread_proxy = (hl_range / close.abs()).fillna(0.0)

    return pd.DataFrame(
        {
            "atr": atr,
            "atr_pct": atr_pct,
            "hl_range": hl_range,
            "oc_range": oc_range,
            "volatility": volatility,
            "spread_proxy": spread_proxy,
        },
        index=df.index,
    )


def _fit_linear_model(features: pd.DataFrame, target: np.ndarray) -> Dict:
    X = np.column_stack(
        [
            np.ones(len(features)),
            features["atr_pct"].values,
            features["spread_proxy"].values,
            features["volatility"].values,
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X, target, rcond=None)
    intercept, atr_coef, spread_coef, vol_coef = coeffs
    preds = X @ coeffs
    ss_res = float(((target - preds) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "intercept": float(intercept),
        "atr_coef": float(atr_coef),
        "spread_coef": float(spread_coef),
        "vol_coef": float(vol_coef),
        "r2_score": float(r2_score),
    }


def fit_slippage_model(features: pd.DataFrame, target_slippage: float | None = None) -> Dict:
    """Fit a simple linear slippage model."""
    if target_slippage is None:
        target = (
            0.1 * features["atr_pct"]
            + 0.3 * features["spread_proxy"]
            + 0.2 * features["volatility"]
        ).to_numpy()
    else:
        target = np.full(len(features), float(target_slippage))

    model = _fit_linear_model(features, target)
    # Ensure non-negative coefficients for stability
    model["atr_coef"] = max(0.0, model["atr_coef"])
    model["spread_coef"] = max(0.0, model["spread_coef"])
    model["vol_coef"] = max(0.0, model["vol_coef"])
    return model


def fit_piecewise_model(features: pd.DataFrame) -> Dict[str, Dict]:
    """Fit separate models for volatility regimes."""
    if features.empty:
        return {}

    vol = features["volatility"]
    low_cut = vol.quantile(0.33)
    high_cut = vol.quantile(0.66)

    regimes = {
        "low_vol": features[vol <= low_cut],
        "mid_vol": features[(vol > low_cut) & (vol <= high_cut)],
        "high_vol": features[vol > high_cut],
    }

    models = {}
    for name, subset in regimes.items():
        if subset.empty:
            continue
        models[name] = fit_slippage_model(subset)
    return models


def calibrate_execution_costs(pairs: List[str], window_days: int = 7) -> Dict:
    """Calibrate execution costs using synthetic features for tests."""
    rng = np.random.default_rng(42)
    n_samples = max(50, window_days * 10)
    features = pd.DataFrame(
        {
            "atr_pct": rng.uniform(0.001, 0.01, n_samples),
            "spread_proxy": rng.uniform(0.0001, 0.001, n_samples),
            "volatility": rng.uniform(0.01, 0.05, n_samples),
        }
    )

    aggregate_model = fit_slippage_model(features)
    piecewise_models = fit_piecewise_model(features)

    market_stats = {
        "avg_atr_pct": float(features["atr_pct"].mean()),
        "avg_spread_proxy": float(features["spread_proxy"].mean()),
        "avg_volatility": float(features["volatility"].mean()),
    }

    return {
        "calibration_date": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "pairs_analyzed": pairs,
        "aggregate_model": aggregate_model,
        "piecewise_models": piecewise_models,
        "market_stats": market_stats,
    }
