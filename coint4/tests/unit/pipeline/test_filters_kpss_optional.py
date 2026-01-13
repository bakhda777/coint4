from __future__ import annotations

import numpy as np
import pandas as pd

from coint2.pipeline import filters as filters_module


def test_filter_pairs_when_kpss_disabled_then_does_not_call_kpss(monkeypatch) -> None:
    periods = 1000
    idx = pd.date_range("2024-01-01", periods=periods, freq="15min")
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 1, periods))
    price_df = pd.DataFrame({"A": base, "B": base + 0.1}, index=idx)

    monkeypatch.setattr(filters_module, "fast_coint", lambda y, x, trend="n": (0.0, 0.01, 0))
    monkeypatch.setattr(filters_module, "calculate_half_life", lambda spread: 10)
    monkeypatch.setattr(filters_module, "count_mean_crossings", lambda spread: 10)
    monkeypatch.setattr(filters_module, "calculate_hurst_exponent", lambda spread: 0.4)

    def _raise(*_args, **_kwargs):
        raise AssertionError("kpss should not be called when threshold is None")

    monkeypatch.setattr(filters_module, "kpss", _raise)

    filtered = filters_module.filter_pairs_by_coint_and_half_life(
        [("A", "B")],
        price_df,
        min_correlation=-1.0,
        min_mean_crossings=1,
        min_half_life=0.01,
        max_half_life=100,
        kpss_pvalue_threshold=None,
        liquidity_usd_daily=0.0,
    )

    assert len(filtered) == 1
