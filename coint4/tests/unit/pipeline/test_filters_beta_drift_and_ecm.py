import numpy as np
import pandas as pd

from coint2.pipeline.filters import _FilterWorkerCfg, _evaluate_pair


def _ar1(*, rho: float, n: int, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(int(n), dtype=float)
    for t in range(1, int(n)):
        x[t] = rho * x[t - 1] + rng.normal(0.0, sigma)
    return x


def _cfg(**overrides):
    base = dict(
        pvalue_threshold=1.0,  # keep coint gate effectively off in unit tests
        min_beta=0.0,
        max_beta=1_000.0,
        max_beta_drift_ratio=None,
        min_half_life=0.0,
        max_half_life=1e9,
        min_mean_crossings=1,
        min_history_ratio=0.0,
        min_correlation=-1.0,  # allow anything
        liquidity_usd_daily=0.0,  # disable market snapshot gates
        max_bid_ask_pct=1.0,
        max_avg_funding_pct=1.0,
        kpss_pvalue_threshold=1.0,  # disable KPSS gate
        max_hurst_exponent=1.0,  # keep hurst gate effectively off
        require_market_metrics=False,
        require_same_quote=False,
        min_volume_usd_24h=0.0,
        min_days_live=0,
        max_funding_rate_abs=0.0,
        max_tick_size_pct=0.0,
        ecm_alpha_tstat_threshold=None,
    )
    base.update(overrides)
    return _FilterWorkerCfg(**base)


def test_evaluate_pair_rejects_unstable_beta_when_max_beta_drift_ratio_set():
    n = 1_000
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(42)
    s2 = np.cumsum(rng.normal(0.0, 1.0, size=n)) + 100.0
    noise = rng.normal(0.0, 0.5, size=n)
    mid = n // 2
    s1 = np.empty(n, dtype=float)
    s1[:mid] = 2.0 * s2[:mid] + noise[:mid]
    s1[mid:] = 4.0 * s2[mid:] + noise[mid:]
    price_df = pd.DataFrame({"S1": s1, "S2": s2}, index=idx)

    cfg = _cfg(max_beta_drift_ratio=0.25)
    status, stage, reason_key, *_rest = _evaluate_pair(("S1", "S2"), price_df, cfg)
    assert status == "reject"
    assert stage == "beta"
    assert reason_key == "beta_drift"


def test_evaluate_pair_ecm_tstat_threshold_filters_pairs():
    n = 1_000
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(7)
    s2 = np.cumsum(rng.normal(0.0, 1.0, size=n)) + 100.0

    # Mean-reverting spread should pass.
    spread_pass = _ar1(rho=0.9, n=n, seed=0, sigma=1.0)
    s1_pass = 2.0 * s2 + spread_pass
    price_df_pass = pd.DataFrame({"S1": s1_pass, "S2": s2}, index=idx)
    cfg = _cfg(ecm_alpha_tstat_threshold=2.0)
    status, *_rest = _evaluate_pair(("S1", "S2"), price_df_pass, cfg)
    assert status == "pass"

    # Very slow mean-reversion should be rejected by ECM gate (tstat not negative enough).
    spread_fail = _ar1(rho=0.995, n=n, seed=0, sigma=1.0)
    s1_fail = 2.0 * s2 + spread_fail
    price_df_fail = pd.DataFrame({"S1": s1_fail, "S2": s2}, index=idx)
    status, stage, reason_key, *_rest = _evaluate_pair(("S1", "S2"), price_df_fail, cfg)
    assert status == "reject"
    assert stage == "ecm"
    assert reason_key == "ecm"

