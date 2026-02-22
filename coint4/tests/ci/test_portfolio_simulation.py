from types import SimpleNamespace

import pandas as pd

from coint2.pipeline.walk_forward_orchestrator import _simulate_realistic_portfolio


def test_simulate_realistic_portfolio_uses_positions_and_exit_pnl():
    index = pd.date_range("2024-01-01", periods=3, freq="D")

    pnl0 = pd.Series([0.0, -1.0, 5.0], index=index)
    pnl1 = pd.Series([0.0, -2.0, 0.0], index=index)

    pos0 = pd.Series([0, 1, 0], index=index)
    pos1 = pd.Series([0, 1, 1], index=index)

    score0 = pd.Series([0.0, 3.0, 0.0], index=index)
    score1 = pd.Series([0.0, 2.0, 0.0], index=index)

    cfg = SimpleNamespace(portfolio=SimpleNamespace(max_active_positions=1))

    portfolio_pnl = _simulate_realistic_portfolio(
        [pnl0, pnl1],
        cfg,
        all_positions=[pos0, pos1],
        all_scores=[score0, score1],
    )

    expected = pd.Series([0.0, -1.0, 5.0], index=index)
    pd.testing.assert_series_equal(portfolio_pnl, expected, check_freq=False)


def test_simulate_realistic_portfolio_applies_deleverage_and_daily_hard_stop():
    index = pd.date_range("2024-01-01 00:00:00", periods=4, freq="15min")
    pnl = pd.Series([-1.0, -1.0, -1.0, -1.0], index=index)
    pos = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)

    cfg = SimpleNamespace(
        portfolio=SimpleNamespace(max_active_positions=1, initial_capital=100.0),
        backtest=SimpleNamespace(
            portfolio_daily_stop_pct=0.02,
            portfolio_deleverage_start_pct=0.01,
            portfolio_deleverage_factor=0.5,
        ),
    )

    portfolio_pnl, diagnostics = _simulate_realistic_portfolio(
        [pnl],
        cfg,
        all_positions=[pos],
        return_diagnostics=True,
    )

    expected = pd.Series([-1.0, -0.5, -0.5, 0.0], index=index)
    pd.testing.assert_series_equal(portfolio_pnl, expected, check_freq=False)
    assert "turnover_units" in diagnostics
    assert "exposure_units" in diagnostics
