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
