import math

import numpy as np
import pytest

from coint2.core.sharpe import annualized_sharpe_ratio, annualized_sharpe_ratio_from_equity


def test_annualized_sharpe_ratio_skips_non_finite_and_non_numeric_values() -> None:
    returns = [0.01, "bad", float("nan"), None, 0.02, float("inf"), -0.01]
    valid_returns = [0.01, 0.02, -0.01]

    expected = math.sqrt(252.0) * np.mean(valid_returns) / np.std(valid_returns, ddof=1)
    actual = annualized_sharpe_ratio(returns, 252.0)

    assert actual == pytest.approx(expected, rel=1e-12)


def test_annualized_sharpe_ratio_uses_zero_risk_free_when_input_is_invalid() -> None:
    returns = [0.01, -0.01, 0.03, -0.02]

    expected = annualized_sharpe_ratio(returns, 365.0)
    actual = annualized_sharpe_ratio(returns, 365.0, risk_free_rate=float("nan"))

    assert actual == pytest.approx(expected, rel=1e-12)


def test_annualized_sharpe_ratio_from_equity_skips_invalid_rows_and_zero_base() -> None:
    equities = [100.0, "invalid", 110.0, 0.0, 20.0, 22.0]
    expected_returns = [0.10, -1.0, 0.10]

    expected = annualized_sharpe_ratio(expected_returns, 365.0)
    actual = annualized_sharpe_ratio_from_equity(equities, 365.0)

    assert actual == pytest.approx(expected, rel=1e-12)
