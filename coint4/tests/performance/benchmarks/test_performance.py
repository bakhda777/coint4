import numpy as np
import pandas as pd
import pytest

from coint2.core import performance

# Константы для тестирования
ANNUALIZING_FACTOR_TRADING_DAYS = 252
ANNUALIZING_FACTOR_CALENDAR_DAYS = 365
TEST_PNL_SERIES = [0.01, -0.02, 0.03, -0.01, 0.02]
TOLERANCE = 1e-10


@pytest.mark.unit
def test_sharpe_ratio_when_annualizing_factor_changes_then_scales_correctly():
    """Тест проверяет, что коэффициент Шарпа правильно масштабируется с разными факторами аннуализации."""
    pnl = pd.Series(TEST_PNL_SERIES)
    daily_sharpe = pnl.mean() / pnl.std()

    result_252 = performance.sharpe_ratio(pnl, ANNUALIZING_FACTOR_TRADING_DAYS)
    result_365 = performance.sharpe_ratio(pnl, ANNUALIZING_FACTOR_CALENDAR_DAYS)

    expected_252 = daily_sharpe * np.sqrt(ANNUALIZING_FACTOR_TRADING_DAYS)
    expected_365 = daily_sharpe * np.sqrt(ANNUALIZING_FACTOR_CALENDAR_DAYS)

    assert result_252 == pytest.approx(expected_252, rel=TOLERANCE), f"Ожидается {expected_252}, получено {result_252}"
    assert result_365 == pytest.approx(expected_365, rel=TOLERANCE), f"Ожидается {expected_365}, получено {result_365}"
