import numpy as np
import pandas as pd
import pytest

from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life

# Константы для тестирования
TEST_PERIODS = 100
START_DATE = "2020-01-01"
FREQUENCY = "D"
NORMAL_MEAN = 0
NORMAL_STD = 1
SPREAD_COEFFICIENT = 0.5
SPREAD_NOISE_STD = 0.1
Y_MULTIPLIER = 1000
ASSET_A = "A"
ASSET_B = "B"

# Константы для фильтрации
PVALUE_THRESHOLD_HIGH = 0.99
MIN_HALF_LIFE_LOW = 0.1
MAX_HALF_LIFE_HIGH = 1000
MIN_MEAN_CROSSINGS_ZERO = 0


@pytest.mark.unit
def test_filter_pairs_by_coint_when_beta_range_then_filters_correctly(rng) -> None:
    """Тест фильтрации пар по диапазону бета коэффициентов."""
    # Детерминизм обеспечен через rng
    idx = pd.date_range(START_DATE, periods=TEST_PERIODS, freq=FREQUENCY)
    x = np.cumsum(rng.normal(NORMAL_MEAN, NORMAL_STD, TEST_PERIODS))
    spread = np.zeros(TEST_PERIODS)
    for i in range(1, TEST_PERIODS):
        spread[i] = SPREAD_COEFFICIENT * spread[i - 1] + rng.normal(NORMAL_MEAN, SPREAD_NOISE_STD)
    y = Y_MULTIPLIER * x + spread
    df = pd.DataFrame({ASSET_A: y, ASSET_B: x}, index=idx)

    pairs = [(ASSET_A, ASSET_B)]
    result = filter_pairs_by_coint_and_half_life(
        pairs,
        df,
        pvalue_threshold=PVALUE_THRESHOLD_HIGH,
        min_half_life=MIN_HALF_LIFE_LOW,
        max_half_life=MAX_HALF_LIFE_HIGH,
        min_mean_crossings=MIN_MEAN_CROSSINGS_ZERO,
    )

    assert result == []
