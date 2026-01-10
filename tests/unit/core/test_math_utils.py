import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from coint2.core.math_utils import (
    calculate_half_life,
    calculate_ssd,
    count_mean_crossings,
    half_life_numba,
    mean_crossings_numba,
    rolling_beta,
    rolling_zscore,
)

# Константы для тестирования
ROLLING_WINDOW_SIZE = 5
SMALL_ROLLING_WINDOW = 3
PERFECT_LINEAR_BETA = 2.0
PERFECT_LINEAR_INTERCEPT = 1.0
AR_COEFFICIENT = 0.8
EXPECTED_MEAN_CROSSINGS = 5
EXPECTED_NUMBA_CROSSINGS = 2
RELATIVE_TOLERANCE = 0.05
SSD_TOP_K_SMALL = 1
SSD_TOP_K_MEDIUM = 2
SSD_BLOCK_SIZE = 2

# Тестовые данные
TEST_SERIES_LENGTH = 10
TEST_SERIES_SMALL = 6
TEST_OSCILLATING_SERIES = [1, -1, 1, -1, 1, -1]
TEST_COMPLEX_SERIES = [1, 2, 1, 0, -1, -2, -1, 0, 1]


@pytest.mark.unit
def test_rolling_beta_when_linear_relation_then_matches_linregress():
    """Тест проверяет, что rolling_beta дает те же результаты, что и scipy.linregress."""
    # Создаем идеальную линейную зависимость
    x = pd.Series(np.arange(TEST_SERIES_LENGTH, dtype=float))
    y = x * PERFECT_LINEAR_BETA + PERFECT_LINEAR_INTERCEPT  # perfect linear relation with beta=2

    beta = rolling_beta(y, x, ROLLING_WINDOW_SIZE)

    # Вычисляем ожидаемую бету через linregress для каждого окна
    expected = []
    for i in range(len(x)):
        if i + 1 < ROLLING_WINDOW_SIZE:
            expected.append(np.nan)
        else:
            window_start = i - ROLLING_WINDOW_SIZE + 1
            window_end = i + 1
            slope, _, _, _, _ = linregress(x[window_start:window_end], y[window_start:window_end])
            expected.append(slope)
    expected = pd.Series(expected, index=x.index)

    pd.testing.assert_series_equal(beta, expected)


@pytest.mark.unit
def test_rolling_zscore_when_basic_series_then_matches_pandas():
    """Тест проверяет, что rolling_zscore дает те же результаты, что и pandas rolling."""
    series = pd.Series([1, 2, 3, 4, 5, 6])
    z = rolling_zscore(series, SMALL_ROLLING_WINDOW)

    # Вычисляем ожидаемый результат через pandas
    means = series.rolling(SMALL_ROLLING_WINDOW).mean()
    stds = series.rolling(SMALL_ROLLING_WINDOW).std()
    expected = (series - means) / stds

    pd.testing.assert_series_equal(z, expected)


def _brute_force_ssd(df: pd.DataFrame) -> pd.Series:
    data = df.to_numpy()
    dot = data.T @ data
    sum_sq = np.diag(dot)
    ssd = sum_sq[:, None] + sum_sq[None, :] - 2 * dot
    iu, ju = np.triu_indices_from(ssd, k=1)
    idx = pd.MultiIndex.from_arrays([df.columns[iu], df.columns[ju]])
    values = ssd[iu, ju]
    return pd.Series(values, index=idx).sort_values()


@pytest.mark.unit
def test_calculate_ssd_when_using_blocks_then_matches_brute_force():
    """Тест проверяет, что блочный алгоритм SSD дает те же результаты, что и brute force."""
    # Создаем тестовые нормализованные цены
    norm_prices = pd.DataFrame(
        {
            "A": [0, 0, 0],
            "B": [1, 1, 1],
            "C": [0, 2, 4],
            "D": [0, 1, 2],
        }
    )

    # Вычисляем ожидаемый результат через brute force
    expected = _brute_force_ssd(norm_prices)

    # Тестируем с top_k=2
    result = calculate_ssd(norm_prices, top_k=SSD_TOP_K_MEDIUM, block_size=SSD_BLOCK_SIZE)
    assert len(result) == SSD_TOP_K_MEDIUM, f"Результат должен содержать {SSD_TOP_K_MEDIUM} элемента"
    pd.testing.assert_series_equal(result, expected.head(SSD_TOP_K_MEDIUM))

    # Тестируем с top_k=1
    single = calculate_ssd(norm_prices, top_k=SSD_TOP_K_SMALL, block_size=SSD_BLOCK_SIZE)
    assert len(single) == SSD_TOP_K_SMALL, f"Результат должен содержать {SSD_TOP_K_SMALL} элемент"
    pd.testing.assert_series_equal(single, expected.head(SSD_TOP_K_SMALL))


@pytest.mark.unit
def test_calculate_half_life_when_deterministic_series_then_correct() -> None:
    """Тест проверяет расчет полупериода для детерминистической серии."""
    series = pd.Series([AR_COEFFICIENT**i for i in range(TEST_SERIES_LENGTH)])
    expected = -np.log(2) / (AR_COEFFICIENT - 1)
    result = calculate_half_life(series)
    assert np.isclose(result, expected), f"Ожидаемый полупериод: {expected}, получен: {result}"


@pytest.mark.unit
def test_count_mean_crossings_when_oscillating_then_correct():
    """Тест проверяет подсчет пересечений среднего для осциллирующей серии."""
    series = pd.Series(TEST_OSCILLATING_SERIES)
    result = count_mean_crossings(series)
    assert result == EXPECTED_MEAN_CROSSINGS, f"Ожидается {EXPECTED_MEAN_CROSSINGS} пересечений, получено {result}"


@pytest.mark.unit
def test_half_life_numba_when_ar_series_then_matches_analytical():
    """Тест проверяет Numba-версию расчета полупериода."""
    # Создаем серию с известным коэффициентом авторегрессии
    series = np.array([AR_COEFFICIENT**i for i in range(TEST_SERIES_LENGTH)])
    result = half_life_numba(series)
    expected = -np.log(2) / (AR_COEFFICIENT - 1)
    assert np.isclose(result, expected, rtol=RELATIVE_TOLERANCE), \
        f"Ожидаемый полупериод: {expected}, получен: {result}"


@pytest.mark.unit
def test_mean_crossings_numba_when_complex_series_then_correct():
    """Тест проверяет Numba-версию подсчета пересечений среднего."""
    series = np.array(TEST_COMPLEX_SERIES)
    result = mean_crossings_numba(series)
    assert result == EXPECTED_NUMBA_CROSSINGS, \
        f"Ожидается {EXPECTED_NUMBA_CROSSINGS} пересечений, получено {result}"
