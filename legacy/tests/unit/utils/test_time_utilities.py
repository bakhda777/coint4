import pandas as pd
import pytest

from coint2.utils.time_utils import ensure_datetime_index, infer_frequency

# Константы для тестирования
TEST_START_DATE = "2021-01-01"
TEST_PERIODS_2 = 2
TEST_PERIODS_5 = 5
FREQUENCY_DAILY = "D"
FREQUENCY_HOURLY = "h"
FREQUENCY_15MIN = "15min"
TIMEZONE_UTC = "UTC"
TEST_VALUES = [1, 2]
IRREGULAR_DATES = ["2021-01-01", "2021-01-02", "2021-01-04"]


@pytest.mark.smoke
def test_datetime_index_when_ensured_then_sorts_and_drops_tz() -> None:
    """Тест обеспечения корректного datetime индекса."""
    COLUMN_A = "a"

    idx = pd.date_range(TEST_START_DATE, periods=TEST_PERIODS_2, freq=FREQUENCY_DAILY, tz=TIMEZONE_UTC)[::-1]
    df = pd.DataFrame({COLUMN_A: TEST_VALUES}, index=idx)

    result = ensure_datetime_index(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    expected = list(pd.date_range(TEST_START_DATE, periods=TEST_PERIODS_2, freq=FREQUENCY_DAILY))
    assert list(result.index) == expected


@pytest.mark.parametrize("freq,expected", [
    (FREQUENCY_DAILY, FREQUENCY_DAILY),
    (FREQUENCY_HOURLY, FREQUENCY_HOURLY),
    (FREQUENCY_15MIN, FREQUENCY_15MIN)
])
@pytest.mark.unit
def test_frequency_when_regular_then_inferred_correctly(freq: str, expected: str) -> None:
    """Тест корректного определения регулярной частоты."""
    idx = pd.date_range(TEST_START_DATE, periods=TEST_PERIODS_5, freq=freq)
    assert infer_frequency(idx) == expected


@pytest.mark.smoke
def test_frequency_when_irregular_then_inferred_as_none() -> None:
    """Тест определения нерегулярной частоты как None."""
    idx = pd.to_datetime(IRREGULAR_DATES)
    assert infer_frequency(idx) is None

