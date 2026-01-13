from pathlib import Path

import pandas as pd
import pytest

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)

# Константы для тестирования
TEST_SYMBOLS = ["AAA", "BBB"]
TEST_SYMBOL_SHIFTS = [0, 1]
INTRADAY_FREQUENCY = "15min"
TRADING_START_TIME = "09:30"
TRADING_END_TIME = "16:00"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2021-01-02"
EXTENDED_END_DATE = "2021-01-03"

# Константы конфигурации
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_RISK_PER_POSITION = 0.01
DEFAULT_MAX_ACTIVE_POSITIONS = 5
DEFAULT_LOOKBACK_DAYS = 1
DEFAULT_COINT_PVALUE_THRESHOLD = 0.05
DEFAULT_SSD_TOP_N = 1
DEFAULT_MIN_HALF_LIFE_DAYS = 1
DEFAULT_MAX_HALF_LIFE_DAYS = 30
DEFAULT_MIN_MEAN_CROSSINGS = 12
DEFAULT_ROLLING_WINDOW = 1
DEFAULT_ZSCORE_THRESHOLD = 1.0
DEFAULT_STOP_LOSS_MULTIPLIER = 3.0
DEFAULT_FILL_LIMIT_PCT = 0.1
DEFAULT_COMMISSION_PCT = 0.0
DEFAULT_SLIPPAGE_PCT = 0.0
DEFAULT_ANNUALIZING_FACTOR = 365
DEFAULT_TRAINING_PERIOD_DAYS = 1
DEFAULT_TESTING_PERIOD_DAYS = 1


def create_intraday_dataset(tmp_path: Path) -> None:
    """Создает тестовый набор внутридневных данных."""
    # Создаем временные ряды для двух торговых дней
    day1 = pd.date_range(f"{TEST_START_DATE} {TRADING_START_TIME}",
                         f"{TEST_START_DATE} {TRADING_END_TIME}",
                         freq=INTRADAY_FREQUENCY)
    day2 = pd.date_range(f"{TEST_END_DATE} {TRADING_START_TIME}",
                         f"{TEST_END_DATE} {TRADING_END_TIME}",
                         freq=INTRADAY_FREQUENCY)
    idx = day1.append(day2)

    # Создаем данные для каждого символа с небольшим сдвигом
    for sym, shift in zip(TEST_SYMBOLS, TEST_SYMBOL_SHIFTS):
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(len(idx)), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series.values})
        df.to_parquet(part_dir / "data.parquet")


def make_cfg(tmp_path: Path) -> AppConfig:
    """Создает конфигурацию для тестирования внутридневных данных."""
    return AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            risk_per_position_pct=DEFAULT_RISK_PER_POSITION,
            max_active_positions=DEFAULT_MAX_ACTIVE_POSITIONS,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            coint_pvalue_threshold=DEFAULT_COINT_PVALUE_THRESHOLD,
            ssd_top_n=DEFAULT_SSD_TOP_N,
            min_half_life_days=DEFAULT_MIN_HALF_LIFE_DAYS,
            max_half_life_days=DEFAULT_MAX_HALF_LIFE_DAYS,
            min_mean_crossings=DEFAULT_MIN_MEAN_CROSSINGS,
        ),
        backtest=BacktestConfig(
            timeframe=INTRADAY_FREQUENCY,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
            stop_loss_multiplier=DEFAULT_STOP_LOSS_MULTIPLIER,
            fill_limit_pct=DEFAULT_FILL_LIMIT_PCT,
            commission_pct=DEFAULT_COMMISSION_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            annualizing_factor=DEFAULT_ANNUALIZING_FACTOR,
        ),
        walk_forward=WalkForwardConfig(
            start_date=TEST_START_DATE,
            end_date=EXTENDED_END_DATE,
            training_period_days=DEFAULT_TRAINING_PERIOD_DAYS,
            testing_period_days=DEFAULT_TESTING_PERIOD_DAYS,
        ),
    )


@pytest.mark.integration
def test_intraday_frequency_when_loaded_then_correct_timestamps(tmp_path: Path) -> None:
    """Тест проверяет корректность загрузки внутридневных данных с правильными временными метками."""
    create_intraday_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    handler = DataHandler(cfg)

    # Определяем временной диапазон для загрузки
    start_timestamp = pd.Timestamp(f"{TEST_START_DATE} {TRADING_START_TIME}")
    end_timestamp = pd.Timestamp(f"{TEST_END_DATE} {TRADING_END_TIME}")

    # Загружаем данные пары
    result = handler.load_pair_data(TEST_SYMBOLS[0], TEST_SYMBOLS[1], start_timestamp, end_timestamp)

    # Создаем ожидаемый индекс
    day1 = pd.date_range(f"{TEST_START_DATE} {TRADING_START_TIME}",
                         f"{TEST_START_DATE} {TRADING_END_TIME}",
                         freq=INTRADAY_FREQUENCY)
    day2 = pd.date_range(f"{TEST_END_DATE} {TRADING_START_TIME}",
                         f"{TEST_END_DATE} {TRADING_END_TIME}",
                         freq=INTRADAY_FREQUENCY)
    expected_index = day1.append(day2)

    # Проверяем соответствие временных меток
    assert list(result.index) == list(expected_index), "Временные метки должны точно соответствовать ожидаемым"

    # Проверяем, что частота не определена автоматически (None для внутридневных данных)
    assert handler.freq is None, "Частота должна быть None для внутридневных данных"

    # Проверяем наличие обеих колонок
    assert TEST_SYMBOLS[0] in result.columns, f"Колонка {TEST_SYMBOLS[0]} должна присутствовать"
    assert TEST_SYMBOLS[1] in result.columns, f"Колонка {TEST_SYMBOLS[1]} должна присутствовать"

    # Проверяем, что данные не пустые
    assert not result.empty, "Результат не должен быть пустым"
