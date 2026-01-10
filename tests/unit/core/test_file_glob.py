from pathlib import Path
# uuid заменён на детерминистические имена файлов

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
TEST_START_DATE = "2020-12-30"
TEST_PERIODS = 5
TEST_LOOKBACK_DAYS = 2
TEST_END_DATE = "2021-01-02"
DEFAULT_INITIAL_CAPITAL = 1
DEFAULT_RISK_PER_POSITION = 0.1
DEFAULT_MAX_ACTIVE_POSITIONS = 1


def create_deterministic_shards(base: Path) -> None:
    """Создает тестовые данные в разных шардах для проверки glob поиска."""
    # Создаем данные с более ранними датами, чтобы покрыть lookback_days
    idx = pd.date_range(TEST_START_DATE, periods=TEST_PERIODS, freq="D")

    for sym in TEST_SYMBOLS:
        # Создаем данные за 2020 год
        part_2020 = base / f"symbol={sym}" / "year=2020" / "month=12"
        part_2020.mkdir(parents=True, exist_ok=True)
        df_2020 = pd.DataFrame({"timestamp": idx[:2], "close": range(2)})
        df_2020.to_parquet(part_2020 / "shard_2020_001.parquet")

        # Создаем данные за 2021 год
        part_2021 = base / f"symbol={sym}" / "year=2021" / "month=01"
        part_2021.mkdir(parents=True, exist_ok=True)
        df_2021 = pd.DataFrame({"timestamp": idx[2:], "close": range(2, 5)})
        df_2021.to_parquet(part_2021 / "shard_2021_001.parquet")


@pytest.mark.integration
def test_rglob_when_searching_multiple_shards_then_finds_all_files(tmp_path: Path) -> None:
    """Тест проверяет, что glob поиск находит файлы во всех шардах."""
    create_deterministic_shards(tmp_path)

    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            risk_per_position_pct=DEFAULT_RISK_PER_POSITION,
            max_active_positions=DEFAULT_MAX_ACTIVE_POSITIONS
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=TEST_LOOKBACK_DAYS,
            coint_pvalue_threshold=0.05,
            ssd_top_n=1,
            min_half_life_days=1,
            max_half_life_days=30,
            min_mean_crossings=12,
        ),
        backtest=BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=3.0,
            fill_limit_pct=0.1,
            commission_pct=0.0,
            slippage_pct=0.0,
            annualizing_factor=365,
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date=TEST_END_DATE,
            training_period_days=1,
            testing_period_days=1,
        ),
    )

    handler = DataHandler(cfg)

    # Загружаем данные за указанный период
    end_date = pd.Timestamp(TEST_END_DATE)
    df = handler.load_all_data_for_period(end_date=end_date)

    # Проверяем, что данные загружены из всех шардов
    assert not df.empty, "Данные должны быть загружены из всех шардов"
    assert len(df.columns) == len(TEST_SYMBOLS), f"Должны быть загружены данные для всех символов: {TEST_SYMBOLS}"

    for symbol in TEST_SYMBOLS:
        assert symbol in df.columns, f"Символ {symbol} должен присутствовать в результате"
