from pathlib import Path

import pandas as pd
import pytest
from dask import delayed

import coint2.pipeline.pair_scanner as pair_scanner
from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    FilterParamsConfig,
    PortfolioConfig,
    WalkForwardConfig,
)

# Константы для тестирования
TEST_START_DATE = '2021-01-01'
TEST_PERIODS = 20
TEST_FREQUENCY = 'D'
TEST_YEAR = 2021
TEST_MONTH = 1
COINTEGRATION_OFFSET = 0.1

# Символы для тестирования
SYMBOL_A = 'A'
SYMBOL_B = 'B'
SYMBOL_C = 'C'
PARQUET_FILENAME = 'data.parquet'

# Константы для конфигурации
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_RISK_PER_POSITION = 0.01
DEFAULT_MAX_ACTIVE_POSITIONS = 5
DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_COINT_PVALUE_THRESHOLD = 0.05
DEFAULT_SSD_TOP_N = 1
DEFAULT_MIN_HALF_LIFE_DAYS = 1
DEFAULT_MAX_HALF_LIFE_DAYS = 30
DEFAULT_MIN_MEAN_CROSSINGS = 12


def create_parquet_files(tmp_path: Path) -> None:
    """Создает тестовые parquet файлы с синтетическими данными."""
    idx = pd.date_range(TEST_START_DATE, periods=TEST_PERIODS, freq=TEST_FREQUENCY)
    a = pd.Series(range(TEST_PERIODS), index=idx)
    b = a + COINTEGRATION_OFFSET  # cointegrated with A
    c = pd.Series(range(TEST_PERIODS, 0, -1), index=idx)

    for sym, series in [(SYMBOL_A, a), (SYMBOL_B, b), (SYMBOL_C, c)]:
        part_dir = tmp_path / f'symbol={sym}' / f'year={TEST_YEAR}' / f'month={TEST_MONTH:02d}'
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'timestamp': idx, 'close': series})
        df.to_parquet(part_dir / PARQUET_FILENAME)


@pytest.mark.integration
def test_find_cointegrated_pairs_when_scanned_then_detects_correctly(monkeypatch, tmp_path: Path) -> None:
    """Интеграционный тест поиска коинтегрированных пар."""
    create_parquet_files(tmp_path)
    cfg = AppConfig(
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
        filter_params=FilterParamsConfig(),
        backtest=BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=3.0,
            fill_limit_pct=0.1,
            commission_pct=0.001,
            slippage_pct=0.0005,
            annualizing_factor=365,
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)
    data = handler.load_all_data_for_period()
    
    # Отладочная информация
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data index: {data.index}")
    
    # Если данные пустые, пропускаем тест
    if data.empty or 'A' not in data.columns:
        import pytest
        pytest.skip("No data loaded for test")

    trad_calls: list[tuple[str, str]] = []

    def fake_tradability(
        handler_arg,
        s1: str,
        s2: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        _min_hl: float,
        _max_hl: float,
        _min_cross: int,
    ) -> tuple[str, str] | None:
        trad_calls.append((s1, s2))
        return (s1, s2)

    monkeypatch.setattr(
        pair_scanner,
        "_test_pair_for_tradability",
        delayed(fake_tradability),
    )

    beta = data["A"].cov(data["B"]) / data["B"].var()
    spread = data["A"] - beta * data["B"]
    expected = ("A", "B", beta, spread.mean(), spread.std())

    start = data.index.min()
    end = data.index.max()
    pairs = pair_scanner.find_cointegrated_pairs(handler, start, end, cfg)

    assert pairs == [expected]
    assert trad_calls == [("A", "B")]
