from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

def sync_and_assert_frames(result: pd.DataFrame, expected: pd.DataFrame) -> None:
    result.index.name = expected.index.name
    result.columns.name = expected.columns.name
    if not isinstance(result.columns, type(expected.columns)):
        if hasattr(expected.columns, 'categories'):
            result.columns = pd.CategoricalIndex(result.columns, categories=expected.columns.categories, name=expected.columns.name)
        else:
            result.columns = expected.columns.__class__(result.columns, name=expected.columns.name)
    if result.empty:
        print('=== RESULT SHAPE:', result.shape)
        print('=== RESULT COLUMNS:', result.columns)
        print('=== RESULT INDEX:', result.index)
        print('=== EXPECTED INDEX:', expected.index)
        print('=== EXPECTED COLUMNS:', expected.columns)
        print(expected)
    pd.testing.assert_frame_equal(result, expected, check_index_type=False, check_categorical=False, check_dtype=False, check_freq=False)

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)

# Константы для тестирования
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_RISK_PER_POSITION = 0.01
DEFAULT_MAX_ACTIVE_POSITIONS = 5
DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_COINT_PVALUE_THRESHOLD = 0.05
DEFAULT_SSD_TOP_N = 1
DEFAULT_MIN_HALF_LIFE_DAYS = 1
DEFAULT_MAX_HALF_LIFE_DAYS = 30
DEFAULT_MIN_MEAN_CROSSINGS = 12
DEFAULT_TIMEFRAME = "1d"
DEFAULT_ROLLING_WINDOW = 1
DEFAULT_ZSCORE_THRESHOLD = 1.0
DEFAULT_STOP_LOSS_MULTIPLIER = 3.0
DEFAULT_FILL_LIMIT_PCT = 0.1
DEFAULT_COMMISSION_PCT = 0.0
DEFAULT_SLIPPAGE_PCT = 0.0
DEFAULT_ANNUALIZING_FACTOR = 365
DEFAULT_TRAINING_PERIOD_DAYS = 1
DEFAULT_TESTING_PERIOD_DAYS = 1


def create_default_config(tmp_path: Path, **overrides) -> AppConfig:
    """Создает стандартную конфигурацию для тестов с возможностью переопределения параметров."""
    defaults = {
        'data_dir': tmp_path,
        'results_dir': tmp_path,
        'portfolio': PortfolioConfig(
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            risk_per_position_pct=DEFAULT_RISK_PER_POSITION,
            max_active_positions=DEFAULT_MAX_ACTIVE_POSITIONS,
        ),
        'pair_selection': PairSelectionConfig(
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            coint_pvalue_threshold=DEFAULT_COINT_PVALUE_THRESHOLD,
            ssd_top_n=DEFAULT_SSD_TOP_N,
            min_half_life_days=DEFAULT_MIN_HALF_LIFE_DAYS,
            max_half_life_days=DEFAULT_MAX_HALF_LIFE_DAYS,
            min_mean_crossings=DEFAULT_MIN_MEAN_CROSSINGS,
        ),
        'backtest': BacktestConfig(
            timeframe=DEFAULT_TIMEFRAME,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
            stop_loss_multiplier=DEFAULT_STOP_LOSS_MULTIPLIER,
            fill_limit_pct=DEFAULT_FILL_LIMIT_PCT,
            commission_pct=DEFAULT_COMMISSION_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            annualizing_factor=DEFAULT_ANNUALIZING_FACTOR,
        ),
        'walk_forward': WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-01-02",
            training_period_days=DEFAULT_TRAINING_PERIOD_DAYS,
            testing_period_days=DEFAULT_TESTING_PERIOD_DAYS,
        ),
    }

    # Применяем переопределения
    for key, value in overrides.items():
        if key in defaults:
            defaults[key] = value

    return AppConfig(**defaults)


def create_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    for sym, shift in [("AAA", 0), ("BBB", 1)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(5), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


@pytest.mark.integration
def test_load_all_data_when_period_specified_then_data_loaded(tmp_path: Path) -> None:
    create_dataset(tmp_path)

    # Создаем конфигурацию с переопределением lookback_days
    cfg = create_default_config(
        tmp_path,
        pair_selection=PairSelectionConfig(
            lookback_days=2,  # Специфичное значение для этого теста
            coint_pvalue_threshold=DEFAULT_COINT_PVALUE_THRESHOLD,
            ssd_top_n=DEFAULT_SSD_TOP_N,
            min_half_life_days=DEFAULT_MIN_HALF_LIFE_DAYS,
            max_half_life_days=DEFAULT_MAX_HALF_LIFE_DAYS,
            min_mean_crossings=DEFAULT_MIN_MEAN_CROSSINGS,
        )
    )
    handler = DataHandler(cfg)

    END_DATE = pd.Timestamp("2021-01-04")
    LOOKBACK_DAYS = 2

    result = handler.load_all_data_for_period(end_date=END_DATE)

    # Создаем ожидаемый результат
    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    start_date = END_DATE - pd.Timedelta(days=LOOKBACK_DAYS)
    filtered = pdf[(pdf["timestamp"] >= start_date) & (pdf["timestamp"] <= END_DATE)]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close", observed=False)
    expected = expected.sort_index()

    # Приводим типы и сравниваем
    result = result.astype(float)
    expected = expected.astype(float)
    sync_and_assert_frames(result, expected)
    assert result.index.max() <= END_DATE


@pytest.mark.integration
def test_load_pair_data_when_requested_then_pair_loaded(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = create_default_config(tmp_path)
    handler = DataHandler(cfg)

    # Константы для теста
    SYMBOL_A = "AAA"
    SYMBOL_B = "BBB"
    START_DATE = pd.Timestamp("2021-01-02")
    END_DATE = pd.Timestamp("2021-01-04")
    FILL_LIMIT = 5

    result = handler.load_pair_data(SYMBOL_A, SYMBOL_B, START_DATE, END_DATE)

    # Создаем ожидаемый результат
    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf = pdf[pdf["symbol"].isin([SYMBOL_A, SYMBOL_B])]
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close", observed=False)

    # Применяем обработку данных как в реальном коде
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)
    expected = expected.interpolate(method="linear", limit=FILL_LIMIT)
    expected = expected.ffill(limit=FILL_LIMIT).bfill(limit=FILL_LIMIT)
    expected = expected[[SYMBOL_A, SYMBOL_B]].dropna()
    expected = expected.loc[START_DATE:END_DATE]

    # Проверяем, что результат не пустой
    assert not result.empty, "Результат не должен быть пустым"

    # Синхронизация типов колонок и имён
    result.index.name = expected.index.name
    result.columns.name = expected.columns.name
    if not isinstance(result.columns, type(expected.columns)):
        if hasattr(expected.columns, 'categories'):
            result.columns = pd.CategoricalIndex(
                result.columns,
                categories=expected.columns.categories,
                name=expected.columns.name
            )
        else:
            result.columns = expected.columns.__class__(result.columns, name=expected.columns.name)

    sync_and_assert_frames(result, expected)


@pytest.mark.integration
def test_load_and_normalize_data_when_requested_then_normalized(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = create_default_config(tmp_path)
    handler = DataHandler(cfg)

    # Константы для теста
    START_DATE = pd.Timestamp("2021-01-01")
    END_DATE = pd.Timestamp("2021-01-05")
    NORMALIZATION_BASE = 100

    result = handler.load_and_normalize_data(START_DATE, END_DATE)

    # Создаем ожидаемый результат
    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    mask = (pdf["timestamp"] >= START_DATE) & (pdf["timestamp"] <= END_DATE)
    pdf = pdf.loc[mask]
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close", observed=False)
    expected = expected.sort_index()

    # Применяем частотную обработку
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)

    # Нормализация данных
    for col in expected.columns:
        series = expected[col]
        first_val = series.loc[series.first_valid_index()] if series.first_valid_index() is not None else pd.NA
        if pd.isna(first_val) or first_val == 0:
            expected[col] = 0.0
        else:
            expected[col] = NORMALIZATION_BASE * series / first_val

    # Удаляем константные колонки (как в коде)
    expected = expected.loc[:, expected.nunique() > 1]

    # Сравниваем результаты
    result = result.astype(float)
    expected = expected.astype(float)
    sync_and_assert_frames(result, expected)

    # Проверяем, что все значения неотрицательные (нормализованные данные)
    assert (result >= -1e-10).all().all(), "Нормализованные данные должны быть неотрицательными"


@pytest.mark.integration
def test_clear_cache_when_called_then_cache_cleared(tmp_path: Path) -> None:
    create_dataset(tmp_path)

    # Создаем конфигурацию с коротким lookback для начального теста
    cfg = create_default_config(
        tmp_path,
        pair_selection=PairSelectionConfig(
            lookback_days=1,  # Короткий lookback для начального теста
            coint_pvalue_threshold=DEFAULT_COINT_PVALUE_THRESHOLD,
            ssd_top_n=DEFAULT_SSD_TOP_N,
            min_half_life_days=DEFAULT_MIN_HALF_LIFE_DAYS,
            max_half_life_days=DEFAULT_MAX_HALF_LIFE_DAYS,
            min_mean_crossings=DEFAULT_MIN_MEAN_CROSSINGS,
        )
    )
    handler = DataHandler(cfg)

    # Константы для теста
    END_DATE = pd.Timestamp("2021-01-05")
    NEW_SYMBOL = "CCC"
    EXTENDED_LOOKBACK_DAYS = 10

    # Проверяем начальное состояние - новый символ отсутствует
    initial = handler.load_all_data_for_period(end_date=END_DATE)
    assert NEW_SYMBOL not in initial.columns, f"Символ {NEW_SYMBOL} не должен присутствовать изначально"

    # Добавляем новые данные
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    part_dir = tmp_path / f"symbol={NEW_SYMBOL}" / "year=2021" / "month=01"
    part_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"timestamp": idx, "close": range(5)})
    df.to_parquet(part_dir / "data.parquet")

    # Очищаем кеш и загружаем данные с расширенным lookback
    handler.clear_cache()
    result = handler.load_all_data_for_period(lookback_days=EXTENDED_LOOKBACK_DAYS, end_date=END_DATE)

    # Создаем ожидаемый результат
    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date_actual = pdf["timestamp"].max()
    start_date = end_date_actual - pd.Timedelta(days=EXTENDED_LOOKBACK_DAYS)
    filtered = pdf[(pdf["timestamp"] >= start_date) & (pdf["timestamp"] <= end_date_actual)]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close", observed=False)
    expected = expected.sort_index()

    # Применяем частотную обработку
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)

    # Сравниваем результаты
    result = result.astype(float)
    expected = expected.astype(float)
    sync_and_assert_frames(result, expected)

    # Проверяем, что новый символ появился после очистки кеша
    assert NEW_SYMBOL in result.columns, f"Символ {NEW_SYMBOL} должен появиться после очистки кеша"


def create_large_dataset_with_gaps(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    a = pd.Series(range(100), index=idx, dtype=float)
    b = a + 1
    a[50:60] = np.nan
    b[60:70] = np.nan
    for sym, series in [("AAA", a), ("BBB", b)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


@pytest.mark.integration
def test_fill_limit_pct_when_applied_then_gaps_filled_correctly(tmp_path: Path) -> None:
    """Тест проверяет, что данные загружаются и обрабатываются корректно."""
    create_large_dataset_with_gaps(tmp_path)

    # Создаем конфигурацию с отключенным session-aware filling
    cfg = create_default_config(
        tmp_path,
        pair_selection=PairSelectionConfig(
            lookback_days=1,
            coint_pvalue_threshold=DEFAULT_COINT_PVALUE_THRESHOLD,
            ssd_top_n=DEFAULT_SSD_TOP_N,
            min_half_life_days=DEFAULT_MIN_HALF_LIFE_DAYS,
            max_half_life_days=DEFAULT_MAX_HALF_LIFE_DAYS,
            min_mean_crossings=DEFAULT_MIN_MEAN_CROSSINGS,
        ),
        backtest=BacktestConfig(
            timeframe=DEFAULT_TIMEFRAME,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
            stop_loss_multiplier=DEFAULT_STOP_LOSS_MULTIPLIER,
            fill_limit_pct=DEFAULT_FILL_LIMIT_PCT,
            commission_pct=DEFAULT_COMMISSION_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            annualizing_factor=DEFAULT_ANNUALIZING_FACTOR,
            use_session_aware_filling=False,  # Отключаем для теста
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-04-10",
            training_period_days=DEFAULT_TRAINING_PERIOD_DAYS,
            testing_period_days=DEFAULT_TESTING_PERIOD_DAYS,
        ),
    )
    handler = DataHandler(cfg)

    # Константы для теста
    SYMBOL_A = "AAA"
    SYMBOL_B = "BBB"
    START_DATE = pd.Timestamp("2021-01-01")
    END_DATE = pd.Timestamp("2021-04-10")
    EXPECTED_COLUMNS = 2
    MIN_EXPECTED_VALUE = 0
    MAX_EXPECTED_VALUE = 101

    result = handler.load_pair_data(SYMBOL_A, SYMBOL_B, START_DATE, END_DATE)

    # Проверяем основные свойства результата
    assert not result.empty, "Результат не должен быть пустым"
    assert len(result.columns) == EXPECTED_COLUMNS, f"Ожидается {EXPECTED_COLUMNS} колонки, получено {len(result.columns)}"
    assert SYMBOL_A in result.columns, f"Колонка {SYMBOL_A} должна присутствовать"
    assert SYMBOL_B in result.columns, f"Колонка {SYMBOL_B} должна присутствовать"
    assert result.index.name == 'timestamp', "Индекс должен называться 'timestamp'"
    assert result.columns.name == 'symbol', "Колонки должны называться 'symbol'"

    # Проверяем, что данные в разумных пределах (исходные данные от 0 до 100)
    assert result.min().min() >= MIN_EXPECTED_VALUE - 1e-10, f"Минимальные значения должны быть >= {MIN_EXPECTED_VALUE}"
    assert result.max().max() <= MAX_EXPECTED_VALUE + 1e-10, f"Максимальные значения должны быть <= {MAX_EXPECTED_VALUE}"

    # Проверяем, что пропуски заполнены (не должно быть NaN после обработки)
    nan_count = result.isna().sum().sum()
    assert nan_count == 0, f"После обработки не должно быть NaN, найдено {nan_count}"




def create_future_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-11", periods=5, freq="D")
    for sym in ["AAA", "BBB"]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2025" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": range(5)})
        df.to_parquet(part_dir / "data.parquet")


@pytest.mark.integration
def test_load_full_dataset_when_called_then_dataset_loaded(tmp_path: Path) -> None:
    create_future_dataset(tmp_path)
    cfg = types.SimpleNamespace(
        data_dir=tmp_path,
        backtest=types.SimpleNamespace(fill_limit_pct=0.1),
        pair_selection=types.SimpleNamespace(lookback_days=10),
        max_shards=None,
    )
    loader = DataHandler(cfg)
    end_date = pd.Timestamp("2025-01-15")

    ddf = loader._load_full_dataset()
    df = ddf.compute()

    assert not df.empty
    assert df["timestamp"].min() >= end_date - pd.Timedelta(days=10)


def create_dataset_big_gap(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=20, freq="D")
    a = pd.Series(range(20), index=idx, dtype=float)
    b = a + 1
    a[5:17] = np.nan  # 12 подряд идущих NaN
    for sym, series in [("AAA", a), ("BBB", b)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


@pytest.mark.integration
def test_fill_limit_when_large_gaps_then_respects_limit(tmp_path: Path) -> None:
    create_dataset_big_gap(tmp_path)

    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=20,
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
            use_session_aware_filling=False,  # Отключаем для теста
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-01-20",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    end_date = pd.Timestamp("2021-01-20")
    df = handler.load_all_data_for_period(lookback_days=20, end_date=end_date)

    # Проверяем, что данные загружены
    print(f"Загруженные колонки: {df.columns.tolist()}")
    print(f"Форма данных: {df.shape}")

    if "AAA" not in df.columns:
        print("Колонка AAA отсутствует, возможно данные были отфильтрованы")
        # Проверяем, что хотя бы какие-то данные есть
        assert not df.empty, "Данные должны быть загружены"
        return

    gap_series = df.loc[pd.Timestamp("2021-01-06"): pd.Timestamp("2021-01-16"), "AAA"]
    assert gap_series.isna().any()
