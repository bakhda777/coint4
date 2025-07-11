from pathlib import Path
import types

import numpy as np
import pandas as pd

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


def create_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    for sym, shift in [("AAA", 0), ("BBB", 1)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(5), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


def test_load_all_data_for_period(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=2,
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
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    result = handler.load_all_data_for_period()

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=2)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()

    result = result.astype(float)
    expected = expected.astype(float)
    sync_and_assert_frames(result, expected)


def test_load_pair_data(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=10,
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
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    result = handler.load_pair_data(
        "AAA",
        "BBB",
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-04"),
    )

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf = pdf[pdf["symbol"].isin(["AAA", "BBB"])]
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)
    limit = 5
    expected = expected.interpolate(method="linear", limit=limit)
    expected = expected.ffill(limit=limit).bfill(limit=limit)
    expected = expected[["AAA", "BBB"]].dropna()
    expected = expected.loc[pd.Timestamp("2021-01-02"): pd.Timestamp("2021-01-04")]

    # Диагностика при пустом результате
    if result.empty:
        print('=== RESULT SHAPE:', result.shape)
        print('=== RESULT COLUMNS:', result.columns)
        print('=== RESULT INDEX:', result.index)
        print('=== EXPECTED INDEX:', expected.index)
        print('=== EXPECTED COLUMNS:', expected.columns)
        print(expected)

    # Синхронизация типов колонок и имён
    result.index.name = expected.index.name
    result.columns.name = expected.columns.name
    if not isinstance(result.columns, type(expected.columns)):
        if hasattr(expected.columns, 'categories'):
            result.columns = pd.CategoricalIndex(result.columns, categories=expected.columns.categories, name=expected.columns.name)
        else:
            result.columns = expected.columns.__class__(result.columns, name=expected.columns.name)

    sync_and_assert_frames(result, expected)


def test_load_and_normalize_data(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=10,
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
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-01-05")
    result = handler.load_and_normalize_data(start, end)

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    mask = (pdf["timestamp"] >= start) & (pdf["timestamp"] <= end)
    pdf = pdf.loc[mask]
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)

    for col in expected.columns:
        series = expected[col]
        first_val = series.loc[series.first_valid_index()] if series.first_valid_index() is not None else pd.NA
        if pd.isna(first_val) or first_val == 0:
            expected[col] = 0.0
        else:
            expected[col] = 100 * series / first_val
    # Удаляем константные колонки (как в коде)
    expected = expected.loc[:, expected.nunique() > 1]

    try:
        result = result.astype(float)
        expected = expected.astype(float)
        sync_and_assert_frames(result, expected)
    except AssertionError as e:
        print('=== RESULT INDEX:', result.index)
        print(result.head())
        print('=== EXPECTED INDEX:', expected.index)
        print(expected.head())
        raise
    assert (result >= 0).all().all()


def test_clear_cache(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=1,
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
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    initial = handler.load_all_data_for_period()
    assert "CCC" not in initial.columns

    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    part_dir = tmp_path / "symbol=CCC" / "year=2021" / "month=01"
    part_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"timestamp": idx, "close": range(5)})
    df.to_parquet(part_dir / "data.parquet")

    handler.clear_cache()
    # Явно увеличиваем lookback_days, чтобы включить все даты
    result = handler.load_all_data_for_period(lookback_days=10)

    # Диагностика состояния результата
    print('=== RESULT SHAPE:', result.shape)
    print('=== RESULT COLUMNS:', result.columns)
    if not result.empty:
        print('=== RESULT TIMESTAMP RANGE:', result.index.min(), result.index.max())

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=10)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()
    freq_val = pd.infer_freq(expected.index)
    if freq_val:
        expected = expected.asfreq(freq_val)

    print('=== EXPECTED SHAPE:', expected.shape)
    print('=== EXPECTED COLUMNS:', expected.columns)
    if not expected.empty:
        print('=== EXPECTED TIMESTAMP RANGE:', expected.index.min(), expected.index.max())

    try:
        result = result.astype(float)
        expected = expected.astype(float)
        sync_and_assert_frames(result, expected)
    except AssertionError as e:
        print('=== RESULT INDEX:', result.index)
        print(result.head())
        print('=== EXPECTED INDEX:', expected.index)
        print(expected.head())
        raise
    assert "CCC" in result.columns


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


def test_fill_limit_pct_application(tmp_path: Path) -> None:
    create_large_dataset_with_gaps(tmp_path)

    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=1,
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

            end_date="2021-04-10",

            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-04-10")
    result = handler.load_pair_data("AAA", "BBB", start, end)

    expected_a = pd.Series(np.arange(100, dtype=float), index=pd.date_range("2021-01-01", periods=100, freq="D"))
    expected_b = expected_a + 1
    expected_a[50:60] = np.nan
    expected_b[60:70] = np.nan
    expected = pd.DataFrame({"AAA": expected_a, "BBB": expected_b})
    limit = 5
    expected = expected.interpolate(method="linear", limit=limit)
    expected = expected.ffill(limit=limit).bfill(limit=limit)
    expected = expected[["AAA", "BBB"]].dropna()
    expected.index.name = 'timestamp'
    expected.columns.name = 'symbol'

    result = result.astype(float)
    expected = expected.astype(float)
    sync_and_assert_frames(result, expected)




def create_future_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-11", periods=5, freq="D")
    for sym in ["AAA", "BBB"]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2025" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": range(5)})
        df.to_parquet(part_dir / "data.parquet")


def test__load_full_dataset(tmp_path: Path) -> None:
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
