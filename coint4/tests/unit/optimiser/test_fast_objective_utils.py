import numpy as np
import pandas as pd
from types import SimpleNamespace

from optimiser.fast_objective import _clean_step_dataframe, _pairs_df_to_tuples, _resolve_step_size_days


def test_clean_step_dataframe_dedup_and_drop_columns():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:00",
            "2024-01-01 00:15",
            "2024-01-01 00:30",
        ]
    )
    df = pd.DataFrame(
        {
            "a": [1.0, 1.1, 1.2, 1.3],
            "b": [np.nan, np.nan, 2.0, 2.1],
            "c": [np.nan, np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    base_config = SimpleNamespace(
        backtest=SimpleNamespace(),
        data_processing=SimpleNamespace(nan_threshold=0.5),
    )

    cleaned = _clean_step_dataframe(df, base_config)

    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates
    assert "c" not in cleaned.columns


def test_pairs_df_to_tuples():
    step_pairs = pd.DataFrame(
        {
            "s1": ["AAA", "BBB"],
            "s2": ["CCC", "DDD"],
            "beta": [1.0, 0.9],
        }
    )
    assert _pairs_df_to_tuples(step_pairs) == [("AAA", "CCC"), ("BBB", "DDD")]


def test_resolve_step_size_days_refit_frequency():
    cfg = SimpleNamespace(
        walk_forward=SimpleNamespace(
            step_size_days=None,
            testing_period_days=10,
            refit_frequency="weekly",
        )
    )
    assert _resolve_step_size_days(cfg) == 7
