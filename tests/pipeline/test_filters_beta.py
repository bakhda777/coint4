import numpy as np
import pandas as pd

from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life


def test_filter_pairs_beta_range() -> None:
    np.random.seed(0)
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = np.cumsum(np.random.normal(0, 1, n))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.5 * spread[i - 1] + np.random.normal(0, 0.1)
    y = 1000 * x + spread
    df = pd.DataFrame({"A": y, "B": x}, index=idx)

    pairs = [("A", "B")]
    result = filter_pairs_by_coint_and_half_life(
        pairs,
        df,
        pvalue_threshold=0.99,
        min_half_life=0.1,
        max_half_life=1000,
        min_mean_crossings=0,
    )

    assert result == []
