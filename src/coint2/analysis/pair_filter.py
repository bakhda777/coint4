from __future__ import annotations

import pandas as pd
from hurst import compute_Hc


def calculate_hurst_exponent(series: pd.Series) -> float:
    """Calculate Hurst exponent for a time series."""
    # Drop NaN/inf and ensure sufficient length
    series = series.dropna()
    if len(series) < 50:
        return 0.5
    H, _c, _data = compute_Hc(series, kind="price", simplified=True)
    return float(H)
