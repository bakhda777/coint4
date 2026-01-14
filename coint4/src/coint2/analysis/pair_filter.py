from __future__ import annotations

import pandas as pd
from hurst import compute_Hc


def calculate_hurst_exponent(series: pd.Series) -> float:
    """Calculate Hurst exponent for a time series."""
    # Drop NaN/inf and ensure sufficient length
    series = series.dropna()
    if len(series) < 50:
        return 0.5
    
    # Check for invalid values that would cause log10 errors
    if not series.var() > 0:
        return 0.5
    if (series <= 0).any():
        shift = -series.min() + 1e-6
        series = series + shift
    
    try:
        H, _c, _data = compute_Hc(series, kind="price", simplified=True)
        # Validate result
        if pd.isna(H) or not (0 <= H <= 1):
            return 0.5
        return float(H)
    except (FloatingPointError, ValueError, RuntimeWarning):
        # Return neutral value if computation fails
        return 0.5
