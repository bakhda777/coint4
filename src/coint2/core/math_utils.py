"""Mathematical helper utilities."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numba import njit


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Calculate rolling beta of ``y`` relative to ``x``.

    The beta is computed as the rolling covariance between ``y`` and ``x``
    divided by the rolling variance of ``x``.

    Parameters
    ----------
    y : pd.Series
        Dependent variable series.
    x : pd.Series
        Independent variable series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Series of rolling beta values aligned to the right edge of the window.
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    # Защита от деления на ноль
    return cov / var.where(var != 0, np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score for ``series``.

    The z-score is computed as the deviation of each value from the rolling
    mean divided by the rolling standard deviation.

    Parameters
    ----------
    series : pd.Series
        Time series to compute the z-score on.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Series of rolling z-score values aligned to the right edge of the
        window.
    """
    # Check for empty series to avoid numpy warnings
    if len(series) == 0:
        return pd.Series(dtype=float)
    
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    
    # Replace NaN std with small value to avoid division by zero
    std = std.fillna(1e-8)
    std = std.replace(0, 1e-8)
    
    return (series - mean) / std


from typing import Optional

def calculate_ssd(
    normalized_prices: pd.DataFrame,
    top_k: int | None = None,
    block_size: int = 512,
) -> pd.Series:
    """Compute top-K pairwise SSDs using a memory-efficient block approach."""

    cols = normalized_prices.columns.to_numpy()
    all_pairs: list[tuple[int, int, float]] = []

    num_cols = len(cols)

    for i0 in range(0, num_cols, block_size):
        i1 = min(i0 + block_size, num_cols)
        a = normalized_prices.iloc[:, i0:i1].to_numpy()

        for j0 in range(i0, num_cols, block_size):
            j1 = min(j0 + block_size, num_cols)
            b = normalized_prices.iloc[:, j0:j1].to_numpy()

            ssd_block = ((a[:, :, None] - b[:, None, :]) ** 2).sum(axis=0)

            if i0 == j0:
                if ssd_block.shape[0] == ssd_block.shape[1]:
                    indices = np.triu_indices_from(ssd_block, k=1)
                    block_pairs = list(
                        zip(
                            indices[0] + i0,
                            indices[1] + j0,
                            ssd_block[indices],
                        )
                    )
                else:
                    block_pairs = []
            else:
                idx_a, idx_b = np.indices(ssd_block.shape)
                block_pairs = list(
                    zip(
                        idx_a.ravel() + i0,
                        idx_b.ravel() + j0,
                        ssd_block.ravel(),
                    )
                )

            all_pairs.extend(block_pairs)

            if top_k is not None and len(all_pairs) > top_k * 2:
                all_pairs = sorted(all_pairs, key=lambda x: x[2])[:top_k]

    all_pairs = sorted(all_pairs, key=lambda x: x[2])[:top_k]

    multi_index = pd.MultiIndex.from_tuples(
        [(cols[i], cols[j]) for i, j, _ in all_pairs]
    )

    return pd.Series([ssd for _, _, ssd in all_pairs], index=multi_index)


def calculate_half_life(series: pd.Series) -> float:
    """Estimate the half-life of mean reversion for a time series.

    The half-life represents the time it takes for a deviation from the
    mean to reduce by half assuming an Ornstein-Uhlenbeck process. The
    implementation follows the common approach of regressing the first
    difference of the series on its lagged values.

    Parameters
    ----------
    series : pd.Series
        Input time series.

    Returns
    -------
    float
        Estimated half-life. ``np.inf`` is returned when the estimated
        mean reversion speed is non-negative.
    """
    # align lagged series with the differenced series
    y_lag = series.shift(1).dropna()
    delta_y = (series - y_lag).dropna()
    common_index = y_lag.index.intersection(delta_y.index)
    y_lag = y_lag.loc[common_index]
    delta_y = delta_y.loc[common_index]

    # add constant and run OLS regression using a lightweight
    # implementation to avoid external dependencies at runtime
    try:  # pragma: no cover - use statsmodels when available
        import statsmodels.api as sm  # type: ignore

        X = sm.add_constant(y_lag.to_numpy())
        model = sm.OLS(delta_y.to_numpy(), X)
        result = model.fit()
        lambda_coef = float(result.params[1])
    except Exception:  # fallback if statsmodels is unavailable
        X = np.column_stack([np.ones(len(y_lag)), y_lag.to_numpy()])
        beta, *_ = np.linalg.lstsq(X, delta_y.to_numpy(), rcond=None)
        lambda_coef = float(beta[1])

    if lambda_coef >= 0:
        return float(np.inf)

    return -np.log(2.0) / lambda_coef


def count_mean_crossings(series: pd.Series) -> int:
    """Count how many times a series crosses its mean value."""
    
    # Check for empty series to avoid numpy warnings
    if len(series) == 0 or series.isna().all():
        return 0

    centered_series = series - series.mean()
    signs = np.sign(centered_series)
    # diff will be non-zero when sign changes
    return int(np.where(np.diff(signs) != 0)[0].size)


# --- НАЧАЛО НОВОГО КОДА ---

@njit(cache=True, fastmath=True)
def half_life_numba(y: np.ndarray) -> float:
    """Numba-optimized half-life calculation."""
    y_lag = y[:-1]
    delta = np.diff(y)

    # OLS regression: delta = alpha + lambda * y_lag + error
    # Using matrix formulation: X = [ones, y_lag], beta = (X'X)^-1 X'y
    n = len(y_lag)
    if n == 0:
        return np.inf

    sum_x = np.sum(y_lag)
    sum_y = np.sum(delta)
    sum_xx = np.sum(y_lag * y_lag)
    sum_xy = np.sum(y_lag * delta)

    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return np.inf

    lambda_coef = (n * sum_xy - sum_x * sum_y) / denominator

    if lambda_coef >= 0:
        return np.inf

    return -np.log(2.0) / lambda_coef


def safe_div(num: float, denom: float, default: float | None = None) -> float:
    """Safely divide two numbers, handling zero and NaN denominators.
    
    Parameters
    ----------
    num : float
        Numerator.
    denom : float
        Denominator.
    default : float, optional
        Value to return if denominator is invalid. If None, returns NaN.
        
    Returns
    -------
    float
        Result of division or default value.
    """
    if denom is None or denom == 0 or np.isnan(denom):
        return default if default is not None else np.nan
    return num / denom
