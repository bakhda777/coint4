"""Vectorized operations for walk-forward analysis optimization."""

import numpy as np
import pandas as pd
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
import warnings


@jit(nopython=True, cache=True)
def _calculate_trade_stats_vectorized(positions: np.ndarray, pnl: np.ndarray) -> Tuple[int, float, float, int, int]:
    """Vectorized calculation of trade statistics.
    
    Args:
        positions: Array of position values
        pnl: Array of PnL values
        
    Returns:
        Tuple of (trade_count, total_pnl, max_daily_gain, win_days, lose_days)
    """
    if len(positions) == 0:
        return 0, 0.0, 0.0, 0, 0
    
    # Calculate trade count
    trade_count = 0
    prev_pos = 0.0
    for i in range(len(positions)):
        current_pos = positions[i]
        if prev_pos == 0.0 and current_pos != 0.0:
            trade_count += 1
        prev_pos = current_pos
    
    # Calculate PnL statistics
    total_pnl = np.sum(pnl)
    max_daily_gain = np.max(pnl) if len(pnl) > 0 else 0.0
    
    # Count win/lose days
    win_days = 0
    lose_days = 0
    for p in pnl:
        if p > 0:
            win_days += 1
        elif p < 0:
            lose_days += 1
    
    return trade_count, total_pnl, max_daily_gain, win_days, lose_days


@jit(nopython=True, cache=True)
def _normalize_pair_data_vectorized(data: np.ndarray) -> np.ndarray:
    """Vectorized normalization of pair data to start at 100.
    
    Args:
        data: 2D array where each column is a price series
        
    Returns:
        Normalized data array
    """
    if data.shape[0] == 0:
        return data
    
    # Normalize each column to start at 100
    normalized = np.empty_like(data)
    for col in range(data.shape[1]):
        first_val = data[0, col]
        if first_val != 0:
            normalized[:, col] = data[:, col] / first_val * 100.0
        else:
            normalized[:, col] = data[:, col]
    
    return normalized


@jit(nopython=True, cache=True, parallel=True)
def _batch_calculate_rolling_stats(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of rolling statistics for multiple series.
    
    Args:
        data: 2D array where each column is a time series
        window: Rolling window size
        
    Returns:
        Tuple of (rolling_mean, rolling_std, rolling_beta)
    """
    n_rows, n_cols = data.shape
    
    # Initialize output arrays
    rolling_mean = np.full((n_rows, n_cols), np.nan)
    rolling_std = np.full((n_rows, n_cols), np.nan)
    rolling_beta = np.full((n_rows, n_cols // 2), np.nan)  # Assuming pairs
    
    # Calculate rolling statistics
    for i in prange(window - 1, n_rows):
        start_idx = i - window + 1
        
        # Rolling mean and std for each column
        for col in range(n_cols):
            window_data = data[start_idx:i+1, col]
            rolling_mean[i, col] = np.mean(window_data)
            rolling_std[i, col] = np.std(window_data)
        
        # Rolling beta for pairs (assuming even number of columns)
        for pair_idx in range(0, n_cols, 2):
            if pair_idx + 1 < n_cols:
                x = data[start_idx:i+1, pair_idx]
                y = data[start_idx:i+1, pair_idx + 1]
                
                # Simple linear regression for beta
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                if denominator != 0:
                    rolling_beta[i, pair_idx // 2] = numerator / denominator
    
    return rolling_mean, rolling_std, rolling_beta


class VectorizedStatsCalculator:
    """Vectorized calculator for pair trading statistics."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self._stats_cache = {}
    
    def calculate_pair_stats_batch(self, pair_results: List[Dict]) -> pd.DataFrame:
        """Calculate statistics for multiple pairs using vectorized operations.
        
        Args:
            pair_results: List of pair result dictionaries
            
        Returns:
            DataFrame with calculated statistics
        """
        if not pair_results:
            return pd.DataFrame()
        
        # Extract data for vectorized processing
        stats_list = []
        
        for result in pair_results:
            if result.get('success', False) and 'trade_stat' in result:
                trade_stat = result['trade_stat']
                
                # Extract basic statistics from trade_stat
                pair = trade_stat.get('pair', 'unknown')
                period = trade_stat.get('period', 'unknown')
                total_pnl = trade_stat.get('total_pnl', 0.0)
                trade_count = trade_stat.get('trade_count', 0)
                win_days = trade_stat.get('win_days', 0)
                lose_days = trade_stat.get('lose_days', 0)
                
                # Calculate additional statistics from PnL series if available
                pnl_series = result.get('pnl_series', pd.Series())
                if not pnl_series.empty:
                    pnl_values = pnl_series.values
                    max_gain = np.max(pnl_values) if len(pnl_values) > 0 else 0.0
                    total_days = len(pnl_values)
                    
                    # Recalculate win/lose days from PnL if not provided
                    if win_days == 0 and lose_days == 0:
                        win_days = int(np.sum(pnl_values > 0))
                        lose_days = int(np.sum(pnl_values < 0))
                else:
                    max_gain = 0.0
                    total_days = win_days + lose_days
                
                stats_list.append({
                    'pair': pair,
                    'period': period,
                    'total_pnl': total_pnl,
                    'trade_count': trade_count,
                    'max_daily_gain': max_gain,
                    'win_days': win_days,
                    'lose_days': lose_days,
                    'total_days': total_days
                })
        
        return pd.DataFrame(stats_list)
    
    def normalize_pair_data_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize multiple pair datasets using vectorized operations.
        
        Args:
            data_dict: Dictionary of pair_name -> DataFrame
            
        Returns:
            Dictionary of normalized DataFrames
        """
        normalized_dict = {}
        
        for pair_name, df in data_dict.items():
            if not df.empty and len(df.columns) >= 2:
                # Convert to numpy for vectorized processing
                data_array = df.values
                normalized_array = _normalize_pair_data_vectorized(data_array)
                
                # Convert back to DataFrame
                normalized_dict[pair_name] = pd.DataFrame(
                    normalized_array, 
                    index=df.index, 
                    columns=df.columns
                )
            else:
                normalized_dict[pair_name] = df.copy()
        
        return normalized_dict
    
    def calculate_rolling_stats_batch(self, data: pd.DataFrame, window: int) -> Dict[str, pd.DataFrame]:
        """Calculate rolling statistics for multiple time series using vectorized operations.
        
        Args:
            data: DataFrame with multiple time series as columns
            window: Rolling window size
            
        Returns:
            Dictionary with 'mean', 'std', 'beta' DataFrames
        """
        if data.empty:
            return {'mean': pd.DataFrame(), 'std': pd.DataFrame(), 'beta': pd.DataFrame()}
        
        # Use cache key
        cache_key = f"{hash(str(data.index[0]))}-{hash(str(data.index[-1]))}-{window}-{len(data.columns)}"
        
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        # Calculate rolling statistics using pandas for better compatibility
        rolling_mean = data.rolling(window=window, min_periods=1).mean()
        rolling_std = data.rolling(window=window, min_periods=1).std()
        
        # Calculate rolling beta for pairs
        n_cols = len(data.columns)
        beta_columns = []
        beta_data = []
        
        # Assume pairs of columns for beta calculation
        for i in range(0, n_cols, 2):
            if i + 1 < n_cols:
                col1, col2 = data.columns[i], data.columns[i + 1]
                
                # Calculate rolling beta using covariance and variance
                rolling_cov = data[col1].rolling(window=window, min_periods=1).cov(data[col2])
                rolling_var = data[col2].rolling(window=window, min_periods=1).var()
                
                # Beta = Cov(X,Y) / Var(Y)
                rolling_beta_series = rolling_cov / rolling_var
                rolling_beta_series = rolling_beta_series.fillna(0)
                
                beta_columns.append(f'beta_{col1}_{col2}')
                beta_data.append(rolling_beta_series)
        
        # Create beta DataFrame
        if beta_data:
            rolling_beta = pd.DataFrame(dict(zip(beta_columns, beta_data)), index=data.index)
        else:
            rolling_beta = pd.DataFrame(index=data.index)
        
        # Convert back to DataFrames
        result = {
            'mean': rolling_mean,
            'std': rolling_std,
            'beta': rolling_beta
        }
        
        # Cache result
        if len(self._stats_cache) < self.cache_size:
            self._stats_cache[cache_key] = result
        
        return result


def vectorized_eval_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate complex expressions using pandas.eval for better performance.
    
    Args:
        df: DataFrame to evaluate expression on
        expression: String expression to evaluate
        
    Returns:
        Series with evaluation results
    """
    try:
        # Use pandas.eval for vectorized evaluation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.eval(expression, local_dict={'df': df})
    except Exception:
        # Fallback to regular evaluation
        return eval(expression)


def batch_process_pairs_vectorized(pair_data_list: List[Tuple], 
                                 processing_func, 
                                 batch_size: int = 50) -> List:
    """Process pairs in batches using vectorized operations.
    
    Args:
        pair_data_list: List of pair data tuples
        processing_func: Function to process each batch
        batch_size: Size of each batch
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(pair_data_list), batch_size):
        batch = pair_data_list[i:i + batch_size]
        batch_results = processing_func(batch)
        results.extend(batch_results)
    
    return results