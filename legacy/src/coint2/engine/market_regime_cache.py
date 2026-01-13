"""Optimized market regime detection with caching and Numba acceleration."""

import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, Tuple, Optional
from collections import defaultdict


@jit(nopython=True, cache=True)
def _hurst_exponent_jit(log_prices: np.ndarray) -> float:
    """Numba-optimized Hurst Exponent calculation.
    
    Args:
        log_prices: Log prices array
        
    Returns:
        Hurst exponent (0.0-1.0)
    """
    if len(log_prices) < 10:
        return 0.5
        
    try:
        # Calculate cumulative deviations from mean
        mean_log_price = np.mean(log_prices)
        cumulative_deviations = np.cumsum(log_prices - mean_log_price)
        
        # Calculate range (R)
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # Calculate standard deviation (S)
        S = np.std(log_prices)
        
        if S == 0.0 or R == 0.0:
            return 0.5
            
        # Calculate R/S ratio
        rs_ratio = R / S
        
        # Hurst exponent approximation: H ≈ log(R/S) / log(n)
        n = len(log_prices)
        hurst = np.log(rs_ratio) / np.log(n)
        
        # Clamp to reasonable range
        return max(0.0, min(1.0, hurst))
        
    except:
        return 0.5


@jit(nopython=True, cache=True)
def _variance_ratio_jit(log_prices: np.ndarray, k: int = 2) -> float:
    """Numba-optimized Variance Ratio calculation.
    
    Args:
        log_prices: Log prices array
        k: Lag parameter
        
    Returns:
        Variance ratio
    """
    if len(log_prices) < k * 3:
        return 1.0
        
    try:
        # Calculate returns
        returns = np.diff(log_prices)
        
        if len(returns) < k * 2:
            return 1.0
            
        # Calculate k-period returns
        k_returns = np.zeros(len(log_prices) - k)
        for i in range(len(k_returns)):
            k_returns[i] = log_prices[i + k] - log_prices[i]
            
        if len(k_returns) == 0 or len(returns) == 0:
            return 1.0
            
        # Calculate variances
        var_1 = np.var(returns)
        var_k = np.var(k_returns)
        
        if var_1 == 0.0:
            return 1.0
            
        # Variance ratio
        vr = var_k / (k * var_1)
        
        return max(0.1, min(3.0, vr))
        
    except:
        return 1.0


@jit(nopython=True, cache=True)
def _rolling_correlation_jit(x: np.ndarray, y: np.ndarray) -> float:
    """Numba-optimized rolling correlation calculation.
    
    Args:
        x, y: Price arrays
        
    Returns:
        Correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
        
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 2:
            return 0.0
            
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # Calculate correlation
        mean_x = np.mean(x_clean)
        mean_y = np.mean(y_clean)
        
        num = np.sum((x_clean - mean_x) * (y_clean - mean_y))
        den_x = np.sum((x_clean - mean_x) ** 2)
        den_y = np.sum((y_clean - mean_y) ** 2)
        
        if den_x == 0.0 or den_y == 0.0:
            return 0.0
            
        corr = num / np.sqrt(den_x * den_y)
        return max(-1.0, min(1.0, corr))
        
    except:
        return 0.0


@jit(nopython=True, cache=True)
def _exponential_weighted_correlation_jit(x: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Numba-optimized exponential weighted correlation calculation.
    
    Args:
        x, y: Price arrays
        alpha: Smoothing factor (0 < alpha <= 1), higher = more weight to recent data
        
    Returns:
        Exponential weighted correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
        
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 2:
            return 0.0
            
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        n = len(x_clean)
        
        if n < 2:
            return 0.0
        
        # Calculate exponential weights (more weight to recent observations)
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = (1 - alpha) ** (n - 1 - i)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted means
        mean_x = np.sum(weights * x_clean)
        mean_y = np.sum(weights * y_clean)
        
        # Calculate weighted covariance and variances
        cov_xy = np.sum(weights * (x_clean - mean_x) * (y_clean - mean_y))
        var_x = np.sum(weights * (x_clean - mean_x) ** 2)
        var_y = np.sum(weights * (y_clean - mean_y) ** 2)
        
        if var_x == 0.0 or var_y == 0.0:
            return 0.0
            
        corr = cov_xy / np.sqrt(var_x * var_y)
        return max(-1.0, min(1.0, corr))
        
    except:
        return 0.0


class MarketRegimeCache:
    """Cache for market regime calculations to avoid redundant computations."""
    
    def __init__(self, hurst_window: int = 720, vr_window: int = 480):
        self.hurst_window = hurst_window
        self.vr_window = vr_window
        
        # Cache for Hurst exponents by asset
        self.hurst_cache: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        # Cache for Variance Ratios by asset
        self.vr_cache: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        # Cache for rolling correlations by pair
        self.corr_cache: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        # Track last calculation indices to avoid redundant work
        self.last_hurst_calc: Dict[str, int] = {}
        self.last_vr_calc: Dict[str, int] = {}
        
    def get_hurst_exponent(self, asset_name: str, index: int, prices: pd.Series) -> float:
        """Get cached or calculate Hurst exponent for asset.
        
        Args:
            asset_name: Unique asset identifier
            index: Current data index
            prices: Price series
            
        Returns:
            Hurst exponent
        """
        cache_key = (asset_name, index)
        
        # Check if already calculated for this index
        if index in self.hurst_cache[asset_name]:
            return self.hurst_cache[asset_name][index]
            
        # Calculate if we have enough data
        if len(prices) >= 10:
            try:
                log_prices = np.log(prices.dropna().values)
                if len(log_prices) >= 10:
                    hurst = _hurst_exponent_jit(log_prices)
                    self.hurst_cache[asset_name][index] = hurst
                    return hurst
            except:
                pass
                
        # Default neutral value
        self.hurst_cache[asset_name][index] = 0.5
        return 0.5
        
    def get_variance_ratio(self, asset_name: str, index: int, prices: pd.Series, k: int = 2) -> float:
        """Get cached or calculate Variance Ratio for asset.
        
        Args:
            asset_name: Unique asset identifier
            index: Current data index
            prices: Price series
            k: Lag parameter
            
        Returns:
            Variance ratio
        """
        cache_key = (asset_name, index, k)
        
        # Check if already calculated for this index
        if index in self.vr_cache[asset_name]:
            return self.vr_cache[asset_name][index]
            
        # Calculate if we have enough data
        if len(prices) >= k * 3:
            try:
                log_prices = np.log(prices.dropna().values)
                if len(log_prices) >= k * 3:
                    vr = _variance_ratio_jit(log_prices, k)
                    self.vr_cache[asset_name][index] = vr
                    return vr
            except:
                pass
                
        # Default neutral value
        self.vr_cache[asset_name][index] = 1.0
        return 1.0
        
    def get_rolling_correlation(self, pair_name: str, index: int, 
                              x_prices: pd.Series, y_prices: pd.Series) -> float:
        """Get cached or calculate rolling correlation for pair.
        
        Args:
            pair_name: Unique pair identifier
            index: Current data index
            x_prices, y_prices: Price series
            
        Returns:
            Correlation coefficient
        """
        # Check if already calculated for this index
        if index in self.corr_cache[pair_name]:
            return self.corr_cache[pair_name][index]
            
        # Calculate correlation
        if len(x_prices) >= 2 and len(y_prices) >= 2:
            try:
                x_vals = x_prices.dropna().values
                y_vals = y_prices.dropna().values
                
                if len(x_vals) >= 2 and len(y_vals) >= 2:
                    # Align arrays
                    min_len = min(len(x_vals), len(y_vals))
                    x_vals = x_vals[-min_len:]
                    y_vals = y_vals[-min_len:]
                    
                    corr = _rolling_correlation_jit(x_vals, y_vals)
                    self.corr_cache[pair_name][index] = corr
                    return corr
            except:
                pass
                
        # Default neutral value
        self.corr_cache[pair_name][index] = 0.0
        return 0.0
        
    def get_exponential_weighted_correlation(self, pair_name: str, index: int, 
                                           x_prices: pd.Series, y_prices: pd.Series, 
                                           alpha: float = 0.1) -> float:
        """Get cached or calculate exponential weighted correlation for pair.
        
        Args:
            pair_name: Unique pair identifier
            index: Current data index
            x_prices, y_prices: Price series
            alpha: Smoothing factor for exponential weighting
            
        Returns:
            Exponential weighted correlation coefficient
        """
        # Use separate cache key for EW correlation
        ew_pair_name = f"{pair_name}_ew_{alpha}"
        
        # Check if already calculated for this index
        if ew_pair_name not in self.corr_cache:
            self.corr_cache[ew_pair_name] = {}
            
        if index in self.corr_cache[ew_pair_name]:
            return self.corr_cache[ew_pair_name][index]
            
        # Calculate EW correlation
        if len(x_prices) >= 2 and len(y_prices) >= 2:
            try:
                x_vals = x_prices.dropna().values
                y_vals = y_prices.dropna().values
                
                if len(x_vals) >= 2 and len(y_vals) >= 2:
                    # Align arrays
                    min_len = min(len(x_vals), len(y_vals))
                    x_vals = x_vals[-min_len:]
                    y_vals = y_vals[-min_len:]
                    
                    corr = _exponential_weighted_correlation_jit(x_vals, y_vals, alpha)
                    self.corr_cache[ew_pair_name][index] = corr
                    return corr
            except:
                pass
                
        # Default neutral value
        self.corr_cache[ew_pair_name][index] = 0.0
        return 0.0
        
    def clear_old_cache(self, current_index: int, keep_last_n: int = 1000):
        """Clear old cache entries to prevent memory bloat.
        
        Args:
            current_index: Current data index
            keep_last_n: Number of recent entries to keep
        """
        cutoff_index = current_index - keep_last_n
        
        # Clear old Hurst cache entries
        for asset_name in self.hurst_cache:
            indices_to_remove = [idx for idx in self.hurst_cache[asset_name] 
                               if idx < cutoff_index]
            for idx in indices_to_remove:
                del self.hurst_cache[asset_name][idx]
                
        # Clear old VR cache entries
        for asset_name in self.vr_cache:
            indices_to_remove = [idx for idx in self.vr_cache[asset_name] 
                               if idx < cutoff_index]
            for idx in indices_to_remove:
                del self.vr_cache[asset_name][idx]
                
        # Clear old correlation cache entries
        for pair_name in self.corr_cache:
            indices_to_remove = [idx for idx in self.corr_cache[pair_name] 
                               if idx < cutoff_index]
            for idx in indices_to_remove:
                del self.corr_cache[pair_name][idx]