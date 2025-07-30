"""Optimized PairBacktester using global rolling statistics cache."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .base_engine import BasePairBacktester
from ..core.global_rolling_cache import get_global_rolling_manager
from ..core.memory_optimization import GLOBAL_PRICE

logger = logging.getLogger(__name__)


class OptimizedPairBacktester(BasePairBacktester):
    """
    Optimized PairBacktester that uses global rolling statistics cache.
    
    This version eliminates redundant rolling calculations by using
    pre-computed global rolling statistics, providing significant
    performance improvements for large-scale backtesting.
    """
    
    def __init__(self, *args, use_global_cache: bool = True, **kwargs):
        """
        Initialize optimized backtester.
        
        Args:
            use_global_cache: Whether to use global rolling cache
            *args, **kwargs: Arguments passed to parent PairBacktester
        """
        super().__init__(*args, **kwargs)
        self.use_global_cache = use_global_cache
        self.cache_manager = None
        self.symbol1 = None
        self.symbol2 = None
        
        if self.use_global_cache:
            self.cache_manager = get_global_rolling_manager()
            
        # Extract symbol names from pair_data columns
        if not self.pair_data.empty and len(self.pair_data.columns) >= 2:
            # Assume first two columns are the pair symbols
            cols = list(self.pair_data.columns)
            if 'y' in cols and 'x' in cols:
                # Standard format: find original symbol names
                self._extract_symbol_names_from_data()
            else:
                # Direct symbol names
                self.symbol1, self.symbol2 = cols[0], cols[1]
                
    def _extract_symbol_names_from_data(self):
        """
        Extract original symbol names from pair data.
        
        This method attempts to find the original symbol names
        that correspond to the 'y' and 'x' columns in pair_data.
        """
        # This is a placeholder - in practice, symbol names should be
        # passed explicitly or stored as metadata
        # For now, we'll use a fallback approach
        
        global GLOBAL_PRICE
        
        if GLOBAL_PRICE is not None and hasattr(self, 'pair_data'):
            # Try to match data patterns to find original symbols
            # This is a heuristic approach and may not always work
            
            if len(GLOBAL_PRICE.columns) >= 2:
                # Use first two symbols as fallback
                self.symbol1 = GLOBAL_PRICE.columns[0]
                self.symbol2 = GLOBAL_PRICE.columns[1]
                logger.warning(f"⚠️ Using fallback symbols: {self.symbol1}, {self.symbol2}")
            else:
                logger.warning("⚠️ Cannot determine symbol names, global cache disabled")
                self.use_global_cache = False
        else:
            logger.warning("⚠️ Global price data not available, global cache disabled")
            self.use_global_cache = False
            
    def set_symbol_names(self, symbol1: str, symbol2: str):
        """
        Explicitly set symbol names for cache lookup.
        
        Args:
            symbol1: First symbol name
            symbol2: Second symbol name
        """
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        logger.info(f"📊 Set symbol names: {symbol1}, {symbol2}")
        
    def _get_cached_rolling_stats(self, window: int, start_idx: Optional[int] = None, 
                                 end_idx: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get cached rolling statistics for the current pair.
        
        Args:
            window: Rolling window size
            start_idx: Start index for data slice
            end_idx: End index for data slice
            
        Returns:
            Dictionary with cached statistics or None if not available
        """
        if not self.use_global_cache or self.cache_manager is None:
            return None
            
        if self.symbol1 is None or self.symbol2 is None:
            logger.warning("⚠️ Symbol names not set, cannot use global cache")
            return None
            
        try:
            if not self.cache_manager.initialized:
                logger.warning("⚠️ Global cache not initialized")
                return None
                
            stats = self.cache_manager.get_pair_rolling_stats(
                self.symbol1, self.symbol2, window, start_idx, end_idx
            )
            
            logger.debug(f"📊 Retrieved cached rolling stats for {self.symbol1}/{self.symbol2}, window={window}")
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cached rolling stats: {e}")
            return None
    
    def _calculate_rolling_statistics_optimized(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculate rolling statistics using global cache when possible.
        
        Args:
            df: DataFrame with pair data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling statistics added
        """
        # Try to use cached statistics first
        cached_stats = self._get_cached_rolling_stats(window)
        
        if cached_stats is not None:
            # Use cached statistics
            logger.info(f"🚀 Using cached rolling statistics for window={window}")
            
            # Align cached data with current DataFrame
            data_length = len(df)
            
            # Ensure cached arrays match data length
            for key, arr in cached_stats.items():
                if len(arr) != data_length:
                    # Truncate or pad as needed
                    if len(arr) > data_length:
                        cached_stats[key] = arr[-data_length:]
                    else:
                        # Pad with NaN at the beginning
                        padded = np.full(data_length, np.nan, dtype=np.float32)
                        padded[-len(arr):] = arr
                        cached_stats[key] = padded
            
            # Add cached statistics to DataFrame
            df = df.copy()
            df['y_mean'] = cached_stats['mean1']
            df['x_mean'] = cached_stats['mean2']
            df['y_std'] = cached_stats['std1']
            df['x_std'] = cached_stats['std2']
            
            return df
        else:
            # Fallback to original rolling calculations
            logger.info(f"📊 Using original rolling calculations for window={window}")
            return self._calculate_rolling_statistics_original(df, window)
    
    def _calculate_rolling_statistics_original(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Original rolling statistics calculation (fallback).
        
        Args:
            df: DataFrame with pair data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling statistics added
        """
        df = df.copy()
        
        # Calculate rolling means and standard deviations
        df['y_mean'] = df['y'].rolling(window=window, min_periods=window).mean()
        df['x_mean'] = df['x'].rolling(window=window, min_periods=window).mean()
        df['y_std'] = df['y'].rolling(window=window, min_periods=window).std(ddof=0)
        df['x_std'] = df['x'].rolling(window=window, min_periods=window).std(ddof=0)
        
        return df
    
    def run(self) -> None:
        """
        Run optimized backtest using global rolling cache when available.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Логируем начало обработки пары
        logger.info(f"🔄 Начинаем оптимизированный бэктест пары {self.pair_name or 'Unknown'} с {len(self.pair_data)} периодами данных")
        
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            logger.warning(f"⚠️ Пустые данные для пары {self.pair_name or 'Unknown'}, пропускаем")
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return
            
        # Log cache usage
        if self.use_global_cache and self.cache_manager and self.cache_manager.initialized:
            cache_info = self.cache_manager.get_cache_info()
            logger.info(f"🚀 Running optimized backtest with global cache: {cache_info['total_memory_mb']:.1f} MB")
        else:
            logger.info("📊 Running standard backtest (no global cache)")
            
        # Rename columns for convenience
        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x"
            }
        )
        
        # Add optimized rolling statistics
        df = self._calculate_rolling_statistics_optimized(df, self.rolling_window)
        
        # Continue with standard backtest logic
        # Calculate spread using rolling regression
        df['spread'] = np.nan
        df['z_score'] = np.nan
        df['position'] = 0
        df['pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        # Initialize tracking variables
        current_position = 0
        cumulative_pnl = 0.0
        
        # Main backtest loop
        for i in range(self.rolling_window, len(df)):
            # Get rolling window data
            y_window = df['y'].iloc[i - self.rolling_window:i]
            x_window = df['x'].iloc[i - self.rolling_window:i]
            
            # Skip if insufficient data
            if y_window.isna().any() or x_window.isna().any():
                continue
                
            # Calculate rolling regression
            try:
                # Use numpy for faster computation
                y_vals = y_window.values
                x_vals = x_window.values
                
                # Add constant term
                X = np.column_stack([np.ones(len(x_vals)), x_vals])
                
                # OLS regression: y = alpha + beta * x
                coeffs = np.linalg.lstsq(X, y_vals, rcond=None)[0]
                alpha, beta = coeffs[0], coeffs[1]
                
                # Calculate current spread
                current_spread = df['y'].iloc[i] - (alpha + beta * df['x'].iloc[i])
                df.loc[df.index[i], 'spread'] = current_spread
                
                # Calculate z-score using rolling statistics
                if i >= self.rolling_window:
                    spread_window = df['spread'].iloc[i - self.rolling_window:i]
                    spread_mean = spread_window.mean()
                    spread_std = spread_window.std()
                    
                    if spread_std > 0:
                        z_score = (current_spread - spread_mean) / spread_std
                        df.loc[df.index[i], 'z_score'] = z_score
                        
                        # Trading logic
                        new_position = self._determine_position(z_score, current_position)
                        
                        if new_position != current_position:
                            # Position change - calculate PnL
                            if current_position != 0:
                                # Close existing position
                                pnl = self._calculate_position_pnl(
                                    current_position, df.iloc[i], df.iloc[i-1]
                                )
                                cumulative_pnl += pnl
                                df.loc[df.index[i], 'pnl'] = pnl
                                
                            current_position = new_position
                            
                        df.loc[df.index[i], 'position'] = current_position
                        df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                        
            except np.linalg.LinAlgError:
                # Skip if regression fails
                continue
                
        # Store results
        self.results = df[['spread', 'z_score', 'position', 'pnl', 'cumulative_pnl']].copy()
        
        # Log performance summary
        if not self.results.empty:
            total_pnl = self.results['cumulative_pnl'].iloc[-1]
            num_trades = (self.results['position'].diff() != 0).sum()
            logger.info(f"✅ {self.pair_name or 'Unknown'}: Завершен оптимизированный бэктест - PnL: {total_pnl:.4f}, Сделок: {num_trades}")
        else:
            logger.info(f"✅ {self.pair_name or 'Unknown'}: Завершен оптимизированный бэктест - нет результатов")
    
    def _determine_position(self, z_score: float, current_position: int) -> int:
        """
        Determine new position based on z-score and current position.
        
        Args:
            z_score: Current z-score
            current_position: Current position (-1, 0, 1)
            
        Returns:
            New position
        """
        # Entry signals
        if current_position == 0:
            if z_score > self.z_threshold:
                return -1  # Short spread (sell y, buy x)
            elif z_score < -self.z_threshold:
                return 1   # Long spread (buy y, sell x)
                
        # Exit signals - ИСПРАВЛЕНО: правильная логика выхода
        elif current_position == 1:  # Long position
            # Exit long when z_score moves back toward mean (becomes less negative)
            if z_score > -abs(self.z_exit):
                return 0  # Close long position
        elif current_position == -1:  # Short position  
            # Exit short when z_score moves back toward mean (becomes less positive)
            if z_score < abs(self.z_exit):
                return 0  # Close short position
            
        return current_position
    
    def _calculate_position_pnl(self, position: int, current_row: pd.Series, 
                               previous_row: pd.Series) -> float:
        """
        Calculate PnL for position change.
        
        Args:
            position: Position size (-1, 0, 1)
            current_row: Current data row
            previous_row: Previous data row
            
        Returns:
            PnL for the position
        """
        if position == 0:
            return 0.0
            
        # Calculate returns
        y_return = (current_row['y'] - previous_row['y']) / previous_row['y']
        x_return = (current_row['x'] - previous_row['x']) / previous_row['x']
        
        # Position PnL (simplified)
        if position == 1:  # Long spread
            pnl = y_return - x_return
        else:  # Short spread
            pnl = x_return - y_return
            
        # Apply costs
        pnl -= self.commission_pct * 2  # Round-trip commission
        pnl -= self.slippage_pct * 2    # Round-trip slippage
        
        return pnl
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about optimization usage.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'use_global_cache': self.use_global_cache,
            'cache_initialized': False,
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'cache_info': None
        }
        
        if self.cache_manager:
            stats['cache_initialized'] = self.cache_manager.initialized
            if self.cache_manager.initialized:
                stats['cache_info'] = self.cache_manager.get_cache_info()
                
        return stats