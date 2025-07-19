"""Global rolling statistics cache integration for pair backtesting."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from .memory_optimization import (
    GLOBAL_PRICE, GLOBAL_STATS,
    determine_required_windows,
    build_global_rolling_stats,
    get_rolling_stats_view,
    verify_rolling_stats_correctness,
    cleanup_global_data
)
from ..utils.config import AppConfig

logger = logging.getLogger(__name__)


class GlobalRollingStatsManager:
    """Manager for global rolling statistics cache."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the global rolling stats manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.available_windows = set()
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize rolling stats cache from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful
        """
        try:
            self.config = config
            
            # Determine required windows from config
            self.available_windows = determine_required_windows(config)
            
            # Build the global rolling stats cache
            success = build_global_rolling_stats(self.available_windows)
            
            if success:
                self.initialized = True
                logger.info(f"✅ Global rolling stats manager initialized with windows: {sorted(self.available_windows)}")
            else:
                logger.error("❌ Failed to build global rolling stats cache")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing global rolling stats manager: {e}")
            return False
        
    def initialize_from_config(self, config_path: str) -> bool:
        """
        Initialize rolling stats cache from configuration file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            True if initialization successful
        """
        try:
            from ..utils.config import load_config_from_yaml
            
            # Load configuration
            config = load_config_from_yaml(config_path)
            
            # Use the new initialize method
            return self.initialize(config)
            
        except Exception as e:
            logger.error(f"❌ Error initializing from config: {e}")
            return False
    
    def get_pair_rolling_stats(self, symbol1: str, symbol2: str, window: int, 
                              start_idx: Optional[int] = None, 
                              end_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get rolling statistics for a specific pair.
        
        Args:
            symbol1: First symbol in pair
            symbol2: Second symbol in pair
            window: Rolling window size
            start_idx: Start index for time slice
            end_idx: End index for time slice
            
        Returns:
            Dictionary with 'mean1', 'mean2', 'std1', 'std2' arrays
        """
        if not self.initialized:
            raise RuntimeError("Global rolling stats manager not initialized")
            
        if window not in self.available_windows:
            raise ValueError(f"Window {window} not available. Available: {sorted(self.available_windows)}")
            
        try:
            # Get rolling means
            mean1 = get_rolling_stats_view('mean', window, [symbol1], start_idx, end_idx)
            mean2 = get_rolling_stats_view('mean', window, [symbol2], start_idx, end_idx)
            
            # Get rolling stds
            std1 = get_rolling_stats_view('std', window, [symbol1], start_idx, end_idx)
            std2 = get_rolling_stats_view('std', window, [symbol2], start_idx, end_idx)
            
            return {
                'mean1': mean1.flatten(),
                'mean2': mean2.flatten(),
                'std1': std1.flatten(),
                'std2': std2.flatten()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting pair rolling stats: {e}")
            raise
    
    def get_symbol_rolling_stats(self, symbol: str, window: int,
                                start_idx: Optional[int] = None,
                                end_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get rolling statistics for a single symbol.
        
        Args:
            symbol: Symbol name
            window: Rolling window size
            start_idx: Start index for time slice
            end_idx: End index for time slice
            
        Returns:
            Dictionary with 'mean', 'std' arrays
        """
        if not self.initialized:
            raise RuntimeError("Global rolling stats manager not initialized")
            
        if window not in self.available_windows:
            raise ValueError(f"Window {window} not available. Available: {sorted(self.available_windows)}")
            
        try:
            mean = get_rolling_stats_view('mean', window, [symbol], start_idx, end_idx)
            std = get_rolling_stats_view('std', window, [symbol], start_idx, end_idx)
            
            return {
                'mean': mean.flatten(),
                'std': std.flatten()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting symbol rolling stats: {e}")
            raise
    
    def verify_cache_correctness(self, test_symbols: Optional[List[str]] = None, 
                                test_windows: Optional[List[int]] = None) -> bool:
        """
        Verify cache correctness against pandas calculations.
        
        Args:
            test_symbols: Symbols to test (if None, tests first 3 available)
            test_windows: Windows to test (if None, tests all available)
            
        Returns:
            True if all tests pass
        """
        if not self.initialized:
            logger.error("❌ Cannot verify: manager not initialized")
            return False
            
        global GLOBAL_PRICE
        
        if GLOBAL_PRICE is None:
            logger.error("❌ Cannot verify: global price data not available")
            return False
            
        # Default test parameters
        if test_symbols is None:
            test_symbols = list(GLOBAL_PRICE.columns[:3])  # Test first 3 symbols
            
        if test_windows is None:
            test_windows = list(self.available_windows)
            
        logger.info(f"🔍 Verifying cache correctness for {len(test_symbols)} symbols, {len(test_windows)} windows")
        
        all_passed = True
        
        for symbol in test_symbols:
            for window in test_windows:
                try:
                    passed = verify_rolling_stats_correctness(window, symbol, tolerance=1e-6)
                    if not passed:
                        all_passed = False
                        logger.error(f"❌ Verification failed: {symbol}, window={window}")
                except Exception as e:
                    all_passed = False
                    logger.error(f"❌ Verification error for {symbol}, window={window}: {e}")
                    
        if all_passed:
            logger.info("✅ All cache verification tests passed")
        else:
            logger.error("❌ Some cache verification tests failed")
            
        return all_passed
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache information
        """
        global GLOBAL_STATS, GLOBAL_PRICE
        
        info = {
            'initialized': self.initialized,
            'available_windows': sorted(self.available_windows),
            'num_cached_arrays': len(GLOBAL_STATS),
            'total_memory_mb': 0.0,
            'price_data_shape': None,
            'cache_keys': list(GLOBAL_STATS.keys())
        }
        
        if GLOBAL_PRICE is not None:
            info['price_data_shape'] = GLOBAL_PRICE.shape
            
        # Calculate total memory usage
        for arr in GLOBAL_STATS.values():
            info['total_memory_mb'] += arr.nbytes / 1e6
            
        return info


# Global instance for easy access
_global_manager: Optional[GlobalRollingStatsManager] = None


def get_global_rolling_manager() -> GlobalRollingStatsManager:
    """
    Get the global rolling stats manager instance.
    
    Returns:
        Global manager instance
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = GlobalRollingStatsManager()
        
    return _global_manager


def initialize_global_rolling_cache(config: Dict[str, Any]) -> bool:
    """
    Initialize the global rolling cache from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if initialization successful
    """
    manager = get_global_rolling_manager()
    return manager.initialize(config)


def cleanup_global_rolling_cache() -> None:
    """Clean up global rolling cache and reset manager."""
    global _global_manager
    
    # Clean up global data
    cleanup_global_data()
    
    # Reset manager
    if _global_manager is not None:
        _global_manager.initialized = False
        _global_manager.available_windows.clear()
        
    logger.info("🧹 Global rolling cache cleaned up")


def get_cached_rolling_stats(symbol1: str, symbol2: str, window: int,
                            start_idx: Optional[int] = None,
                            end_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Convenience function to get cached rolling stats for a pair.
    
    Args:
        symbol1: First symbol
        symbol2: Second symbol
        window: Rolling window
        start_idx: Start index
        end_idx: End index
        
    Returns:
        Dictionary with rolling statistics
    """
    manager = get_global_rolling_manager()
    return manager.get_pair_rolling_stats(symbol1, symbol2, window, start_idx, end_idx)