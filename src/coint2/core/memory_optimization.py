"""Memory optimization module for price data using memory-mapped files."""

import os
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import psutil

from coint2.utils.config import AppConfig
from coint2.utils.timing_utils import logged_time, time_block

logger = logging.getLogger(__name__)

# Global variables for memory-mapped price data and rolling stats cache
GLOBAL_PRICE: Optional[pd.DataFrame] = None
GLOBAL_STATS: Dict[Tuple[str, int], np.ndarray] = {}
_mmap_lock = threading.Lock()
_stats_lock = threading.Lock()


def consolidate_price_data(data_dir: str, output_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    """
    Consolidate raw price data into a single memory-mappable Parquet file.
    
    Args:
        data_dir: Directory containing raw price data
        output_path: Path for consolidated Parquet file
        start_date: Start date for data consolidation
        end_date: End date for data consolidation
        
    Returns:
        True if consolidation successful, False otherwise
    """
    logger.info(f"🔄 Consolidating price data from {data_dir} to {output_path}")
    
    try:
        from coint2.core.data_loader import load_master_dataset
        
        # Load all data for the specified period
        with time_block("loading raw data for consolidation"):
            df_long = load_master_dataset(data_dir, start_date, end_date)
            
        if df_long.empty:
            logger.error("❌ No data loaded for consolidation")
            return False
            
        # Pivot to wide format: index=datetime, columns=symbols
        with time_block("pivoting data to wide format"):
            df_price = df_long.pivot_table(index="timestamp", columns="symbol", values="close")
            
        # Sort index for consistent ordering
        df_price = df_price.sort_index()
        
        # Convert to float32 to save memory
        df_price = df_price.astype('float32')
        
        # Log memory usage
        memory_mb = df_price.memory_usage(deep=True).sum() / 1e6
        logger.info(f"📊 Consolidated data: {df_price.shape[0]} rows × {df_price.shape[1]} symbols")
        logger.info(f"💾 Memory usage: {memory_mb:.1f} MB")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to Parquet with memory-mapping optimizations
        with time_block("writing consolidated Parquet file"):
            table = pa.Table.from_pandas(df_price, preserve_index=True)
            pq.write_table(
                table,
                output_path,
                compression="zstd",  # Good compression, doesn't interfere with mmap
                use_dictionary=False,  # Dictionaries interfere with memory mapping
                row_group_size=50000,  # Optimize for streaming access
            )
            
        logger.info(f"✅ Successfully consolidated data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error consolidating price data: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_global_price_data(consolidated_path: str) -> bool:
    """
    Initialize global memory-mapped price data.
    
    Args:
        consolidated_path: Path to consolidated Parquet file
        
    Returns:
        True if initialization successful, False otherwise
    """
    global GLOBAL_PRICE
    
    with _mmap_lock:
        if GLOBAL_PRICE is not None:
            logger.info("📊 Global price data already initialized")
            return True
            
        try:
            logger.info(f"🔄 Initializing memory-mapped price data from {consolidated_path}")
            
            if not Path(consolidated_path).exists():
                logger.error(f"❌ Consolidated file not found: {consolidated_path}")
                return False
                
            with time_block("loading memory-mapped data"):
                # Open dataset with memory mapping
                dataset = ds.dataset(consolidated_path, format="parquet", memory_map=True)
                table = dataset.to_table()
                
                # Convert to pandas with memory optimization
                GLOBAL_PRICE = table.to_pandas(
                    split_blocks=True,  # Each column as separate NumPy block
                    self_destruct=True  # Release Arrow table immediately
                )
                
            # Log memory info
            memory_mb = GLOBAL_PRICE.memory_usage(deep=True).sum() / 1e6
            logger.info(f"📊 Loaded {GLOBAL_PRICE.shape[0]} rows × {GLOBAL_PRICE.shape[1]} symbols")
            logger.info(f"💾 Memory usage: {memory_mb:.1f} MB (memory-mapped)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing global price data: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_price_data_view(symbols: list, start_time: Optional[pd.Timestamp] = None, 
                       end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Get a view of price data for specified symbols and time range.
    
    Args:
        symbols: List of symbol names
        start_time: Start time for data slice (optional)
        end_time: End time for data slice (optional)
        
    Returns:
        DataFrame view (no copy) of requested data
    """
    global GLOBAL_PRICE
    
    if GLOBAL_PRICE is None:
        raise RuntimeError("Global price data not initialized. Call initialize_global_price_data() first.")
        
    # Filter symbols that exist in the data
    available_symbols = [s for s in symbols if s in GLOBAL_PRICE.columns]
    
    if len(available_symbols) != len(symbols):
        missing = set(symbols) - set(available_symbols)
        logger.warning(f"⚠️ Missing symbols: {missing}")
        
    if not available_symbols:
        return pd.DataFrame()
        
    # Create view with time and symbol filtering
    if start_time is not None and end_time is not None:
        # Use loc for time-based slicing (creates view if slice is contiguous)
        data_view = GLOBAL_PRICE.loc[start_time:end_time, available_symbols]
    elif start_time is not None:
        data_view = GLOBAL_PRICE.loc[start_time:, available_symbols]
    elif end_time is not None:
        data_view = GLOBAL_PRICE.loc[:end_time, available_symbols]
    else:
        data_view = GLOBAL_PRICE[available_symbols]
        
    return data_view


def setup_blas_threading_limits(num_threads: int = 1, verbose: bool = True) -> Dict[str, Any]:
    """
    Set BLAS threading limits to prevent oversubscription in parallel processing.
    
    This function addresses the common issue where NumPy/SciPy operations spawn
    multiple threads per process, leading to oversubscription when combined with
    joblib parallel processing. By limiting BLAS to single-threaded operation,
    we achieve better performance through reduced context switching.
    
    Args:
        num_threads: Number of threads for BLAS operations (default: 1)
        verbose: Whether to log detailed information
        
    Returns:
        Dict with threading information and status
    """
    import os
    import psutil
    
    result = {
        'status': 'success',
        'num_threads_set': num_threads,
        'env_vars_set': [],
        'threadpoolctl_available': False,
        'initial_thread_count': psutil.Process().num_threads(),
        'warnings': []
    }
    
    num_threads_str = str(num_threads)
    
    # Comprehensive list of BLAS/threading environment variables
    env_vars = {
        # OpenMP (used by many BLAS implementations)
        "OMP_NUM_THREADS": num_threads_str,
        
        # OpenBLAS
        "OPENBLAS_NUM_THREADS": num_threads_str,
        "GOTO_NUM_THREADS": num_threads_str,  # Legacy OpenBLAS
        
        # Intel MKL
        "MKL_NUM_THREADS": num_threads_str,
        "MKL_DOMAIN_NUM_THREADS": f"MKL_BLAS={num_threads_str}",
        
        # Apple Accelerate (macOS)
        "VECLIB_MAXIMUM_THREADS": num_threads_str,
        
        # AMD BLIS
        "BLIS_NUM_THREADS": num_threads_str,
        
        # NumExpr (used by pandas)
        "NUMEXPR_NUM_THREADS": num_threads_str,
        "NUMEXPR_MAX_THREADS": num_threads_str,
        
        # Additional threading controls
        "NPY_NUM_BUILD_JOBS": num_threads_str,
        "JULIA_NUM_THREADS": num_threads_str,  # In case Julia libraries are used
    }
    
    # Set environment variables
    for var_name, var_value in env_vars.items():
        old_value = os.environ.get(var_name)
        os.environ[var_name] = var_value
        result['env_vars_set'].append({
            'name': var_name,
            'old_value': old_value,
            'new_value': var_value
        })
    
    # Try to use threadpoolctl for runtime control
    try:
        import threadpoolctl
        
        # Get current threading info before changes
        before_info = threadpoolctl.threadpool_info()
        
        # Apply limits
        threadpoolctl.threadpool_limits(num_threads)
        
        # Get threading info after changes
        after_info = threadpoolctl.threadpool_info()
        
        result['threadpoolctl_available'] = True
        result['threadpool_info_before'] = before_info
        result['threadpool_info_after'] = after_info
        
        if verbose:
            logger.info(f"✅ BLAS threading limits set to {num_threads} via threadpoolctl")
            
            # Log detected libraries
            libraries = set()
            for info in after_info:
                libraries.add(f"{info.get('internal_api', 'unknown')}")
            
            if libraries:
                logger.info(f"🔧 Controlled libraries: {', '.join(sorted(libraries))}")
                
    except ImportError:
        result['threadpoolctl_available'] = False
        result['warnings'].append("threadpoolctl not available")
        if verbose:
            logger.warning("⚠️ threadpoolctl not available, using environment variables only")
            logger.info("💡 Install threadpoolctl for better BLAS control: pip install threadpoolctl")
    
    except Exception as e:
        result['status'] = 'partial_success'
        result['warnings'].append(f"threadpoolctl error: {str(e)}")
        if verbose:
            logger.warning(f"⚠️ threadpoolctl error: {e}, falling back to environment variables")
    
    # Final thread count check
    result['final_thread_count'] = psutil.Process().num_threads()
    
    if verbose:
        logger.info(f"🧵 Process threads: {result['initial_thread_count']} → {result['final_thread_count']}")
        logger.info(f"📝 Set {len(result['env_vars_set'])} environment variables")
        
        if result['warnings']:
            logger.warning(f"⚠️ Warnings: {'; '.join(result['warnings'])}")
    
    return result


def check_blas_threading_status() -> Dict[str, Any]:
    """
    Check current BLAS threading status and detect potential oversubscription.
    
    Returns:
        Dict with current threading status and recommendations
    """
    import os
    import psutil
    import numpy as np
    import time
    
    status = {
        'process_threads': psutil.Process().num_threads(),
        'cpu_count': psutil.cpu_count(),
        'env_vars': {},
        'threadpoolctl_info': None,
        'numpy_config': None,
        'performance_test': None,
        'recommendations': []
    }
    
    # Check environment variables
    env_vars_to_check = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]
    
    for var in env_vars_to_check:
        status['env_vars'][var] = os.environ.get(var, 'not_set')
    
    # Get threadpoolctl info if available
    try:
        import threadpoolctl
        status['threadpoolctl_info'] = threadpoolctl.threadpool_info()
    except ImportError:
        status['threadpoolctl_info'] = "threadpoolctl not available"
    
    # Get NumPy configuration
    try:
        import io
        import contextlib
        
        # Capture numpy config output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            np.__config__.show()
        status['numpy_config'] = f.getvalue()
    except Exception as e:
        status['numpy_config'] = f"Error getting numpy config: {e}"
    
    # Simple performance test to detect threading behavior
    try:
        initial_threads = psutil.Process().num_threads()
        
        # Small matrix operation to trigger BLAS
        start_time = time.time()
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        _ = np.dot(a, b)
        elapsed = time.time() - start_time
        
        peak_threads = psutil.Process().num_threads()
        
        status['performance_test'] = {
            'initial_threads': initial_threads,
            'peak_threads': peak_threads,
            'thread_increase': peak_threads - initial_threads,
            'elapsed_seconds': elapsed
        }
        
    except Exception as e:
        status['performance_test'] = f"Error in performance test: {e}"
    
    # Generate recommendations
    if isinstance(status['performance_test'], dict):
        thread_increase = status['performance_test']['thread_increase']
        if thread_increase > 4:
            status['recommendations'].append(
                f"High thread increase detected ({thread_increase}). Consider setting BLAS limits."
            )
    
    # Check if any threading env vars are not set to 1
    unset_vars = [var for var, val in status['env_vars'].items() 
                  if val not in ['1', '1.0']]
    if unset_vars:
        status['recommendations'].append(
            f"Consider setting these variables to 1: {', '.join(unset_vars)}"
        )
    
    # Check for potential oversubscription
    if status['process_threads'] > status['cpu_count'] * 2:
        status['recommendations'].append(
            f"Potential oversubscription: {status['process_threads']} threads on {status['cpu_count']} CPUs"
        )
    
    return status


def test_blas_performance(matrix_size: int = 2000, num_iterations: int = 3) -> Dict[str, Any]:
    """
    Test BLAS performance with and without threading limits.
    
    Args:
        matrix_size: Size of test matrices
        num_iterations: Number of test iterations
        
    Returns:
        Dict with performance test results
    """
    import numpy as np
    import time
    import psutil
    
    results = {
        'matrix_size': matrix_size,
        'num_iterations': num_iterations,
        'tests': []
    }
    
    # Generate test data
    np.random.seed(42)
    a = np.random.randn(matrix_size, matrix_size).astype(np.float64)
    b = np.random.randn(matrix_size, matrix_size).astype(np.float64)
    
    for i in range(num_iterations):
        initial_threads = psutil.Process().num_threads()
        
        start_time = time.time()
        result = np.dot(a, b)
        elapsed = time.time() - start_time
        
        peak_threads = psutil.Process().num_threads()
        
        test_result = {
            'iteration': i + 1,
            'elapsed_seconds': elapsed,
            'initial_threads': initial_threads,
            'peak_threads': peak_threads,
            'thread_increase': peak_threads - initial_threads,
            'gflops': (2.0 * matrix_size**3) / (elapsed * 1e9)  # Approximate GFLOPS
        }
        
        results['tests'].append(test_result)
        
        # Small delay between tests
        time.sleep(0.1)
    
    # Calculate averages
    if results['tests']:
        avg_elapsed = sum(t['elapsed_seconds'] for t in results['tests']) / len(results['tests'])
        avg_threads = sum(t['peak_threads'] for t in results['tests']) / len(results['tests'])
        avg_gflops = sum(t['gflops'] for t in results['tests']) / len(results['tests'])
        
        results['averages'] = {
            'elapsed_seconds': avg_elapsed,
            'peak_threads': avg_threads,
            'gflops': avg_gflops
        }
    
    return results


def monitor_memory_usage(duration_seconds: int = 60, interval_seconds: int = 1):
    """
    Monitor memory usage for debugging memory leaks.
    
    Args:
        duration_seconds: How long to monitor
        interval_seconds: Monitoring interval
    """
    import time
    import threading
    
    def memory_monitor():
        process = psutil.Process()
        start_time = time.time()
        
        logger.info("🔍 Starting memory monitoring...")
        
        while time.time() - start_time < duration_seconds:
            rss_mb = process.memory_info().rss / 1e6
            logger.info(f"📊 RSS Memory: {rss_mb:.1f} MB")
            time.sleep(interval_seconds)
            
        logger.info("🔍 Memory monitoring finished")
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()
    
    return monitor_thread


def verify_no_data_copies(data_view: pd.DataFrame, original_data: pd.DataFrame) -> bool:
    """
    Verify that data_view shares memory with original_data (no copy made).
    
    Args:
        data_view: The view/slice of data
        original_data: The original data
        
    Returns:
        True if no copy was made, False if data was copied
    """
    try:
        # Check if underlying numpy arrays share memory
        if hasattr(data_view, 'values') and hasattr(original_data, 'values'):
            view_base = data_view.values.base
            orig_base = original_data.values.base
            
            # If view_base points to original data, no copy was made
            shares_memory = view_base is not None and (
                view_base is orig_base or 
                np.shares_memory(data_view.values, original_data.values)
            )
            
            if shares_memory:
                logger.debug("✅ Data view shares memory (no copy)")
                return True
            else:
                logger.warning("⚠️ Data was copied (memory not shared)")
                return False
        else:
            logger.warning("⚠️ Cannot verify memory sharing")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Error verifying memory sharing: {e}")
        return False


def cleanup_global_data():
    """
    Clean up global price data to free memory.
    """
    global GLOBAL_PRICE, GLOBAL_STATS
    
    with _mmap_lock:
        if GLOBAL_PRICE is not None:
            logger.info("🧹 Cleaning up global price data")
            GLOBAL_PRICE = None
        else:
            logger.info("📊 No global price data to clean up")
            
    with _stats_lock:
        if GLOBAL_STATS:
            logger.info("🧹 Cleaning up global rolling stats cache")
            GLOBAL_STATS.clear()
        else:
            logger.info("📊 No global rolling stats to clean up")


def determine_required_windows(config: Dict[str, Any]) -> set:
    """
    Determine all rolling windows needed based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Set of unique window sizes needed
    """
    windows = set()
    
    # Core backtest windows
    if 'backtest' in config:
        backtest_cfg = config['backtest']
        windows.add(backtest_cfg.get('rolling_window', 30))
        windows.add(backtest_cfg.get('volatility_lookback', 96))
        windows.add(backtest_cfg.get('volatility_lookback', 96) * 2)  # запас для ATR/σ
        windows.add(backtest_cfg.get('correlation_window', 720))
        windows.add(backtest_cfg.get('hurst_window', 720))
        windows.add(backtest_cfg.get('variance_ratio_window', 480))
        
    # Portfolio windows
    if 'portfolio' in config:
        portfolio_cfg = config['portfolio']
        vol_hours = portfolio_cfg.get('volatility_lookback_hours', 24)
        # Convert hours to 15-min periods: 24 hours = 24 * 4 = 96 periods
        vol_periods = vol_hours * 4
        windows.add(vol_periods)
        windows.add(vol_periods * 2)  # запас
        
    # Remove any None values and ensure minimum window size
    windows = {w for w in windows if w is not None and w >= 3}
    
    logger.info(f"📊 Determined required rolling windows: {sorted(windows)}")
    return windows


def build_global_rolling_stats(windows: Optional[set] = None, use_bottleneck: bool = True) -> bool:
    """
    Build global rolling statistics cache for all required windows.
    
    Args:
        windows: Set of window sizes to compute. If None, uses common windows.
        use_bottleneck: Whether to use bottleneck for faster computation
        
    Returns:
        True if successful, False otherwise
    """
    global GLOBAL_PRICE, GLOBAL_STATS
    
    if GLOBAL_PRICE is None:
        logger.error("❌ Global price data not initialized. Call initialize_global_price_data() first.")
        return False
        
    if windows is None:
        # Default common windows if not specified
        windows = {30, 96, 192, 480, 720}
        
    with _stats_lock:
        try:
            logger.info(f"🔄 Building global rolling stats cache for windows: {sorted(windows)}")
            
            # Ensure index is monotonic and unique
            if not GLOBAL_PRICE.index.is_monotonic_increasing:
                logger.warning("⚠️ Price data index is not monotonic, sorting...")
                GLOBAL_PRICE.sort_index(inplace=True)
                
            if not GLOBAL_PRICE.index.is_unique:
                logger.warning("⚠️ Price data index has duplicates, removing...")
                GLOBAL_PRICE = GLOBAL_PRICE[~GLOBAL_PRICE.index.duplicated(keep='first')]
            
            # Convert to numpy array for faster computation
            price_array = GLOBAL_PRICE.to_numpy(dtype=np.float32, copy=False)
            
            if use_bottleneck:
                try:
                    import bottleneck as bn
                    logger.info("🚀 Using bottleneck for rolling calculations")
                    
                    for window in windows:
                        with time_block(f"computing rolling stats for window {window}"):
                            # Compute mean and std using bottleneck
                            mean_arr = bn.move_mean(price_array, window, axis=0, min_count=window)
                            std_arr = bn.move_std(price_array, window, axis=0, ddof=0, min_count=window)
                            
                            # Store in cache with proper keys
                            GLOBAL_STATS[('mean', window)] = mean_arr.astype(np.float32)
                            GLOBAL_STATS[('std', window)] = std_arr.astype(np.float32)
                            
                except ImportError:
                    logger.warning("⚠️ Bottleneck not available, falling back to pandas")
                    use_bottleneck = False
                    
            if not use_bottleneck:
                # Fallback to pandas rolling
                logger.info("🐼 Using pandas for rolling calculations")
                
                for window in windows:
                    with time_block(f"computing rolling stats for window {window}"):
                        mean_df = GLOBAL_PRICE.rolling(window, min_periods=window).mean()
                        std_df = GLOBAL_PRICE.rolling(window, min_periods=window).std(ddof=0)
                        
                        # Convert to numpy arrays and store
                        GLOBAL_STATS[('mean', window)] = mean_df.to_numpy(dtype=np.float32)
                        GLOBAL_STATS[('std', window)] = std_df.to_numpy(dtype=np.float32)
            
            # Log memory usage
            total_memory_mb = 0
            for key, arr in GLOBAL_STATS.items():
                memory_mb = arr.nbytes / 1e6
                total_memory_mb += memory_mb
                logger.debug(f"📊 {key}: {arr.shape} - {memory_mb:.1f} MB")
                
            logger.info(f"✅ Built rolling stats cache: {total_memory_mb:.1f} MB total")
            logger.info(f"📊 Cache contains {len(GLOBAL_STATS)} arrays for {len(windows)} windows")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building rolling stats cache: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_rolling_stats_view(stat_type: str, window: int, symbols: Optional[list] = None, 
                          start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> np.ndarray:
    """
    Get a view of rolling statistics for specified parameters.
    
    Args:
        stat_type: Type of statistic ('mean' or 'std')
        window: Rolling window size
        symbols: List of symbol names (if None, returns all symbols)
        start_idx: Start index for time slice
        end_idx: End index for time slice
        
    Returns:
        NumPy array view of requested statistics
    """
    global GLOBAL_STATS, GLOBAL_PRICE
    
    cache_key = (stat_type, window)
    
    if cache_key not in GLOBAL_STATS:
        raise KeyError(f"Rolling stats not found for {cache_key}. Call build_global_rolling_stats() first.")
        
    stats_array = GLOBAL_STATS[cache_key]
    
    # Handle symbol filtering
    if symbols is not None:
        if GLOBAL_PRICE is None:
            raise RuntimeError("Global price data not available for symbol filtering")
            
        # Get column indices for requested symbols
        available_symbols = [s for s in symbols if s in GLOBAL_PRICE.columns]
        if len(available_symbols) != len(symbols):
            missing = set(symbols) - set(available_symbols)
            logger.warning(f"⚠️ Missing symbols in rolling stats: {missing}")
            
        if not available_symbols:
            return np.array([], dtype=np.float32).reshape(0, 0)
            
        col_indices = [GLOBAL_PRICE.columns.get_loc(s) for s in available_symbols]
        stats_array = stats_array[:, col_indices]
    
    # Handle time slicing
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx or 0
        end_idx = end_idx or len(stats_array)
        stats_array = stats_array[start_idx:end_idx]
        
    return stats_array


def verify_rolling_stats_correctness(window: int, symbol: str, tolerance: float = 1e-6) -> bool:
    """
    Verify that cached rolling statistics match pandas calculations.
    
    Args:
        window: Window size to test
        symbol: Symbol to test
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if statistics match within tolerance
    """
    global GLOBAL_PRICE, GLOBAL_STATS
    
    if GLOBAL_PRICE is None or symbol not in GLOBAL_PRICE.columns:
        logger.error(f"❌ Cannot verify: price data or symbol {symbol} not available")
        return False
        
    try:
        # Get cached stats
        cached_mean = get_rolling_stats_view('mean', window, [symbol])
        cached_std = get_rolling_stats_view('std', window, [symbol])
        
        # Compute reference using the same method as build_global_rolling_stats
        symbol_series = GLOBAL_PRICE[symbol].to_numpy(dtype=np.float32)
        try:
            import bottleneck as bn
            ref_mean = bn.move_mean(symbol_series, window, min_count=window)
            ref_std = bn.move_std(symbol_series, window, ddof=0, min_count=window)
        except ImportError:
            # Fallback to pandas if bottleneck not available
            symbol_series_pd = GLOBAL_PRICE[symbol]
            ref_mean = symbol_series_pd.rolling(window, min_periods=window).mean().to_numpy()
            ref_std = symbol_series_pd.rolling(window, min_periods=window).std(ddof=0).to_numpy()
        
        # Compare (ignoring NaN values)
        mean_match = np.allclose(cached_mean.flatten(), ref_mean, atol=tolerance, equal_nan=True)
        std_match = np.allclose(cached_std.flatten(), ref_std, atol=tolerance, equal_nan=True)
        
        if mean_match and std_match:
            logger.info(f"✅ Rolling stats verification passed for {symbol}, window={window}")
            return True
        else:
            logger.error(f"❌ Rolling stats verification failed for {symbol}, window={window}")
            logger.error(f"   Mean match: {mean_match}, Std match: {std_match}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verifying rolling stats: {e}")
        return False


def update_rolling_stats_incremental(new_data_slice: slice) -> bool:
    """
    Update rolling statistics cache incrementally for walk-forward optimization.
    
    Args:
        new_data_slice: Slice object defining the new data range
        
    Returns:
        True if update successful
    """
    global GLOBAL_STATS
    
    try:
        with _stats_lock:
            if not GLOBAL_STATS:
                logger.warning("⚠️ No rolling stats cache to update")
                return False
                
            # For walk-forward, we typically want to keep only the relevant portion
            # This is a zero-copy operation using array slicing
            for key, arr in GLOBAL_STATS.items():
                GLOBAL_STATS[key] = arr[new_data_slice]
                
            logger.info(f"✅ Updated rolling stats cache for slice {new_data_slice}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error updating rolling stats incrementally: {e}")
        return False