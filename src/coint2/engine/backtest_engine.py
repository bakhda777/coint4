import numpy as np
import pandas as pd
import statsmodels.api as sm
import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from functools import lru_cache

from ..core import performance
from .market_regime_cache import MarketRegimeCache


@dataclass
class TradeState:
    """State of an active trade."""
    entry_date: pd.Timestamp
    entry_index: int
    entry_z: float
    entry_spread: float
    position_size: float
    stop_loss_z: float
    capital_at_risk_used: float  # Capital at risk when trade was opened
    entry_price_s1: float
    entry_price_s2: float
    beta: float


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(
        self,
        pair_data: pd.DataFrame,
        rolling_window: int,
        z_threshold: float,
        z_exit: float = 0.0,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        bid_ask_spread_pct_s1: float = 0.001,
        bid_ask_spread_pct_s2: float = 0.001,
        annualizing_factor: int = 365,
        capital_at_risk: float = 1.0,
        stop_loss_multiplier: float = 2.0,
        take_profit_multiplier: float = None,
        cooldown_periods: int = 0,
        wait_for_candle_close: bool = False,
        max_margin_usage: float = float("inf"),
        half_life: float | None = None,
        time_stop_multiplier: float | None = None,
        # NEW: Portfolio integration for position limits
        portfolio = None,
        pair_name: str = "",
        # NEW: Enhanced risk management parameters
        use_kelly_sizing: bool = True,
        max_kelly_fraction: float = 0.25,
        volatility_lookback: int = 96,  # 24 hours for 15-min data
        adaptive_thresholds: bool = True,
        var_confidence: float = 0.05,
        max_var_multiplier: float = 3.0,
        # NEW: Volatility-based position sizing parameters
        volatility_based_sizing: bool = False,
        volatility_lookback_hours: int = 24,
        min_position_size_pct: float = 0.005,
        max_position_size_pct: float = 0.02,
        volatility_adjustment_factor: float = 2.0,
        # Market regime detection parameters
        market_regime_detection: bool = True,
        hurst_window: int = 720,  # 30 days for 15-min data
        hurst_trending_threshold: float = 0.5,
        variance_ratio_window: int = 480,  # 20 days for 15-min data
        variance_ratio_trending_min: float = 1.2,
        variance_ratio_mean_reverting_max: float = 0.8,
        # Structural break protection parameters
        structural_break_protection: bool = True,
        cointegration_test_frequency: int = 2688,  # 7 days for 15-min data
        adf_pvalue_threshold: float = 0.05,
        exclusion_period_days: int = 30,
        max_half_life_days: int = 10,
        min_correlation_threshold: float = 0.6,
        correlation_window: int = 720,  # 30 days for 15-min data
        # Performance optimization parameters
        regime_check_frequency: int = 1,  # Check regime every N bars (1=every bar, 48=every 12 hours)
        use_market_regime_cache: bool = True,  # Enable caching for Hurst/VR calculations
        adf_check_frequency: int = 2688,  # ADF test frequency (separate from cointegration_test_frequency)
        cache_cleanup_frequency: int = 1000,  # Clean cache every N bars
        # NEW: Lazy ADF parameters
        lazy_adf_threshold: float = 0.1,  # Trigger ADF only if correlation change > threshold
        # NEW: Neutral band parameters
        hurst_neutral_band: float = 0.05,  # Neutral band around 0.5 for Hurst
        vr_neutral_band: float = 0.2,  # Neutral band around 1.0 for VR
        # NEW: Exponential weighted correlation parameters
        use_exponential_weighted_correlation: bool = False,  # Use EW correlation instead of rolling
        ew_correlation_alpha: float = 0.1,  # Smoothing factor for EW correlation
        # NEW: Configuration object for accessing config parameters
        config = None,
    ) -> None:
        """Initialize backtester.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with two columns containing price series for the pair.
        rolling_window : int
            Window size for rolling parameter estimation.
        z_threshold : float
            Z-score absolute threshold for entry signals.
        z_exit : float
            Z-score absolute threshold for exit signals (default 0.0).
        commission_pct : float
            Commission percentage for trades.
        slippage_pct : float
            Slippage percentage for trades.
        bid_ask_spread_pct_s1 : float
            Bid-ask spread percentage for first instrument (default 0.001).
        bid_ask_spread_pct_s2 : float
            Bid-ask spread percentage for second instrument (default 0.001).
        cooldown_periods : int
            Number of periods to wait after closing position before re-entering.
        """
        self.pair_data = pair_data.copy()
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.z_exit = z_exit
        self.cooldown_periods = cooldown_periods
        self.results: pd.DataFrame | None = None
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.bid_ask_spread_pct_s1 = bid_ask_spread_pct_s1
        self.bid_ask_spread_pct_s2 = bid_ask_spread_pct_s2
        self.annualizing_factor = annualizing_factor
        self.capital_at_risk = capital_at_risk
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.wait_for_candle_close = wait_for_candle_close
        self.max_margin_usage = max_margin_usage
        self.half_life = half_life
        self.time_stop_multiplier = time_stop_multiplier
        # NEW: Portfolio integration for position limits
        self.portfolio = portfolio
        self.pair_name = pair_name
        # NEW: Enhanced risk management parameters
        self.use_kelly_sizing = use_kelly_sizing
        self.max_kelly_fraction = max_kelly_fraction
        self.volatility_lookback = volatility_lookback
        self.adaptive_thresholds = adaptive_thresholds
        self.var_confidence = var_confidence
        self.max_var_multiplier = max_var_multiplier
        
        # NEW: Enhanced entry/exit rules
        if config is not None:
            self.zscore_entry_threshold = getattr(config, 'zscore_entry_threshold', z_threshold)
            self.min_spread_move_sigma = getattr(config, 'min_spread_move_sigma', 1.2)
            self.min_position_hold_minutes = getattr(config, 'min_position_hold_minutes', 60)
            self.anti_churn_cooldown_minutes = getattr(config, 'anti_churn_cooldown_minutes', 60)
            
            # NEW: Enhanced stop-loss rules
            self.pair_stop_loss_usd = getattr(config, 'pair_stop_loss_usd', 75.0)
            self.pair_stop_loss_zscore = getattr(config, 'pair_stop_loss_zscore', 3.0)
            self.portfolio_daily_stop_pct = getattr(config, 'portfolio_daily_stop_pct', 0.02)
            
            # NEW: Time-based filters
            self.enable_funding_time_filter = getattr(config, 'enable_funding_time_filter', True)
            self.funding_blackout_minutes = getattr(config, 'funding_blackout_minutes', 30)
            self.funding_reset_hours = getattr(config, 'funding_reset_hours', [0, 8, 16])
            self.enable_macro_event_filter = getattr(config, 'enable_macro_event_filter', True)
            self.macro_blackout_minutes = getattr(config, 'macro_blackout_minutes', 30)
            
            # NEW: Pair quarantine system
            self.enable_pair_quarantine = getattr(config, 'enable_pair_quarantine', True)
            self.quarantine_pnl_threshold_sigma = getattr(config, 'quarantine_pnl_threshold_sigma', 3.0)
            self.quarantine_drawdown_threshold_pct = getattr(config, 'quarantine_drawdown_threshold_pct', 0.08)
            self.quarantine_period_days = getattr(config, 'quarantine_period_days', 7)
            self.quarantine_rolling_window_days = getattr(config, 'quarantine_rolling_window_days', 30)
            
            # NEW: Enhanced cost modeling
            self.enable_realistic_costs = getattr(config, 'enable_realistic_costs', True)
            self.commission_rate_per_leg = getattr(config, 'commission_rate_per_leg', 0.0004)
            self.slippage_half_spread_multiplier = getattr(config, 'slippage_half_spread_multiplier', 2.0)
            self.funding_cost_enabled = getattr(config, 'funding_cost_enabled', True)
            
            # NEW: Enhanced position sizing with beta recalculation
            self.beta_recalc_frequency_hours = getattr(config, 'beta_recalc_frequency_hours', 48)
            self.beta_window_days_min = getattr(config, 'beta_window_days_min', 60)
            self.beta_window_days_max = getattr(config, 'beta_window_days_max', 120)
            self.use_ols_beta_sizing = getattr(config, 'use_ols_beta_sizing', True)
        else:
            # Default values when no config is provided
            self.zscore_entry_threshold = z_threshold
            self.min_spread_move_sigma = 1.2
            self.min_position_hold_minutes = 60
            self.anti_churn_cooldown_minutes = 60
            self.pair_stop_loss_usd = 75.0
            self.pair_stop_loss_zscore = 3.0
            self.portfolio_daily_stop_pct = 0.02
            self.enable_funding_time_filter = True
            self.funding_blackout_minutes = 30
            self.funding_reset_hours = [0, 8, 16]
            self.enable_macro_event_filter = True
            self.macro_blackout_minutes = 30
            self.enable_pair_quarantine = True
            self.quarantine_pnl_threshold_sigma = 3.0
            self.quarantine_drawdown_threshold_pct = 0.08
            self.quarantine_period_days = 7
            self.quarantine_rolling_window_days = 30
            self.enable_realistic_costs = True
            self.commission_rate_per_leg = 0.0004
            self.slippage_half_spread_multiplier = 2.0
            self.funding_cost_enabled = True
            self.beta_recalc_frequency_hours = 48
            self.beta_window_days_min = 60
            self.beta_window_days_max = 120
            self.use_ols_beta_sizing = True
        
        # NEW: State tracking for enhanced features
        self.last_beta_recalc_time = None
        self.last_flat_time = None
        self.last_spread_at_flat = None
        self.position_entry_time = None
        self.daily_portfolio_high = None
        self.daily_portfolio_equity = 0.0
        self.quarantined_pairs = {}  # pair_name -> quarantine_end_date
        self.pair_pnl_history = {}  # pair_name -> list of daily PnL
        self.pair_equity_peaks = {}  # pair_name -> peak equity
        self.macro_event_blackout_end = None
        
        # NEW: Enhanced trade logging
        self.complete_trades_log = []  # For open->close cycle logging
        # NEW: Volatility-based position sizing parameters
        self.volatility_based_sizing = volatility_based_sizing
        self.volatility_lookback_hours = volatility_lookback_hours
        self.min_position_size_pct = min_position_size_pct
        self.max_position_size_pct = max_position_size_pct
        self.volatility_adjustment_factor = volatility_adjustment_factor
        # Market regime detection parameters
        self.market_regime_detection = market_regime_detection
        self.hurst_window = hurst_window
        self.hurst_trending_threshold = hurst_trending_threshold
        self.variance_ratio_window = variance_ratio_window
        self.variance_ratio_trending_min = variance_ratio_trending_min
        self.variance_ratio_mean_reverting_max = variance_ratio_mean_reverting_max
        # Structural break protection parameters
        self.structural_break_protection = structural_break_protection
        self.cointegration_test_frequency = cointegration_test_frequency
        self.adf_pvalue_threshold = adf_pvalue_threshold
        # Performance optimization parameters
        self.regime_check_frequency = regime_check_frequency
        self.use_market_regime_cache = use_market_regime_cache
        self.adf_check_frequency = adf_check_frequency
        self.cache_cleanup_frequency = cache_cleanup_frequency
        self.lazy_adf_threshold = lazy_adf_threshold
        self.hurst_neutral_band = hurst_neutral_band
        self.vr_neutral_band = vr_neutral_band
        self.use_exponential_weighted_correlation = use_exponential_weighted_correlation
        self.ew_correlation_alpha = ew_correlation_alpha
        
        # Initialize market regime cache if enabled
        self.market_regime_cache = None
        if self.use_market_regime_cache:
            self.market_regime_cache = MarketRegimeCache(
                hurst_window=self.hurst_window,
                vr_window=self.variance_ratio_window
            )
        
        # Track last regime check index
        self.last_regime_check_index = -1
        # Track last correlation for lazy ADF testing
        self.last_correlation = None
        self.exclusion_period_days = exclusion_period_days
        self.max_half_life_days = max_half_life_days
        self.min_correlation_threshold = min_correlation_threshold
        self.correlation_window = correlation_window
        # Handle empty DataFrame case
        if pair_data.empty or len(pair_data.columns) < 2:
            self.s1 = "s1"  # Default column names
            self.s2 = "s2"
        else:
            self.s1 = pair_data.columns[0]
            self.s2 = pair_data.columns[1]
        self.trades_log: list[dict] = []
        
        # NEW: Risk management state variables
        self.rolling_returns: pd.Series = pd.Series(dtype=float)
        self.rolling_volatility: pd.Series = pd.Series(dtype=float)
        self.kelly_fractions: pd.Series = pd.Series(dtype=float)
        self.adaptive_z_thresholds: pd.Series = pd.Series(dtype=float)
        
        # Market regime detection state variables
        self.hurst_exponents: pd.Series = pd.Series(dtype=float)
        self.variance_ratios: pd.Series = pd.Series(dtype=float)
        self.market_regime: pd.Series = pd.Series(dtype=str)  # 'trending', 'mean_reverting', 'neutral'
        
        # Structural break protection state variables
        self.rolling_correlations: pd.Series = pd.Series(dtype=float)
        self.adf_pvalues: pd.Series = pd.Series(dtype=float)
        self.half_life_estimates: pd.Series = pd.Series(dtype=float)
        self.excluded_pairs: dict = {}  # pair_name -> exclusion_end_date
        self.last_cointegration_test: int = 0
        
        # FIXED: Use OrderedDict with fixed size limit for memory control
        self._ols_cache = OrderedDict()
        self._ols_cache_max_size = 1000  # Fixed limit for 15-minute data
        self._last_window_hash = None
        
        # Incremental backtesting functionality (fixes look-ahead bias)
        self.capital_at_risk_history: pd.Series = pd.Series(dtype=float)
        self.active_trade: Optional[TradeState] = None
        self.incremental_pnl: pd.Series = pd.Series(dtype=float)
        self.incremental_positions: pd.Series = pd.Series(dtype=float)
        self.incremental_trades: pd.Series = pd.Series(dtype=float)
        self.incremental_costs: pd.Series = pd.Series(dtype=float)
        self.cooldown_end_date: Optional[pd.Timestamp] = None
        self.incremental_trades_log: List[Dict] = []
        
        # Validate trading parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate trading parameters for logical consistency."""
        # Check data size vs rolling window (skip for empty data in incremental mode)
        if not self.pair_data.empty and len(self.pair_data) < self.rolling_window + 2:
            raise ValueError(f"Data length ({len(self.pair_data)}) must be at least rolling_window + 2 ({self.rolling_window + 2})")
        
        # Check threshold consistency
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        
        if self.z_exit < 0:
            raise ValueError("z_exit must be non-negative")
            
        if self.z_exit >= self.z_threshold:
            raise ValueError(f"z_exit ({self.z_exit}) must be less than z_threshold ({self.z_threshold}). "
                           "Current configuration will prevent all position entries.")
        
        # Check multipliers consistency
        if self.stop_loss_multiplier <= 0:
            raise ValueError("stop_loss_multiplier must be positive")
            
        if self.take_profit_multiplier is not None and self.take_profit_multiplier <= 0:
            raise ValueError("take_profit_multiplier must be positive")
            
        # FIXED: More flexible take_profit validation - allow values >= 1.0 for some strategies
        if (self.take_profit_multiplier is not None and
            self.take_profit_multiplier <= 0.0):
            raise ValueError(f"take_profit_multiplier ({self.take_profit_multiplier}) must be positive")
        
        # Warning for potentially problematic values
        if (self.take_profit_multiplier is not None and
            self.take_profit_multiplier >= 1.0):
            import warnings
            warnings.warn(f"take_profit_multiplier ({self.take_profit_multiplier}) >= 1.0 may prevent "
                         "take-profit from triggering in mean-reverting strategies", 
                         UserWarning, stacklevel=2)
        
        # Check cost parameters
        if self.commission_pct < 0 or self.slippage_pct < 0:
            raise ValueError("Commission and slippage percentages must be non-negative")
            
        if self.bid_ask_spread_pct_s1 < 0 or self.bid_ask_spread_pct_s2 < 0:
            raise ValueError("Bid-ask spread percentages must be non-negative")
            
        # Check for unrealistically high trading costs (including bid-ask spread)
        total_cost_pct = (self.commission_pct + self.slippage_pct +
                         max(self.bid_ask_spread_pct_s1, self.bid_ask_spread_pct_s2))
        if total_cost_pct > 0.05:  # 5% - more reasonable limit
            raise ValueError(f"Total trading costs ({total_cost_pct:.4f}) exceed 5%. "
                           "This is likely unrealistic for most markets.")
        
        # Check rolling window reasonableness
        if self.rolling_window < 3:
            raise ValueError("rolling_window must be at least 3 for meaningful regression")
        
        if not self.pair_data.empty and self.rolling_window > len(self.pair_data) // 2:
            raise ValueError(f"rolling_window ({self.rolling_window}) is too large relative to data size ({len(self.pair_data)})")
        
        # FIXED: Добавлена проверка на минимальное количество наблюдений для статистической значимости
        if self.rolling_window < 10:
            import warnings
            warnings.warn(f"rolling_window ({self.rolling_window}) is very small and may lead to unreliable statistics", 
                         UserWarning, stacklevel=2)
        
        # FIXED: Проверка на разумность временных параметров
        if (self.half_life is not None and self.time_stop_multiplier is not None and 
            self.half_life * self.time_stop_multiplier < 1.0):
            raise ValueError(f"time_stop_limit ({self.half_life * self.time_stop_multiplier}) is too small - should be at least 1 time unit")
        
        # NEW: Validate enhanced risk management parameters
        if self.max_kelly_fraction <= 0 or self.max_kelly_fraction > 1.0:
            raise ValueError(f"max_kelly_fraction ({self.max_kelly_fraction}) must be between 0 and 1")
        
        if self.volatility_lookback < 10:
            raise ValueError(f"volatility_lookback ({self.volatility_lookback}) must be at least 10 periods")
        
        if self.var_confidence <= 0 or self.var_confidence >= 1.0:
            raise ValueError(f"var_confidence ({self.var_confidence}) must be between 0 and 1")
        
        if self.max_var_multiplier <= 1.0:
            raise ValueError(f"max_var_multiplier ({self.max_var_multiplier}) must be greater than 1.0")

    def _calculate_ols_with_cache(self, y_win: pd.Series, x_win: pd.Series) -> tuple[float, float, float]:
        """Calculate OLS regression with caching for optimization.
        
        FIXED: Improved cache management to prevent memory leaks and secure hashing.
        """
        # FIXED: Use secure hashing with hashlib instead of unsafe str() fallback
        y_array = y_win.values
        x_array = x_win.values
        
        # FIXED: Optimized hashing for better performance
        try:
            # Use array shape and basic statistics for faster hashing
            # This avoids expensive tobytes() operations while maintaining uniqueness
            y_stats = (y_array.mean(), y_array.std(), y_array.min(), y_array.max())
            x_stats = (x_array.mean(), x_array.std(), x_array.min(), x_array.max())
            hash_input = f"{len(y_array)}_{y_stats}_{x_stats}".encode()
        except (AttributeError, ValueError):
            # Fallback to simple length-based hash for degenerate cases
            hash_input = f"{len(y_array)}_{y_array.sum()}_{x_array.sum()}".encode()
        
        window_hash = hashlib.md5(hash_input).hexdigest()
        
        # Check if we have cached results for this window
        if window_hash in self._ols_cache:
            # FIXED: Move to end for LRU behavior
            self._ols_cache.move_to_end(window_hash)
            return self._ols_cache[window_hash]
        
        # Calculate OLS regression
        x_const = sm.add_constant(x_win)
        model = sm.OLS(y_win, x_const).fit()
        
        # FIXED: Check if model has enough parameters to avoid IndexError
        if len(model.params) < 2:
            # For degenerate cases (e.g., constant data), return default values
            beta = 0.0
            spread_win = y_win - beta * x_win
            mean = spread_win.mean() if len(spread_win) > 0 else 0.0
            std_raw = spread_win.std() if len(spread_win) > 0 else 0.0
            # Enhanced protection: use price-based minimum std to avoid unrealistic values
            price_scale = max(y_win.mean(), x_win.mean()) if len(y_win) > 0 and len(x_win) > 0 else 1.0
            min_std = max(1e-8, price_scale * 1e-6)  # Adaptive minimum based on price scale
            std = max(std_raw, min_std)
        else:
            beta = model.params.iloc[1]
            spread_win = y_win - beta * x_win
            mean = spread_win.mean() if len(spread_win) > 0 else 0.0
            std_raw = spread_win.std() if len(spread_win) > 0 else 0.0
            # Enhanced protection: use price-based minimum std to avoid unrealistic values
            price_scale = max(y_win.mean(), x_win.mean()) if len(y_win) > 0 and len(x_win) > 0 else 1.0
            min_std = max(1e-8, price_scale * 1e-6)  # Adaptive minimum based on price scale
            std = max(std_raw, min_std)
        
        # Cache the results
        result = (beta, mean, std)
        
        # FIXED: Use fixed size LRU cache to prevent unlimited memory growth
        if len(self._ols_cache) >= self._ols_cache_max_size:
            # Remove oldest entry (LRU behavior)
            self._ols_cache.popitem(last=False)
        
        self._ols_cache[window_hash] = result
        return result

    def clear_ols_cache(self) -> None:
        """Clear the OLS cache to free memory."""
        self._ols_cache.clear()
        self._last_window_hash = None

    def get_ols_cache_info(self) -> dict:
        """Get information about OLS cache usage.
        
        Returns
        -------
        dict
            Dictionary containing cache size, max size, and hit rate information.
        """
        return {
            "current_size": len(self._ols_cache),
            "max_size": self._ols_cache_max_size,
            "usage_pct": (len(self._ols_cache) / self._ols_cache_max_size) * 100,
            "memory_efficient": len(self._ols_cache) < self._ols_cache_max_size
        }
    
    def _is_profitable_exit(self, current_pnl: float, position: float, price_s1: float, 
                           price_s2: float, beta: float) -> bool:
        """Check if exit would be profitable after accounting for closing costs.
        
        Args:
            current_pnl: Current unrealized PnL of the trade
            position: Current position size (size_s1)
            price_s1: Current price of asset 1
            price_s2: Current price of asset 2
            beta: Current hedge ratio
            
        Returns:
            bool: True if exit would be profitable after costs
        """
        if position == 0:
            return True
            
        # Calculate closing costs
        position_s2 = -position * beta
        notional_s1 = abs(position * price_s1)
        notional_s2 = abs(position_s2 * price_s2)
        
        closing_commission = (notional_s1 + notional_s2) * self.commission_pct
        closing_slippage = (notional_s1 + notional_s2) * self.slippage_pct
        closing_bid_ask = (notional_s1 * self.bid_ask_spread_pct_s1 + 
                          notional_s2 * self.bid_ask_spread_pct_s2)
        
        total_closing_costs = closing_commission + closing_slippage + closing_bid_ask
        
        # Exit is profitable if current PnL exceeds closing costs
        return current_pnl > total_closing_costs

    def _pair_is_tradeable(self, symbol_a: str, symbol_b: str, stats_a: dict, stats_b: dict) -> bool:
        """Check if a pair meets all tradability criteria.
        
        Args:
            symbol_a: First symbol in pair
            symbol_b: Second symbol in pair  
            stats_a: Statistics for first symbol
            stats_b: Statistics for second symbol
            
        Returns:
            bool: True if pair is tradeable, False otherwise
        """
        def get_quote_currency(symbol: str) -> str:
            """Extract quote currency from symbol."""
            if symbol.endswith('USDT'):
                return 'USDT'
            elif symbol.endswith('USDC'):
                return 'USDC'
            elif symbol.endswith('BUSD'):
                return 'BUSD'
            elif symbol.endswith('FDUSD'):
                return 'FDUSD'
            else:
                return 'OTHER'
        
        # 1. Same stable quote currency check
        quote_a = get_quote_currency(symbol_a)
        quote_b = get_quote_currency(symbol_b)
        if quote_a != quote_b:
            return False
        
        # 2. Liquidity check: >= 20M USD daily volume
        if (stats_a.get('vol_usd_24h', 0) < 20_000_000 or 
            stats_b.get('vol_usd_24h', 0) < 20_000_000):
            return False
        
        # 3. Listing age check: >= 30 days
        if (stats_a.get('days_live', 0) < 30 or 
            stats_b.get('days_live', 0) < 30):
            return False
        
        # 4. Funding spread check: use config value
        funding_threshold = getattr(self, 'max_funding_rate_abs', 0.0003)
        if (abs(stats_a.get('funding', 0)) >= funding_threshold or 
            abs(stats_b.get('funding', 0)) >= funding_threshold):
            return False
        
        # 5. Technical conditions
        # Tick size / price: use config value
        tick_threshold = getattr(self, 'max_tick_size_pct', 0.0005)
        if (stats_a.get('tick_pct', 1) >= tick_threshold or 
            stats_b.get('tick_pct', 1) >= tick_threshold):
            return False
        
        # Half-life: use config value
        half_life_threshold = getattr(self, 'max_half_life_hours', 72.0)
        if (stats_a.get('half_life_h', 999) >= half_life_threshold or 
            stats_b.get('half_life_h', 999) >= half_life_threshold):
            return False
        
        return True

    def _calculate_position_size(self, entry_z: float, spread_curr: float, mean: float,
                               std: float, beta: float, price_s1: float, price_s2: float) -> float:
        """Calculate position size based on risk management parameters.
        
        FIXED: Returns single position size (not tuple) for spread trading.
        Position size represents the quantity of the first asset (S1).
        The second asset quantity is automatically calculated as -beta * position_size.
        
        Returns:
            float: Position size for the first asset
        """
        EPSILON = 1e-8
        
        # Count active pairs for risk allocation
        active_pair_count = getattr(self, 'active_pair_count', 1)
        
        # Calculate notional per pair
        notional = self.capital_at_risk / max(active_pair_count, 1)
        
        # Calculate stop loss price for risk calculation
        stop_loss_z = np.sign(entry_z) * self.stop_loss_multiplier
        stop_loss_price = mean + stop_loss_z * std
        
        # Calculate risk per unit (difference between entry and stop loss)
        risk_per_unit = abs(spread_curr - stop_loss_price)
        
        # FIXED: Protection against microscopic risk with min_risk_per_unit
        min_risk_per_unit = max(0.1 * std, EPSILON)
        if risk_per_unit < min_risk_per_unit:
            risk_per_unit = min_risk_per_unit
        
        # Calculate position size based on risk
        if risk_per_unit > EPSILON:
            position_size = notional / risk_per_unit
        else:
            position_size = notional / price_s1 if price_s1 > EPSILON else 0.0
        
        # Apply trade value limit
        trade_value = price_s1 + abs(beta) * price_s2
        if trade_value > EPSILON:
            trade_value_limit = notional / trade_value
            position_size = min(position_size, trade_value_limit)
        
        # Apply margin limits based on total trade value
        # Total trade value = |position_size| * price_s1 + |beta * position_size| * price_s2
        total_trade_value = abs(position_size) * price_s1 + abs(beta * position_size) * price_s2
        if hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage) and total_trade_value > EPSILON:
            margin_limit = self.capital_at_risk * self.max_margin_usage
            if total_trade_value > margin_limit:
                scale_factor = margin_limit / total_trade_value
                position_size *= scale_factor
        
        # Apply volatility-based adjustment if enabled
        if self.volatility_based_sizing:
            volatility_multiplier = self._calculate_volatility_multiplier()
            position_size *= volatility_multiplier
        
        # FIXED: Apply correlation adjustment for position sizing
        # Higher correlation increases risk, so reduce position size
        if hasattr(self, 'pair_data') and len(self.pair_data) >= self.correlation_window:
            s1_prices = self.pair_data.iloc[:, 0].tail(self.correlation_window)
            s2_prices = self.pair_data.iloc[:, 1].tail(self.correlation_window)
            correlation = self._calculate_rolling_correlation(s1_prices, s2_prices, self.correlation_window)
            
            # Apply correlation adjustment: higher correlation -> smaller position
            # Correlation ranges from -1 to 1, we want to reduce size for high positive correlation
            if correlation is not None and not np.isnan(correlation) and correlation > 0.5:  # High positive correlation increases risk
                correlation_adjustment = 1.0 - (correlation - 0.5) * 1.0  # Reduce by up to 50%
                position_size *= correlation_adjustment
        
        # FIXED: Final margin limit check after all adjustments
        # Ensure the final position size still respects margin limits
        if hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage):
            final_total_trade_value = abs(position_size) * price_s1 + abs(beta * position_size) * price_s2
            if final_total_trade_value > EPSILON:
                margin_limit = self.capital_at_risk * self.max_margin_usage
                if final_total_trade_value > margin_limit:
                    final_scale_factor = margin_limit / final_total_trade_value
                    position_size *= final_scale_factor
        
        return position_size

    def _calculate_kelly_fraction(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion fraction for position sizing.
        
        Args:
            returns: Historical returns series
            
        Returns:
            Kelly fraction (capped at max_kelly_fraction)
        """
        if len(returns) < 10:  # Need minimum data
            return self.capital_at_risk
        
        # Remove outliers (beyond 3 sigma)
        mean_return = returns.mean()
        std_return = returns.std()
        filtered_returns = returns[
            (returns >= mean_return - 3 * std_return) & 
            (returns <= mean_return + 3 * std_return)
        ]
        
        if len(filtered_returns) < 5:
            return self.capital_at_risk
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        if len(filtered_returns) == 0:
            return self.capital_at_risk
            
        win_rate = (filtered_returns > 0).mean()
        if win_rate == 0 or win_rate == 1:
            return self.capital_at_risk
        
        avg_win = filtered_returns[filtered_returns > 0].mean() if (filtered_returns > 0).any() else 0
        avg_loss = abs(filtered_returns[filtered_returns < 0].mean()) if (filtered_returns < 0).any() else 1
        
        if avg_loss == 0:
            return self.capital_at_risk
        
        # Kelly fraction
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Cap the Kelly fraction
        kelly_f = max(0, min(kelly_f, self.max_kelly_fraction))
        
        return kelly_f if kelly_f > 0 else self.capital_at_risk
    
    def _calculate_volatility_multiplier(self) -> float:
        """Calculate position size multiplier based on current volatility.
        
        Returns:
            Multiplier for position size (between min_position_size_pct and max_position_size_pct)
        """
        if not hasattr(self, 'pair_data') or len(self.pair_data) < self.volatility_lookback_hours * 4:
            return 1.0  # Default multiplier if insufficient data
        
        # Calculate recent volatility (using 15-min bars, so 4 bars per hour)
        lookback_periods = self.volatility_lookback_hours * 4
        recent_data = self.pair_data.tail(lookback_periods)
        
        # Calculate returns for both assets
        returns_y = recent_data.iloc[:, 0].pct_change().dropna()
        returns_x = recent_data.iloc[:, 1].pct_change().dropna()
        
        if len(returns_y) < 10 or len(returns_x) < 10:
            return 1.0
        
        # Calculate average volatility of both assets
        vol_y = returns_y.std() * np.sqrt(365 * 24 * 4)  # Annualized volatility
        vol_x = returns_x.std() * np.sqrt(365 * 24 * 4)  # Annualized volatility
        avg_volatility = (vol_y + vol_x) / 2
        
        # Calculate historical average volatility for comparison
        if len(self.pair_data) > lookback_periods * 2:
            historical_data = self.pair_data.iloc[:-lookback_periods]
            hist_returns_y = historical_data.iloc[:, 0].pct_change().dropna()
            hist_returns_x = historical_data.iloc[:, 1].pct_change().dropna()
            
            if len(hist_returns_y) > 10 and len(hist_returns_x) > 10:
                hist_vol_y = hist_returns_y.std() * np.sqrt(365 * 24 * 4)
                hist_vol_x = hist_returns_x.std() * np.sqrt(365 * 24 * 4)
                historical_avg_volatility = (hist_vol_y + hist_vol_x) / 2
            else:
                historical_avg_volatility = avg_volatility
        else:
            historical_avg_volatility = avg_volatility
        
        # Avoid division by zero
        if historical_avg_volatility == 0:
            return 1.0
        
        # Calculate volatility ratio (current vs historical)
        volatility_ratio = avg_volatility / historical_avg_volatility
        
        # Apply inverse relationship: higher volatility -> smaller position
        # Use volatility_adjustment_factor to control sensitivity
        if volatility_ratio > 1:
            # Higher than normal volatility -> reduce position size
            multiplier = 1.0 / (1.0 + (volatility_ratio - 1.0) * self.volatility_adjustment_factor)
        else:
            # Lower than normal volatility -> increase position size
            multiplier = 1.0 + (1.0 - volatility_ratio) * self.volatility_adjustment_factor
        
        # Ensure multiplier stays within reasonable bounds
        min_multiplier = self.min_position_size_pct / 0.01  # Relative to base 1% risk
        max_multiplier = self.max_position_size_pct / 0.01  # Relative to base 1% risk
        
        return max(min_multiplier, min(max_multiplier, multiplier))
    
    def _calculate_adaptive_threshold(self, z_scores: pd.Series, current_volatility: float) -> float:
        """Calculate adaptive z-score threshold based on market volatility.
        
        Args:
            z_scores: Historical z-scores
            current_volatility: Current market volatility
            
        Returns:
            Adaptive z-score threshold
        """
        if not self.adaptive_thresholds or len(z_scores) < self.volatility_lookback:
            return self.z_threshold
        
        # Calculate rolling volatility of z-scores
        z_volatility = z_scores.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        if pd.isna(z_volatility) or z_volatility == 0:
            return self.z_threshold
        
        # Adjust threshold based on volatility regime
        # Higher volatility -> higher threshold (more conservative)
        if len(z_scores) == 0:
            return self.z_threshold
        
        rolling_std = z_scores.rolling(window=self.volatility_lookback * 2).std()
        if len(rolling_std.dropna()) == 0:
            return self.z_threshold
        
        base_volatility = rolling_std.median()
        if pd.isna(base_volatility) or base_volatility == 0:
            return self.z_threshold
        
        volatility_ratio = z_volatility / base_volatility
        
        # Scale threshold: 1.0x to max_var_multiplier based on volatility
        volatility_multiplier = min(self.max_var_multiplier, max(1.0, volatility_ratio))
        
        adaptive_threshold = self.z_threshold * volatility_multiplier
        
        return adaptive_threshold
    
    def _calculate_var_position_size(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate position size based on Value at Risk.
        
        Args:
            returns: Historical returns series
            confidence: VaR confidence level
            
        Returns:
            Position size multiplier
        """
        if confidence is None:
            confidence = self.var_confidence
        
        if len(returns) < 20:  # Need minimum data for VaR
            return self.capital_at_risk
        
        # Calculate VaR using historical simulation
        var_percentile = returns.quantile(confidence)
        
        if var_percentile >= 0:  # No downside risk detected
            return self.capital_at_risk
        
        # Target maximum loss of 2% of capital per trade
        target_max_loss = 0.02
        position_size = target_max_loss / abs(var_percentile)
        
        # Cap position size
        position_size = min(position_size, self.capital_at_risk)
        
        return max(0.1, position_size)  # Minimum 10% of base size

    def _calculate_trade_duration(self, current_time, entry_time, current_index: int, entry_index: int) -> float:
        """Calculate trade duration with consistent time units.
        
        FIXED: Unified time handling - always return hours for 15-minute data.
        """
        try:
            if isinstance(current_time, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                # Return duration in hours for datetime indices
                return (current_time - entry_time).total_seconds() / 3600.0
            elif hasattr(current_time, '__sub__') and hasattr(entry_time, '__sub__'):
                # Try to calculate difference for other time-like objects
                diff = current_time - entry_time
                if hasattr(diff, 'total_seconds'):
                    return diff.total_seconds() / 3600.0
                else:
                    # FIXED: For numerical indices, assume 15-minute data
                    # This ensures consistency with time_stop_multiplier
                    return float(diff) * 0.25  # 1 period = 15 minutes = 0.25 hours
            else:
                # FIXED: For non-datetime indices, return periods in hours (15-minute data)
                return float(current_index - entry_index) * 0.25
        except (TypeError, AttributeError, ValueError):
            # FIXED: Enhanced fallback with better error handling (15-minute data)
            return float(current_index - entry_index) * 0.25

    def _enter_position(self, df: pd.DataFrame, i: int, signal: int, z_curr: float,
                       spread_curr: float, mean: float, std: float, beta: float) -> float:
        """Enter a new position with enhanced risk management.
        
        FIXED: Uses corrected position sizing that returns single value.
        FIXED: Added capital sufficiency check before entering position.
        """
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        
        # FIXED: Check capital sufficiency before entering position
        if not self._check_capital_sufficiency(price_s1, price_s2, beta):
            return 0.0
        
        # Calculate base position size
        base_position_size = self._calculate_position_size(z_curr, spread_curr, mean, std, beta, price_s1, price_s2)
        
        if base_position_size == 0:
            return 0.0
        
        # Apply Kelly sizing if enabled and we have sufficient data
        kelly_multiplier = 1.0
        if self.use_kelly_sizing and len(self.rolling_returns) >= 10:
            kelly_fraction = self._calculate_kelly_fraction(self.rolling_returns)
            kelly_multiplier = kelly_fraction / self.capital_at_risk  # Normalize
            
        # Apply VaR-based sizing if we have sufficient data
        var_multiplier = 1.0
        if len(self.rolling_returns) >= 20:
            var_position_size = self._calculate_var_position_size(self.rolling_returns)
            var_multiplier = var_position_size / self.capital_at_risk  # Normalize
        
        # Combine all multipliers
        final_size = base_position_size * kelly_multiplier * var_multiplier
        
        # Ensure minimum position size (at least 10% of base size)
        final_size = max(abs(final_size), abs(base_position_size) * 0.1)
        
        # FIXED: Final margin limit check after all multipliers
        # Ensure the final position size still respects margin limits
        EPSILON = 1e-8
        if hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage):
            final_total_trade_value = abs(final_size) * price_s1 + abs(beta * final_size) * price_s2
            if final_total_trade_value > EPSILON:
                margin_limit = self.capital_at_risk * self.max_margin_usage
                if final_total_trade_value > margin_limit:
                    final_scale_factor = margin_limit / final_total_trade_value
                    final_size *= final_scale_factor
        
        # Apply signal direction
        new_position = signal * final_size

        # Log entry details
        df.loc[df.index[i], "entry_price_s1"] = price_s1
        df.loc[df.index[i], "entry_price_s2"] = price_s2
        df.loc[df.index[i], "entry_z"] = z_curr
        
        # FIXED: Unified time handling for entry date with proper dtype handling
        if isinstance(df.index, pd.DatetimeIndex):
            df.loc[df.index[i], "entry_date"] = df.index[i]
        else:
            df.loc[df.index[i], "entry_date"] = float(i)
            
        return new_position

    def _check_capital_sufficiency(self, price_s1: float, price_s2: float, beta: float) -> bool:
        """Check if there is sufficient capital to enter a position.
        
        FIXED: Added capital sufficiency check before position entry.
        Uses the same logic as _calculate_position_size to ensure consistency.
        
        Args:
            price_s1: Price of first asset
            price_s2: Price of second asset
            beta: Hedge ratio
            
        Returns:
            bool: True if sufficient capital is available
        """
        EPSILON = 1e-8
        
        # Use same logic as _calculate_position_size
        available_capital = self.capital_at_risk
        if hasattr(self, 'portfolio') and self.portfolio is not None:
            available_capital = getattr(self.portfolio, 'available_capital', self.capital_at_risk)
        
        # Count active pairs for risk allocation
        active_pair_count = getattr(self, 'active_pair_count', 1)
        notional = available_capital / max(active_pair_count, 1)
        
        # Calculate trade value per unit
        trade_value = price_s1 + abs(beta) * price_s2
        if trade_value <= EPSILON:
            return False
        
        # Calculate minimum viable position size based on available capital
        # Apply margin limits first to determine maximum allowed trade value
        max_allowed_trade_value = available_capital
        if hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage):
            max_allowed_trade_value = available_capital * self.max_margin_usage
        
        # Calculate minimum meaningful position size (0.01 units)
        min_position_size = 0.01
        min_trade_cost = min_position_size * trade_value
        
        # Check if minimum trade cost exceeds our capital limits
        if min_trade_cost > max_allowed_trade_value:
            return False
        
        # For very expensive trades, apply stricter limits
        capital_usage_pct = min_trade_cost / available_capital
        
        # Reject if minimum position would use more than 50% of capital
        # (This is more lenient since _calculate_position_size will scale down if needed)
        if capital_usage_pct > 0.50:
            return False
        
        # Additional check: reject if trade value per unit is extremely high relative to capital
        trade_value_pct = trade_value / available_capital
        if trade_value_pct > 2.0:  # If one unit costs more than 200% of available capital
            return False
        
        # For very high-value trades, ensure meaningful capital usage
        # Only apply this check for extremely expensive trades (>50% of capital per unit)
        if trade_value > available_capital * 0.5:  # If trade value per unit > 50% of capital
            if capital_usage_pct < 0.01:  # Require at least 1% capital usage
                return False
        
        return True

    def _calculate_trading_costs(self, position_s1_change: float, position_s2_change: float,
                               price_s1: float, price_s2: float) -> tuple[float, float, float, float]:
        """Calculate detailed trading costs with full breakdown.
        
        FIXED: Corrected cost calculation for pair trading.
        For pair trading, we trade both assets simultaneously:
        - S1 position change: position_s1_change
        - S2 position change: -beta * position_s1_change (opposite direction)
        
        Returns:
            tuple: (commission_costs, slippage_costs, bid_ask_costs, total_costs)
        """
        # Calculate notional values for both assets
        notional_change_s1 = abs(position_s1_change * price_s1)
        notional_change_s2 = abs(position_s2_change * price_s2)
        
        # Commission and slippage apply to total notional traded
        total_notional = notional_change_s1 + notional_change_s2
        commission_costs = total_notional * self.commission_pct
        slippage_costs = total_notional * self.slippage_pct
        
        # FIXED: Bid-ask spread cost calculation
        # Apply bid-ask spread to each asset separately based on their individual spreads
        # Use half of the spread to represent crossing cost (entry/exit)
        bid_ask_costs = (notional_change_s1 * self.bid_ask_spread_pct_s1 * 0.5 + 
                        notional_change_s2 * self.bid_ask_spread_pct_s2 * 0.5)
        
        total_costs = commission_costs + slippage_costs + bid_ask_costs
        
        return commission_costs, slippage_costs, bid_ask_costs, total_costs

    def _log_exit(self, df: pd.DataFrame, i: int, z_curr: float, exit_reason: str,
                  entry_datetime, entry_index: int) -> None:
        """FIXED: Extracted method to eliminate code duplication in exit logging."""
        df.loc[df.index[i], "exit_reason"] = exit_reason
        df.loc[df.index[i], "exit_price_s1"] = df["y"].iat[i]
        df.loc[df.index[i], "exit_price_s2"] = df["x"].iat[i]
        df.loc[df.index[i], "exit_z"] = z_curr
        
        # Calculate trade duration with unified time handling
        if entry_datetime is not None:
            df.loc[df.index[i], "trade_duration"] = self._calculate_trade_duration(
                df.index[i], entry_datetime, i, entry_index
            )

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Логируем начало обработки пары
        logger.info(f"🔄 Начинаем бэктест пары {self.pair_name or 'Unknown'} с {len(self.pair_data)} периодами данных")
        
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            logger.warning(f"⚠️ Пустые данные для пары {self.pair_name or 'Unknown'}, пропускаем")
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return

        # Переименовываем столбцы для удобства
        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x",
            }
        ).copy()

        # Prepare columns for rolling parameters
        df["beta"] = np.nan
        df["mean"] = np.nan
        df["std"] = np.nan
        df["spread"] = np.nan
        df["z_score"] = np.nan

        # FIXED: Prevent look-ahead bias by using only historical data
        # Calculate parameters using data up to (but not including) current period
        for i in range(self.rolling_window, len(df)):
            # FIXED: Use only historical data to avoid look-ahead bias
            # Window should end at i-1, not i, to exclude current period
            y_win = df["y"].iloc[i - self.rolling_window : i]
            x_win = df["x"].iloc[i - self.rolling_window : i]
            
            # Use cached OLS calculation
            beta, mean, std = self._calculate_ols_with_cache(y_win, x_win)
            
            # FIXED: Более гибкая обработка низкой волатильности
            # Используем адаптивный порог на основе средних значений цен
            y_slice = df["y"].iloc[i-self.rolling_window:i]
            x_slice = df["x"].iloc[i-self.rolling_window:i]
            
            # Проверяем на пустые срезы перед вычислением mean
            if len(y_slice) == 0 or len(x_slice) == 0:
                continue
            
            price_scale = max(y_slice.mean(), x_slice.mean())
            min_std_threshold = max(1e-6, price_scale * 1e-4)  # Адаптивный порог
            
            if std < min_std_threshold:
                continue  # Skip this iteration, leaving values as NaN
            
            current_spread = df["y"].iat[i] - beta * df["x"].iat[i]
            z = (current_spread - mean) / std
            
            # FIXED: Use iloc for more efficient assignment
            idx = df.index[i]
            df.loc[idx, "beta"] = beta
            df.loc[idx, "mean"] = mean
            df.loc[idx, "std"] = std
            df.loc[idx, "spread"] = current_spread
            df.loc[idx, "z_score"] = z

        # Инициализируем столбцы для результатов
        df["position"] = 0.0
        df["trades"] = 0.0
        df["pnl"] = 0.0
        df["costs"] = 0.0

        # Добавляем столбцы для расширенного логирования
        df["entry_price_s1"] = np.nan
        df["entry_price_s2"] = np.nan
        df["exit_price_s1"] = np.nan
        df["exit_price_s2"] = np.nan
        df["entry_z"] = np.nan
        df["exit_z"] = np.nan
        df["exit_reason"] = ""
        df["trade_duration"] = 0.0
        # FIXED: Use proper dtype for entry_date to avoid warnings
        if isinstance(df.index, pd.DatetimeIndex):
            df["entry_date"] = pd.NaT
        else:
            df["entry_date"] = np.nan
        
        # Enhanced cost tracking columns
        df["commission_costs"] = 0.0
        df["slippage_costs"] = 0.0
        df["bid_ask_costs"] = 0.0
        df["impact_costs"] = 0.0
        
        # NEW: Market regime detection and structural break protection columns
        df["market_regime"] = "neutral"
        df["hurst_exponent"] = np.nan
        df["variance_ratio"] = np.nan
        df["rolling_correlation"] = np.nan
        df["half_life_estimate"] = np.nan
        df["adf_pvalue"] = np.nan
        df["structural_break_detected"] = False

        total_cost_pct = self.commission_pct + self.slippage_pct

        position = 0.0
        entry_z = 0.0
        stop_loss_z = 0.0
        cooldown_remaining = 0  # Tracks remaining cooldown periods

        # Вынесем get_loc вычисления из цикла для оптимизации
        position_col_idx = df.columns.get_loc("position")
        trades_col_idx = df.columns.get_loc("trades")
        costs_col_idx = df.columns.get_loc("costs")
        pnl_col_idx = df.columns.get_loc("pnl")

        # Variables for detailed trade logging
        entry_datetime = None
        entry_spread = 0.0
        entry_position_size = 0.0
        entry_index = 0
        current_trade_pnl = 0.0

        for i in range(1, len(df)):
            if (
                pd.isna(df["spread"].iat[i])
                or pd.isna(df["spread"].iat[i - 1])
                or pd.isna(df["z_score"].iat[i])
            ):
                df.iat[i, position_col_idx] = position
                df.iat[i, trades_col_idx] = 0.0
                df.iat[i, costs_col_idx] = 0.0
                df.iat[i, pnl_col_idx] = 0.0
                continue

            beta = df["beta"].iat[i]
            mean = df["mean"].iat[i]
            std = df["std"].iat[i]
            spread_prev = df["spread"].iat[i - 1]
            spread_curr = df["spread"].iat[i]
            z_curr = df["z_score"].iat[i]

            # Calculate PnL using individual asset returns: pnl = size_s1 * ΔP1 + size_s2 * ΔP2
            # where size_s2 = -beta * size_s1
            price_s1_curr = df["y"].iat[i]
            price_s2_curr = df["x"].iat[i]
            price_s1_prev = df["y"].iat[i - 1]
            price_s2_prev = df["x"].iat[i - 1]
            
            delta_p1 = price_s1_curr - price_s1_prev
            delta_p2 = price_s2_curr - price_s2_prev
            
            # position represents size_s1, size_s2 = -beta * size_s1
            beta = df["beta"].iat[i]
            size_s1 = position
            size_s2 = -beta * size_s1
            
            pnl = size_s1 * delta_p1 + size_s2 * delta_p2

            # NEW: Market regime detection and structural break protection
            market_regime = self._detect_market_regime(i, df)
            structural_break_detected = self._check_structural_breaks(i, df)
            
            # Skip trading if in trending regime or structural break detected
            trading_allowed = True
            if market_regime == 'trending':
                trading_allowed = False
            if structural_break_detected and position != 0:
                # Force close position due to structural break
                new_position = 0.0
                if position != 0:
                    # Log exit due to structural break
                    df.loc[df.index[i], "exit_reason"] = "structural_break"
                    df.loc[df.index[i], "exit_z"] = z_curr
                    df.loc[df.index[i], "exit_price_s1"] = price_s1_curr
                    df.loc[df.index[i], "exit_price_s2"] = price_s2_curr
                    
                    # Calculate trade duration
                    if entry_datetime is not None:
                        if isinstance(df.index, pd.DatetimeIndex):
                            trade_duration = (df.index[i] - entry_datetime).total_seconds() / 3600  # hours
                        else:
                            trade_duration = i - entry_index
                        df.loc[df.index[i], "trade_duration"] = trade_duration
                        
                    # Log detailed trade
                    self.trades_log.append({
                        'entry_date': entry_datetime,
                        'exit_date': df.index[i],
                        'entry_z': entry_z,
                        'exit_z': z_curr,
                        'position_size': entry_position_size,
                        'pnl': current_trade_pnl + pnl,
                        'exit_reason': 'structural_break',
                        'trade_duration': trade_duration if 'trade_duration' in locals() else 0
                    })
                    
                    # Reset trade tracking
                    entry_datetime = None
                    entry_spread = 0.0
                    entry_position_size = 0.0
                    current_trade_pnl = 0.0
                    
            elif not trading_allowed and position == 0:
                # Don't enter new positions in trending regime
                new_position = 0.0
            else:
                new_position = position
                
                # NEW: Check margin limits for existing positions when prices change
                if position != 0 and hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage):
                    # Calculate current total trade value for existing position
                    current_total_trade_value = abs(position) * price_s1_curr + abs(beta * position) * price_s2_curr
                    margin_limit = self.capital_at_risk * self.max_margin_usage
                    
                    # If current position exceeds margin limit, scale it down
                    if current_total_trade_value > margin_limit:
                        scale_factor = margin_limit / current_total_trade_value
                        new_position = position * scale_factor
                        
                        # Log margin limit adjustment
                        if hasattr(self, 'pair_name'):
                            logger.info(f"⚠️ {self.pair_name}: Position scaled down due to margin limit. "
                                      f"Original: {position:.6f}, New: {new_position:.6f}, "
                                      f"Scale factor: {scale_factor:.4f}")
                
            # NEW: Record market regime and structural break analysis results
            idx = df.index[i]
            df.loc[idx, "market_regime"] = market_regime
            df.loc[idx, "structural_break_detected"] = structural_break_detected
            
            # Copy analysis results from state variables if available
            if idx in self.hurst_exponents.index:
                df.loc[idx, "hurst_exponent"] = self.hurst_exponents.at[idx]
            if idx in self.variance_ratios.index:
                df.loc[idx, "variance_ratio"] = self.variance_ratios.at[idx]
            if idx in self.rolling_correlations.index:
                df.loc[idx, "rolling_correlation"] = self.rolling_correlations.at[idx]
            if idx in self.half_life_estimates.index:
                df.loc[idx, "half_life_estimate"] = self.half_life_estimates.at[idx]
            if idx in self.adf_pvalues.index:
                df.loc[idx, "adf_pvalue"] = self.adf_pvalues.at[idx]

            # FIXED: Time-based stop-loss с унифицированными единицами времени
            if (
                position != 0
                and self.half_life is not None
                and self.time_stop_multiplier is not None
                and entry_datetime is not None
            ):
                # FIXED: Unified time handling for consistent duration calculation
                trade_duration_periods = self._calculate_trade_duration_periods(
                    df.index[i], entry_datetime, i, entry_index
                )
                
                # half_life is in periods, so time_stop_limit is also in periods
                time_stop_limit_periods = self.half_life * self.time_stop_multiplier
                if trade_duration_periods >= time_stop_limit_periods:
                    new_position = 0.0
                    cooldown_remaining = self.cooldown_periods
                    # FIXED: Use extracted _log_exit method
                    self._log_exit(df, i, z_curr, 'time_stop', entry_datetime, entry_index)
                    # Close position in portfolio
                    if self.portfolio is not None and position != 0:
                        self.portfolio.close_position(self.pair_name)

            # Уменьшаем cooldown счетчик
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Закрытие позиций в конце теста для чистоты метрик
            if i == len(df) - 1 and position != 0:
                new_position = 0.0  # Форс-закрытие в последнем периоде
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "end_of_test", entry_datetime, entry_index)
                # Close position in portfolio
                if self.portfolio is not None:
                    self.portfolio.close_position(self.pair_name)
            # NEW: Enhanced stop-loss rules - USD-based and Z-score based
            elif position != 0:
                # Calculate unrealized PnL in USD
                price_s1_curr = df["y"].iat[i]
                price_s2_curr = df["x"].iat[i]
                
                # Calculate current position value and unrealized PnL
                if hasattr(self, 'entry_price_s1') and hasattr(self, 'entry_price_s2'):
                    unrealized_pnl = (position * (price_s1_curr - self.entry_price_s1) + 
                                     (-position * beta) * (price_s2_curr - self.entry_price_s2))
                else:
                    unrealized_pnl = current_trade_pnl
                
                # Stop-loss conditions: 75 USDT loss OR 3σ Z-score
                usd_stop_loss = unrealized_pnl <= -75.0
                z_stop_loss = (position > 0 and z_curr <= stop_loss_z) or (position < 0 and z_curr >= stop_loss_z)
                
                if usd_stop_loss or z_stop_loss:
                    new_position = 0.0
                    cooldown_remaining = self.cooldown_periods
                    exit_reason = "usd_stop_loss" if usd_stop_loss else "z_stop_loss"
                    self._log_exit(df, i, z_curr, exit_reason, entry_datetime, entry_index)
                    
                    # Track exit time for protection
                    self.last_position_exit_time = df.index[i]
                    if not isinstance(df.index, pd.DatetimeIndex):
                        self.last_position_exit_index = i
                    
                    # Close position in portfolio
                    if self.portfolio is not None:
                        self.portfolio.close_position(self.pair_name)
            # NEW: Minimum holding time requirement (1 hour)
            elif position != 0 and entry_datetime is not None:
                # Check minimum holding time
                if isinstance(df.index, pd.DatetimeIndex):
                    holding_time_hours = (df.index[i] - entry_datetime).total_seconds() / 3600
                    min_holding_met = holding_time_hours >= (self.min_position_hold_minutes / 60.0)  # Use config value
                else:
                    # For non-datetime index, assume 15-minute periods
                    holding_periods = i - entry_index
                    min_holding_met = holding_periods >= 4  # 4 * 15min = 1 hour
                
                if min_holding_met:
                    # ENHANCED: Take-profit логика с учетом комиссий
                    # Выход при движении z-score к нулю, но только если прибыль покрывает комиссии
                    if (
                        self.take_profit_multiplier is not None
                        and abs(z_curr) <= abs(entry_z) * self.take_profit_multiplier
                        and self._is_profitable_exit(current_trade_pnl, position, price_s1_curr, price_s2_curr, beta)
                    ):
                        new_position = 0.0
                        cooldown_remaining = self.cooldown_periods
                        self._log_exit(df, i, z_curr, "take_profit", entry_datetime, entry_index)
                        
                        # Track exit time for protection
                        self.last_position_exit_time = df.index[i]
                        if not isinstance(df.index, pd.DatetimeIndex):
                            self.last_position_exit_index = i
                        
                        # Close position in portfolio
                        if self.portfolio is not None:
                            self.portfolio.close_position(self.pair_name)
                    # Z-score exit: закрываем позицию если z-score вернулся к заданному уровню
                    elif abs(z_curr) <= self.z_exit:
                        new_position = 0.0
                        cooldown_remaining = self.cooldown_periods
                        self._log_exit(df, i, z_curr, "z_exit", entry_datetime, entry_index)
                        
                        # Track exit time for protection
                        self.last_position_exit_time = df.index[i]
                        if not isinstance(df.index, pd.DatetimeIndex):
                            self.last_position_exit_index = i
                        
                        # Close position in portfolio
                        if self.portfolio is not None:
                            self.portfolio.close_position(self.pair_name)

            # Проверяем сигналы входа только если не в последнем периоде и не в cooldown
            if i < len(df) - 1 and cooldown_remaining == 0:
                # NEW: Enhanced entry rules according to requirements
                # 1. Check if pair is tradeable (basic check - detailed filtering done at pair selection)
                # NOTE: Temporarily disabled until symbol stats are available
                # if not self._pair_is_tradeable(df.index[i]):
                #     new_position = position  # Keep current position
                # else:
                if True:  # Allow trading for now
                    # 2. Calculate adaptive threshold
                    current_threshold = self.z_threshold  # Use configured threshold
                    if self.adaptive_thresholds and i >= self.volatility_lookback:
                        z_scores_window = df["z_score"].iloc[:i+1]
                        current_volatility = z_scores_window.rolling(window=self.volatility_lookback).std().iloc[-1]
                        if not pd.isna(current_volatility):
                            adaptive_threshold = self._calculate_adaptive_threshold(z_scores_window, current_volatility)
                            current_threshold = adaptive_threshold
                    
                    # 3. Check minimum spread movement since last flat (temporarily disabled)
                    spread_movement_ok = True  # Temporarily disable this filter
                    # if hasattr(self, 'last_flat_spread') and self.last_flat_spread is not None:
                    #     spread_change = abs(spread_curr - self.last_flat_spread)
                    #     min_movement = getattr(self, 'min_spread_move_sigma', 1.2) * std  # Use config value
                    #     spread_movement_ok = spread_change >= min_movement
                    
                    # 4. Check minimum time since last position (temporarily disabled)
                    time_protection_ok = True  # Temporarily disable this filter
                    # if hasattr(self, 'last_position_exit_time') and self.last_position_exit_time is not None:
                    #     if isinstance(df.index, pd.DatetimeIndex):
                    #         time_since_exit = (df.index[i] - self.last_position_exit_time).total_seconds() / 3600
                    #         cooldown_hours = self.anti_churn_cooldown_minutes / 60.0
                    #         time_protection_ok = time_since_exit >= cooldown_hours  # Use config value
                    #     else:
                    #         # For non-datetime index, assume 15-minute periods
                    #         periods_since_exit = i - getattr(self, 'last_position_exit_index', 0)
                    #         cooldown_periods = self.anti_churn_cooldown_minutes // 15  # Convert to 15-min periods
                    #         time_protection_ok = periods_since_exit >= cooldown_periods  # Use config value
                    
                    signal = 0
                    if z_curr > current_threshold and spread_movement_ok and time_protection_ok:
                        signal = -1
                    elif z_curr < -current_threshold and spread_movement_ok and time_protection_ok:
                        signal = 1

                    z_prev = df["z_score"].iat[i - 1]
                    long_confirmation = (signal == 1) and (z_curr > z_prev)
                    short_confirmation = (signal == -1) and (z_curr < z_prev)

                    # Check portfolio position limits before entering new position
                    can_enter_new_position = True
                    if self.portfolio is not None and new_position == 0:
                        can_enter_new_position = self.portfolio.can_open_position()
                    
                    if can_enter_new_position and ((new_position == 0 and (long_confirmation or short_confirmation)) or (
                        new_position != 0
                        and (long_confirmation or short_confirmation)
                        and np.sign(new_position) != signal
                    )):
                        new_position = self._enter_position(
                            df, i, signal, z_curr, spread_curr, mean, std, beta
                        )
                        entry_z = z_curr
                        # NEW: Enhanced stop-loss rules (75 USDT or configurable σ)
                        stop_loss_z = float(np.sign(entry_z) * self.pair_stop_loss_zscore)  # Use config value
                        
                        # Track last flat spread for movement calculation
                        flat_threshold = getattr(self, 'flat_zscore_threshold', 0.5)
                        if abs(z_curr) < flat_threshold:  # Consider as "flat" when z-score is small
                            self.last_flat_spread = spread_curr
                        
                        # Register position with portfolio if available
                        if self.portfolio is not None and new_position != 0 and position == 0:
                            self.portfolio.open_position(self.pair_name, {
                                'entry_date': df.index[i],
                                'entry_z': entry_z,
                                'position_size': new_position
                            })
                    else:
                        # Update last flat spread when no position and z-score is small
                        flat_threshold = getattr(self, 'flat_zscore_threshold', 0.5)
                        if position == 0 and abs(z_curr) < flat_threshold:
                            self.last_flat_spread = spread_curr

            # FIXED: Optimized DataFrame operations - calculate values once to avoid redundant operations
            price_s1 = df["y"].iat[i]
            price_s2 = df["x"].iat[i]
            trade_value = price_s1 + abs(beta) * price_s2
            
            # Note: Margin limit is now handled in _calculate_position_size method

            trades = abs(new_position - position)
            position_s1_change = new_position - position
            position_s2_change = -new_position * beta - (-position * beta)

            # Complete trading cost calculation including bid-ask spread
            notional_change_s1 = abs(position_s1_change * price_s1)
            notional_change_s2 = abs(position_s2_change * price_s2)
            
            # Calculate all cost components
            commission_costs = (notional_change_s1 + notional_change_s2) * self.commission_pct
            slippage_costs = (notional_change_s1 + notional_change_s2) * self.slippage_pct
            bid_ask_costs = (notional_change_s1 * self.bid_ask_spread_pct_s1 + 
                           notional_change_s2 * self.bid_ask_spread_pct_s2)
            
            total_costs = commission_costs + slippage_costs + bid_ask_costs

            # PnL after accounting for trading costs
            step_pnl = pnl - total_costs

            # 2. Обновляем основной DataFrame с детализированными издержками
            df.iat[i, position_col_idx] = new_position
            df.iat[i, trades_col_idx] = trades
            df.iat[i, costs_col_idx] = total_costs
            df.iat[i, pnl_col_idx] = step_pnl
            
            # Store detailed cost breakdown
            commission_col_idx = df.columns.get_loc("commission_costs")
            slippage_col_idx = df.columns.get_loc("slippage_costs")
            bid_ask_col_idx = df.columns.get_loc("bid_ask_costs")
            impact_col_idx = df.columns.get_loc("impact_costs")
            
            df.iat[i, commission_col_idx] = commission_costs
            df.iat[i, slippage_col_idx] = slippage_costs
            df.iat[i, bid_ask_col_idx] = bid_ask_costs
            df.iat[i, impact_col_idx] = 0.0   # Not implemented in this model

            # 3. Накапливаем PnL для детального лога (из версии codex)
            # Update trade PnL accumulator if a trade is open
            if position != 0 or (position == 0 and new_position != 0):
                current_trade_pnl += step_pnl

            # Handle entry logging with enhanced tracking
            if position == 0 and new_position != 0:
                entry_datetime = df.index[i]
                entry_spread = spread_curr
                entry_position_size = new_position
                entry_index = i
                
                # Store entry prices for PnL calculation
                self.entry_price_s1 = price_s1_curr
                self.entry_price_s2 = price_s2_curr
                
                # Логируем вход в позицию
                position_type = "LONG" if new_position > 0 else "SHORT"
                logger.info(f"📈 {self.pair_name or 'Unknown'}: Вход в {position_type} позицию на z-score={z_curr:.3f}, размер={abs(new_position):.6f}")
            # Handle exit logging with enhanced cost tracking
            if position != 0 and new_position == 0 and entry_datetime is not None:
                exit_datetime = df.index[i]
                # FIXED: Unified time handling with consistent units
                duration_hours = self._calculate_trade_duration(
                    exit_datetime, entry_datetime, i, entry_index
                )
                
                # Calculate realistic costs for this trade
                realistic_costs = self._calculate_realistic_costs(
                    abs(entry_position_size), price_s1_curr, 
                    abs(entry_position_size * beta), price_s2_curr,
                    (self.bid_ask_spread_pct_s1 + self.bid_ask_spread_pct_s2) / 4,  # spread_half
                    0.0001,  # funding_rate_long (default)
                    0.0001,  # funding_rate_short (default)
                    duration_hours
                )

                trade_info = {
                    "pair": f"{self.s1}-{self.s2}",
                    "entry_datetime": entry_datetime,
                    "exit_datetime": exit_datetime,
                    "position_type": "long" if entry_position_size > 0 else "short",
                    "entry_price_spread": entry_spread,
                    "exit_price_spread": spread_curr,
                    "gross_pnl": current_trade_pnl,
                    "commission_cost": realistic_costs.get('commission', 0),
                    "slippage_cost": realistic_costs.get('slippage', 0),
                    "funding_cost": realistic_costs.get('funding', 0),
                    "net_pnl": current_trade_pnl - sum(realistic_costs.values()),
                    "pnl": current_trade_pnl - sum(realistic_costs.values()),  # Add 'pnl' field for test compatibility
                    "exit_reason": df.loc[df.index[i], "exit_reason"],
                    "trade_duration_hours": duration_hours,
                }
                self.trades_log.append(trade_info)
                
                # Логируем выход из позиции
                exit_reason = df.loc[df.index[i], "exit_reason"] or "z_exit"
                net_pnl = current_trade_pnl - sum(realistic_costs.values())
                pnl_sign = "💰" if net_pnl > 0 else "💸" if net_pnl < 0 else "➖"
                logger.info(f"📉 {self.pair_name or 'Unknown'}: Выход из позиции ({exit_reason}) на z-score={z_curr:.3f}, Net PnL={net_pnl:.4f} {pnl_sign}, длительность={duration_hours:.1f}ч")

                # Сбрасываем счетчики для следующей сделки
                current_trade_pnl = 0.0
                entry_datetime = None
                entry_spread = 0.0
                entry_position_size = 0.0
                
                # NEW: Update rolling returns for Kelly sizing
                if step_pnl != 0 and abs(entry_position_size) > 0:  # Only add non-zero returns and avoid division by zero
                    denominator = abs(entry_position_size) * (price_s1_curr + abs(beta) * price_s2_curr)
                    if denominator > 0:  # Additional safety check
                        trade_return = step_pnl / denominator
                        self.rolling_returns = pd.concat([self.rolling_returns, pd.Series([trade_return])])
                        # Keep only recent returns for efficiency
                        if len(self.rolling_returns) > 200:
                            self.rolling_returns = self.rolling_returns.iloc[-100:]

            # NEW: Update rolling volatility and other risk metrics
            if i > 0 and not pd.isna(step_pnl):
                period_return = step_pnl / max(abs(position * price_s1_curr), 1.0)  # Avoid division by zero
                self.rolling_volatility = pd.concat([self.rolling_volatility, pd.Series([abs(period_return)])])
                if len(self.rolling_volatility) > self.volatility_lookback * 2:
                    self.rolling_volatility = self.rolling_volatility.iloc[-self.volatility_lookback:]

            position = new_position

        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df
        
        # Логируем итоговую статистику по паре
        total_pnl = df["cumulative_pnl"].iloc[-1] if not df.empty else 0.0
        num_trades = len(self.trades_log)
        profitable_trades = len([t for t in self.trades_log if t.get('pnl', 0) > 0])
        win_rate = (profitable_trades / num_trades * 100) if num_trades > 0 else 0.0
        
        logger.info(f"✅ {self.pair_name or 'Unknown'}: Завершен бэктест - PnL: {total_pnl:.4f}, Сделок: {num_trades}, Винрейт: {win_rate:.1f}%")

    def get_results(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest not yet run")

        # Check if required columns exist, if not create them with default values
        required_columns = {
            "trades": 0.0,
            "costs": 0.0,
            "commission_costs": 0.0,
            "slippage_costs": 0.0,
            "bid_ask_costs": 0.0,
            "impact_costs": 0.0,
            "cumulative_pnl": 0.0,
            "y": 0.0,
            "x": 0.0,
            "beta": 0.0
        }
        
        for col, default_value in required_columns.items():
            if col not in self.results.columns:
                if col == "cumulative_pnl" and "pnl" in self.results.columns:
                    # Calculate cumulative PnL from PnL column if it exists
                    self.results[col] = self.results["pnl"].cumsum()
                else:
                    self.results[col] = default_value

        return {
            "spread": self.results["spread"],
            "z_score": self.results["z_score"],
            "position": self.results["position"],
            "trades": self.results["trades"],
            "costs": self.results["costs"],
            "pnl": self.results["pnl"],
            "cumulative_pnl": self.results["cumulative_pnl"],
            "trades_log": self.trades_log,
            # Enhanced cost breakdown
            "commission_costs": self.results["commission_costs"],
            "slippage_costs": self.results["slippage_costs"],
            "bid_ask_costs": self.results["bid_ask_costs"],
            "impact_costs": self.results["impact_costs"],
            # Price and regression data for PnL verification
            "y": self.results["y"],
            "x": self.results["x"],
            "beta": self.results["beta"],
        }

    def get_performance_metrics(self) -> dict:
        if self.results is None or self.results.empty:
            raise ValueError("Backtest has not been run or produced no results")

        pnl = self.results["pnl"].dropna()
        cum_pnl = self.results["cumulative_pnl"].dropna()

        # Calculate trade-related metrics
        num_trades = len([t for t in self.trades_log if t.get('action') == 'open'])
        
        # Calculate average trade duration
        trade_durations = []
        for trade in self.trades_log:
            if trade.get('action') == 'close' and 'duration' in trade:
                trade_durations.append(trade['duration'])
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
        
        # Calculate total return
        total_pnl_val = cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0
        total_return = (total_pnl_val / self.capital_at_risk) * 100 if self.capital_at_risk > 0 else 0.0

        # Если после dropna ничего не осталось, возвращаем нулевые метрики
        if pnl.empty:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "num_trades": num_trades,
                "avg_trade_duration": avg_trade_duration,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "kelly_criterion": 0.0,
            }

        # FIXED: Improved Sharpe Ratio calculation using dynamic capital
        # Use capital_at_risk_history if available, otherwise fall back to fixed capital
        if len(self.capital_at_risk_history) > 0:
            # Align PnL with capital history and calculate returns dynamically
            aligned_capital = self.capital_at_risk_history.reindex(pnl.index, method='ffill')
            # Fill any remaining NaN values with the base capital_at_risk
            aligned_capital = aligned_capital.fillna(self.capital_at_risk)
            # Avoid division by zero
            aligned_capital = aligned_capital.replace(0, self.capital_at_risk)
            returns = pnl / aligned_capital
        else:
            # Fallback to original method if no capital history
            returns = pnl / self.capital_at_risk
            
        sharpe = performance.sharpe_ratio(returns, self.annualizing_factor)
        
        # Calculate new metrics
        win_rate_val = performance.win_rate(pnl)
        expectancy_val = performance.expectancy(pnl)
        kelly_val = performance.kelly_criterion(pnl)

        return {
            "sharpe_ratio": 0.0 if np.isnan(sharpe) else sharpe,
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": total_pnl_val,
            "total_return": total_return,
            "num_trades": num_trades,
            "avg_trade_duration": avg_trade_duration,
            "win_rate": win_rate_val,
            "expectancy": expectancy_val,
            "kelly_criterion": kelly_val,
        }

    # Incremental backtesting methods (fixes look-ahead bias)
    def set_capital_at_risk(self, date: pd.Timestamp, capital: float) -> None:
        """Set capital at risk for a specific date.
        
        This method should be called before processing each day's data
        to ensure the correct capital amount is used for position sizing.
        """
        self.capital_at_risk_history[date] = capital
        
    def get_capital_at_risk_for_date(self, date: pd.Timestamp) -> float:
        """Get the capital at risk that was available on a specific date.
        
        FIXED: Improved logic for finding the correct capital amount.
        """
        if len(self.capital_at_risk_history) == 0:
            return self.capital_at_risk  # Fallback to initial value
            
        # If exact date exists, return it
        if date in self.capital_at_risk_history.index:
            return self.capital_at_risk_history[date]
        
        # Find the most recent date that is <= the requested date
        available_dates = self.capital_at_risk_history.index
        valid_dates = available_dates[available_dates <= date]
        
        if len(valid_dates) > 0:
            # Use the most recent valid date
            most_recent_date = valid_dates.max()
            return self.capital_at_risk_history[most_recent_date]
        else:
            # No historical data before this date, use initial value
            return self.capital_at_risk
             
    def process_single_period(self, date: pd.Timestamp, price_s1: float, price_s2: float) -> Dict:
        """Process a single time period incrementally.
        
        This method fixes look-ahead bias by using only capital available at trade entry time.
        
        Returns:
            Dict with keys: position, trade, pnl, costs, trade_opened, trade_closed
        """
        # Ensure we have enough historical data
        if len(self.pair_data) < self.rolling_window + 1:
            return {
                'position': 0.0,
                'trade': 0.0, 
                'pnl': 0.0,
                'costs': 0.0,
                'trade_opened': False,
                'trade_closed': False
            }
            
        # Calculate current market parameters
        recent_data = self.pair_data.tail(self.rolling_window)
        if len(recent_data) < self.rolling_window:
            return {
                'position': 0.0,
                'trade': 0.0,
                'pnl': 0.0, 
                'costs': 0.0,
                'trade_opened': False,
                'trade_closed': False
            }
            
        y_win = recent_data.iloc[:, 0]
        x_win = recent_data.iloc[:, 1]
        
        try:
            beta, mean, std = self._calculate_ols_with_cache(y_win, x_win)
        except (ValueError, np.linalg.LinAlgError):
            return {
                'position': 0.0,
                'trade': 0.0,
                'pnl': 0.0,
                'costs': 0.0, 
                'trade_opened': False,
                'trade_closed': False
            }
            
        # Check for valid parameters
        if not (np.isfinite(beta) and np.isfinite(mean) and np.isfinite(std) and std > 1e-6):
            return {
                'position': 0.0,
                'trade': 0.0,
                'pnl': 0.0,
                'costs': 0.0,
                'trade_opened': False, 
                'trade_closed': False
            }
            
        current_spread = price_s1 - beta * price_s2
        z_score = (current_spread - mean) / std
        
        # Initialize result
        result = {
            'position': 0.0,
            'trade': 0.0,
            'pnl': 0.0,
            'costs': 0.0,
            'trade_opened': False,
            'trade_closed': False,
            'z_score': z_score,
            'spread': current_spread,
            'beta': beta,
            'mean': mean,
            'std': std
        }
        
        # Calculate P&L if we have an active trade
        if self.active_trade is not None:
            prev_spread = self.active_trade.entry_spread
            if date in self.incremental_pnl.index and len(self.incremental_pnl) > 0:
                # Get previous spread from the last available data
                if len(self.pair_data) >= 2:
                    prev_data = self.pair_data.iloc[-2]
                    prev_spread = prev_data.iloc[0] - self.active_trade.beta * prev_data.iloc[1]
                elif len(self.pair_data) >= 1:
                    prev_data = self.pair_data.iloc[-1]
                    prev_spread = prev_data.iloc[0] - self.active_trade.beta * prev_data.iloc[1]
                
            # FIXED: Calculate PnL using spread change for pair trading
            # For pair trading, PnL = position_size * (current_spread - entry_spread)
            # This correctly captures the profit from spread convergence/divergence
            current_spread = price_s1 - self.active_trade.beta * price_s2
            spread_change = current_spread - self.active_trade.entry_spread
            
            pnl = self.active_trade.position_size * spread_change
            result['pnl'] = pnl
            result['position'] = self.active_trade.position_size
            
        # Check exit conditions
        trade_closed = False
        if self.active_trade is not None:
            # FIXED: Corrected stop loss logic based on Z-score movement
            # Stop loss triggers when Z-score moves further away from mean (against our position)
            # For long spread (positive position): entered when z < -threshold, stop when z becomes even more negative
            # For short spread (negative position): entered when z > +threshold, stop when z becomes even more positive
            stop_loss_triggered = False
            if self.active_trade.position_size > 0:  # Long spread position
                # Long spread: stop if Z-score becomes more negative than stop_loss_z
                stop_loss_triggered = z_score <= self.active_trade.stop_loss_z
            else:  # Short spread position
                # Short spread: stop if Z-score becomes more positive than stop_loss_z
                stop_loss_triggered = z_score >= self.active_trade.stop_loss_z
                
            if stop_loss_triggered:
                self._close_trade(date, z_score, 'stop_loss')
                trade_closed = True
                result['trade_closed'] = True
                result['stop_loss_triggered'] = True
                
            # FIXED: Take profit logic - exit when Z-score moves toward mean
            # Take profit when Z-score has moved a certain fraction back toward zero
            elif (self.take_profit_multiplier is not None and
                  abs(z_score) <= abs(self.active_trade.entry_z) * (1 - self.take_profit_multiplier)):
                self._close_trade(date, z_score, 'take_profit')
                trade_closed = True
                result['trade_closed'] = True
                
            # Z-score exit
            elif abs(z_score) <= self.z_exit:
                self._close_trade(date, z_score, 'z_exit')
                trade_closed = True
                result['trade_closed'] = True
                
            # FIXED: Time stop with proper frequency handling
            elif (self.half_life is not None and self.time_stop_multiplier is not None):
                # Calculate trade duration in periods using index positions
                try:
                    current_index = self.pair_data.index.get_loc(date)
                except KeyError:
                    # If exact date not found, use the last available index
                    current_index = len(self.pair_data) - 1
                trade_duration_periods = current_index - self.active_trade.entry_index
                
                # half_life is in periods, so time_stop_limit is also in periods
                time_stop_limit_periods = self.half_life * self.time_stop_multiplier
                if trade_duration_periods >= time_stop_limit_periods:
                    self._close_trade(date, z_score, 'time_stop')
                    trade_closed = True
                    result['trade_closed'] = True
                    
        # Check entry conditions (only if no active trade, not in cooldown, and no stop loss triggered this period)
        if (self.active_trade is None and 
            (self.cooldown_end_date is None or date > self.cooldown_end_date) and
            not result.get('stop_loss_triggered', False)):
            
            signal = 0
            if z_score > self.z_threshold:
                signal = -1
            elif z_score < -self.z_threshold:
                signal = 1
                
            if signal != 0:
                # FIXED: Check capital sufficiency before calculating position size
                if not self._check_capital_sufficiency(price_s1, price_s2, beta):
                    # Insufficient capital, skip this trade
                    pass
                else:
                    # Get capital at risk for this specific date
                    capital_for_trade = self.get_capital_at_risk_for_date(date)
                    
                    # Calculate position size using capital available at trade entry
                    old_capital = self.capital_at_risk
                    self.capital_at_risk = capital_for_trade
                    position_size = self._calculate_position_size(
                        z_score, current_spread, mean, std, beta, price_s1, price_s2
                    )
                    self.capital_at_risk = old_capital  # Restore original
                    
                    # FIXED: Final margin limit check for incremental backtesting
                    # Ensure the position size respects margin limits
                    EPSILON = 1e-8
                    if hasattr(self, 'max_margin_usage') and np.isfinite(self.max_margin_usage) and position_size > 0:
                        total_trade_value = abs(position_size) * price_s1 + abs(beta * position_size) * price_s2
                        if total_trade_value > EPSILON:
                            margin_limit = capital_for_trade * self.max_margin_usage
                            if total_trade_value > margin_limit:
                                scale_factor = margin_limit / total_trade_value
                                position_size *= scale_factor
                    
                    if position_size > 0:
                        self._open_trade(date, z_score, current_spread, signal * position_size,
                                       capital_for_trade, price_s1, price_s2, beta)
                        result['position'] = signal * position_size
                        result['trade'] = abs(signal * position_size)
                        result['trade_opened'] = True
                    
        # Update incremental series
        self.incremental_pnl[date] = result['pnl']
        self.incremental_positions[date] = result['position']
        self.incremental_trades[date] = result['trade']
        self.incremental_costs[date] = result['costs']
        
        return result
         
    def _open_trade(self, date: pd.Timestamp, entry_z: float, entry_spread: float,
                   position_size: float, capital_used: float, price_s1: float, 
                   price_s2: float, beta: float) -> None:
        """Open a new trade."""
        # FIXED: Correct stop loss calculation
        # For long positions (entry_z < 0): stop_loss_z should be more negative (entry_z - stop_loss_multiplier)
        # For short positions (entry_z > 0): stop_loss_z should be more positive (entry_z + stop_loss_multiplier)
        if entry_z < 0:  # Long position
            stop_loss_z = entry_z - abs(self.stop_loss_multiplier)
        else:  # Short position
            stop_loss_z = entry_z + abs(self.stop_loss_multiplier)
        
        # Get correct entry index for the current date
        try:
            entry_index = self.pair_data.index.get_loc(date)
        except KeyError:
            # If exact date not found, use the last available index
            entry_index = len(self.pair_data) - 1
        
        self.active_trade = TradeState(
            entry_date=date,
            entry_index=entry_index,
            entry_z=entry_z,
            entry_spread=entry_spread,
            position_size=position_size,
            stop_loss_z=stop_loss_z,
            capital_at_risk_used=capital_used,
            entry_price_s1=price_s1,
            entry_price_s2=price_s2,
            beta=beta
        )
        
        # Log trade opening
        self.incremental_trades_log.append({
            'action': 'open',
            'date': date,
            'entry_z': entry_z,
            'position_size': position_size,
            'capital_used': capital_used,
            'entry_price_s1': price_s1,
            'entry_price_s2': price_s2,
            'beta': beta
        })
        
    def _close_trade(self, date: pd.Timestamp, exit_z: float, exit_reason: str) -> None:
        """Close the active trade."""
        if self.active_trade is None:
            return
            
        # Calculate trade PnL
        trade_pnl = 0.0
        if hasattr(self.active_trade, 'entry_date') and self.active_trade.entry_date in self.incremental_pnl.index:
            # Sum PnL from entry date to current date
            entry_date = self.active_trade.entry_date
            pnl_slice = self.incremental_pnl.loc[entry_date:date]
            trade_pnl = pnl_slice.sum() if not pnl_slice.empty else 0.0
            
        # Log trade closing
        self.incremental_trades_log.append({
            'action': 'close',
            'date': date,
            'exit_z': exit_z,
            'exit_reason': exit_reason,
            'position_size': self.active_trade.position_size,
            'capital_used': self.active_trade.capital_at_risk_used,
            'entry_date': self.active_trade.entry_date,
            'pnl': trade_pnl
        })
        
        # Set cooldown
        if self.cooldown_periods > 0:
            # Assuming 15-minute data, calculate cooldown end date
            cooldown_timedelta = pd.Timedelta(minutes=15 * self.cooldown_periods)
            self.cooldown_end_date = date + cooldown_timedelta
            
        self.active_trade = None
        
    def get_incremental_results(self) -> Dict:
        """Get results from incremental processing."""
        total_pnl = self.incremental_pnl.sum() if len(self.incremental_pnl) > 0 else 0.0
        trade_count = len([t for t in self.incremental_trades_log if t['action'] == 'open'])
        
        return {
            'pnl': self.incremental_pnl,
            'position': self.incremental_positions,
            'trades': self.incremental_trades,
            'costs': self.incremental_costs,
            'trades_log': self.incremental_trades_log,
            'total_pnl': total_pnl,
            'trade_count': trade_count
        }
        
    def reset_incremental_state(self) -> None:
        """Reset incremental state for a new backtest."""
        self.active_trade = None
        self.cooldown_end_date = None
        self.incremental_pnl = pd.Series(dtype=float)
        self.incremental_positions = pd.Series(dtype=float)
        self.incremental_trades = pd.Series(dtype=float)
        self.incremental_costs = pd.Series(dtype=float)
        self.incremental_trades_log = []
        self.capital_at_risk_history = pd.Series(dtype=float)
        
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst Exponent for market regime detection.
        
        Returns:
            float: Hurst exponent (H > 0.5 = trending, H < 0.5 = mean-reverting)
        """
        if len(prices) < 10:
            return 0.5  # Neutral if insufficient data
            
        try:
            # Calculate log returns
            log_prices = np.log(prices.dropna())
            if len(log_prices) < 10:
                return 0.5
                
            # Calculate cumulative deviations from mean
            mean_log_price = log_prices.mean()
            cumulative_deviations = (log_prices - mean_log_price).cumsum()
            
            # Calculate range (R)
            R = cumulative_deviations.max() - cumulative_deviations.min()
            
            # Calculate standard deviation (S)
            S = log_prices.std()
            
            if S == 0 or R == 0:
                return 0.5
                
            # Calculate R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent approximation: H ≈ log(R/S) / log(n)
            n = len(log_prices)
            hurst = np.log(rs_ratio) / np.log(n)
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except (ValueError, ZeroDivisionError):
            return 0.5
            
    def _calculate_variance_ratio(self, prices: pd.Series, k: int = 2) -> float:
        """Calculate Variance Ratio Test for market regime detection.
        
        Args:
            prices: Price series
            k: Lag parameter (default 2)
            
        Returns:
            float: Variance ratio (VR > 1.2 = trending, VR < 0.8 = mean-reverting)
        """
        if len(prices) < k * 3:
            return 1.0  # Neutral if insufficient data
            
        try:
            # Calculate log returns
            log_prices = np.log(prices.dropna())
            returns = log_prices.diff().dropna()
            
            if len(returns) < k * 2:
                return 1.0
                
            # Calculate k-period returns
            k_returns = log_prices.diff(k).dropna()
            
            if len(k_returns) == 0 or len(returns) == 0:
                return 1.0
                
            # Calculate variances
            var_1 = returns.var()
            var_k = k_returns.var()
            
            if var_1 == 0:
                return 1.0
                
            # Variance ratio
            vr = var_k / (k * var_1)
            
            return max(0.1, min(3.0, vr))  # Clamp to reasonable range
            
        except (ValueError, ZeroDivisionError):
            return 1.0
            
    def _calculate_rolling_correlation(self, s1: pd.Series, s2: pd.Series, window: int) -> float:
        """Calculate rolling correlation between two series.
        
        Returns:
            float: Correlation coefficient
        """
        if len(s1) < window or len(s2) < window:
            return 0.0
            
        try:
            corr = s1.tail(window).corr(s2.tail(window))
            return corr if not pd.isna(corr) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
            
    def _calculate_half_life(self, spread: pd.Series, window: int) -> float:
        """Calculate half-life of mean reversion for spread.
        
        Returns:
            float: Half-life in periods
        """
        if len(spread) < window:
            return float('inf')
            
        try:
            spread_window = spread.tail(window).dropna()
            if len(spread_window) < 10:
                return float('inf')
                
            # Calculate lagged spread
            spread_lag = spread_window.shift(1).dropna()
            spread_current = spread_window[1:]
            
            if len(spread_lag) == 0 or len(spread_current) == 0:
                return float('inf')
                
            # OLS regression: spread_t = alpha + beta * spread_{t-1} + error
            X = sm.add_constant(spread_lag)
            y = spread_current
            
            try:
                model = sm.OLS(y, X).fit()
                beta = model.params.iloc[1] if len(model.params) > 1 else 0.0
                
                if beta >= 1.0 or beta <= 0.0:
                    return float('inf')
                    
                # Half-life = -ln(2) / ln(beta)
                half_life = -np.log(2) / np.log(beta)
                return max(0.1, min(1000.0, half_life))  # Clamp to reasonable range
                
            except:
                return float('inf')
                
        except (ValueError, ZeroDivisionError):
            return float('inf')
            
    def _perform_adf_test(self, spread: pd.Series, window: int) -> float:
        """Perform Augmented Dickey-Fuller test on spread.
        
        Returns:
            float: p-value of ADF test
        """
        if len(spread) < window:
            return 1.0  # High p-value indicates non-stationarity
            
        try:
            from statsmodels.tsa.stattools import adfuller
            
            spread_window = spread.tail(window).dropna()
            if len(spread_window) < 10:
                return 1.0
                
            # Perform ADF test
            result = adfuller(spread_window, maxlag=min(12, len(spread_window)//3))
            p_value = result[1]
            
            return max(0.0, min(1.0, p_value))
            
        except:
            return 1.0
            
    def _detect_market_regime(self, i: int, df: pd.DataFrame) -> str:
        """Detect current market regime based on Hurst Exponent and Variance Ratio.
        
        Returns:
            str: 'trending', 'mean_reverting', or 'neutral'
        """
        if not self.market_regime_detection or i < self.hurst_window:
            return 'neutral'
            
        # Check if we need to recalculate regime (frequency optimization)
        if (i - self.last_regime_check_index) < self.regime_check_frequency:
            # Return last calculated regime if available
            idx = df.index[i]
            if idx in self.market_regime.index and not pd.isna(self.market_regime.loc[idx]):
                return self.market_regime.loc[idx]
            # If no previous regime, return neutral
            return 'neutral'
            
        try:
            # Update last check index
            self.last_regime_check_index = i
            
            # Get price series for analysis
            s1_prices = df['y'].iloc[max(0, i - self.hurst_window):i]
            s2_prices = df['x'].iloc[max(0, i - self.hurst_window):i]
            
            # Calculate Hurst Exponent for both assets (with caching if enabled)
            if self.use_market_regime_cache and self.market_regime_cache:
                # Use cached calculations
                asset1_name = f"{self.pair_name}_y"
                asset2_name = f"{self.pair_name}_x"
                
                hurst_s1 = self.market_regime_cache.get_hurst_exponent(asset1_name, i, s1_prices)
                hurst_s2 = self.market_regime_cache.get_hurst_exponent(asset2_name, i, s2_prices)
                
                vr_s1 = self.market_regime_cache.get_variance_ratio(asset1_name, i, s1_prices)
                vr_s2 = self.market_regime_cache.get_variance_ratio(asset2_name, i, s2_prices)
                
                # Periodic cache cleanup
                if i % self.cache_cleanup_frequency == 0:
                    self.market_regime_cache.clear_old_cache(i)
            else:
                # Use original calculations
                hurst_s1 = self._calculate_hurst_exponent(s1_prices)
                hurst_s2 = self._calculate_hurst_exponent(s2_prices)
                vr_s1 = self._calculate_variance_ratio(s1_prices)
                vr_s2 = self._calculate_variance_ratio(s2_prices)
            
            avg_hurst = (hurst_s1 + hurst_s2) / 2
            avg_vr = (vr_s1 + vr_s2) / 2
            
            # Store values for analysis
            idx = df.index[i]
            self.hurst_exponents.at[idx] = avg_hurst
            self.variance_ratios.at[idx] = avg_vr
            
            # Determine regime based on both indicators with neutral bands
            hurst_upper = self.hurst_trending_threshold + self.hurst_neutral_band
            hurst_lower = self.hurst_trending_threshold - self.hurst_neutral_band
            vr_upper = 1.0 + self.vr_neutral_band
            vr_lower = 1.0 - self.vr_neutral_band
            
            # Strong trending: both indicators clearly above thresholds
            if (avg_hurst > hurst_upper and avg_vr > self.variance_ratio_trending_min):
                regime = 'trending'
            # Strong mean-reverting: both indicators clearly below thresholds
            elif (avg_hurst < hurst_lower and avg_vr < self.variance_ratio_mean_reverting_max):
                regime = 'mean_reverting'
            # Neutral zone: indicators in neutral bands or conflicting signals
            else:
                regime = 'neutral'
                
            self.market_regime.loc[idx] = regime
            
            # Fill intermediate indices with the same regime value for consistency
            if self.regime_check_frequency > 1:
                # Fill forward from current index to next check point
                end_idx = min(len(df), i + self.regime_check_frequency)
                for j in range(i, end_idx):
                    if j < len(df):
                        fill_idx = df.index[j]
                        self.market_regime.loc[fill_idx] = regime
            
            return regime
            
        except:
            return 'neutral'
            
    def _check_structural_breaks(self, i: int, df: pd.DataFrame) -> bool:
        """Check for structural breaks in cointegration relationship.
        
        Returns:
            bool: True if structural break detected (should close position)
        """
        if not self.structural_break_protection:
            return False
            
        try:
            idx = df.index[i]
            
            # Check rolling correlation (with caching if enabled)
            if i >= self.correlation_window:
                s1_prices = df['y'].iloc[i - self.correlation_window:i]
                s2_prices = df['x'].iloc[i - self.correlation_window:i]
                
                if self.use_market_regime_cache and self.market_regime_cache:
                    # Use cached correlation calculation (EW or rolling)
                    if self.use_exponential_weighted_correlation:
                        correlation = self.market_regime_cache.get_exponential_weighted_correlation(
                            self.pair_name, i, s1_prices, s2_prices, self.ew_correlation_alpha
                        )
                    else:
                        correlation = self.market_regime_cache.get_rolling_correlation(
                            self.pair_name, i, s1_prices, s2_prices
                        )
                else:
                    # Use original calculation
                    correlation = self._calculate_rolling_correlation(s1_prices, s2_prices, self.correlation_window)
                    
                self.rolling_correlations.at[idx] = correlation
                
                if correlation < self.min_correlation_threshold:
                    return True
                    
            # Check half-life of spread
            if i >= self.correlation_window and not pd.isna(df['spread'].iloc[i]):
                spread_series = df['spread'].iloc[max(0, i - self.correlation_window):i]
                half_life = self._calculate_half_life(spread_series, min(len(spread_series), self.correlation_window))
                self.half_life_estimates.at[idx] = half_life
                
                # Convert half-life from periods to days (assuming 15-minute data)
                half_life_days = half_life * 15 / (60 * 24)  # Convert to days
                if half_life_days > self.max_half_life_days:
                    return True
                    
            # Periodic cointegration test (optimized frequency)
            if (i - self.last_cointegration_test >= self.adf_check_frequency and 
                i >= self.adf_check_frequency):
                
                # Lazy ADF: only test if correlation has changed significantly
                should_test_adf = True
                if hasattr(self, 'last_correlation') and self.last_correlation is not None:
                    current_corr = self.rolling_correlations.at[idx] if idx in self.rolling_correlations.index else 0.0
                    corr_change = abs(current_corr - self.last_correlation)
                    # Only test if correlation changed by more than lazy_adf_threshold
                    should_test_adf = corr_change > self.lazy_adf_threshold
                
                if should_test_adf:
                    spread_series = df['spread'].iloc[max(0, i - self.adf_check_frequency):i]
                    p_value = self._perform_adf_test(spread_series, len(spread_series))
                    self.adf_pvalues.at[idx] = p_value
                    self.last_cointegration_test = i
                    
                    # Store current correlation for next comparison
                    if idx in self.rolling_correlations.index:
                        self.last_correlation = self.rolling_correlations.at[idx]
                    
                    if p_value > self.adf_pvalue_threshold:
                        return True
                else:
                    # Skip ADF test but update last test time to avoid frequent checks
                    self.last_cointegration_test = i
                    
            return False
            
        except:
            return False
    
    def _calculate_trade_duration_periods(self, current_time, entry_time, current_index, entry_index):
        """
        FIXED: Unified method for calculating trade duration in periods.
        
        Args:
            current_time: Current timestamp
            entry_time: Entry timestamp
            current_index: Current index position
            entry_index: Entry index position
            
        Returns:
            float: Trade duration in periods
        """
        try:
            if isinstance(current_time, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                # For datetime index, calculate based on actual time difference
                trade_duration_seconds = (current_time - entry_time).total_seconds()
                # For 15-minute data: 1 period = 15 minutes = 900 seconds
                period_seconds = 15 * 60
                return trade_duration_seconds / period_seconds
            else:
                # For integer index, use index difference
                return current_index - entry_index
        except Exception:
            # Fallback to index difference
            return current_index - entry_index
    
    def _pair_is_tradeable(self, symbol_a: str, symbol_b: str, stats_a: dict, stats_b: dict) -> bool:
        """Check if a pair meets all tradeability criteria.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            stats_a: Statistics for first symbol
            stats_b: Statistics for second symbol
            
        Returns:
            True if pair is tradeable, False otherwise
        """
        def get_quote_currency(symbol: str) -> str:
            """Extract quote currency from symbol."""
            if symbol.endswith('USDT'):
                return 'USDT'
            elif symbol.endswith('USDC'):
                return 'USDC'
            elif symbol.endswith('BUSD'):
                return 'BUSD'
            elif symbol.endswith('FDUSD'):
                return 'FDUSD'
            return 'UNKNOWN'
        
        # 1. Same quote currency check
        quote_a = get_quote_currency(symbol_a)
        quote_b = get_quote_currency(symbol_b)
        if quote_a != quote_b:
            return False
        
        # 2. Volume check
        vol_threshold = getattr(self, 'min_volume_usd_24h', 20_000_000)
        if (stats_a.get('vol_usd_24h', 0) < vol_threshold or 
            stats_b.get('vol_usd_24h', 0) < vol_threshold):
            return False
        
        # 3. Days live check
        days_threshold = getattr(self, 'min_days_live', 30)
        if (stats_a.get('days_live', 0) < days_threshold or 
            stats_b.get('days_live', 0) < days_threshold):
            return False
        
        # 4. Funding rate check
        funding_threshold = getattr(self, 'max_funding_rate_abs', 0.0003)
        if (abs(stats_a.get('funding', 0)) >= funding_threshold or 
            abs(stats_b.get('funding', 0)) >= funding_threshold):
            return False
        
        # 5. Tick size check
        tick_threshold = getattr(self, 'max_tick_size_pct', 0.0005)
        if (stats_a.get('tick_pct', 1.0) >= tick_threshold or 
            stats_b.get('tick_pct', 1.0) >= tick_threshold):
            return False
        
        # 6. Half-life check
        half_life_threshold = getattr(self, 'max_half_life_hours', 72)
        if (stats_a.get('half_life_h', 1000) >= half_life_threshold or 
            stats_b.get('half_life_h', 1000) >= half_life_threshold):
            return False
        
        return True
    
    def _recalculate_beta(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """Recalculate beta using OLS regression on log prices.
        
        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            
        Returns:
            Beta coefficient
        """
        if len(prices_a) < 10 or len(prices_b) < 10:
            return 1.0  # Default beta
        
        try:
            log_prices_a = np.log(prices_a)
            log_prices_b = np.log(prices_b)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(log_prices_a) & np.isfinite(log_prices_b)
            if np.sum(valid_mask) < 10:
                return 1.0
            
            log_prices_a = log_prices_a[valid_mask]
            log_prices_b = log_prices_b[valid_mask]
            
            # OLS regression: log_prices_a = alpha + beta * log_prices_b
            X = np.column_stack([np.ones(len(log_prices_b)), log_prices_b])
            beta_coef = np.linalg.lstsq(X, log_prices_a, rcond=None)[0][1]
            
            # Sanity check on beta
            if not np.isfinite(beta_coef) or beta_coef <= 0 or beta_coef > 10:
                return 1.0
            
            return beta_coef
        except Exception:
            return 1.0
    
    def _is_funding_blackout_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if current time is in funding blackout period.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if in blackout period
        """
        if not self.enable_funding_time_filter:
            return False
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        for reset_hour in self.funding_reset_hours:
            # Check if within blackout window around reset hour
            blackout_start = (reset_hour * 60 - self.funding_blackout_minutes) % (24 * 60)
            blackout_end = (reset_hour * 60 + self.funding_blackout_minutes) % (24 * 60)
            
            current_minute = hour * 60 + minute
            
            if blackout_start <= blackout_end:
                if blackout_start <= current_minute <= blackout_end:
                    return True
            else:  # Crosses midnight
                if current_minute >= blackout_start or current_minute <= blackout_end:
                    return True
        
        return False
    
    def _is_macro_blackout_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if current time is in macro event blackout period.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if in blackout period
        """
        if not self.enable_macro_event_filter:
            return False
        
        if self.macro_event_blackout_end is not None:
            return timestamp <= self.macro_event_blackout_end
        
        return False
    
    def _is_pair_quarantined(self, pair_name: str, timestamp: pd.Timestamp) -> bool:
        """Check if pair is currently quarantined.
        
        Args:
            pair_name: Name of the pair
            timestamp: Current timestamp
            
        Returns:
            True if pair is quarantined
        """
        if not self.enable_pair_quarantine:
            return False
        
        if pair_name in self.quarantined_pairs:
            quarantine_end = self.quarantined_pairs[pair_name]
            if timestamp <= quarantine_end:
                return True
            else:
                # Remove expired quarantine
                del self.quarantined_pairs[pair_name]
        
        return False
    
    def _update_pair_quarantine_status(self, pair_name: str, current_pnl: float, 
                                      current_equity: float, timestamp: pd.Timestamp):
        """Update quarantine status for a pair based on performance.
        
        Args:
            pair_name: Name of the pair
            current_pnl: Current PnL for the pair
            current_equity: Current equity for the pair
            timestamp: Current timestamp
        """
        if not self.enable_pair_quarantine:
            return
        
        # Update PnL history
        if pair_name not in self.pair_pnl_history:
            self.pair_pnl_history[pair_name] = []
        
        # Add today's PnL (simplified - in real implementation would track daily)
        self.pair_pnl_history[pair_name].append(current_pnl)
        
        # Keep only recent history
        max_history = self.quarantine_rolling_window_days
        if len(self.pair_pnl_history[pair_name]) > max_history:
            self.pair_pnl_history[pair_name] = self.pair_pnl_history[pair_name][-max_history:]
        
        # Update equity peak
        if pair_name not in self.pair_equity_peaks:
            self.pair_equity_peaks[pair_name] = current_equity
        else:
            self.pair_equity_peaks[pair_name] = max(self.pair_equity_peaks[pair_name], current_equity)
        
        # Check quarantine conditions
        pnl_history = self.pair_pnl_history[pair_name]
        if len(pnl_history) >= 5:  # Need some history
            pnl_std = np.std(pnl_history)
            pnl_threshold = -self.quarantine_pnl_threshold_sigma * pnl_std
            
            # Check PnL condition
            if current_pnl < pnl_threshold:
                quarantine_end = timestamp + pd.Timedelta(days=self.quarantine_period_days)
                self.quarantined_pairs[pair_name] = quarantine_end
                return
        
        # Check drawdown condition
        peak_equity = self.pair_equity_peaks[pair_name]
        if peak_equity > 0:
            drawdown = (current_equity - peak_equity) / peak_equity
            if drawdown < -self.quarantine_drawdown_threshold_pct:
                quarantine_end = timestamp + pd.Timedelta(days=self.quarantine_period_days)
                self.quarantined_pairs[pair_name] = quarantine_end
    
    def _calculate_realistic_costs(self, qty_long: float, price_long: float, 
                                  qty_short: float, price_short: float,
                                  spread_half: float, funding_rate_long: float,
                                  funding_rate_short: float, holding_hours: float) -> dict:
        """Calculate realistic trading costs.
        
        Args:
            qty_long: Quantity of long position
            price_long: Price of long asset
            qty_short: Quantity of short position
            price_short: Price of short asset
            spread_half: Half of the bid-ask spread
            funding_rate_long: Funding rate for long position
            funding_rate_short: Funding rate for short position
            holding_hours: Hours position was held
            
        Returns:
            Dictionary with cost breakdown
        """
        if not self.enable_realistic_costs:
            return {'commission': 0, 'slippage': 0, 'funding': 0, 'total': 0}
        
        # Commission cost (per leg)
        notional_long = abs(qty_long * price_long)
        notional_short = abs(qty_short * price_short)
        commission_cost = (notional_long + notional_short) * self.commission_rate_per_leg
        
        # Slippage cost (based on spread)
        slippage_cost = self.slippage_half_spread_multiplier * spread_half
        
        # Funding cost
        funding_cost = 0
        if self.funding_cost_enabled and holding_hours > 0:
            funding_periods = holding_hours / 8  # Funding every 8 hours
            funding_cost_long = notional_long * funding_rate_long * funding_periods
            funding_cost_short = notional_short * funding_rate_short * funding_periods
            funding_cost = funding_cost_long + funding_cost_short
        
        total_cost = commission_cost + slippage_cost + abs(funding_cost)
        
        return {
            'commission': commission_cost,
            'slippage': slippage_cost,
            'funding': funding_cost,
            'total': total_cost
        }
