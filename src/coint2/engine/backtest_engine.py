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
        # NEW: Enhanced risk management parameters
        self.use_kelly_sizing = use_kelly_sizing
        self.max_kelly_fraction = max_kelly_fraction
        self.volatility_lookback = volatility_lookback
        self.adaptive_thresholds = adaptive_thresholds
        self.var_confidence = var_confidence
        self.max_var_multiplier = max_var_multiplier
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
        self.exclusion_period_days = exclusion_period_days
        self.max_half_life_days = max_half_life_days
        self.min_correlation_threshold = min_correlation_threshold
        self.correlation_window = correlation_window
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
        # Check data size vs rolling window
        if len(self.pair_data) < self.rolling_window + 2:
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
            
        # FIXED: Corrected take_profit validation - it should be less than 1.0 for proper logic
        if (self.take_profit_multiplier is not None and
            self.take_profit_multiplier >= 1.0):
            raise ValueError(f"take_profit_multiplier ({self.take_profit_multiplier}) must be less than 1.0 "
                           "to ensure take-profit triggers before entry level.")
        
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
        
        if self.rolling_window > len(self.pair_data) // 2:
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
        
        # Create secure hash using hashlib
        try:
            # Primary method: use tobytes() for efficient hashing
            y_bytes = y_array.tobytes()
            x_bytes = x_array.tobytes()
            hash_input = y_bytes + x_bytes + str(len(y_array)).encode()
        except (AttributeError, ValueError):
            # FIXED: Secure fallback using hashlib instead of str()
            hash_input = (str(y_array.tolist()) + str(x_array.tolist())).encode()
        
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
            mean = spread_win.mean()
            std = max(spread_win.std(), 1e-8)  # Avoid zero std
        else:
            beta = model.params.iloc[1]
            spread_win = y_win - beta * x_win
            mean = spread_win.mean()
            std = max(spread_win.std(), 1e-8)  # FIXED: Avoid zero std in all cases
        
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

    def _calculate_position_size(self, entry_z: float, spread_curr: float, mean: float,
                               std: float, beta: float, price_s1: float, price_s2: float) -> float:
        """Calculate position size based on risk management parameters.
        
        FIXED: Added minimum risk_per_unit threshold to prevent excessive position sizes
        when spread is very close to stop loss.
        ENHANCED: Account for trading costs when calculating thresholds.
        """
        EPSILON = 1e-8
        
        stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
        stop_loss_price = mean + stop_loss_z * std
        risk_per_unit_raw = abs(spread_curr - stop_loss_price)
        trade_value = price_s1 + abs(beta) * price_s2
        
        # Calculate round-turn trading costs per unit
        # Round-turn = open + close, so 2x the one-way costs
        round_turn_commission_rate = 2 * (self.commission_pct + self.slippage_pct + 
                                         (self.bid_ask_spread_pct_s1 + self.bid_ask_spread_pct_s2) / 2)
        round_turn_cost_per_unit = trade_value * round_turn_commission_rate
        
        # Ensure minimum profit target is 2-3x round-turn costs
        min_profit_multiplier = 2.5  # 2.5x round-turn costs as minimum
        min_risk_per_unit_costs = min_profit_multiplier * round_turn_cost_per_unit
        
        # FIXED: Prevent excessive position sizes by enforcing minimum risk per unit
        # When risk_per_unit is too small, position size becomes dangerously large
        min_risk_per_unit_volatility = 0.1 * std  # Use 10% of volatility as minimum risk
        min_risk_per_unit = max(risk_per_unit_raw, min_risk_per_unit_costs, 
                               min_risk_per_unit_volatility, EPSILON)
        risk_per_unit = min_risk_per_unit
        
        # Use consistent epsilon protection for division by zero with stricter checks
        size_risk = self.capital_at_risk / risk_per_unit if risk_per_unit > EPSILON and np.isfinite(risk_per_unit) else 0.0
        size_value = self.capital_at_risk / trade_value if trade_value > EPSILON and np.isfinite(trade_value) else 0.0
        
        # Margin limit calculation with proper epsilon protection
        if np.isfinite(self.max_margin_usage) and trade_value > EPSILON:
            margin_limit = self.capital_at_risk * self.max_margin_usage / trade_value
        else:
            margin_limit = float('inf')
        
        # NEW: Apply volatility-based position sizing if enabled
        if self.volatility_based_sizing:
            volatility_multiplier = self._calculate_volatility_multiplier()
            base_size = min(size_risk, size_value, margin_limit)
            return base_size * volatility_multiplier
        
        return min(size_risk, size_value, margin_limit)

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
        base_volatility = z_scores.rolling(window=self.volatility_lookback * 2).std().median()
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
        
        FIXED: Унифицированная обработка времени - всегда возвращаем часы.
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
                    # FIXED: Для числовых индексов предполагаем 15-минутные данные
                    # Это обеспечивает консистентность с time_stop_multiplier
                    return float(diff) * 0.25  # 1 период = 15 минут = 0.25 часа
            else:
                # FIXED: Для non-datetime индексов возвращаем периоды в часах (15-минутные данные)
                return float(current_index - entry_index) * 0.25
        except (TypeError, AttributeError, ValueError):
            # FIXED: Enhanced fallback with better error handling (15-минутные данные)
            return float(current_index - entry_index) * 0.25

    def _enter_position(self, df: pd.DataFrame, i: int, signal: int, z_curr: float,
                       spread_curr: float, mean: float, std: float, beta: float) -> float:
        """Enter a new position with enhanced risk management.
        
        ENHANCED: Integrated Kelly sizing and VaR-based position sizing.
        """
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        
        # Calculate base position size
        base_size = self._calculate_position_size(z_curr, spread_curr, mean, std, beta, price_s1, price_s2)
        
        # NEW: Apply enhanced risk management
        final_size = base_size
        
        # Apply Kelly sizing if enabled and we have sufficient data
        if self.use_kelly_sizing and len(self.rolling_returns) >= 10:
            kelly_fraction = self._calculate_kelly_fraction(self.rolling_returns)
            final_size *= kelly_fraction
            
        # Apply VaR-based sizing if we have sufficient data
        if len(self.rolling_returns) >= 20:
            var_multiplier = self._calculate_var_position_size(self.rolling_returns)
            final_size *= var_multiplier
            
        # Ensure minimum position size
        final_size = max(final_size, base_size * 0.1)  # At least 10% of base size
        
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

    def _calculate_trading_costs(self, position_s1_change: float, position_s2_change: float,
                               price_s1: float, price_s2: float) -> tuple[float, float, float, float]:
        """Calculate detailed trading costs with full breakdown.
        
        FIXED: Унифицированный расчет издержек с полной детализацией.
        
        Returns:
            tuple: (commission_costs, slippage_costs, bid_ask_costs, total_costs)
        """
        notional_change_s1 = abs(position_s1_change * price_s1)
        notional_change_s2 = abs(position_s2_change * price_s2)
        
        # Детальный расчет всех компонентов издержек
        commission_costs = (notional_change_s1 + notional_change_s2) * self.commission_pct
        slippage_costs = (notional_change_s1 + notional_change_s2) * self.slippage_pct
        bid_ask_costs = (notional_change_s1 * self.bid_ask_spread_pct_s1 + 
                        notional_change_s2 * self.bid_ask_spread_pct_s2)
        total_costs = commission_costs + slippage_costs + bid_ask_costs
        
        return commission_costs, slippage_costs, bid_ask_costs, total_costs

    def _log_exit(self, df: pd.DataFrame, i: int, z_curr: float, exit_reason: str,
                  entry_datetime, entry_index: int) -> None:
        """FIXED: Extracted method to eliminate code duplication in exit logging."""
        df.loc[df.index[i], "exit_reason"] = exit_reason
        df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
        df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
        df.loc[df.index[i], "exit_z"] = z_curr
        
        # Calculate trade duration with unified time handling
        if entry_datetime is not None:
            df.loc[df.index[i], "trade_duration"] = self._calculate_trade_duration(
                df.index[i], entry_datetime, i, entry_index
            )

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
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

        # FIXED: Use iloc for better performance and avoid potential indexing issues
        for i in range(self.rolling_window, len(df)):
            # FIXED: More efficient window slicing using iloc
            y_win = df["y"].iloc[i - self.rolling_window : i]
            x_win = df["x"].iloc[i - self.rolling_window : i]
            
            # Use cached OLS calculation
            beta, mean, std = self._calculate_ols_with_cache(y_win, x_win)
            
            # FIXED: Более гибкая обработка низкой волатильности
            # Используем адаптивный порог на основе средних значений цен
            price_scale = max(df["y"].iloc[i-self.rolling_window:i].mean(), 
                             df["x"].iloc[i-self.rolling_window:i].mean())
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
                
            # NEW: Record market regime and structural break analysis results
            idx = df.index[i]
            df.loc[idx, "market_regime"] = market_regime
            df.loc[idx, "structural_break_detected"] = structural_break_detected
            
            # Copy analysis results from state variables if available
            if idx in self.hurst_exponents.index:
                df.loc[idx, "hurst_exponent"] = self.hurst_exponents.loc[idx]
            if idx in self.variance_ratios.index:
                df.loc[idx, "variance_ratio"] = self.variance_ratios.loc[idx]
            if idx in self.rolling_correlations.index:
                df.loc[idx, "rolling_correlation"] = self.rolling_correlations.loc[idx]
            if idx in self.half_life_estimates.index:
                df.loc[idx, "half_life_estimate"] = self.half_life_estimates.loc[idx]
            if idx in self.adf_pvalues.index:
                df.loc[idx, "adf_pvalue"] = self.adf_pvalues.loc[idx]

            # FIXED: Time-based stop-loss с унифицированными единицами времени
            if (
                position != 0
                and self.half_life is not None
                and self.time_stop_multiplier is not None
                and entry_datetime is not None
            ):
                # Используем унифицированную функцию расчета времени (в часах)
                trade_duration_hours = self._calculate_trade_duration(
                    df.index[i], entry_datetime, i, entry_index
                )
                # half_life предполагается в днях, конвертируем в часы
                time_stop_limit_hours = self.half_life * 24 * self.time_stop_multiplier
                if trade_duration_hours >= time_stop_limit_hours:
                    new_position = 0.0
                    cooldown_remaining = self.cooldown_periods
                    # FIXED: Use extracted _log_exit method
                    self._log_exit(df, i, z_curr, 'time_stop', entry_datetime, entry_index)

            # Уменьшаем cooldown счетчик
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Закрытие позиций в конце теста для чистоты метрик
            if i == len(df) - 1 and position != 0:
                new_position = 0.0  # Форс-закрытие в последнем периоде
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "end_of_test", entry_datetime, entry_index)
            # Стоп-лосс выходы
            elif position > 0 and z_curr <= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "stop_loss", entry_datetime, entry_index)
            elif position < 0 and z_curr >= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "stop_loss", entry_datetime, entry_index)
            # ENHANCED: Take-profit логика с учетом комиссий
            # Выход при движении z-score к нулю, но только если прибыль покрывает комиссии
            elif (
                self.take_profit_multiplier is not None
                and position != 0
                and abs(z_curr) <= abs(entry_z) * self.take_profit_multiplier
                and self._is_profitable_exit(current_trade_pnl, position, price_s1_curr, price_s2_curr, beta)
            ):
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "take_profit", entry_datetime, entry_index)
            # Z-score exit: закрываем позицию если z-score вернулся к заданному уровню
            elif position != 0 and abs(z_curr) <= self.z_exit:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # FIXED: Use extracted _log_exit method
                self._log_exit(df, i, z_curr, "z_exit", entry_datetime, entry_index)

            # Проверяем сигналы входа только если не в последнем периоде и не в cooldown
            if i < len(df) - 1 and cooldown_remaining == 0:
                # NEW: Calculate adaptive threshold if enabled
                current_threshold = self.z_threshold
                if self.adaptive_thresholds and i >= self.volatility_lookback:
                    z_scores_window = df["z_score"].iloc[:i+1]
                    current_volatility = z_scores_window.rolling(window=self.volatility_lookback).std().iloc[-1]
                    if not pd.isna(current_volatility):
                        current_threshold = self._calculate_adaptive_threshold(z_scores_window, current_volatility)
                
                signal = 0
                if z_curr > current_threshold:
                    signal = -1
                elif z_curr < -current_threshold:
                    signal = 1

                z_prev = df["z_score"].iat[i - 1]
                long_confirmation = (signal == 1) and (z_curr > z_prev)
                short_confirmation = (signal == -1) and (z_curr < z_prev)

                # FIXED: Extracted position entry logic to reduce duplication
                if (new_position == 0 and (long_confirmation or short_confirmation)) or (
                    new_position != 0
                    and (long_confirmation or short_confirmation)
                    and np.sign(new_position) != signal
                ):
                    new_position = self._enter_position(
                        df, i, signal, z_curr, spread_curr, mean, std, beta
                    )
                    entry_z = z_curr
                    stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)

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

            # Handle entry logging
            if position == 0 and new_position != 0:
                entry_datetime = df.index[i]
                entry_spread = spread_curr
                entry_position_size = new_position
                entry_index = i
            # Handle exit logging
            if position != 0 and new_position == 0 and entry_datetime is not None:
                exit_datetime = df.index[i]
                # FIXED: Unified time handling with consistent units
                duration_hours = self._calculate_trade_duration(
                    exit_datetime, entry_datetime, i, entry_index
                )

                trade_info = {
                    "pair": f"{self.s1}-{self.s2}",
                    "entry_datetime": entry_datetime,
                    "exit_datetime": exit_datetime,
                    "position_type": "long" if entry_position_size > 0 else "short",
                    "entry_price_spread": entry_spread,
                    "exit_price_spread": spread_curr,
                    "pnl": current_trade_pnl,
                    "exit_reason": df.loc[df.index[i], "exit_reason"],
                    "trade_duration_hours": duration_hours,
                }
                self.trades_log.append(trade_info)

                # Сбрасываем счетчики для следующей сделки
                current_trade_pnl = 0.0
                entry_datetime = None
                entry_spread = 0.0
                entry_position_size = 0.0
                
                # NEW: Update rolling returns for Kelly sizing
                if step_pnl != 0:  # Only add non-zero returns
                    trade_return = step_pnl / (abs(entry_position_size) * (price_s1_curr + abs(beta) * price_s2_curr))
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

    def get_results(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest not yet run")

        # Check if required columns exist, if not create them with default values
        required_columns = {
            "trades": 0.0,
            "commission_costs": 0.0,
            "slippage_costs": 0.0,
            "bid_ask_costs": 0.0,
            "impact_costs": 0.0,
            "cumulative_pnl": 0.0
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

        # Если после dropna ничего не осталось, возвращаем нулевые метрики
        if pnl.empty:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "kelly_criterion": 0.0,
            }

        # FIXED: Правильный расчет Sharpe Ratio - передаем доходности, а не PnL
        # Конвертируем PnL в доходности относительно capital_at_risk
        returns = pnl / self.capital_at_risk
        sharpe = performance.sharpe_ratio(returns, self.annualizing_factor)
        
        # Calculate new metrics
        win_rate_val = performance.win_rate(pnl)
        expectancy_val = performance.expectancy(pnl)
        kelly_val = performance.kelly_criterion(pnl)

        return {
            "sharpe_ratio": 0.0 if np.isnan(sharpe) else sharpe,
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
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
        """Get the capital at risk that was available on a specific date."""
        if date in self.capital_at_risk_history:
            return self.capital_at_risk_history[date]
        
        # If exact date not found, use the most recent available
        available_dates = self.capital_at_risk_history.index
        if len(available_dates) == 0:
            return self.capital_at_risk  # Fallback to initial value
            
        recent_dates = available_dates[available_dates <= date]
        if len(recent_dates) > 0:
            return self.capital_at_risk_history[recent_dates[-1]]
        else:
            return self.capital_at_risk  # Fallback to initial value
             
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
                prev_data = self.pair_data.iloc[-2] if len(self.pair_data) >= 2 else self.pair_data.iloc[-1]
                prev_spread = prev_data.iloc[0] - self.active_trade.beta * prev_data.iloc[1]
                
            # Calculate PnL using individual asset returns: pnl = size_s1 * ΔP1 + size_s2 * ΔP2
            # where size_s2 = -beta * size_s1
            delta_p1 = price_s1 - self.active_trade.entry_price_s1
            delta_p2 = price_s2 - self.active_trade.entry_price_s2
            
            size_s1 = self.active_trade.position_size
            size_s2 = -self.active_trade.beta * size_s1
            
            pnl = size_s1 * delta_p1 + size_s2 * delta_p2
            result['pnl'] = pnl
            result['position'] = self.active_trade.position_size
            
        # Check exit conditions
        trade_closed = False
        if self.active_trade is not None:
            # Stop loss
            if ((self.active_trade.position_size > 0 and z_score <= self.active_trade.stop_loss_z) or
                (self.active_trade.position_size < 0 and z_score >= self.active_trade.stop_loss_z)):
                self._close_trade(date, z_score, 'stop_loss')
                trade_closed = True
                result['trade_closed'] = True
                
            # Take profit
            elif (self.take_profit_multiplier is not None and
                  abs(z_score) <= abs(self.active_trade.entry_z) * self.take_profit_multiplier):
                self._close_trade(date, z_score, 'take_profit')
                trade_closed = True
                result['trade_closed'] = True
                
            # Z-score exit
            elif abs(z_score) <= self.z_exit:
                self._close_trade(date, z_score, 'z_exit')
                trade_closed = True
                result['trade_closed'] = True
                
            # Time stop
            elif (self.half_life is not None and self.time_stop_multiplier is not None):
                trade_duration_hours = (date - self.active_trade.entry_date).total_seconds() / 3600.0
                time_stop_limit_hours = self.half_life * 24 * self.time_stop_multiplier
                if trade_duration_hours >= time_stop_limit_hours:
                    self._close_trade(date, z_score, 'time_stop')
                    trade_closed = True
                    result['trade_closed'] = True
                    
        # Check entry conditions (only if no active trade and not in cooldown)
        if (self.active_trade is None and 
            (self.cooldown_end_date is None or date > self.cooldown_end_date)):
            
            signal = 0
            if z_score > self.z_threshold:
                signal = -1
            elif z_score < -self.z_threshold:
                signal = 1
                
            if signal != 0:
                # Get capital at risk for this specific date
                capital_for_trade = self.get_capital_at_risk_for_date(date)
                
                # Calculate position size using capital available at trade entry
                old_capital = self.capital_at_risk
                self.capital_at_risk = capital_for_trade
                position_size = self._calculate_position_size(
                    z_score, current_spread, mean, std, beta, price_s1, price_s2
                )
                self.capital_at_risk = old_capital  # Restore original
                
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
        stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
        
        self.active_trade = TradeState(
            entry_date=date,
            entry_index=len(self.pair_data) - 1,
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
            
        # Log trade closing
        self.incremental_trades_log.append({
            'action': 'close',
            'date': date,
            'exit_z': exit_z,
            'exit_reason': exit_reason,
            'position_size': self.active_trade.position_size,
            'capital_used': self.active_trade.capital_at_risk_used,
            'entry_date': self.active_trade.entry_date
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
            
        try:
            # Get price series for analysis
            s1_prices = df['y'].iloc[max(0, i - self.hurst_window):i]
            s2_prices = df['x'].iloc[max(0, i - self.hurst_window):i]
            
            # Calculate Hurst Exponent for both assets
            hurst_s1 = self._calculate_hurst_exponent(s1_prices)
            hurst_s2 = self._calculate_hurst_exponent(s2_prices)
            avg_hurst = (hurst_s1 + hurst_s2) / 2
            
            # Calculate Variance Ratio for both assets
            vr_s1 = self._calculate_variance_ratio(s1_prices)
            vr_s2 = self._calculate_variance_ratio(s2_prices)
            avg_vr = (vr_s1 + vr_s2) / 2
            
            # Store values for analysis
            idx = df.index[i]
            self.hurst_exponents.loc[idx] = avg_hurst
            self.variance_ratios.loc[idx] = avg_vr
            
            # Determine regime based on both indicators
            if (avg_hurst > self.hurst_trending_threshold and 
                avg_vr > self.variance_ratio_trending_min):
                regime = 'trending'
            elif (avg_hurst < self.hurst_trending_threshold and 
                  avg_vr < self.variance_ratio_mean_reverting_max):
                regime = 'mean_reverting'
            else:
                regime = 'neutral'
                
            self.market_regime.loc[idx] = regime
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
            
            # Check rolling correlation
            if i >= self.correlation_window:
                s1_prices = df['y'].iloc[i - self.correlation_window:i]
                s2_prices = df['x'].iloc[i - self.correlation_window:i]
                correlation = self._calculate_rolling_correlation(s1_prices, s2_prices, self.correlation_window)
                self.rolling_correlations.loc[idx] = correlation
                
                if correlation < self.min_correlation_threshold:
                    return True
                    
            # Check half-life of spread
            if i >= self.correlation_window and not pd.isna(df['spread'].iloc[i]):
                spread_series = df['spread'].iloc[max(0, i - self.correlation_window):i]
                half_life = self._calculate_half_life(spread_series, min(len(spread_series), self.correlation_window))
                self.half_life_estimates.loc[idx] = half_life
                
                # Convert half-life from periods to days (assuming 15-minute data)
                half_life_days = half_life * 15 / (60 * 24)  # Convert to days
                if half_life_days > self.max_half_life_days:
                    return True
                    
            # Periodic cointegration test
            if (i - self.last_cointegration_test >= self.cointegration_test_frequency and 
                i >= self.cointegration_test_frequency):
                
                spread_series = df['spread'].iloc[max(0, i - self.cointegration_test_frequency):i]
                p_value = self._perform_adf_test(spread_series, len(spread_series))
                self.adf_pvalues.loc[idx] = p_value
                self.last_cointegration_test = i
                
                if p_value > self.adf_pvalue_threshold:
                    return True
                    
            return False
            
        except:
            return False
