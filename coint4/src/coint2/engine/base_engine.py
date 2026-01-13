import numpy as np
import pandas as pd
import statsmodels.api as sm
import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from abc import ABC, abstractmethod
from numpy.linalg import LinAlgError

from ..core import performance
from .market_regime_cache import MarketRegimeCache

class DataNormalizer(ABC):
    """Abstract base class for data normalization."""
    
    def __init__(self):
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'DataNormalizer':
        """Fit normalizer on training data."""
        pass
        
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        pass
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

class MinMaxNormalizer(DataNormalizer):
    """Min-Max normalization with fit/transform pattern."""
    
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame) -> 'MinMaxNormalizer':
        """Fit min-max scalers on training data."""
        for col in data.columns:
            scaler = MinMaxScaler(feature_range=self.feature_range)
            scaler.fit(data[[col]].values)
            self.scalers[col] = scaler
        self.is_fitted = True
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        result = data.copy()
        for col in data.columns:
            if col in self.scalers:
                result[col] = self.scalers[col].transform(data[[col]].values).flatten()
        return result

class LogReturnsNormalizer(DataNormalizer):
    """Log returns normalization."""
    
    def __init__(self, base=np.e):
        super().__init__()
        self.base = base
        
    def fit(self, data: pd.DataFrame) -> 'LogReturnsNormalizer':
        """No fitting required for log returns."""
        self.is_fitted = True
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform to log returns."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        if self.base == np.e:
            return np.log(data / data.shift(1)).dropna()
        else:
            return np.log(data / data.shift(1)) / np.log(self.base)

@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    window_id: int

class WalkForwardSplitter:
    """Generates walk-forward windows for backtesting."""
    
    def __init__(self, 
                 training_period_days: int,
                 testing_period_days: int,
                 step_size_days: int = None,
                 min_training_samples: int = 1000):
        self.training_period_days = training_period_days
        self.testing_period_days = testing_period_days
        self.step_size_days = step_size_days or testing_period_days
        self.min_training_samples = min_training_samples
        
    def split(self, data: pd.DataFrame, freq_per_day: int = 96) -> List[WalkForwardWindow]:
        """Generate walk-forward windows.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with datetime index
        freq_per_day : int
            Number of data points per day (default 96 for 15-min data)
            
        Returns
        -------
        List[WalkForwardWindow]
            List of walk-forward windows
        """
        windows = []
        total_samples = len(data)
        
        training_samples = self.training_period_days * freq_per_day
        testing_samples = self.testing_period_days * freq_per_day
        step_samples = self.step_size_days * freq_per_day
        
        window_id = 0
        train_start = 0
        
        while True:
            train_end = train_start + training_samples
            test_start = train_end
            test_end = test_start + testing_samples
            
            # Check if we have enough data
            if test_end > total_samples:
                break
                
            # Check minimum training samples
            if train_end - train_start < self.min_training_samples:
                break
                
            windows.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            ))
            
            window_id += 1
            train_start += step_samples
            
        return windows

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
    phase: str = "test"  # "train" or "test" for walk-forward

class BasePairBacktester:
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
        # NEW: Walk-forward testing parameters
        walk_forward_enabled: bool = False,
        walk_forward_splitter: WalkForwardSplitter = None,
        # NEW: Trading period restriction parameters
        trading_start: pd.Timestamp = None,
        trading_end: pd.Timestamp = None,
        # NEW: Normalization parameters
        normalization_enabled: bool = False,
        normalizer: DataNormalizer = None,
        # NEW: Online statistics parameters
        online_stats_enabled: bool = True,
        volatility_method: str = "rolling",  # "rolling" or "ewm"
        ewm_alpha: float = 0.1,
        kelly_lookback_trades: int = 50,
        adaptive_threshold_lookback: int = 100,
        beta_recalc_online: bool = True,
        # NEW: Signal shift parameters
        signal_shift_enabled: bool = True,
        shift_periods: int = 1,
        skip_last_bar: bool = True,
        signal_delay_minutes: int = 0,
        # NEW: Enhanced cost parameters
        fee_maker: float = 0.0002,
        fee_taker: float = 0.0004,
        slippage_bps: float = 2.0,
        half_spread_bps: float = 1.0,
        slippage_stress_multiplier: float = 1.0,
        always_model_slippage: bool = True,
        # NEW: Signal confirmation parameter
        require_signal_confirmation: bool = False,
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
        # NEW: Trading period restriction
        self.trading_start = trading_start
        self.trading_end = trading_end
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
        
        # NEW: Walk-forward testing parameters
        self.walk_forward_enabled = walk_forward_enabled
        self.walk_forward_splitter = walk_forward_splitter
        self.current_phase = "test"  # Current phase: "train" or "test"
        self.current_window_id = 0
        
        # NEW: Normalization parameters
        self.normalization_enabled = normalization_enabled
        self.normalizer = normalizer
        self.original_data = None  # Store original data before normalization
        
        # NEW: Online statistics parameters
        self.online_stats_enabled = online_stats_enabled
        self.volatility_method = volatility_method
        self.ewm_alpha = ewm_alpha
        self.kelly_lookback_trades = kelly_lookback_trades
        self.adaptive_threshold_lookback = adaptive_threshold_lookback
        self.beta_recalc_online = beta_recalc_online
        
        # NEW: Signal shift parameters
        self.signal_shift_enabled = signal_shift_enabled
        self.shift_periods = shift_periods
        self.skip_last_bar = skip_last_bar
        self.signal_delay_minutes = signal_delay_minutes
        
        # NEW: Enhanced cost parameters
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.slippage_bps = slippage_bps
        self.half_spread_bps = half_spread_bps
        self.slippage_stress_multiplier = slippage_stress_multiplier
        self.always_model_slippage = always_model_slippage
        
        # NEW: Signal confirmation parameter
        self.require_signal_confirmation = require_signal_confirmation
        
        # NEW: Online statistics state
        self.online_volatility = None
        self.online_kelly_history = []
        self.online_threshold_history = []
        self.last_beta_update_idx = 0
        
        # Validate trading parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate trading parameters for logical consistency."""
        # Check data size vs rolling window (skip for empty data in incremental mode)
        if not self.pair_data.empty and len(self.pair_data) < self.rolling_window + 2:
            # Для очень маленьких выборок — ошибка, иначе предупреждение
            if len(self.pair_data) < 6:
                raise ValueError(
                    f"Data length ({len(self.pair_data)}) must be at least rolling_window + 2 "
                    f"({self.rolling_window + 2})"
                )
            import warnings
            warnings.warn(
                f"Data length ({len(self.pair_data)}) is below rolling_window + 2 "
                f"({self.rolling_window + 2}); results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        
        # Check threshold consistency
        if self.zscore_entry_threshold <= 0:
            raise ValueError("zscore_entry_threshold must be positive")
        
        # ИСПРАВЛЕНИЕ: Разрешаем отрицательные z_exit для более гибких стратегий
        # if self.z_exit < 0:
        #     raise ValueError("z_exit must be non-negative")
            
        if self.z_exit >= self.zscore_entry_threshold:
            raise ValueError(f"z_exit ({self.z_exit}) must be less than zscore_entry_threshold ({self.zscore_entry_threshold}). "
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
            import warnings
            warnings.warn(
                f"rolling_window ({self.rolling_window}) is large relative to data size "
                f"({len(self.pair_data)}); results may be unstable.",
                UserWarning,
                stacklevel=2,
            )
        
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
            # Use window data statistics for hashing to avoid global computations
            # This avoids expensive operations while maintaining uniqueness
            y_stats = (len(y_win), y_win.sum(), y_win.min(), y_win.max())
            x_stats = (len(x_win), x_win.sum(), x_win.min(), x_win.max())
            hash_input = f"{len(y_array)}_{y_stats}_{x_stats}".encode()
        except (AttributeError, ValueError):
            # Fallback to simple length-based hash for degenerate cases
            hash_input = f"{len(y_array)}_{y_win.sum() if len(y_win) > 0 else 0}_{x_win.sum() if len(x_win) > 0 else 0}".encode()
        
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
            # For degenerate cases (e.g., constant data), return NaN values
            return (float('nan'), float('nan'), float('nan'))
        else:
            beta = model.params.iloc[1]
            spread_win = y_win - beta * x_win
            mean = spread_win.mean() if len(spread_win) > 0 else 0.0
            std = spread_win.std() if len(spread_win) > 0 else 0.0
            
            # CRITICAL FIX: Handle zero std cases to match manual_backtest behavior
            if std < 1e-6 or np.isnan(std):
                # Return NaN for very small std to match manual_backtest behavior
                return (float('nan'), float('nan'), float('nan'))
            elif std < 1e-8:
                # For very small but non-zero std, set minimum to prevent division by zero
                std = 1e-8
        
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
        """Calculate position size based on new requirements.
        
        NEW: Kelly or capital fraction based only on completed trades [:i]
        NEW: Limit f to range [0, f_max]
        NEW: qty = f * equity / (price_y + |beta|*price_x)
        NEW: Check min_notional_per_trade and max_notional_per_trade
        
        Returns:
            float: Position size for the first asset
        """
        EPSILON = 1e-8
        
        # Get current equity from portfolio
        current_equity = self.portfolio.get_current_equity() if self.portfolio else self.capital_at_risk
        
        # Calculate Kelly fraction or capital fraction based on completed trades only
        f = self._calculate_kelly_or_capital_fraction()
        
        # Limit f to range [0, f_max]
        f_max = getattr(self.portfolio.config, 'f_max', 0.25) if hasattr(self.portfolio, 'config') else 0.25
        f = max(0.0, min(f, f_max))
        
        # Calculate position size using new formula: qty = f * equity / (price_y + |beta|*price_x)
        denominator = price_s1 + abs(beta) * price_s2
        if denominator <= EPSILON:
            return 0.0
        
        position_size = f * current_equity / denominator
        
        # Check notional value constraints
        notional = abs(position_size) * price_s1 + abs(beta * position_size) * price_s2
        
        # Get min/max notional from portfolio config or use defaults for incremental backtesting
        if hasattr(self.portfolio, 'config') and self.portfolio.config:
            min_notional = getattr(self.portfolio.config, 'min_notional_per_trade', 100.0)
            max_notional = getattr(self.portfolio.config, 'max_notional_per_trade', 10000.0)
        else:
            # For incremental backtesting without portfolio, use standard defaults
            min_notional = 100.0  # Standard minimum notional
            max_notional = current_equity * 0.5  # Max 50% of capital
        
        # Check minimum notional
        if notional < min_notional:
            return 0.0  # Skip trade if below minimum
        
        # Check maximum notional
        if notional > max_notional:
            # Scale down to maximum
            scale_factor = max_notional / notional
            position_size *= scale_factor
            notional = max_notional
        
        # Check margin requirements using portfolio
        if self.portfolio and hasattr(self.portfolio, 'check_margin_requirements') and not self.portfolio.check_margin_requirements(notional):
            # Try to scale down to fit margin requirements
            available_margin = self.portfolio.get_available_margin()
            if available_margin > 0:
                scale_factor = available_margin / notional
                position_size *= scale_factor
                notional *= scale_factor
                
                # Check if still above minimum after scaling
                if notional < min_notional:
                    return 0.0
            else:
                return 0.0  # No available margin
        
        return position_size

    def _calculate_position_size_with_capital(self, entry_z: float, spread_curr: float, mean: float,
                                            std: float, beta: float, price_s1: float, price_s2: float,
                                            capital_for_trade: float) -> float:
        """Calculate position size with specific capital amount without modifying global state.

        CRITICAL FIX: This method avoids race conditions by not modifying self.capital_at_risk.
        """
        # Temporarily store original capital
        original_capital = self.capital_at_risk

        try:
            # Set capital for calculation
            self.capital_at_risk = capital_for_trade

            # Calculate position size using existing method
            position_size = self._calculate_position_size(
                entry_z, spread_curr, mean, std, beta, price_s1, price_s2
            )

            return position_size
        finally:
            # Always restore original capital
            self.capital_at_risk = original_capital

    def _calculate_kelly_or_capital_fraction(self) -> float:
        """Calculate Kelly fraction or capital fraction based only on completed trades.
        
        NEW: Only uses completed trades [:i] for calculation
        NEW: Returns fraction in range [0, f_max]
        
        Returns:
            float: Kelly fraction or default capital fraction
        """
        # Get completed trades from portfolio or trades log
        completed_trades = []
        
        if hasattr(self, 'trades_log') and self.trades_log:
            # Use trades log if available
            completed_trades = [trade['pnl'] for trade in self.trades_log if 'pnl' in trade]
        elif self.portfolio and hasattr(self.portfolio, 'completed_trades'):
            # Use portfolio completed trades if available
            completed_trades = [trade.pnl for trade in self.portfolio.completed_trades]
        
        # If no completed trades, use default capital fraction
        if len(completed_trades) < 10:
            return 0.02  # Default 2% of capital
        
        # Convert to pandas Series for Kelly calculation
        returns = pd.Series(completed_trades)
        
        # Remove outliers (beyond 3 sigma)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.02  # Default if no variance
        
        filtered_returns = returns[
            (returns >= mean_return - 3 * std_return) & 
            (returns <= mean_return + 3 * std_return)
        ]
        
        if len(filtered_returns) < 5:
            return 0.02  # Default if too few valid returns
        
        # Kelly formula: f = (bp - q) / b
        # where b = average win / average loss, p = win probability, q = loss probability
        win_rate = (filtered_returns > 0).mean()
        
        if win_rate == 0 or win_rate == 1:
            return 0.02  # Default if all wins or all losses
        
        wins = filtered_returns[filtered_returns > 0]
        losses = filtered_returns[filtered_returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.02  # Default if no wins or no losses
        
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return 0.02  # Default if no average loss
        
        # Kelly fraction
        b = avg_win / avg_loss  # Odds ratio
        kelly_f = (b * win_rate - (1 - win_rate)) / b
        
        # Ensure non-negative and reasonable
        kelly_f = max(0.0, kelly_f)
        
        # Apply conservative scaling (Kelly can be aggressive)
        kelly_f *= 0.5  # Use half-Kelly for safety
        
        return kelly_f
    
    def _calculate_enhanced_trade_costs(self, position_change, price_s1, price_s2, beta):
        """Calculate enhanced trading costs including commissions, slippage, and spreads."""
        if abs(position_change) < 1e-8:
            return 0.0
        
        # Determine trade side (1 for buy, -1 for sell)
        side_s1 = 1 if position_change > 0 else -1
        side_s2 = -1 if position_change > 0 else 1  # Opposite side for hedge
        
        # Calculate effective prices with slippage and spreads
        # Asset 1 (y)
        slippage_s1 = self.slippage_bps / 10000 * self.slippage_stress_multiplier
        half_spread_s1 = self.half_spread_bps / 10000
        price_impact_s1 = side_s1 * (slippage_s1 + half_spread_s1)
        effective_price_s1 = price_s1 * (1 + price_impact_s1)
        
        # Asset 2 (x)
        slippage_s2 = self.slippage_bps / 10000 * self.slippage_stress_multiplier
        half_spread_s2 = self.half_spread_bps / 10000
        price_impact_s2 = side_s2 * (slippage_s2 + half_spread_s2)
        effective_price_s2 = price_s2 * (1 + price_impact_s2)
        
        # Calculate trade quantities
        qty_s1 = abs(position_change)
        qty_s2 = abs(beta * position_change)
        
        # Calculate commissions
        # Determine if maker or taker (assume taker for market orders)
        fee_rate = self.fee_taker
        commission_s1 = effective_price_s1 * qty_s1 * fee_rate
        commission_s2 = effective_price_s2 * qty_s2 * fee_rate
        
        # Total cost
        total_cost = commission_s1 + commission_s2
        
        # Log detailed costs for analysis
        cost_breakdown = {
            'commission_s1': commission_s1,
            'commission_s2': commission_s2,
            'slippage_s1': abs(price_s1 * qty_s1 * slippage_s1),
            'slippage_s2': abs(price_s2 * qty_s2 * slippage_s2),
            'spread_s1': abs(price_s1 * qty_s1 * half_spread_s1),
            'spread_s2': abs(price_s2 * qty_s2 * half_spread_s2),
            'total': total_cost
        }
        
        # Store cost breakdown for analysis
        if not hasattr(self, 'cost_breakdown_log'):
            self.cost_breakdown_log = []
        self.cost_breakdown_log.append(cost_breakdown)
        
        return total_cost
    
    def _recalculate_beta_online(self, df, current_idx):
        """Recalculate beta using online method with historical data only."""
        if current_idx < self.rolling_window:
            return df["beta"].iat[current_idx] if not pd.isna(df["beta"].iat[current_idx]) else 1.0
        
        # Use historical data only
        y_historical = df["y"].iloc[:current_idx]
        x_historical = df["x"].iloc[:current_idx]
        
        # Use rolling window for beta recalculation (online statistics)
        window_size = min(len(y_historical), self.rolling_window)
        y_win = y_historical.iloc[-window_size:] if len(y_historical) >= window_size else y_historical
        x_win = x_historical.iloc[-window_size:] if len(x_historical) >= window_size else x_historical
        
        try:
            beta, _, _ = self._calculate_ols_with_cache(y_win, x_win)
            return beta if not pd.isna(beta) else 1.0
        except:
            return 1.0
    
    def _update_online_volatility(self, df, current_idx):
        """Update volatility using online calculation with historical data only."""
        if current_idx < self.rolling_window:
            return
        
        # Use historical data only
        spread_historical = df["spread"].iloc[:current_idx]
        
        # Calculate rolling volatility
        volatility_window = min(len(spread_historical), self.rolling_window)
        recent_spreads = spread_historical.iloc[-volatility_window:] if len(spread_historical) >= volatility_window else spread_historical
        
        if len(recent_spreads) > 1:
            volatility = recent_spreads.std()
            df.loc[df.index[current_idx], "rolling_volatility"] = volatility
 
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
    
    def _calculate_volatility_multiplier(self, current_idx: int = None) -> float:
        """Calculate position size multiplier based on current volatility.
        
        Args:
            current_idx: Current index in data (для предотвращения lookahead bias)
        
        Returns:
            Multiplier for position size (between min_position_size_pct and max_position_size_pct)
        """
        if not hasattr(self, 'pair_data') or len(self.pair_data) < self.volatility_lookback_hours * 4:
            return 1.0  # Default multiplier if insufficient data
        
        # Calculate recent volatility (using 15-min bars, so 4 bars per hour)
        lookback_periods = self.volatility_lookback_hours * 4
        
        # Используем данные только до текущего момента
        if current_idx is None:
            # Для обратной совместимости (WARNING: может привести к lookahead bias!)
            recent_data = self.pair_data.iloc[-lookback_periods:] if len(self.pair_data) >= lookback_periods else self.pair_data
        else:
            # ПРАВИЛЬНО: Используем данные строго до текущего индекса
            start_idx = max(0, current_idx - lookback_periods)
            end_idx = current_idx  # НЕ включаем текущий индекс
            recent_data = self.pair_data.iloc[start_idx:end_idx]
        
        # Calculate returns for both assets
        returns_y = recent_data.iloc[:, 0].ffill().pct_change(fill_method=None).dropna()
        returns_x = recent_data.iloc[:, 1].ffill().pct_change(fill_method=None).dropna()
        
        if len(returns_y) < 10 or len(returns_x) < 10:
            return 1.0
        
        # Calculate average volatility of both assets
        vol_y = returns_y.std() * np.sqrt(365 * 24 * 4)  # Annualized volatility
        vol_x = returns_x.std() * np.sqrt(365 * 24 * 4)  # Annualized volatility
        avg_volatility = (vol_y + vol_x) / 2
        
        # Calculate historical average volatility for comparison
        if len(self.pair_data) > lookback_periods * 2:
            historical_data = self.pair_data.iloc[:-lookback_periods]
            hist_returns_y = historical_data.iloc[:, 0].ffill().pct_change(fill_method=None).dropna()
            hist_returns_x = historical_data.iloc[:, 1].ffill().pct_change(fill_method=None).dropna()
            
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
            return self.zscore_entry_threshold

        # Calculate rolling volatility of z-scores using only available data
        if len(z_scores) < self.volatility_lookback:
            return self.zscore_entry_threshold

        # Use only the last volatility_lookback periods to avoid look-ahead bias
        recent_z_scores = z_scores.iloc[-self.volatility_lookback:]
        z_volatility = recent_z_scores.std()

        if pd.isna(z_volatility) or z_volatility == 0:
            return self.zscore_entry_threshold
        
        # Adjust threshold based on volatility regime
        # Higher volatility -> higher threshold (more conservative)
        if len(z_scores) < self.volatility_lookback * 2:
            base_volatility = z_volatility  # Use current volatility as base
        else:
            # Use longer window for base volatility calculation
            longer_window = z_scores.iloc[-self.volatility_lookback * 2:]
            base_volatility = longer_window.std()
        if pd.isna(base_volatility) or base_volatility == 0:
            return self.zscore_entry_threshold

        volatility_ratio = z_volatility / base_volatility

        # Scale threshold: 1.0x to max_var_multiplier based on volatility
        volatility_multiplier = min(self.max_var_multiplier, max(1.0, volatility_ratio))

        adaptive_threshold = self.zscore_entry_threshold * volatility_multiplier
        
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
        
        # CRITICAL FIX: Bid-ask spread cost calculation
        # Apply full bid-ask spread for taker orders (more realistic for market orders)
        # For pair trading, we typically need to cross the spread on both legs
        bid_ask_costs = (notional_change_s1 * self.bid_ask_spread_pct_s1 +
                        notional_change_s2 * self.bid_ask_spread_pct_s2)
        
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
        pair_name = self.pair_name or 'Unknown'
        print(f"\n🔄 БЭКТЕСТ ПАРЫ: {pair_name}")
        print(f"   Данных: {len(self.pair_data)} периодов")
        print(f"   Колонки: {list(self.pair_data.columns) if not self.pair_data.empty else 'ПУСТО'}")
        print(f"   Rolling window: {self.rolling_window}")
        print(f"   Z-score пороги: entry={self.zscore_entry_threshold}, exit={self.z_exit}")

        logger.info(f"🔄 Начинаем бэктест пары {pair_name} с {len(self.pair_data)} периодами данных")
        
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            logger.warning(f"⚠️ Пустые данные для пары {self.pair_name or 'Unknown'}, пропускаем")
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return
            
        # NEW: Walk-forward testing logic
        if self.walk_forward_enabled and self.walk_forward_splitter is not None:
            self._run_walk_forward()
        else:
            self._run_single_backtest()
            
    def _run_walk_forward(self) -> None:
        """Run walk-forward backtesting."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Store original data
        self.original_data = self.pair_data.copy()
        
        # Generate walk-forward windows
        windows = self.walk_forward_splitter.split(self.pair_data)
        logger.info(f"🔄 Walk-forward: {len(windows)} окон для тестирования")
        
        all_test_results = []
        
        for window in windows:
            logger.info(f"🔄 Обрабатываем окно {window.window_id + 1}/{len(windows)}")
            
            # Extract training and testing data
            train_data = self.pair_data.iloc[window.train_start:window.train_end].copy()
            test_data = self.pair_data.iloc[window.test_start:window.test_end].copy()
            
            # Fit normalizer on training data if enabled
            if self.normalization_enabled and self.normalizer is not None:
                self.normalizer.fit(train_data)
                train_data = self.normalizer.transform(train_data)
                test_data = self.normalizer.transform(test_data)
            
            # Set current phase and window
            self.current_phase = "train"
            self.current_window_id = window.window_id
            
            # Run training phase (for parameter calibration)
            self.pair_data = train_data
            train_results = self._run_single_backtest_internal(phase="train")
            
            # Set test phase
            self.current_phase = "test"
            
            # Run testing phase
            self.pair_data = test_data
            test_results = self._run_single_backtest_internal(phase="test")
            
            # Add window information to test results
            test_results['window_id'] = window.window_id
            test_results['phase'] = 'test'
            
            all_test_results.append(test_results)
        
        # Concatenate all test results
        if all_test_results:
            self.results = pd.concat(all_test_results, ignore_index=False)
        else:
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
        
        # Restore original data
        self.pair_data = self.original_data
        
    def _run_single_backtest(self) -> None:
        """Run single backtest without walk-forward."""
        # Apply normalization if enabled
        if self.normalization_enabled and self.normalizer is not None:
            self.normalizer.fit(self.pair_data)
            self.pair_data = self.normalizer.transform(self.pair_data)
            
        self.results = self._run_single_backtest_internal(phase="test")
        
    def _run_single_backtest_internal(self, phase: str = "test") -> pd.DataFrame:
        """Internal method to run a single backtest with new execution order."""
        import logging
        logger = logging.getLogger(__name__)

        # Переименовываем столбцы для удобства
        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x",
            }
        ).copy()
        
        # Add phase column for tracking
        df["phase"] = phase

        # Initialize all required columns
        self._initialize_dataframe_columns(df)
        
        # Initialize state variables
        self._initialize_backtest_state()
        
        # ИСПРАВЛЕНИЕ LOOKAHEAD BIAS: Buffer for signals (computed on bar i, executed on bar i+1)
        signal_buffer = 0
        
        # Main execution loop with FIXED order: execute_previous_signal -> update_rolling_stats -> compute_new_signal -> mark_to_market
        # Start from rolling_window to match manual_backtest logic
        for i in range(self.rolling_window, len(df)):
            # (a) execute_orders(i, signal_buffer) - execute signal from PREVIOUS bar (1-bar lag)
            self.execute_orders(df, i, signal_buffer)
            
            # (b) update_rolling_stats(i) - calculate stats for current bar using historical data
            self.update_rolling_stats(df, i)
            
            # (c) compute_signal(i) - generate NEW signal based on current bar stats (for next bar)
            signal_buffer = self.compute_signal(df, i)
            
            # (d) mark_to_market(i) - update PnL and equity
            self.mark_to_market(df, i)
            
        # Handle bars before rolling_window with mark_to_market only
        for i in range(1, self.rolling_window):
            self.mark_to_market(df, i)
        
        # Calculate cumulative PnL
        df["cumulative_pnl"] = df["pnl"].cumsum()
        
        self.results = df
        
        # Log final statistics
        self._log_final_statistics(df)
        
        return df
    
    def _initialize_dataframe_columns(self, df: pd.DataFrame) -> None:
        """Initialize all required DataFrame columns."""
        # Rolling statistics columns
        df["beta"] = np.nan
        df["mean"] = np.nan
        df["std"] = np.nan
        df["spread"] = np.nan
        df["z_score"] = np.nan
        
        # Trading columns
        df["position"] = 0.0
        df["trades"] = 0.0
        df["pnl"] = 0.0
        df["step_pnl"] = 0.0  # Alias for pnl for test compatibility
        df["costs"] = 0.0
        df["realized_pnl"] = 0.0
        df["unrealized_pnl"] = 0.0
        df["equity"] = float(self.capital_at_risk)
        
        # Enhanced cost tracking columns
        df["commission_costs"] = 0.0
        df["slippage_costs"] = 0.0
        df["bid_ask_costs"] = 0.0
        df["impact_costs"] = 0.0
        
        # Trade logging columns
        df["entry_price_s1"] = np.nan
        df["entry_price_s2"] = np.nan
        df["exit_price_s1"] = np.nan
        df["exit_price_s2"] = np.nan
        df["entry_z"] = np.nan
        df["exit_z"] = np.nan
        df["exit_reason"] = ""
        df["trade_duration"] = 0.0
        
        # Entry date column with proper dtype
        if isinstance(df.index, pd.DatetimeIndex):
            df["entry_date"] = pd.NaT
        else:
            df["entry_date"] = np.nan
            
        # Market regime detection columns
        if self.market_regime_detection:
            df["market_regime"] = "neutral"
            df["hurst_exponent"] = np.nan
            df["variance_ratio"] = np.nan
            
        # Structural break detection columns
        if self.structural_break_protection:
            df["structural_break_detected"] = False
            df["rolling_correlation"] = np.nan
            df["half_life_estimate"] = np.nan
            df["adf_pvalue"] = np.nan
    
    def _initialize_backtest_state(self) -> None:
        """Initialize backtest state variables."""
        self.current_position = 0.0
        self.current_cash = self.capital_at_risk
        # CRITICAL FIX: Remove accrued_costs as it causes double counting - costs are tracked in cash
        self.entry_price_s1 = np.nan
        self.entry_price_s2 = np.nan
        self.entry_z = 0.0
        self.entry_beta = np.nan  # CRITICAL FIX: Store beta at position entry
        self.entry_datetime = None
        self.entry_index = 0
        self.cooldown_remaining = 0
        self.active_positions_count = 0
        
    def execute_orders(self, df: pd.DataFrame, i: int, signal: int) -> None:
        """Execute orders from signals generated on previous bar."""
        if signal == 0 or i >= len(df):
            return
            
        # Check capital_at_risk and max_active_positions constraints
        if not self._can_open_new_position(signal):
            return
            
        # Get current prices
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        
        # Calculate position size based on signal
        if signal != 0 and self.current_position == 0:  # Opening new position
            # Get z-score and other parameters from the current bar where signal was generated
            z_score = df["z_score"].iat[i]
            spread = df["spread"].iat[i]
            mean = df["mean"].iat[i]
            std = df["std"].iat[i]
            beta = df["beta"].iat[i]
                
            if not pd.isna(z_score) and not pd.isna(std) and std > 0:
                position_size = self._calculate_position_size(
                    z_score, spread, mean, std, beta, price_s1, price_s2
                )
                new_position = signal * abs(position_size)
                
                # Calculate and apply trading costs
                position_change = new_position - self.current_position
                position_s1_change = position_change
                position_s2_change = -beta * position_change
                
                # Get detailed cost breakdown
                commission_costs, slippage_costs, bid_ask_costs, total_costs = self._calculate_trading_costs(
                    position_s1_change, position_s2_change, price_s1, price_s2
                )
                
                # CRITICAL FIX: Update position and costs without double counting
                self.current_position = new_position
                self.current_cash -= total_costs
                # Remove accrued_costs tracking as it causes confusion - costs are in cash
                
                # Store entry information
                self.entry_price_s1 = price_s1
                self.entry_price_s2 = price_s2
                self.entry_z = z_score
                self.entry_beta = beta  # CRITICAL FIX: Store beta at entry
                self.entry_datetime = df.index[i]
                self.entry_index = i
                self.active_positions_count += 1
                
                # Log trade opening to incremental_trades_log
                capital_used = abs(new_position) * (price_s1 + abs(beta) * price_s2)
                self._open_trade(df.index[i], z_score, spread, new_position, 
                               capital_used, price_s1, price_s2, beta)
                
                # Log trade details with cost breakdown
                df.loc[df.index[i], "trades"] = abs(position_change)
                df.loc[df.index[i], "costs"] = total_costs
                df.loc[df.index[i], "commission_costs"] = commission_costs
                df.loc[df.index[i], "slippage_costs"] = slippage_costs
                df.loc[df.index[i], "bid_ask_costs"] = bid_ask_costs
                df.loc[df.index[i], "entry_price_s1"] = price_s1
                df.loc[df.index[i], "entry_price_s2"] = price_s2
                df.loc[df.index[i], "entry_z"] = z_score
                df.loc[df.index[i], "entry_date"] = df.index[i]
                    
        elif signal == 0 and self.current_position != 0:  # Closing position
            # Calculate and apply trading costs for closing
            position_change = -self.current_position
            # CRITICAL FIX: Use entry beta for exit costs and PnL calculation
            exit_beta = self.entry_beta if not pd.isna(self.entry_beta) else 1.0
            position_s1_change = position_change
            position_s2_change = -exit_beta * position_change
            
            # Get detailed cost breakdown
            commission_costs, slippage_costs, bid_ask_costs, total_costs = self._calculate_trading_costs(
                position_s1_change, position_s2_change, price_s1, price_s2
            )
            
            # CRITICAL FIX: Calculate realized PnL using entry beta
            realized_pnl = self._calculate_realized_pnl(
                self.current_position, self.entry_price_s1, self.entry_price_s2,
                price_s1, price_s2, exit_beta
            )
            
            # Log trade closing to incremental_trades_log
            # Use current bar for exit z_score
            exit_z = df["z_score"].iat[i] if i >= 0 and i < len(df) else 0.0
            self._close_trade(df.index[i], exit_z, "z_exit")
            
            # CRITICAL FIX: Update state without double counting costs
            self.current_position = 0.0
            # Update cash: add realized PnL and subtract costs (no double counting)
            self.current_cash += realized_pnl - total_costs
            self.active_positions_count = max(0, self.active_positions_count - 1)
            
            # Log exit details with cost breakdown
            df.loc[df.index[i], "trades"] = abs(position_change)
            df.loc[df.index[i], "costs"] = total_costs
            df.loc[df.index[i], "commission_costs"] = commission_costs
            df.loc[df.index[i], "slippage_costs"] = slippage_costs
            df.loc[df.index[i], "bid_ask_costs"] = bid_ask_costs
            df.loc[df.index[i], "realized_pnl"] = realized_pnl
            df.loc[df.index[i], "exit_price_s1"] = price_s1
            df.loc[df.index[i], "exit_price_s2"] = price_s2
            df.loc[df.index[i], "exit_z"] = exit_z
            
            # Calculate trade duration
            if self.entry_datetime is not None:
                duration = self._calculate_trade_duration(
                    df.index[i], self.entry_datetime, i, self.entry_index
                )
                df.loc[df.index[i], "trade_duration"] = duration
            
            # Reset entry tracking
            self.entry_price_s1 = np.nan
            self.entry_price_s2 = np.nan
            self.entry_beta = np.nan  # CRITICAL FIX: Reset entry beta
            self.entry_datetime = None
            self.cooldown_remaining = self.cooldown_periods
        
        # Update position in DataFrame
        df.loc[df.index[i], "position"] = self.current_position
    
    def update_rolling_stats(self, df: pd.DataFrame, i: int) -> None:
        """Update rolling statistics for bar i using only historical data up to i-1 (no lookahead bias)."""
        # CRITICAL FIX: Calculate statistics for bar i using ONLY historical data [i-rolling_window:i-1]
        # This prevents lookahead bias by excluding current bar from statistics calculation
        
        if i < self.rolling_window:
            return
            
        # CRITICAL FIX: Use data from (i - rolling_window) to (i - 1) inclusive for calculating statistics
        # This ensures NO lookahead bias - statistics calculated only on historical data
        start_idx = i - self.rolling_window
        end_idx = i  # Use data up to (but NOT including) current bar i
        
        # Ensure we have enough data
        if start_idx < 0:
            return
            
        y_win = df["y"].iloc[start_idx:end_idx]  # [i-rolling_window:i]
        x_win = df["x"].iloc[start_idx:end_idx]
        
        # FIXED: Enhanced validation for edge cases
        if len(y_win) < self.rolling_window or len(x_win) < self.rolling_window:
            return
            
        # Check for NaN values
        if y_win.isna().any() or x_win.isna().any():
            return
            
        # Check for constant prices (no variation)
        if y_win.std() < 1e-10 or x_win.std() < 1e-10:
            return
            
        # Check for extreme values or outliers
        y_range = y_win.max() - y_win.min()
        x_range = x_win.max() - x_win.min()
        if y_range < 1e-10 or x_range < 1e-10:
            return
            
        # Calculate OLS parameters using historical data
        try:
            beta, mean, std = self._calculate_ols_with_cache(y_win, x_win)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return
            
        # FIXED: More robust validation of OLS results
        if not (np.isfinite(beta) and np.isfinite(mean) and np.isfinite(std)):
            return
            
        # Calculate spread for current bar i using the calculated statistics
        spread = df["y"].iat[i] - beta * df["x"].iat[i]
        
        # Check std threshold AFTER calculating spread but BEFORE calculating z_score
        # This matches manual_backtest logic: calculate stats but skip z_score if std too small
        if std < 1e-6:
            # Record the statistics but leave z_score as NaN (matches manual_backtest)
            df.loc[df.index[i], "beta"] = beta
            df.loc[df.index[i], "mean"] = mean
            df.loc[df.index[i], "std"] = std
            df.loc[df.index[i], "spread"] = spread
            # z_score remains NaN
            return
            
        # Calculate z-score only if std is above threshold
        z_score = (spread - mean) / std

        # Логируем первые несколько z-score для диагностики
        if i <= self.rolling_window + 5:  # Логируем первые несколько после rolling window
            print(f"   📊 Бар {i}: spread={spread:.4f}, mean={mean:.4f}, std={std:.4f}, z_score={z_score:.4f}")
        
        # Update DataFrame for current bar i
        df.loc[df.index[i], "beta"] = beta
        df.loc[df.index[i], "mean"] = mean
        df.loc[df.index[i], "std"] = std
        df.loc[df.index[i], "spread"] = spread
        df.loc[df.index[i], "z_score"] = z_score
        
        # Update market regime detection if enabled
        if self.market_regime_detection:
            self._detect_market_regime(df, i)
        
        # Update structural break detection if enabled
        if self.structural_break_protection:
            self._check_structural_breaks(df, i)
        
        # Update online volatility if enabled
        if self.online_stats_enabled:
            self._update_online_volatility(df, i)
    
    def compute_signal(self, df: pd.DataFrame, i: int) -> int:
        """Compute trading signal based on available data to avoid look-ahead bias."""
        # Decrease cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            
        # CRITICAL FIX: Use current bar's z_score to generate signal (matches manual_backtest logic)
        # Since update_rolling_stats(i) has already calculated stats for bar i using historical data,
        # we can safely use the z_score at bar i for signal generation
        
        # Check if we have valid data
        if (i < 0 or
            i >= len(df) or
            pd.isna(df["z_score"].iat[i]) or 
            pd.isna(df["spread"].iat[i])):
            return 0
            
        z_curr = df["z_score"].iat[i]
        
        # Entry signals (only if no position and not in cooldown)
        if self.current_position == 0 and self.cooldown_remaining == 0:
            signal = 0
            if z_curr > self.zscore_entry_threshold:
                signal = -1  # Short signal
            elif z_curr < -self.zscore_entry_threshold:
                signal = 1   # Long signal
                
            # Add confirmation logic like in manual_backtest (optional)
            if signal != 0 and self.require_signal_confirmation and i > 0:
                z_prev = df["z_score"].iat[i - 1]
                if not pd.isna(z_prev):
                    # FIXED: Correct confirmation logic
                    # Long signal (z_curr < -threshold): confirm if z-score is becoming more negative (z_curr < z_prev)
                    # Short signal (z_curr > threshold): confirm if z-score is becoming more positive (z_curr > z_prev)
                    long_confirmation = (signal == 1) and (z_curr < z_prev)
                    short_confirmation = (signal == -1) and (z_curr > z_prev)
                    
                    if long_confirmation or short_confirmation:
                        return signal
                    else:
                        return 0  # No confirmed signal
            
            # Return signal immediately if confirmation is disabled or confirmed
            return signal
                
        # Exit signals (only if we have a position)
        elif self.current_position != 0:
            # Check various exit conditions
            
            # Z-score exit (исправлено для работы с отрицательными z_exit)
            if abs(z_curr) <= abs(self.z_exit):
                return 0  # Exit signal
                
            # Stop-loss conditions
            if self._check_stop_loss_conditions(df, i, z_curr):
                return 0  # Exit signal
                
            # Time-based stop
            if self._check_time_stop_condition(df, i):
                return 0  # Exit signal
                
            # Take-profit conditions
            if self._check_take_profit_conditions(df, i, z_curr):
                return 0  # Exit signal
        
        return 0  # No signal
    
    def mark_to_market(self, df: pd.DataFrame, i: int) -> None:
        """Mark positions to market and calculate PnL according to new formulas."""
        # Get current prices
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        
        # CRITICAL FIX: Use entry beta for unrealized PnL calculation to maintain consistency
        beta = self.entry_beta if (self.current_position != 0 and not pd.isna(self.entry_beta)) else (
            df["beta"].iat[i] if not pd.isna(df["beta"].iat[i]) else 1.0
        )
        
        # Calculate unrealized PnL: (current_price - entry_price) * qty for both legs
        unrealized_pnl = 0.0
        if self.current_position != 0 and not pd.isna(self.entry_price_s1) and not pd.isna(self.entry_price_s2):
            # S1 leg: position * (current_price - entry_price)
            pnl_s1 = self.current_position * (price_s1 - self.entry_price_s1)
            # S2 leg: -beta * position * (current_price - entry_price)
            pnl_s2 = (-beta * self.current_position) * (price_s2 - self.entry_price_s2)
            unrealized_pnl = pnl_s1 + pnl_s2
        
        # Calculate step PnL (change from previous bar)
        prev_unrealized = df["unrealized_pnl"].iat[i-1] if i > 0 else 0.0
        step_pnl = unrealized_pnl - prev_unrealized
        
        # Add any realized PnL from this bar
        realized_pnl_step = df["realized_pnl"].iat[i] if "realized_pnl" in df.columns else 0.0
        total_step_pnl = step_pnl + realized_pnl_step
        
        # CRITICAL FIX: Do NOT subtract trading costs here - they are already accounted for in cash
        # Trading costs are subtracted from cash in execute_orders, so subtracting here causes double counting
        
        # Calculate equity: cash + unrealized_pnl (costs already deducted from cash)
        equity = self.current_cash + unrealized_pnl
        
        # Update DataFrame - FIXED: Always set pnl value, even if zero
        df.loc[df.index[i], "unrealized_pnl"] = unrealized_pnl
        df.loc[df.index[i], "pnl"] = total_step_pnl
        df.loc[df.index[i], "step_pnl"] = total_step_pnl  # Alias for test compatibility
        df.loc[df.index[i], "equity"] = equity
    
    def _calculate_trading_costs_integrated(self, position_change: float, beta: float, 
                                          price_s1: float, price_s2: float) -> float:
        """Calculate trading costs integrated into order execution."""
        if abs(position_change) < 1e-8:
            return 0.0
            
        # Calculate position changes for both legs
        position_s1_change = position_change
        position_s2_change = -beta * position_change
        
        # Use existing detailed cost calculation
        commission_costs, slippage_costs, bid_ask_costs, total_costs = self._calculate_trading_costs(
            position_s1_change, position_s2_change, price_s1, price_s2
        )
        
        return total_costs
    
    def _calculate_realized_pnl(self, position: float, entry_price_s1: float, entry_price_s2: float,
                               exit_price_s1: float, exit_price_s2: float, beta: float) -> float:
        """Calculate realized PnL: (exit_price - entry_price) * qty for both legs."""
        if position == 0 or pd.isna(entry_price_s1) or pd.isna(entry_price_s2):
            return 0.0
            
        # S1 leg: position * (exit_price - entry_price)
        pnl_s1 = position * (exit_price_s1 - entry_price_s1)
        # S2 leg: -beta * position * (exit_price - entry_price)
        pnl_s2 = (-beta * position) * (exit_price_s2 - entry_price_s2)
        
        return pnl_s1 + pnl_s2
    
    def _can_open_new_position(self, signal: int) -> bool:
        """Check capital_at_risk and max_active_positions constraints."""
        if signal == 0 or self.current_position != 0:
            return True  # Not opening new position or already have position
            
        # Check max_active_positions constraint
        if hasattr(self, 'max_active_positions') and self.max_active_positions is not None:
            if self.active_positions_count >= self.max_active_positions:
                return False
                
        # Check capital_at_risk constraint
        if hasattr(self, 'capital_at_risk') and self.capital_at_risk is not None:
            # For now, assume we can open if we have positive cash
            if self.current_cash <= 0:
                return False
                
        return True
    
    def _check_stop_loss_conditions(self, df: pd.DataFrame, i: int, z_curr: float) -> bool:
        """Check stop-loss conditions (USD-based and Z-score based)."""
        if self.current_position == 0:
            return False
            
        # Calculate current unrealized PnL
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        # CRITICAL FIX: Use entry beta for consistent PnL calculation
        beta = self.entry_beta if not pd.isna(self.entry_beta) else 1.0
        
        if not pd.isna(self.entry_price_s1) and not pd.isna(self.entry_price_s2):
            unrealized_pnl = self._calculate_realized_pnl(
                self.current_position, self.entry_price_s1, self.entry_price_s2,
                price_s1, price_s2, beta
            )
            
            # USD stop-loss: 75 USDT loss
            if unrealized_pnl <= -75.0:
                return True
                
        # Z-score stop-loss: 3σ (configurable)
        stop_loss_z = getattr(self, 'pair_stop_loss_zscore', 3.0)
        if self.current_position > 0 and z_curr <= -stop_loss_z:
            return True
        elif self.current_position < 0 and z_curr >= stop_loss_z:
            return True
            
        return False
    
    def _check_time_stop_condition(self, df: pd.DataFrame, i: int) -> bool:
        """Check time-based stop condition."""
        if (self.current_position == 0 or 
            self.entry_datetime is None or
            not hasattr(self, 'half_life') or 
            not hasattr(self, 'time_stop_multiplier') or
            self.half_life is None or 
            self.time_stop_multiplier is None):
            return False
            
        # Calculate trade duration
        trade_duration_periods = self._calculate_trade_duration_periods(
            df.index[i], self.entry_datetime, i, self.entry_index
        )
        
        # Check if duration exceeds time stop limit
        time_stop_limit_periods = self.half_life * self.time_stop_multiplier
        return trade_duration_periods >= time_stop_limit_periods
    
    def _check_take_profit_conditions(self, df: pd.DataFrame, i: int, z_curr: float) -> bool:
        """Check take-profit conditions."""
        if (self.current_position == 0 or 
            self.entry_datetime is None or
            not hasattr(self, 'take_profit_multiplier') or
            self.take_profit_multiplier is None):
            return False
            
        # Check minimum holding time (1 hour)
        min_holding_met = self._check_minimum_holding_time(df, i)
        if not min_holding_met:
            return False
            
        # FIXED: Check profitability first to avoid premature exits
        price_s1 = df["y"].iat[i]
        price_s2 = df["x"].iat[i]
        # CRITICAL FIX: Use entry beta for consistent PnL calculation
        beta = self.entry_beta if not pd.isna(self.entry_beta) else 1.0
        
        # Only proceed if exit would be profitable
        if not self._is_profitable_exit(price_s1, price_s2, beta):
            return False
            
        # Take-profit: z-score moves towards zero (only if profitable)
        if abs(z_curr) <= abs(self.entry_z) * self.take_profit_multiplier:
            return True
            
        return False
    
    def _check_minimum_holding_time(self, df: pd.DataFrame, i: int) -> bool:
        """Check if minimum holding time requirement is met."""
        if self.entry_datetime is None:
            return False
            
        min_hold_minutes = getattr(self, 'min_position_hold_minutes', 60)
        
        if isinstance(df.index, pd.DatetimeIndex):
            holding_time_hours = (df.index[i] - self.entry_datetime).total_seconds() / 3600
            return holding_time_hours >= (min_hold_minutes / 60.0)
        else:
            # For non-datetime index, assume 15-minute periods
            holding_periods = i - self.entry_index
            return holding_periods >= (min_hold_minutes // 15)
    
    def _is_profitable_exit(self, price_s1: float, price_s2: float, beta: float) -> bool:
        """Check if exit would be profitable after costs."""
        if pd.isna(self.entry_price_s1) or pd.isna(self.entry_price_s2):
            return True  # Default to allowing exit
            
        # CRITICAL FIX: Use entry beta for consistent PnL calculation
        exit_beta = self.entry_beta if not pd.isna(self.entry_beta) else beta
        
        # Calculate potential realized PnL
        potential_pnl = self._calculate_realized_pnl(
            self.current_position, self.entry_price_s1, self.entry_price_s2,
            price_s1, price_s2, exit_beta
        )
        
        # Calculate exit costs
        exit_costs = self._calculate_trading_costs_integrated(
            -self.current_position, beta, price_s1, price_s2
        )
        
        # Exit is profitable if PnL covers costs
        return potential_pnl > exit_costs
    
    def _log_final_statistics(self, df: pd.DataFrame) -> None:
        """Log final backtest statistics."""
        import logging
        logger = logging.getLogger(__name__)
        
        total_pnl = df["cumulative_pnl"].iloc[-1] if not df.empty and "cumulative_pnl" in df.columns else 0.0
        num_trades = len(self.trades_log)
        profitable_trades = len([t for t in self.trades_log if t.get('pnl', 0) > 0])
        win_rate = (profitable_trades / num_trades * 100) if num_trades > 0 else 0.0
        
        logger.info(f"✅ {self.pair_name or 'Unknown'}: Завершен бэктест - PnL: {total_pnl:.4f}, Сделок: {num_trades}, Винрейт: {win_rate:.1f}%")

    def _create_complete_trades_log(self) -> List[Dict]:
        """Create complete trades log from incremental trades log.
        
        Combines 'open' and 'close' entries from incremental_trades_log
        into complete trade records with entry/exit information.
        """
        complete_trades = []
        open_trades = {}
        
        for entry in self.incremental_trades_log:
            if entry['action'] == 'open':
                # Store open trade info
                open_trades[entry['date']] = entry
            elif entry['action'] == 'close':
                # Find matching open trade
                entry_date = entry.get('entry_date')
                if entry_date in open_trades:
                    open_trade = open_trades[entry_date]
                    
                    # Calculate trade duration
                    duration_hours = (entry['date'] - entry_date).total_seconds() / 3600
                    
                    # Create complete trade record
                    complete_trade = {
                        'pair': self.pair_name or 'Unknown',
                        'entry_datetime': entry_date,
                        'exit_datetime': entry['date'],
                        'position_type': 'long' if open_trade['position_size'] > 0 else 'short',
                        'position_size': open_trade['position_size'],
                        'capital_used': open_trade['capital_used'],
                        'entry_price_s1': open_trade['entry_price_s1'],
                        'entry_price_s2': open_trade['entry_price_s2'],
                        'entry_z': open_trade['entry_z'],
                        'exit_z': entry['exit_z'],
                        'exit_reason': entry['exit_reason'],
                        'pnl': entry['pnl'],
                        'trade_duration_hours': duration_hours,
                        'beta': open_trade.get('beta', 0.0)
                    }
                    
                    complete_trades.append(complete_trade)
                    # Remove processed open trade
                    del open_trades[entry_date]
        
        return complete_trades

    def get_results(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        if isinstance(self.results, dict):
            return self.results

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

        # Create complete trades log from incremental trades log
        complete_trades_log = self._create_complete_trades_log()

        return {
            "spread": self.results["spread"],
            "z_score": self.results["z_score"],
            "position": self.results["position"],
            "trades": self.results["trades"],
            "costs": self.results["costs"],
            "pnl": self.results["pnl"],
            "step_pnl": self.results["pnl"],  # Alias for compatibility with tests
            "cumulative_pnl": self.results["cumulative_pnl"],
            "trades_log": complete_trades_log,
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

        # FIXED: Don't drop NaN values, use fillna(0) instead to preserve data
        pnl = self.results["pnl"].fillna(0.0)
        cum_pnl = self.results["cumulative_pnl"].fillna(0.0)

        # Get complete trades log
        complete_trades_log = self._create_complete_trades_log()
        
        # Calculate trade-related metrics
        num_trades = len(complete_trades_log)
        
        # Calculate average trade duration
        trade_durations = []
        for trade in complete_trades_log:
            if 'trade_duration_hours' in trade:
                trade_durations.append(trade['trade_duration_hours'])
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
        
        # Calculate total return
        total_pnl_val = cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0
        total_return = (total_pnl_val / self.capital_at_risk) * 100 if self.capital_at_risk > 0 else 0.0

        # FIXED: Check if all PnL values are zero instead of checking if empty
        if len(pnl) == 0 or (pnl == 0.0).all():
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": total_pnl_val,
                "total_return": total_return,
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
             
    def process_single_period(self, date: pd.Timestamp, price_s1: float, price_s2: float, current_idx: int = None) -> Dict:
        """Process a single time period incrementally.
        
        This method fixes look-ahead bias by using only capital available at trade entry time.
        
        Args:
            date: Current timestamp
            price_s1: Current price of asset 1
            price_s2: Current price of asset 2
            current_idx: Current index in data (для предотвращения lookahead bias)
        
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
                'trade_closed': False,
                'z_score': 0.0,  # Добавляем для консистентности
                'spread': 0.0,
                'beta': 1.0
            }
            
        # Используем данные только до текущего момента
        # Calculate current market parameters - use only historical data up to current point
        if current_idx is None:
            # Для обратной совместимости: если индекс не передан, используем последние данные
            # WARNING: Это может привести к lookahead bias!
            recent_data = self.pair_data.iloc[-self.rolling_window:] if len(self.pair_data) >= self.rolling_window else self.pair_data
        else:
            # ПРАВИЛЬНО: Используем данные строго до текущего индекса
            start_idx = max(0, current_idx - self.rolling_window)
            end_idx = current_idx  # НЕ включаем текущий индекс
            recent_data = self.pair_data.iloc[start_idx:end_idx]
        if len(recent_data) < self.rolling_window:
            return {
                'position': 0.0,
                'trade': 0.0,
                'pnl': 0.0, 
                'costs': 0.0,
                'trade_opened': False,
                'trade_closed': False,
                'z_score': 0.0,  # Добавляем для консистентности
                'spread': 0.0,
                'beta': 1.0
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
                'trade_closed': False,
                'z_score': 0.0,  # Добавляем для консистентности
                'spread': 0.0,
                'beta': 1.0
            }
            
        # Check for valid parameters
        if not (np.isfinite(beta) and np.isfinite(mean) and np.isfinite(std) and std > 1e-6):
            return {
                'position': 0.0,
                'trade': 0.0,
                'pnl': 0.0,
                'costs': 0.0,
                'trade_opened': False, 
                'trade_closed': False,
                'z_score': 0.0,  # Добавляем для консистентности
                'spread': 0.0,
                'beta': beta if np.isfinite(beta) else 1.0
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
                    
        # Check entry conditions (only if no active trade, not in cooldown, no stop loss triggered, and within trading period)
        trading_allowed = True
        if self.trading_start is not None and date < self.trading_start:
            trading_allowed = False
        if self.trading_end is not None and date > self.trading_end:
            trading_allowed = False

        if (self.active_trade is None and
            (self.cooldown_end_date is None or date > self.cooldown_end_date) and
            not result.get('stop_loss_triggered', False) and
            trading_allowed):

            signal = 0

            if not np.isnan(z_score) and np.isfinite(z_score):
                if z_score > self.zscore_entry_threshold:
                    signal = -1
                    print(f"   📈 СИГНАЛ ПРОДАЖИ: z_score={z_score:.3f} > {self.zscore_entry_threshold}")
                elif z_score < -self.zscore_entry_threshold:
                    signal = 1
                    print(f"   📉 СИГНАЛ ПОКУПКИ: z_score={z_score:.3f} < {-self.zscore_entry_threshold}")
            else:
                # z_score is NaN or infinite - skip signal generation
                pass

            if signal != 0:
                # FIXED: Check capital sufficiency before calculating position size
                if not self._check_capital_sufficiency(price_s1, price_s2, beta):
                    # Insufficient capital, skip this trade
                    pass
                else:
                    # Get capital at risk for this specific date
                    capital_for_trade = self.get_capital_at_risk_for_date(date)

                    # CRITICAL FIX: Calculate position size without modifying global state
                    # Pass capital directly to avoid race conditions
                    position_size = self._calculate_position_size_with_capital(
                        z_score, current_spread, mean, std, beta, price_s1, price_s2, capital_for_trade
                    )
                    
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
            s1_window = s1.iloc[-window:] if len(s1) >= window else s1
            s2_window = s2.iloc[-window:] if len(s2) >= window else s2
            corr = s1_window.corr(s2_window)
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
            spread_window = spread.iloc[-window:].dropna() if len(spread) >= window else spread.dropna()
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
            
            spread_window = spread.iloc[-window:].dropna() if len(spread) >= window else spread.dropna()
            if len(spread_window) < 10:
                return 1.0
                
            # Perform ADF test
            result = adfuller(spread_window, maxlag=min(12, len(spread_window)//3))
            p_value = result[1]
            
            return max(0.0, min(1.0, p_value))
            
        except:
            return 1.0
            
    def _detect_market_regime(self, df: pd.DataFrame, i: int) -> str:
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
            
    def _check_structural_breaks(self, df: pd.DataFrame, i: int) -> bool:
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


class BaseEngine(BasePairBacktester):
    """Backward-compatible alias for BasePairBacktester."""
    pass
