"""Configuration utilities using Pydantic models."""

from pathlib import Path

import yaml as pyyaml  # type: ignore
from pydantic import BaseModel, DirectoryPath, Field, model_validator  # type: ignore


class PairSelectionConfig(BaseModel):
    """Configuration for pair selection parameters."""

    lookback_days: int
    # DEPRECATED: Filters moved to FilterParamsConfig
    coint_pvalue_threshold: float | None = None 
    adf_pvalue_threshold: float | None = None
    min_half_life_days: float | None = None
    max_half_life_days: float | None = None
    min_mean_crossings: int | None = None
    ssd_top_n: int

    # Extended filters
    adaptive_quantiles: bool | None = None
    bar_minutes: int | None = None
    liquidity_usd_daily: float | None = None
    max_bid_ask_pct: float | None = None
    max_avg_funding_pct: float | None = None
    save_filter_reasons: bool | None = None
    max_hurst_exponent: float | None = 0.5
    min_abs_spread_mult: float | None = None
    cost_filter: bool | None = None
    kpss_pvalue_threshold: float | None = None
    pvalue_top_n: int | None = None
    save_std_histogram: bool | None = None
    
    # NEW: Exotic filter flags
    use_kpss_filter: bool = False
    use_hurst_filter: bool = False
    
    # NEW: Pair tradeability filter parameters
    enable_pair_tradeability_filter: bool = True
    min_volume_usd_24h: float = 20_000_000  # 20M USD daily volume
    min_days_live: int = 30  # 30 days since listing
    max_funding_rate_abs: float = 0.0003  # 0.03% absolute funding rate
    max_tick_size_pct: float = 0.0005  # 0.05% tick size relative to price
    max_half_life_hours: float = 72.0  # 72 hours max half-life

    # NEW: Whitelist/Blacklist Filtering
    allowed_quotes: list[str] = Field(default_factory=lambda: ['USDT', 'USDC', 'DAI'])
    blocked_assets: list[str] = Field(default_factory=lambda: ['BRZ', 'EUR', 'METH', 'STETH', 'WBTC'])
    
    # NEW: Pair Universe Configuration
    pair_universe: dict | None = None
    max_total_pairs: int = Field(default=1000, ge=1)  # Global limit on number of pairs
    
    # -------- Validators --------
    @model_validator(mode="after")
    def _check_cost_filter_params(self):  # type: ignore
        """Ensure required parameters are present when `cost_filter` is enabled."""
        if self.cost_filter:
            if self.min_abs_spread_mult is None:
                raise ValueError(
                    "`min_abs_spread_mult` must be specified in the config when `cost_filter` is true"
                )
        return self


class FilterParamsConfig(BaseModel):
    """Configurable thresholds for pair filtering."""

    pvalue_threshold: float = 0.05
    min_beta: float = 0.1
    max_beta: float = 10.0
    min_half_life_days: float = 1.0
    max_half_life_days: float = 252.0
    max_half_life_bars: int | None = None # NEW
    max_hurst_exponent: float = 0.5
    min_mean_crossings: int = 10
    min_daily_volume_usd: float = 50000.0
    
    # Exotic/New filters
    use_kpss_filter: bool = False
    use_hurst_filter: bool = False
    min_profit_potential_pct: float = 0.0
    exclude_same_base_stables: bool = False
    stablecoins: list[str] = Field(default_factory=lambda: ["USDT", "USDC", "DAI"])


class DataProcessingConfig(BaseModel):
    """Configuration for data processing and normalization."""
    
    normalization_method: str = "minmax"
    fill_method: str = "ffill"
    min_history_ratio: float = 0.8
    handle_constant: bool = True


class PortfolioConfig(BaseModel):
    """Configuration for portfolio and risk management."""

    initial_capital: float
    risk_per_position_pct: float
    max_active_positions: int
    max_margin_usage: float = 1.0
    # NEW: Leverage and margin requirements
    leverage_limit: float = 2.0
    # NEW: Position sizing parameters
    f_max: float = 0.25  # Maximum Kelly fraction
    min_notional_per_trade: float = 100.0  # Minimum notional value per trade
    max_notional_per_trade: float = 10000.0  # Maximum notional value per trade
    # NEW: Volatility-based position sizing parameters
    volatility_based_sizing: bool = False
    volatility_lookback_hours: int = 24
    min_position_size_pct: float = 0.005
    max_position_size_pct: float = 0.02
    volatility_adjustment_factor: float = 2.0

    # -------- Validators --------
    @model_validator(mode="after")
    def _validate_portfolio_params(self):  # type: ignore
        """Validate portfolio configuration parameters."""
        if self.max_margin_usage <= 0:
            raise ValueError("`max_margin_usage` must be greater than 0")
        
        if not (0 < self.risk_per_position_pct <= 1):
            raise ValueError("`risk_per_position_pct` must be between 0 and 1")
        
        if self.leverage_limit <= 0:
            raise ValueError("`leverage_limit` must be greater than 0")
        
        if not (0 <= self.f_max <= 1):
            raise ValueError("`f_max` must be between 0 and 1")
        
        if self.min_notional_per_trade <= 0:
            raise ValueError("`min_notional_per_trade` must be greater than 0")
        
        if self.max_notional_per_trade <= self.min_notional_per_trade:
            raise ValueError("`max_notional_per_trade` must be greater than `min_notional_per_trade`")
        
        return self


class BacktestConfig(BaseModel):
    """Configuration for backtesting parameters."""

    timeframe: str
    rolling_window: int
    # REMOVED: zscore_threshold (deprecated, use zscore_entry_threshold)
    stop_loss_multiplier: float
    # NEW: Unified Risk Management
    stop_loss_type: str = Field(default="zscore", description="zscore, percentage, atr, mixed")
    pnl_stop_loss_r_multiple: float = Field(default=1.5, gt=0.0)
    
    fill_limit_pct: float = Field(..., ge=0.0, le=1.0)
    # REMOVED: commission_pct (use commission_rate_per_leg)
    # REMOVED: slippage_pct (use slippage_pct only)
    slippage_pct: float = Field(default=0.0005, ge=0.0) # 0.05%
    annualizing_factor: int  # Новое поле
    time_stop_multiplier: float | None = None
    z_exit: float = Field(default=0.0) # Unified exit threshold
    take_profit_multiplier: float | None = None
    cooldown_hours: int = 0
    # NEW: Enhanced risk management parameters
    use_kelly_sizing: bool = True
    max_kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0) # Relaxed ge=0.0 to allow 0.0
    risk_per_position_pct: float | None = None # Added to allow override here
    volatility_lookback: int = Field(default=96, ge=10)
    adaptive_thresholds: bool = True
    var_confidence: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_var_multiplier: float = Field(default=3.0, gt=1.0)
    
    # NEW: Risk Management Object (from config file structure)
    risk_management: dict | None = None
    
    # NEW: Enhanced entry/exit rules
    zscore_entry_threshold: float = Field(default=2.3, gt=0.0)  # New higher entry threshold
    max_zscore_entry: float = Field(default=100.0, gt=0.0) # Max z-score for entry
    zscore_stop_loss: float = Field(default=4.0, gt=0.0) # Z-score stop loss
    min_spread_move_sigma: float = Field(default=1.2, gt=0.0)  # Minimum spread movement since last flat
    min_position_hold_minutes: int = Field(default=60, ge=0)  # Minimum hold time in minutes
    anti_churn_cooldown_minutes: int = Field(default=60, ge=0)  # Anti-churn protection
    
    # NEW: Enhanced stop-loss rules
    pair_stop_loss_usd: float | None = Field(default=None)  # Pair-level stop loss in USD (Optional)
    pair_stop_loss_zscore: float = Field(default=3.0, gt=0.0)  # Z-score based stop loss
    pair_step_r_limit: float | None = Field(default=None) # NEW: Hard limit for pair PnL per step
    portfolio_daily_stop_pct: float = Field(default=0.02, gt=0.0, lt=1.0)  # 2% daily portfolio stop
    
    # NEW: Time-based filters
    enable_funding_time_filter: bool = True
    funding_blackout_minutes: int = Field(default=30, ge=0)  # Minutes before/after funding reset
    funding_reset_hours: list[int] = Field(default_factory=lambda: [0, 8, 16])  # UTC hours for funding reset
    enable_macro_event_filter: bool = True
    macro_blackout_minutes: int = Field(default=30, ge=0)  # Minutes after macro events
    
    # NEW: Pair quarantine system
    enable_pair_quarantine: bool = True
    quarantine_pnl_threshold_sigma: float = Field(default=3.0, gt=0.0)  # -3 sigma PnL threshold
    quarantine_drawdown_threshold_pct: float = Field(default=0.08, gt=0.0, lt=1.0)  # 8% drawdown threshold
    quarantine_period_days: int = Field(default=7, ge=1)  # 7 days quarantine
    quarantine_rolling_window_days: int = Field(default=30, ge=1)  # 30 days rolling window for PnL stats
    
    # NEW: Trade Limits
    max_round_trips_per_pair_step: int = Field(default=1000, ge=0)
    max_new_entries_per_pair_day: int = Field(default=1000, ge=0)
    max_total_round_trips_per_step: int = Field(default=100000, ge=0)
    
    # NEW: Enhanced cost modeling
    enable_realistic_costs: bool = True
    commission_rate_per_leg: float = Field(default=0.0004, ge=0.0)  # 0.04% per leg (Default Source)
    slippage_half_spread_multiplier: float = Field(default=2.0, ge=0.0)  # 2x half spread for slippage
    funding_cost_enabled: bool = True
    
    # NEW: Enhanced position sizing with beta recalculation
    beta_recalc_frequency_hours: int = Field(default=48, ge=1)  # Recalculate beta every 48 hours
    beta_window_days_min: int = Field(default=60, ge=30)  # Minimum 60 days for beta calculation
    beta_window_days_max: int = Field(default=120, ge=60)  # Maximum 120 days for beta calculation
    use_ols_beta_sizing: bool = True  # Use OLS beta for position sizing
    
    # Market regime detection parameters
    market_regime_detection: bool = True
    hurst_window: int = Field(default=720, ge=100)  # Минимум 100 периодов для Hurst
    hurst_trending_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    variance_ratio_window: int = Field(default=480, ge=50)
    variance_ratio_trending_min: float = Field(default=1.2, gt=1.0)
    variance_ratio_mean_reverting_max: float = Field(default=0.8, lt=1.0, gt=0.0)
    
    # Structural break protection parameters
    structural_break_protection: bool = True
    cointegration_test_frequency: int = Field(default=2688, ge=100)  # Минимум 100 периодов
    adf_pvalue_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    exclusion_period_days: int = Field(default=30, ge=1)
    max_half_life_days: int = Field(default=10, ge=1)
    min_correlation_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    correlation_window: int = Field(default=720, ge=50)

    # -------- Validators --------
    @model_validator(mode="after")
    def _validate_backtest_params(self):  # type: ignore
        """Validate backtesting configuration parameters."""
        # Check zscore_entry_threshold > z_exit if both are set
        if self.z_exit is not None and self.zscore_entry_threshold <= self.z_exit:
            raise ValueError("`zscore_entry_threshold` must be greater than `z_exit`")
        
        # Check that all multipliers are positive
        if self.stop_loss_multiplier <= 0:
            raise ValueError("`stop_loss_multiplier` must be positive")
        
        if self.time_stop_multiplier is not None and self.time_stop_multiplier <= 0:
            raise ValueError("`time_stop_multiplier` must be positive")
        
        if self.take_profit_multiplier is not None and self.take_profit_multiplier <= 0:
            raise ValueError("`take_profit_multiplier` must be positive")
        
        # NEW: Validate enhanced risk management parameters
        if self.max_kelly_fraction < 0.0 or self.max_kelly_fraction > 1.0:
            raise ValueError("`max_kelly_fraction` must be between 0 and 1")
        
        if self.volatility_lookback < 10:
            raise ValueError("`volatility_lookback` must be at least 10 periods")
        
        if self.var_confidence <= 0 or self.var_confidence >= 1.0:
            raise ValueError("`var_confidence` must be between 0 and 1")
        
        if self.max_var_multiplier <= 1.0:
            raise ValueError("`max_var_multiplier` must be greater than 1.0")
        
        # Validate market regime detection parameters
        if self.hurst_window < 100:
            raise ValueError("`hurst_window` must be at least 100 periods")
        
        if self.variance_ratio_window < 50:
            raise ValueError("`variance_ratio_window` must be at least 50 periods")
        
        if self.variance_ratio_trending_min <= 1.0:
            raise ValueError("`variance_ratio_trending_min` must be greater than 1.0")
        
        if self.variance_ratio_mean_reverting_max >= 1.0:
            raise ValueError("`variance_ratio_mean_reverting_max` must be less than 1.0")
        
        # Validate structural break protection parameters
        if self.cointegration_test_frequency < 100:
            raise ValueError("`cointegration_test_frequency` must be at least 100 periods")
        
        if self.correlation_window < 50:
            raise ValueError("`correlation_window` must be at least 50 periods")
        
        return self


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward analysis."""

    start_date: str
    end_date: str
    training_period_days: int
    testing_period_days: int
    # NEW: Explicit step control parameters
    step_size_days: int | None = None
    max_steps: int | None = None


class PairUniverseConfig(BaseModel):
    """Configuration for pair universe and blacklists."""
    universe_path: str = "data/universe.json"
    max_pairs_per_symbol: int = Field(default=4, ge=1)
    max_total_pairs: int = Field(default=1000, ge=1)
    volatile_blacklist: list[str] = Field(default_factory=list)


class TradeLimitsConfig(BaseModel):
    """Configuration for trade limits."""
    max_round_trips_per_pair_step: int = Field(default=1000, ge=0)
    max_new_entries_per_pair_day: int = Field(default=1000, ge=0)
    max_total_round_trips_per_step: int = Field(default=100000, ge=0)


class RiskLimitsConfig(BaseModel):
    """Configuration for risk limits."""
    pair_step_r_multiple: float = Field(default=-3.0)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data_dir: DirectoryPath
    results_dir: Path
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    portfolio: PortfolioConfig
    pair_universe: PairUniverseConfig | None = None  # Moved to top-level
    pair_selection: PairSelectionConfig
    filter_params: FilterParamsConfig = Field(default_factory=FilterParamsConfig)
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig
    trade_limits: TradeLimitsConfig | None = None
    risk_limits: RiskLimitsConfig | None = None
    max_shards: int | None = None


def convert_paths_to_strings(data):
    """Recursively convert pathlib.Path objects to strings in a data structure.
    
    This is needed for YAML serialization since pathlib.Path objects
    cannot be directly serialized to YAML.
    
    Parameters
    ----------
    data : dict, list, or any
        Data structure that may contain Path objects
        
    Returns
    -------
    dict, list, or any
        Data structure with Path objects converted to strings
    """
    if isinstance(data, dict):
        return {key: convert_paths_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_paths_to_strings(item) for item in data]
    elif isinstance(data, Path):
        return str(data)
    else:
        return data


def load_config(path: Path | str) -> AppConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Parsed configuration object.
    """
    path = Path(path) if isinstance(path, str) else path
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = pyyaml.safe_load(f)
    return AppConfig(**raw_cfg)
