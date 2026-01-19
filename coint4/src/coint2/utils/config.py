"""Configuration utilities using Pydantic models."""

from pathlib import Path

import yaml as pyyaml  # type: ignore
from pydantic import BaseModel, DirectoryPath, Field, model_validator  # type: ignore


class PairSelectionConfig(BaseModel):
    """Configuration for pair selection parameters."""

    lookback_days: int
    coint_pvalue_threshold: float
    ssd_top_n: int
    min_half_life_days: float
    max_half_life_days: float
    min_mean_crossings: int
    min_correlation: float = 0.5  # НОВЫЙ параметр для фильтрации по корреляции

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

    # Pair stability across WFA steps (optional).
    pair_stability_window_steps: int | None = None
    pair_stability_min_steps: int | None = None

    # Optional cap on number of pairs to trade per WFA step.
    max_pairs: int | None = None
    
    # NEW: Pair tradeability filter parameters
    enable_pair_tradeability_filter: bool = True
    min_volume_usd_24h: float = 20_000_000  # 20M USD daily volume
    min_days_live: int = 30  # 30 days since listing
    max_funding_rate_abs: float = 0.0003  # 0.03% absolute funding rate
    max_tick_size_pct: float = 0.0005  # 0.05% tick size relative to price
    max_half_life_hours: float = 72.0  # 72 hours max half-life

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

    @model_validator(mode="after")
    def _check_pair_stability(self):  # type: ignore
        window = self.pair_stability_window_steps
        minimum = self.pair_stability_min_steps
        if window is None and minimum is None:
            return self
        if window is None or minimum is None:
            raise ValueError(
                "`pair_stability_window_steps` and `pair_stability_min_steps` must be set together"
            )
        if window < 1:
            raise ValueError("`pair_stability_window_steps` must be >= 1")
        if minimum < 1:
            raise ValueError("`pair_stability_min_steps` must be >= 1")
        if minimum > window:
            raise ValueError("`pair_stability_min_steps` must be <= `pair_stability_window_steps`")
        return self


class FilterParamsConfig(BaseModel):
    """Configurable thresholds for pair filtering."""

    min_beta: float = 0.1
    max_beta: float = 10.0
    min_half_life_days: float = 1
    max_half_life_days: float = 252
    max_hurst_exponent: float = 0.5
    min_mean_crossings: int = 10


class DataProcessingConfig(BaseModel):
    """Configuration for data processing and normalization."""
    
    normalization_method: str = "rolling_zscore"  # КРИТИЧНО: только production-совместимый метод!
    fill_method: str = "ffill"
    min_history_ratio: float = 0.8
    handle_constant: bool = True


class CleanWindowConfig(BaseModel):
    """Optional clean data window for filtering."""

    start_date: str
    end_date: str

    @model_validator(mode="after")
    def _validate_window(self):  # type: ignore
        if self.start_date >= self.end_date:
            raise ValueError("`clean_window.start_date` must be before `clean_window.end_date`")
        return self


class DataFilterConfig(BaseModel):
    """Dataset-level filters for clean window and symbol exclusions."""

    clean_window: CleanWindowConfig | None = None
    exclude_symbols: list[str] = Field(default_factory=list)
    exclude_reason: str | None = None

    @model_validator(mode="after")
    def _normalize_symbols(self):  # type: ignore
        if self.exclude_symbols:
            normalized = []
            for symbol in self.exclude_symbols:
                symbol = str(symbol).strip()
                if symbol:
                    normalized.append(symbol)
            self.exclude_symbols = sorted(set(normalized))
        return self


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
        
        if not (0 < self.f_max <= 1):
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
    zscore_threshold: float
    stop_loss_multiplier: float
    fill_limit_pct: float = Field(..., ge=0.0, le=1.0)
    commission_pct: float  # Новое поле
    slippage_pct: float  # Новое поле
    annualizing_factor: int  # Новое поле
    time_stop_multiplier: float | None = None
    zscore_exit: float | None = None
    take_profit_multiplier: float | None = None
    cooldown_hours: int = 0
    wait_for_candle_close: bool = False
    min_volatility: float = Field(default=0.001, ge=0.0)
    slippage_stress_multiplier: float = Field(default=1.0, ge=0.0)
    always_model_slippage: bool = True
    # NEW: Enhanced risk management parameters
    use_kelly_sizing: bool = True
    max_kelly_fraction: float = Field(default=0.25, ge=0.01, le=1.0)
    volatility_lookback: int = Field(default=96, ge=10)
    adaptive_thresholds: bool = True
    var_confidence: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_var_multiplier: float = Field(default=3.0, gt=1.0)
    
    # NEW: Enhanced entry/exit rules
    zscore_entry_threshold: float = Field(default=2.3, gt=0.0)  # New higher entry threshold
    min_spread_move_sigma: float = Field(default=1.2, gt=0.0)  # Minimum spread movement since last flat
    min_position_hold_minutes: int = Field(default=60, ge=0)  # Minimum hold time in minutes
    anti_churn_cooldown_minutes: int = Field(default=60, ge=0)  # Anti-churn protection
    
    # NEW: Enhanced stop-loss rules
    pair_stop_loss_usd: float = Field(default=75.0, gt=0.0)  # Pair-level stop loss in USD
    pair_stop_loss_zscore: float = Field(default=3.0, gt=0.0)  # Z-score based stop loss
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
    
    # NEW: Enhanced cost modeling
    enable_realistic_costs: bool = True
    commission_rate_per_leg: float = Field(default=0.0004, ge=0.0)  # 0.04% per leg
    slippage_half_spread_multiplier: float = Field(default=2.0, ge=0.0)  # 2x half spread for slippage
    funding_cost_enabled: bool = True
    
    # NEW: Enhanced position sizing with beta recalculation
    beta_recalc_frequency_hours: int = Field(default=48, ge=1)  # Recalculate beta every 48 hours
    beta_window_days_min: int = Field(default=60, ge=30)  # Minimum 60 days for beta calculation
    beta_window_days_max: int = Field(default=120, ge=60)  # Maximum 120 days for beta calculation
    use_ols_beta_sizing: bool = True  # Use OLS beta for position sizing
    
    # Market regime detection parameters
    market_regime_detection: bool = True
    hurst_window: int = Field(default=720, ge=1)  # Минимум зависит от режима, проверяется валидатором
    hurst_trending_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    variance_ratio_window: int = Field(default=480, ge=50)
    variance_ratio_trending_min: float = Field(default=1.2, gt=1.0)
    variance_ratio_mean_reverting_max: float = Field(default=0.8, lt=1.0, gt=0.0)
    
    # Structural break protection parameters
    structural_break_protection: bool = True
    cointegration_test_frequency: int = Field(default=2688, ge=1)  # Минимум зависит от режима, проверяется валидатором
    adf_pvalue_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    exclusion_period_days: int = Field(default=30, ge=1)
    max_half_life_days: int = Field(default=10, ge=1)
    min_correlation_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    correlation_window: int = Field(default=720, ge=50)

    # Performance optimization parameters
    regime_check_frequency: int = Field(default=96, ge=1)
    use_market_regime_cache: bool = True
    adf_check_frequency: int = Field(default=2688, ge=1)
    cache_cleanup_frequency: int = Field(default=1000, ge=1)
    lazy_adf_threshold: float = Field(default=0.1, gt=0.0)
    hurst_neutral_band: float = Field(default=0.05, ge=0.0)
    vr_neutral_band: float = Field(default=0.2, ge=0.0)
    use_exponential_weighted_correlation: bool = False
    ew_correlation_alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    n_jobs: int = -1

    # -------- Validators --------
    @model_validator(mode="after")
    def _validate_backtest_params(self):  # type: ignore
        """Validate backtesting configuration parameters."""
        # Check zscore_threshold > zscore_exit if both are set
        if self.zscore_exit is not None and self.zscore_threshold <= self.zscore_exit:
            raise ValueError("`zscore_threshold` must be greater than `zscore_exit`")
        
        # Check that all multipliers are positive
        if self.stop_loss_multiplier <= 0:
            raise ValueError("`stop_loss_multiplier` must be positive")
        
        if self.time_stop_multiplier is not None and self.time_stop_multiplier <= 0:
            raise ValueError("`time_stop_multiplier` must be positive")
        
        if self.take_profit_multiplier is not None and self.take_profit_multiplier <= 0:
            raise ValueError("`take_profit_multiplier` must be positive")
        
        # NEW: Validate enhanced risk management parameters
        if self.max_kelly_fraction <= 0 or self.max_kelly_fraction > 1.0:
            raise ValueError("`max_kelly_fraction` must be between 0 and 1")
        
        if self.volatility_lookback < 10:
            raise ValueError("`volatility_lookback` must be at least 10 periods")
        
        if self.var_confidence <= 0 or self.var_confidence >= 1.0:
            raise ValueError("`var_confidence` must be between 0 and 1")
        
        if self.max_var_multiplier <= 1.0:
            raise ValueError("`max_var_multiplier` must be greater than 1.0")
        
        # Validate market regime detection parameters
        if self.market_regime_detection and self.hurst_window < 100:
            raise ValueError("`hurst_window` must be at least 100 periods")
        
        if self.variance_ratio_window < 50:
            raise ValueError("`variance_ratio_window` must be at least 50 periods")
        
        if self.variance_ratio_trending_min <= 1.0:
            raise ValueError("`variance_ratio_trending_min` must be greater than 1.0")
        
        if self.variance_ratio_mean_reverting_max >= 1.0:
            raise ValueError("`variance_ratio_mean_reverting_max` must be less than 1.0")
        
        # Validate structural break protection parameters
        if self.structural_break_protection and self.cointegration_test_frequency < 100:
            raise ValueError("`cointegration_test_frequency` must be at least 100 periods")
        
        if self.correlation_window < 50:
            raise ValueError("`correlation_window` must be at least 50 periods")
        
        return self


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward analysis."""

    enabled: bool = True
    start_date: str
    end_date: str
    training_period_days: int
    testing_period_days: int
    step_size_days: int | None = None
    max_steps: int | None = None
    min_training_samples: int | None = None
    refit_frequency: str | None = None
    gap_minutes: int = 15
    pairs_file: str | None = None
    train_days: int | None = None
    test_days: int | None = None

    @model_validator(mode="after")
    def fill_legacy_fields(self):
        if self.train_days is None:
            self.train_days = self.training_period_days
        if self.test_days is None:
            self.test_days = self.testing_period_days
        if self.step_size_days is None:
            self.step_size_days = self.testing_period_days
        return self


class TimeConfig(BaseModel):
    """Time-related settings."""

    timeframe: str = "15min"
    gap_minutes: int = 15


class RiskConfig(BaseModel):
    """Risk guard settings."""

    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.05
    max_no_data_minutes: int = 10
    min_trade_count_per_day: int = 1
    position_size_usd: float = 1000.0


class GuardsConfig(BaseModel):
    """Safety guard flags."""

    enabled: bool = True
    use_reference_on_error: bool = True
    max_position_value: float = 10000.0
    require_price_validation: bool = True


class LoggingConfig(BaseModel):
    """Logging settings."""

    trade_details: bool = False
    debug_level: str = "INFO"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data_dir: DirectoryPath
    results_dir: Path
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    data_filters: DataFilterConfig | None = None
    time: TimeConfig = Field(default_factory=TimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    guards: GuardsConfig = Field(default_factory=GuardsConfig)
    portfolio: PortfolioConfig
    pair_selection: PairSelectionConfig
    filter_params: FilterParamsConfig = Field(default_factory=FilterParamsConfig)
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    max_shards: int | None = None

    @property
    def backtesting(self):
        """Compatibility view for backtesting settings."""
        class BacktestingView:
            def __init__(self, data_processing_cfg: DataProcessingConfig, backtest_cfg: BacktestConfig):
                self.normalization_method = data_processing_cfg.normalization_method
                self.commission_pct = backtest_cfg.commission_pct
                self.slippage_pct = backtest_cfg.slippage_pct

        return BacktestingView(self.data_processing, self.backtest)


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
