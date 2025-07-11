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

    # Extended filters
    min_correlation: float | None = None
    max_correlation: float | None = None
    adaptive_quantiles: bool | None = None
    bar_minutes: int | None = None
    liquidity_usd_daily: float | None = None
    max_bid_ask_pct: float | None = None
    max_avg_funding_pct: float | None = None
    save_filter_reasons: bool | None = None
    min_spread_std: float | None = None
    max_spread_std: float | None = None
    max_hurst_exponent: float | None = 0.5
    min_abs_spread_mult: float | None = None
    cost_filter: bool | None = None
    kpss_pvalue_threshold: float | None = None
    pvalue_top_n: int | None = None
    save_std_histogram: bool | None = None

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


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward analysis."""

    start_date: str
    end_date: str
    training_period_days: int
    testing_period_days: int


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data_dir: DirectoryPath
    results_dir: Path
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    portfolio: PortfolioConfig
    pair_selection: PairSelectionConfig
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig
    max_shards: int | None = None


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
