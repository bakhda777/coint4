"""
Configuration validation using Pydantic for type safety and validation.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import yaml
import json

from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    DataProcessingConfig,
    FilterParamsConfig,
    GuardsConfig,
    LoggingConfig,
    PairSelectionConfig,
    PortfolioConfig,
    RiskConfig,
    TimeConfig,
    WalkForwardConfig,
)


class BacktestingConfig(BaseModel):
    """Backtesting configuration schema."""
    
    normalization_method: str = Field(
        default="rolling_zscore",
        description="Normalization method (must be production-safe)"
    )
    commission_pct: float = Field(
        default=0.001,
        ge=0,
        le=0.01,
        description="Commission percentage (0-1%)"
    )
    slippage_pct: float = Field(
        default=0.0005,
        ge=0,
        le=0.01,
        description="Slippage percentage (0-1%)"
    )
    
    @validator('normalization_method')
    def validate_normalization(cls, v):
        """Ensure production-safe normalization."""
        allowed_methods = ['rolling_zscore', 'expanding_zscore', 'none']
        if v not in allowed_methods:
            raise ValueError(f"Normalization method must be one of {allowed_methods}")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_total_costs(cls, values):
        """Ensure total costs are reasonable."""
        total = values.get('commission_pct', 0) + values.get('slippage_pct', 0)
        if total > 0.005:  # 0.5% total
            raise ValueError(f"Total costs too high: {total:.2%}")
        return values


class SignalConfig(BaseModel):
    """Trading signal configuration."""
    
    zscore_threshold: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Z-score entry threshold"
    )
    zscore_exit: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Z-score exit threshold"
    )
    rolling_window: int = Field(
        default=60,
        ge=10,
        le=500,
        description="Rolling window for calculations"
    )
    max_holding_days: int = Field(
        default=100,
        ge=1,
        le=365,
        description="Maximum holding period"
    )
    
    @root_validator(skip_on_failure=True)
    def validate_hysteresis(cls, values):
        """Ensure proper hysteresis between entry and exit."""
        entry = values.get('zscore_threshold', 2.0)
        exit_val = values.get('zscore_exit', 0.0)
        
        if exit_val >= entry:
            raise ValueError(f"zscore_exit ({exit_val}) must be less than zscore_threshold ({entry})")
        
        if entry - exit_val < 0.5:
            raise ValueError(f"Insufficient hysteresis: {entry - exit_val:.2f} (min 0.5)")
        
        return values


class PairSelectionConfig(BaseModel):
    """Pair selection configuration."""
    
    coint_pvalue_threshold: float = Field(
        default=0.05,
        gt=0,
        le=0.2,
        description="Cointegration p-value threshold"
    )
    ssd_top_n: int = Field(
        default=50000,
        ge=1000,
        le=100000,
        description="Top N pairs by SSD"
    )
    min_half_life_days: float = Field(
        default=2.0,
        ge=0.5,
        le=30,
        description="Minimum half-life in days"
    )
    max_half_life_days: float = Field(
        default=30.0,
        ge=1,
        le=365,
        description="Maximum half-life in days"
    )
    min_mean_crossings: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum mean crossings"
    )
    
    @root_validator(skip_on_failure=True)
    def validate_half_life_range(cls, values):
        """Ensure valid half-life range."""
        min_hl = values.get('min_half_life_days', 2.0)
        max_hl = values.get('max_half_life_days', 30.0)
        
        if min_hl >= max_hl:
            raise ValueError(f"min_half_life ({min_hl}) must be less than max_half_life ({max_hl})")
        
        return values


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""
    
    train_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Training period in days"
    )
    test_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Testing period in days"
    )
    gap_minutes: int = Field(
        default=15,
        ge=0,
        le=1440,
        description="Gap between train/test in minutes"
    )
    n_folds: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of walk-forward folds"
    )
    
    @validator('gap_minutes')
    def validate_gap(cls, v):
        """Ensure gap is aligned with trading frequency."""
        if v % 15 != 0:
            raise ValueError(f"Gap must be multiple of 15 minutes (trading frequency)")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_periods(cls, values):
        """Ensure sensible train/test ratio."""
        train = values.get('train_days', 90)
        test = values.get('test_days', 30)
        
        if test > train:
            raise ValueError(f"Test period ({test}) cannot exceed train period ({train})")
        
        if train < 60:
            raise ValueError(f"Training period ({train}) should be at least 60 days for statistical significance")
        
        if test < 30:
            raise ValueError(f"Test period ({test}) should be at least 30 days for reliable metrics")
        
        return values


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    
    base_slippage: float = Field(
        default=0.0003,
        ge=0,
        le=0.01,
        description="Base slippage"
    )
    atr_multiplier: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="ATR impact multiplier"
    )
    vol_multiplier: float = Field(
        default=0.5,
        ge=0,
        le=2.0,
        description="Volatility impact multiplier"
    )
    latency_ms: int = Field(
        default=10,
        ge=0,
        le=1000,
        description="Execution latency in ms"
    )
    partial_fill_prob: float = Field(
        default=0.05,
        ge=0,
        le=1.0,
        description="Partial fill probability"
    )


class MainConfig(BaseModel):
    """Main configuration combining all sections."""
    
    backtesting: BacktestingConfig
    signals: SignalConfig
    pair_selection: PairSelectionConfig
    walk_forward: WalkForwardConfig
    execution: Optional[ExecutionConfig] = None
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'MainConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'MainConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    def validate_for_production(self) -> List[str]:
        """Validate configuration for production use."""
        warnings = []
        
        # Check normalization
        if self.backtesting.normalization_method != 'rolling_zscore':
            warnings.append(f"⚠️ Non-production normalization: {self.backtesting.normalization_method}")
        
        # Check costs
        total_costs = self.backtesting.commission_pct + self.backtesting.slippage_pct
        if total_costs < 0.0008:
            warnings.append(f"⚠️ Unrealistically low costs: {total_costs:.2%}")
        
        # Check pair selection
        if self.pair_selection.ssd_top_n < 25000:
            warnings.append(f"⚠️ Low pair diversity: {self.pair_selection.ssd_top_n}")
        
        # Check walk-forward
        if self.walk_forward.gap_minutes != 15:
            warnings.append(f"⚠️ Non-standard gap: {self.walk_forward.gap_minutes} minutes")
        
        return warnings


SECTION_MODELS: dict[str, Any] = {
    "data_processing": DataProcessingConfig,
    "portfolio": PortfolioConfig,
    "pair_selection": PairSelectionConfig,
    "filter_params": FilterParamsConfig,
    "backtest": BacktestConfig,
    "walk_forward": WalkForwardConfig,
    "time": TimeConfig,
    "risk": RiskConfig,
    "guards": GuardsConfig,
    "logging": LoggingConfig,
}

DEPRECATED_TOP_LEVEL_KEYS = {
    "filters",
    "normalization",
    "online_statistics",
    "signal_shift",
}

DEPRECATED_SECTION_KEYS = {
    "pair_selection": {"min_beta", "max_beta", "min_profit_potential_pct"},
    "backtest": {"flat_zscore_threshold"},
}

ALLOWED_SEARCH_SPACE_SECTIONS = {
    "filters",
    "signals",
    "risk_management",
    "portfolio",
    "costs",
    "normalization",
    "metrics",
    "optimization",
}

DEPRECATED_SEARCH_SPACE_SECTIONS = {"risk": "risk_management"}


def _collect_extra_keys(raw_cfg: dict) -> List[str]:
    extras: List[str] = []
    top_fields = set(AppConfig.model_fields.keys())
    for key in raw_cfg:
        if key in DEPRECATED_TOP_LEVEL_KEYS:
            continue
        if key not in top_fields:
            extras.append(key)
    for section, model in SECTION_MODELS.items():
        if section not in raw_cfg or not isinstance(raw_cfg[section], dict):
            continue
        model_fields = set(model.model_fields.keys())
        for key in raw_cfg[section]:
            if key not in model_fields:
                extras.append(f"{section}.{key}")
    return extras


def _collect_deprecated_keys(raw_cfg: dict) -> List[str]:
    deprecated: List[str] = []
    for key in DEPRECATED_TOP_LEVEL_KEYS:
        if key in raw_cfg:
            deprecated.append(key)
    for section, keys in DEPRECATED_SECTION_KEYS.items():
        section_cfg = raw_cfg.get(section, {})
        if not isinstance(section_cfg, dict):
            continue
        for key in keys:
            if key in section_cfg:
                deprecated.append(f"{section}.{key}")
    return deprecated


def validate_for_production_cfg(cfg: AppConfig) -> List[str]:
    """Validate configuration for production use."""
    warnings: List[str] = []

    if cfg.data_processing.normalization_method != "rolling_zscore":
        warnings.append(
            f"⚠️ Non-production normalization: {cfg.data_processing.normalization_method}"
        )

    total_costs = cfg.backtest.commission_pct + cfg.backtest.slippage_pct
    if total_costs < 0.0008:
        warnings.append(f"⚠️ Unrealistically low costs: {total_costs:.2%}")

    if cfg.pair_selection.ssd_top_n < 25000:
        warnings.append(f"⚠️ Low pair diversity: {cfg.pair_selection.ssd_top_n}")

    if cfg.walk_forward.gap_minutes != 15:
        warnings.append(f"⚠️ Non-standard gap: {cfg.walk_forward.gap_minutes} minutes")

    return warnings


def validate_search_space_file(config_path: str) -> Tuple[bool, List[str]]:
    """Validate Optuna search space file for expected sections."""
    try:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return False, [f"❌ Validation error: {exc}"]

    if not isinstance(raw, dict):
        return False, ["Search space must be a mapping"]

    warnings: List[str] = []
    for key in raw:
        if key in DEPRECATED_SEARCH_SPACE_SECTIONS:
            target = DEPRECATED_SEARCH_SPACE_SECTIONS[key]
            warnings.append(f"⚠️ Deprecated section '{key}'; use '{target}' instead")
        elif key not in ALLOWED_SEARCH_SPACE_SECTIONS:
            warnings.append(f"⚠️ Unknown search space section: {key}")

    risk_section = raw.get("risk")
    if isinstance(risk_section, dict) and "max_position_size_pct" in risk_section:
        warnings.append("⚠️ Move max_position_size_pct to the 'portfolio' section")

    if warnings:
        return True, warnings
    return True, ["✅ Search space looks consistent"]


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """Validate a configuration file.
    
    Returns:
        Tuple of (is_valid, list_of_errors_or_warnings)
    """
    try:
        if config_path.endswith(".yaml"):
            raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        elif config_path.endswith(".json"):
            raw_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        else:
            return False, ["Unsupported file format (use .yaml or .json)"]

        cfg = AppConfig(**raw_cfg)

        warnings: List[str] = []
        deprecated = _collect_deprecated_keys(raw_cfg)
        if deprecated:
            warnings.append(f"⚠️ Deprecated keys: {', '.join(sorted(deprecated))}")

        extras = _collect_extra_keys(raw_cfg)
        if extras:
            warnings.append(f"⚠️ Unknown keys: {', '.join(sorted(extras))}")

        warnings.extend(validate_for_production_cfg(cfg))

        if warnings:
            return True, warnings
        return True, ["✅ Configuration valid for production"]

    except Exception as e:
        return False, [f"❌ Validation error: {str(e)}"]


def main():
    """Demonstrate config validation."""

    print("=" * 60)
    print("CONFIGURATION VALIDATION DEMO")
    print("=" * 60)

    test_configs = [
        "configs/main_2024.yaml",
        "configs/prod.yaml",
    ]

    print("\n📁 Validating Configs:")
    for config_path in test_configs:
        if not Path(config_path).exists():
            continue
        is_valid, messages = validate_config_file(config_path)
        print(f"\n  {config_path}:")
        print(f"    Valid: {'✅' if is_valid else '❌'}")
        for msg in messages[:5]:
            print(f"    {msg}")

    search_spaces = [
        "configs/search_spaces/fast.yaml",
        "configs/search_spaces/web_ui.yaml",
        "configs/search_space_fast.yaml",
    ]

    print("\n📁 Validating Search Spaces:")
    for space_path in search_spaces:
        if not Path(space_path).exists():
            continue
        is_valid, messages = validate_search_space_file(space_path)
        print(f"\n  {space_path}:")
        print(f"    Valid: {'✅' if is_valid else '❌'}")
        for msg in messages[:5]:
            print(f"    {msg}")


if __name__ == "__main__":
    main()
