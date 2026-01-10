"""
Configuration validation using Pydantic for type safety and validation.
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import yaml
import json


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


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """Validate a configuration file.
    
    Returns:
        Tuple of (is_valid, list_of_errors_or_warnings)
    """
    try:
        # Load and validate
        if config_path.endswith('.yaml'):
            config = MainConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            config = MainConfig.from_json(config_path)
        else:
            return False, ["Unsupported file format (use .yaml or .json)"]
        
        # Check for production warnings
        warnings = config.validate_for_production()
        
        if warnings:
            return True, warnings
        else:
            return True, ["✅ Configuration valid for production"]
            
    except Exception as e:
        return False, [f"❌ Validation error: {str(e)}"]


def main():
    """Demonstrate config validation."""
    
    # Create sample config
    config = MainConfig(
        backtesting=BacktestingConfig(
            normalization_method="rolling_zscore",
            commission_pct=0.001,
            slippage_pct=0.0005
        ),
        signals=SignalConfig(
            zscore_threshold=2.0,
            zscore_exit=0.0,
            rolling_window=60,
            max_holding_days=100
        ),
        pair_selection=PairSelectionConfig(
            coint_pvalue_threshold=0.05,
            ssd_top_n=50000,
            min_half_life_days=2.0,
            max_half_life_days=30.0,
            min_mean_crossings=10
        ),
        walk_forward=WalkForwardConfig(
            train_days=90,
            test_days=30,
            gap_minutes=15,
            n_folds=5
        ),
        execution=ExecutionConfig(
            base_slippage=0.0003,
            atr_multiplier=0.1,
            vol_multiplier=0.5,
            latency_ms=10,
            partial_fill_prob=0.05
        )
    )
    
    print("=" * 60)
    print("CONFIGURATION VALIDATION DEMO")
    print("=" * 60)
    
    # Validate
    warnings = config.validate_for_production()
    
    print("\n✅ Configuration Schema:")
    print(json.dumps(config.dict(), indent=2))
    
    print("\n📊 Production Validation:")
    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("  ✅ All checks passed!")
    
    # Test validation of existing config
    test_configs = [
        "configs/main_2024.yaml",
        "configs/ultra_fast.yaml"
    ]
    
    print("\n📁 Validating Existing Configs:")
    for config_path in test_configs:
        if Path(config_path).exists():
            is_valid, messages = validate_config_file(config_path)
            print(f"\n  {config_path}:")
            print(f"    Valid: {'✅' if is_valid else '❌'}")
            for msg in messages[:3]:  # Show first 3 messages
                print(f"    {msg}")


if __name__ == "__main__":
    main()