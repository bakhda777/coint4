"""Smoke tests for production configuration."""

import pytest
from pathlib import Path


@pytest.mark.smoke
def test_prod_config_exists():
    """Test that prod.yaml exists."""
    config_path = Path("configs/prod.yaml")
    assert config_path.exists(), "Production config not found"


@pytest.mark.smoke
def test_prod_config_valid():
    """Test that prod.yaml is valid and loadable."""
    from coint2.utils.config import load_config
    
    config = load_config("configs/prod.yaml")
    
    # Check critical settings
    assert config.backtesting.normalization_method == "rolling_zscore"
    assert config.time.gap_minutes == 15
    assert config.guards.enabled == True
    assert config.backtesting.commission_pct > 0
    assert config.backtesting.slippage_pct > 0


@pytest.mark.smoke
def test_prod_config_safety_guards():
    """Test that production config has all safety guards."""
    from coint2.utils.config import load_config
    
    config = load_config("configs/prod.yaml")
    
    # Risk limits
    assert config.risk.max_daily_loss_pct > 0
    assert config.risk.max_drawdown_pct > 0
    
    # Guards
    assert config.guards.use_reference_on_error == True
    assert config.guards.max_position_value > 0
    assert config.guards.require_price_validation == True
    
    # Walk-forward settings
    assert config.walk_forward.train_days >= 60
    assert config.walk_forward.test_days >= 30
    
    # Pair selection
    assert config.pair_selection.ssd_top_n >= 25000