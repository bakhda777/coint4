from pathlib import Path

import pytest
from pydantic import ValidationError

from coint2.utils.config import AppConfig, BacktestConfig, load_config


def test_load_config(tmp_path):
    """Configuration file should load into AppConfig."""
    # Создаем временные директории для тестирования
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    
    # Создаем тестовую конфигурацию
    test_config = {
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "portfolio": {
            "initial_capital": 10000.0,
            "risk_per_position_pct": 0.01,
            "max_active_positions": 5
        },
        "pair_selection": {
            "lookback_days": 90,
            "coint_pvalue_threshold": 0.05,
            "ssd_top_n": 10000,
            "min_half_life_days": 1,
            "max_half_life_days": 30,
            "min_mean_crossings": 12,
            "max_hurst_exponent": 0.5
        },
        "backtest": {
            "timeframe": "1d",
            "rolling_window": 30,
            "zscore_threshold": 1.5,
            "stop_loss_multiplier": 3.0,
            "time_stop_multiplier": 2.0,
            "fill_limit_pct": 0.2,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "annualizing_factor": 365
        },
        "walk_forward": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "training_period_days": 30,
            "testing_period_days": 10
        }
    }
    
    # Сохраняем конфигурацию во временный файл
    config_file = tmp_path / "test_config.yaml"
    with config_file.open("w") as f:
        import yaml
        yaml.dump(test_config, f)
    
    cfg = load_config(config_file)
    assert isinstance(cfg, AppConfig)
    assert cfg.pair_selection.lookback_days == 90
    assert cfg.pair_selection.ssd_top_n == 10000
    assert cfg.pair_selection.min_half_life_days == 1
    assert cfg.pair_selection.max_half_life_days == 30
    assert cfg.pair_selection.min_mean_crossings == 12
    assert cfg.pair_selection.max_hurst_exponent == 0.5
    assert cfg.backtest.rolling_window == 30
    assert cfg.backtest.stop_loss_multiplier == 3.0
    assert cfg.backtest.commission_pct == 0.001
    assert cfg.backtest.slippage_pct == 0.0005
    assert cfg.backtest.annualizing_factor == 365
    assert cfg.portfolio.initial_capital == 10000.0
    assert cfg.portfolio.risk_per_position_pct == 0.01
    assert cfg.portfolio.max_active_positions == 5
    assert cfg.filter_params.min_beta == 0.1
    assert cfg.filter_params.max_beta == 10.0
    assert cfg.filter_params.min_half_life_days == 1
    assert cfg.filter_params.max_half_life_days == 252
    assert cfg.filter_params.max_hurst_exponent == 0.5
    assert cfg.filter_params.min_mean_crossings == 10


def test_fill_limit_pct_validation() -> None:
    """fill_limit_pct should be between 0 and 1."""
    with pytest.raises(ValidationError):
        BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=2.0,
            fill_limit_pct=1.5,
            commission_pct=0.001,
            slippage_pct=0.0005,
            annualizing_factor=365,
        )

