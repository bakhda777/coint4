from pathlib import Path

import pytest
from pydantic import ValidationError

from coint2.utils.config import AppConfig, BacktestConfig, load_config

# Константы для тестирования конфигурации
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_RISK_PER_POSITION = 0.01
DEFAULT_MAX_ACTIVE_POSITIONS = 5
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_COINT_PVALUE_THRESHOLD = 0.05
DEFAULT_SSD_TOP_N = 10000
DEFAULT_MIN_HALF_LIFE_DAYS = 1
DEFAULT_MAX_HALF_LIFE_DAYS = 30
DEFAULT_MIN_MEAN_CROSSINGS = 12
DEFAULT_MAX_HURST_EXPONENT = 0.5

# Константы для бэктеста
DEFAULT_TIMEFRAME = "1d"
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_ZSCORE_THRESHOLD = 1.5
DEFAULT_STOP_LOSS_MULTIPLIER = 3.0
DEFAULT_TIME_STOP_MULTIPLIER = 2.0
DEFAULT_FILL_LIMIT_PCT = 0.2
DEFAULT_COMMISSION_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005
DEFAULT_ANNUALIZING_FACTOR = 365

# Константы для walk forward
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2021-01-31"
DEFAULT_TRAINING_PERIOD_DAYS = 30
DEFAULT_TESTING_PERIOD_DAYS = 10

# Имена директорий
DATA_DIR_NAME = "data"
RESULTS_DIR_NAME = "results"


@pytest.mark.unit
def test_load_config_when_valid_then_creates_app_config(tmp_path):
    """Configuration file should load into AppConfig."""
    # Создаем временные директории для тестирования
    data_dir = tmp_path / DATA_DIR_NAME
    results_dir = tmp_path / RESULTS_DIR_NAME
    data_dir.mkdir()
    results_dir.mkdir()

    # Создаем тестовую конфигурацию
    test_config = {
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "portfolio": {
            "initial_capital": DEFAULT_INITIAL_CAPITAL,
            "risk_per_position_pct": DEFAULT_RISK_PER_POSITION,
            "max_active_positions": DEFAULT_MAX_ACTIVE_POSITIONS
        },
        "pair_selection": {
            "lookback_days": DEFAULT_LOOKBACK_DAYS,
            "coint_pvalue_threshold": DEFAULT_COINT_PVALUE_THRESHOLD,
            "ssd_top_n": DEFAULT_SSD_TOP_N,
            "min_half_life_days": DEFAULT_MIN_HALF_LIFE_DAYS,
            "max_half_life_days": DEFAULT_MAX_HALF_LIFE_DAYS,
            "min_mean_crossings": DEFAULT_MIN_MEAN_CROSSINGS,
            "max_hurst_exponent": DEFAULT_MAX_HURST_EXPONENT
        },
        "backtest": {
            "timeframe": DEFAULT_TIMEFRAME,
            "rolling_window": DEFAULT_ROLLING_WINDOW,
            "zscore_threshold": DEFAULT_ZSCORE_THRESHOLD,
            "stop_loss_multiplier": DEFAULT_STOP_LOSS_MULTIPLIER,
            "time_stop_multiplier": DEFAULT_TIME_STOP_MULTIPLIER,
            "fill_limit_pct": DEFAULT_FILL_LIMIT_PCT,
            "commission_pct": DEFAULT_COMMISSION_PCT,
            "slippage_pct": DEFAULT_SLIPPAGE_PCT,
            "annualizing_factor": DEFAULT_ANNUALIZING_FACTOR
        },
        "walk_forward": {
            "start_date": DEFAULT_START_DATE,
            "end_date": DEFAULT_END_DATE,
            "training_period_days": DEFAULT_TRAINING_PERIOD_DAYS,
            "testing_period_days": DEFAULT_TESTING_PERIOD_DAYS
        }
    }
    
    # Сохраняем конфигурацию во временный файл
    CONFIG_FILENAME = "test_config.yaml"
    # Константы для фильтров (значения по умолчанию)
    DEFAULT_MIN_BETA = 0.1
    DEFAULT_MAX_BETA = 10.0
    DEFAULT_FILTER_MIN_HALF_LIFE = 1
    DEFAULT_FILTER_MAX_HALF_LIFE = 252
    DEFAULT_FILTER_MIN_MEAN_CROSSINGS = 10

    config_file = tmp_path / CONFIG_FILENAME
    with config_file.open("w") as f:
        import yaml
        yaml.dump(test_config, f)

    cfg = load_config(config_file)
    assert isinstance(cfg, AppConfig)
    assert cfg.pair_selection.lookback_days == DEFAULT_LOOKBACK_DAYS
    assert cfg.pair_selection.ssd_top_n == DEFAULT_SSD_TOP_N
    assert cfg.pair_selection.min_half_life_days == DEFAULT_MIN_HALF_LIFE_DAYS
    assert cfg.pair_selection.max_half_life_days == DEFAULT_MAX_HALF_LIFE_DAYS
    assert cfg.pair_selection.min_mean_crossings == DEFAULT_MIN_MEAN_CROSSINGS
    assert cfg.pair_selection.max_hurst_exponent == DEFAULT_MAX_HURST_EXPONENT
    assert cfg.backtest.rolling_window == DEFAULT_ROLLING_WINDOW
    assert cfg.backtest.stop_loss_multiplier == DEFAULT_STOP_LOSS_MULTIPLIER
    assert cfg.backtest.commission_pct == DEFAULT_COMMISSION_PCT
    assert cfg.backtest.slippage_pct == DEFAULT_SLIPPAGE_PCT
    assert cfg.backtest.annualizing_factor == DEFAULT_ANNUALIZING_FACTOR
    assert cfg.portfolio.initial_capital == DEFAULT_INITIAL_CAPITAL
    assert cfg.portfolio.risk_per_position_pct == DEFAULT_RISK_PER_POSITION
    assert cfg.portfolio.max_active_positions == DEFAULT_MAX_ACTIVE_POSITIONS
    assert cfg.filter_params.min_beta == DEFAULT_MIN_BETA
    assert cfg.filter_params.max_beta == DEFAULT_MAX_BETA
    assert cfg.filter_params.min_half_life_days == DEFAULT_FILTER_MIN_HALF_LIFE
    assert cfg.filter_params.max_half_life_days == DEFAULT_FILTER_MAX_HALF_LIFE
    assert cfg.filter_params.max_hurst_exponent == DEFAULT_MAX_HURST_EXPONENT
    assert cfg.filter_params.min_mean_crossings == DEFAULT_FILTER_MIN_MEAN_CROSSINGS


@pytest.mark.unit
def test_fill_limit_pct_when_invalid_then_validation_error() -> None:
    """fill_limit_pct should be between 0 and 1."""
    INVALID_FILL_LIMIT = 1.5
    MIN_ROLLING_WINDOW = 1
    MIN_ZSCORE_THRESHOLD = 1.0
    MIN_STOP_LOSS_MULTIPLIER = 2.0

    with pytest.raises(ValidationError):
        BacktestConfig(
            timeframe=DEFAULT_TIMEFRAME,
            rolling_window=MIN_ROLLING_WINDOW,
            zscore_threshold=MIN_ZSCORE_THRESHOLD,
            stop_loss_multiplier=MIN_STOP_LOSS_MULTIPLIER,
            fill_limit_pct=INVALID_FILL_LIMIT,
            commission_pct=DEFAULT_COMMISSION_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            annualizing_factor=DEFAULT_ANNUALIZING_FACTOR,
        )

