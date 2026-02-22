from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    TRADEABILITY_MIN_LIQUIDITY_USD_DAILY,
    TRADEABILITY_MAX_BID_ASK_PCT,
    TRADEABILITY_MAX_AVG_FUNDING_PCT,
    PAIR_STABILITY_MIN_WINDOW_STEPS,
    PAIR_STABILITY_MIN_STEPS,
    load_config,
)

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

# Константы для фильтров
DEFAULT_MIN_BETA = 0.1
DEFAULT_MAX_BETA = 10.0
DEFAULT_FILTER_MIN_HALF_LIFE = 1
DEFAULT_FILTER_MAX_HALF_LIFE = 252
DEFAULT_FILTER_MIN_MEAN_CROSSINGS = 10

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


@pytest.mark.unit
def test_load_config_with_base_config_and_null_overrides(tmp_path, monkeypatch):
    """batch config with base_config should merge and keep base values for null overrides."""
    data_dir = tmp_path / DATA_DIR_NAME
    results_dir = tmp_path / RESULTS_DIR_NAME
    data_dir.mkdir()
    results_dir.mkdir()

    base_config = {
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "portfolio": {
            "initial_capital": DEFAULT_INITIAL_CAPITAL,
            "risk_per_position_pct": DEFAULT_RISK_PER_POSITION,
            "max_active_positions": DEFAULT_MAX_ACTIVE_POSITIONS,
        },
        "pair_selection": {
            "lookback_days": DEFAULT_LOOKBACK_DAYS,
            "coint_pvalue_threshold": DEFAULT_COINT_PVALUE_THRESHOLD,
            "ssd_top_n": DEFAULT_SSD_TOP_N,
            "min_half_life_days": DEFAULT_MIN_HALF_LIFE_DAYS,
            "max_half_life_days": DEFAULT_MAX_HALF_LIFE_DAYS,
            "min_mean_crossings": DEFAULT_MIN_MEAN_CROSSINGS,
            "min_correlation": 0.4,
            "max_hurst_exponent": DEFAULT_MAX_HURST_EXPONENT,
        },
        "filter_params": {
            "min_beta": 0.2,
            "max_beta": 9.0,
            "min_half_life_days": DEFAULT_FILTER_MIN_HALF_LIFE,
            "max_half_life_days": DEFAULT_FILTER_MAX_HALF_LIFE,
            "max_hurst_exponent": DEFAULT_MAX_HURST_EXPONENT,
            "min_mean_crossings": DEFAULT_FILTER_MIN_MEAN_CROSSINGS,
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
            "annualizing_factor": DEFAULT_ANNUALIZING_FACTOR,
        },
        "walk_forward": {
            "start_date": DEFAULT_START_DATE,
            "end_date": DEFAULT_END_DATE,
            "training_period_days": DEFAULT_TRAINING_PERIOD_DAYS,
            "testing_period_days": DEFAULT_TESTING_PERIOD_DAYS,
        },
    }

    base_path = tmp_path / "configs" / "prod_base.yaml"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    override_path = (
        tmp_path
        / "configs"
        / "_autopilot_batches"
        / "20260219_budget1000_bl11_r09_pairgate02_micro24"
        / "candidate.yaml"
    )
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_config = {
        "base_config": "configs/prod_base.yaml",
        "portfolio": {"risk_per_position_pct": 0.02},
        "pair_selection": {"min_correlation": 0.22, "coint_pvalue_threshold": 0.6, "max_pairs": 64},
        "filter_params": {"min_beta": None, "max_hurst_exponent": None},
        "walk_forward": {"start_date": "2022-06-01", "end_date": "2023-04-30"},
        "search": {"min_queue_entries": 24},
    }
    override_path.write_text(yaml.safe_dump(override_config, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    cfg = load_config(override_path)

    assert cfg.portfolio.initial_capital == DEFAULT_INITIAL_CAPITAL
    assert cfg.portfolio.risk_per_position_pct == 0.02
    assert cfg.pair_selection.min_correlation == 0.22
    assert cfg.pair_selection.max_pairs == 64
    assert cfg.filter_params.min_beta == 0.2
    assert cfg.filter_params.max_hurst_exponent == DEFAULT_MAX_HURST_EXPONENT
    assert cfg.walk_forward.start_date == "2022-06-01"
    assert cfg.walk_forward.training_period_days == DEFAULT_TRAINING_PERIOD_DAYS


@pytest.mark.unit
def test_pair_selection_when_tradeability_loose_then_guardrail_reports_violations() -> None:
    cfg = PairSelectionConfig(
        lookback_days=90,
        coint_pvalue_threshold=0.05,
        ssd_top_n=10000,
        min_half_life_days=1.0,
        max_half_life_days=30.0,
        min_mean_crossings=10,
        liquidity_usd_daily=100_000,
        max_bid_ask_pct=0.9,
        max_avg_funding_pct=0.2,
    )

    violations = cfg.tradeability_floor_violations()
    assert any("liquidity_usd_daily<" in item for item in violations)
    assert any("max_bid_ask_pct>" in item for item in violations)
    assert any("max_avg_funding_pct>" in item for item in violations)

    liquidity, max_bid_ask, max_funding = cfg.resolved_tradeability_thresholds()
    assert liquidity == TRADEABILITY_MIN_LIQUIDITY_USD_DAILY
    assert max_bid_ask == TRADEABILITY_MAX_BID_ASK_PCT
    assert max_funding == TRADEABILITY_MAX_AVG_FUNDING_PCT


@pytest.mark.unit
def test_pair_selection_when_stability_is_one_step_then_guardrail_clamps_to_two() -> None:
    cfg = PairSelectionConfig(
        lookback_days=90,
        coint_pvalue_threshold=0.05,
        ssd_top_n=10000,
        min_half_life_days=1.0,
        max_half_life_days=30.0,
        min_mean_crossings=10,
        pair_stability_window_steps=1,
        pair_stability_min_steps=1,
    )

    resolved_window, resolved_min = cfg.resolved_pair_stability()
    assert resolved_window == PAIR_STABILITY_MIN_WINDOW_STEPS
    assert resolved_min == PAIR_STABILITY_MIN_STEPS
