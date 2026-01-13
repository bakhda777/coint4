from pathlib import Path

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)


def _make_cfg(data_dir: Path) -> AppConfig:
    return AppConfig(
        data_dir=data_dir,
        results_dir=data_dir,
        portfolio=PortfolioConfig(
            initial_capital=1.0,
            risk_per_position_pct=0.01,
            max_active_positions=1,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=1,
            coint_pvalue_threshold=0.05,
            ssd_top_n=1,
            min_half_life_days=1,
            max_half_life_days=30,
            min_mean_crossings=1,
        ),
        backtest=BacktestConfig(
            timeframe="15min",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=2.0,
            fill_limit_pct=0.1,
            commission_pct=0.0,
            slippage_pct=0.0,
            annualizing_factor=365,
        ),
        walk_forward=WalkForwardConfig(
            start_date="2024-01-01",
            end_date="2024-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )


def test_get_all_symbols_when_legacy_structure_then_returns_unique_sorted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_downloaded"
    (data_dir / "symbol=AAA").mkdir(parents=True, exist_ok=True)
    (data_dir / "symbol=BBB").mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(data_dir)
    handler = DataHandler(cfg)

    symbols = handler.get_all_symbols()
    assert symbols == ["AAA", "BBB"]
