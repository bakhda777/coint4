"""Command line interface for the coint2 package."""

from __future__ import annotations

from pathlib import Path

import click
import argparse
import logging
import sys

from coint2.core.data_loader import DataHandler
from coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester as PairBacktester
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from coint2.utils.config import load_config, AppConfig
from coint2.utils.timing_utils import setup_timing_logger


@click.group()
@click.option(
    "--expected-freq",
    default=None,
    help="Assert that loaded data has this frequency (e.g. '15T').",
)
@click.pass_context
def main(ctx: click.Context, expected_freq: str | None) -> None:
    """Entry point for the ``coint2`` command."""

    ctx.obj = {"expected_freq": expected_freq}


@main.command()
@click.option("--pair", required=True, help="Pair in the format SYMBOL1,SYMBOL2")
@click.pass_context
def backtest(ctx: click.Context, pair: str) -> None:
    """Quick backtest over the entire dataset (for debugging only)."""

    cfg = load_config(Path("configs/main.yaml"))
    s1, s2 = [p.strip() for p in pair.split(",")]
    handler = DataHandler(cfg)

    ddf = handler._load_full_dataset()
    if ddf.columns.empty:
        click.echo("No data available")
        return

    end_date = ddf["timestamp"].max().compute()
    start_date = ddf["timestamp"].min().compute()

    data = handler.load_pair_data(s1, s2, start_date, end_date)
    if data.empty:
        click.echo("No data available for the pair")
        return

    bt = PairBacktester(
        data,
        rolling_window=cfg.backtest.rolling_window,
        z_threshold=cfg.backtest.zscore_threshold,
        z_exit=getattr(cfg.backtest, 'zscore_exit', 0.0),
        commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0),
        slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0),
        stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
        annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365),
        # Convert hours to periods based on 15-minute timeframe
        cooldown_periods=int(getattr(cfg.backtest, 'cooldown_hours', 0) * 60 / 15),  # 15-минутный таймфрейм
    )
    bt.run()
    metrics = bt.get_performance_metrics()
    for k, v in metrics.items():
        click.echo(f"{k}: {v}")

    expected_freq = ctx.obj.get("expected_freq")
    if expected_freq and handler.freq != expected_freq:
        raise click.ClickException(
            f"Expected frequency {expected_freq}, got {handler.freq}"
        )


@main.command(name="run")
@click.option(
    "--config",
    "config_path",
    default="configs/main.yaml",
    help="Path to the configuration YAML file.",
    type=click.Path(exists=True),
)
@click.option("--timing", "-t", is_flag=True, help="Enable detailed timing logs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def run_cmd(ctx: click.Context, config_path: str, timing: bool, verbose: bool) -> None:
    """Run walk-forward analysis pipeline."""
    
    # Setup enhanced logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Setup timing logger if requested
    if timing or verbose:
        setup_timing_logger(level=logging.INFO)
        click.echo("🔍 Детализированное логирование времени включено")

    cfg = load_config(Path(config_path))
    expected_freq = ctx.obj.get("expected_freq")

    if expected_freq:
        tmp_handler = DataHandler(cfg)
        tmp_handler.load_all_data_for_period(1)
        if tmp_handler.freq != expected_freq:
            raise click.ClickException(
                f"Expected frequency {expected_freq}, got {tmp_handler.freq}"
            )

    metrics = run_walk_forward(cfg)
    for key, value in metrics.items():
        click.echo(f"{key}: {value}")


@main.command(name="analyze-2024")
@click.option("--timing", "-t", is_flag=True, default=True, help="Enable detailed timing logs (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", default="configs/main_2024.yaml", help="Configuration file for 2024 analysis")
@click.pass_context
def analyze_2024_cmd(ctx: click.Context, timing: bool, verbose: bool, config: str) -> None:
    """Run comprehensive 2024 walk-forward analysis with timing profiling."""
    
    # Setup enhanced logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Setup timing logger
    if timing:
        from coint2.utils.timing_utils import setup_timing_logger
        setup_timing_logger(level=logging.INFO)
        click.echo("🔍 Детализированное логирование времени включено")
    
    # Load config
    config_path = Path(config)
    if not config_path.exists():
        raise click.ClickException(f"Configuration file not found: {config_path}")
    
    cfg = load_config(config_path)
    click.echo(f"✅ Конфигурация загружена из {config_path}")
    
    # Display configuration summary
    click.echo(f"\n⚙️  КОНФИГУРАЦИЯ АНАЛИЗА:")
    click.echo(f"📅 Период: {cfg.walk_forward.start_date} → {cfg.walk_forward.end_date}")
    click.echo(f"💰 Капитал: ${cfg.portfolio.initial_capital:,.0f}")
    click.echo(f"🔍 SSD top-N: {cfg.pair_selection.ssd_top_n:,}")
    click.echo(f"⚡ Z-score: ±{cfg.backtest.zscore_threshold}")
    
    # Calculate expected steps
    from datetime import datetime, timedelta
    start_date = datetime.strptime(cfg.walk_forward.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(cfg.walk_forward.end_date, "%Y-%m-%d")
    
    steps = 0
    current = start_date
    while current < end_date:
        train_end = current + timedelta(days=cfg.walk_forward.training_period_days)
        test_end = train_end + timedelta(days=cfg.walk_forward.testing_period_days)
        if test_end <= end_date:
            steps += 1
        current = test_end
    
    click.echo(f"\n📈 Ожидается {steps} walk-forward шагов")
    click.echo(f"⏱️  Примерное время: {steps * 2:.0f}-{steps * 5:.0f} минут")
    
    # Run analysis
    try:
        click.echo("\n🎯 Запуск анализа...")
        metrics = run_walk_forward(cfg)
        
        click.echo("\n✅ Анализ завершен!")
        click.echo("\n📈 Финальные метрики:")
        for key, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.4f}")
            else:
                click.echo(f"  {key}: {value}")
                
    except Exception as e:
        raise click.ClickException(f"Ошибка при выполнении анализа: {e}")


if __name__ == "__main__":
    main()
