#!/usr/bin/env python3
"""Unified backtesting script with multiple modes."""

import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_backtest(
    mode: str = "single",
    config: str = "configs/main.yaml",
    pairs_file: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    paper: bool = False,
    window: str = "month",
    canary: bool = False,
    output_dir: str = "outputs/backtest",
    **kwargs
):
    """Run backtest with specified mode and options.
    
    Modes:
    - single: Single pair backtest
    - portfolio: Portfolio backtest
    - paper: Paper trading simulation
    
    Options:
    - window: Time window (day/week/month) for paper trading
    - canary: Quick canary test with limited data
    """
    
    try:
        from src.coint2.engine.base_engine import BasePairBacktester
        from src.coint2.utils.config import load_config
        
        # Load configuration
        cfg = load_config(config)
        
        # Configure dates based on mode
        if canary:
            # Quick test with recent 30 days
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_dt = datetime.now() - timedelta(days=30)
                start_date = start_dt.strftime('%Y-%m-%d')
        
        if paper:
            # Paper trading mode
            window_days = {
                'day': 1,
                'week': 7,
                'month': 30
            }
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                days = window_days.get(window, 30)
                start_dt = datetime.now() - timedelta(days=days)
                start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"🚀 Starting backtest")
        print(f"   Mode: {mode}")
        print(f"   Config: {config}")
        if pairs_file:
            print(f"   Pairs: {pairs_file}")
        print(f"   Period: {start_date} to {end_date}")
        if paper:
            print(f"   Paper trading: {window} window")
        if canary:
            print(f"   ⚠️ Canary mode (limited data)")
        
        # Run appropriate backtest
        if mode == "single":
            # Single pair backtest
            backtester = BasePairBacktester(cfg)
            results = backtester.run(
                start_date=start_date,
                end_date=end_date,
                pair=kwargs.get('pair', 'BTCUSDT/ETHUSDT')
            )
            
        elif mode == "portfolio":
            # Portfolio backtest
            from src.coint2.portfolio.portfolio_backtester import PortfolioBacktester
            
            if not pairs_file:
                pairs_file = "benchmarks/pairs_universe.yaml"
            
            backtester = PortfolioBacktester(
                config=cfg,
                pairs_file=pairs_file
            )
            results = backtester.run(
                start_date=start_date,
                end_date=end_date
            )
            
        elif mode == "paper" or paper:
            # Paper trading simulation
            from src.coint2.live.paper_trader import PaperTrader
            
            trader = PaperTrader(
                config=cfg,
                pairs_file=pairs_file,
                window=window
            )
            results = trader.run(
                start_date=start_date,
                end_date=end_date
            )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f"backtest_{mode}_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Backtest complete")
        print(f"   Results saved to: {results_file}")
        
        # Print summary
        if isinstance(results, dict):
            if 'sharpe' in results:
                print(f"   Sharpe: {results['sharpe']:.3f}")
            if 'total_pnl' in results:
                print(f"   Total PnL: ${results['total_pnl']:.2f}")
            if 'n_trades' in results:
                print(f"   Trades: {results['n_trades']}")
        
        return 0
        
    except ImportError as e:
        print(f"⚠️ Module not found: {e}")
        print("Falling back to script execution...")
        
        # Fallback to old scripts
        import subprocess
        
        if paper and window == "week":
            script = "scripts/run_paper_week.py"
        elif paper or canary:
            script = "scripts/run_paper_canary.py"
        else:
            script = "scripts/run_optuna_backtest.py"
        
        if not Path(script).exists():
            print(f"❌ Script not found: {script}")
            return 1
        
        cmd = [sys.executable, script]
        if config:
            cmd.extend(['--config', config])
        if start_date:
            cmd.extend(['--start-date', start_date])
        if end_date:
            cmd.extend(['--end-date', end_date])
        
        return subprocess.call(cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified backtesting script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair backtest
  %(prog)s --mode single --pair BTCUSDT/ETHUSDT
  
  # Portfolio backtest
  %(prog)s --mode portfolio --pairs-file benchmarks/pairs_universe.yaml
  
  # Paper trading week
  %(prog)s --paper --window week
  
  # Quick canary test
  %(prog)s --canary --mode single
"""
    )
    
    parser.add_argument('--mode', 
                       choices=['single', 'portfolio', 'paper'],
                       default='single',
                       help='Backtest mode')
    
    parser.add_argument('--config', default='configs/main.yaml',
                       help='Configuration file')
    
    parser.add_argument('--pairs-file',
                       help='Pairs file for portfolio/paper mode')
    
    parser.add_argument('--pair', default='BTCUSDT/ETHUSDT',
                       help='Pair for single mode')
    
    parser.add_argument('--start-date',
                       help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date',
                       help='End date (YYYY-MM-DD)')
    
    parser.add_argument('--paper', action='store_true',
                       help='Enable paper trading mode')
    
    parser.add_argument('--window', 
                       choices=['day', 'week', 'month'],
                       default='month',
                       help='Window for paper trading')
    
    parser.add_argument('--canary', action='store_true',
                       help='Quick canary test with limited data')
    
    parser.add_argument('--output-dir', default='outputs/backtest',
                       help='Output directory')
    
    args = parser.parse_args(argv)
    
    return run_backtest(**vars(args))


if __name__ == '__main__':
    sys.exit(main())