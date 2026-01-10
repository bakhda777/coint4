#!/usr/bin/env python3
"""
Validate universe selection results with backtesting.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coint2.core.data_loader import DataHandler
from src.coint2.engine.base_engine import BasePairBacktester
from src.coint2.utils.config import load_config


def load_top_pairs(universe_dir: str, top_n: int = 10):
    """Load top pairs from universe selection."""
    pairs_file = Path(universe_dir) / 'pairs_universe.yaml'
    
    with open(pairs_file) as f:
        data = yaml.safe_load(f)
    
    pairs = []
    for pair_data in data['pairs'][:top_n]:
        pairs.append(pair_data['pair'])
    
    return pairs


def backtest_pair(pair: str, data_handler, start_date, end_date, config):
    """Run backtest for a single pair."""
    symbol1, symbol2 = pair.split('/')
    
    # Load data
    df = data_handler.load_all_data_for_period(
        lookback_days=90,
        end_date=pd.Timestamp(end_date)
    )
    
    if symbol1 not in df.columns or symbol2 not in df.columns:
        return None
    
    # Prepare pair data
    pair_data = pd.DataFrame({
        'symbol1': df[symbol1].values,
        'symbol2': df[symbol2].values
    })
    
    # Run backtest
    backtester = BasePairBacktester(
        pair_data=pair_data,
        rolling_window=config.backtest.rolling_window,
        z_threshold=config.backtest.zscore_threshold,
        z_exit=config.backtest.zscore_exit,
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct
    )
    
    backtester.run()
    
    # Get results from backtester.results DataFrame
    if backtester.results is None or backtester.results.empty:
        return None
    
    results_df = backtester.results
    total_pnl = results_df['cumulative_pnl'].iloc[-1] if 'cumulative_pnl' in results_df.columns else 0
    
    # Calculate metrics
    if 'pnl' in results_df.columns:
        returns = results_df['pnl'].dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        num_trades = (results_df['position'].diff() != 0).sum() // 2
    else:
        sharpe = 0
        num_trades = 0
    
    return {
        'pair': pair,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe,
        'num_trades': num_trades,
        'win_rate': 0,  # Would need trade-level data
        'max_drawdown': 0  # Would need to calculate
    }


def main():
    parser = argparse.ArgumentParser(description='Validate universe selection')
    parser.add_argument('--universe-dir', required=True, help='Universe output directory')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top pairs to test')
    parser.add_argument('--period-start', required=True, help='Test period start')
    parser.add_argument('--period-end', required=True, help='Test period end')
    parser.add_argument('--config', default='configs/main_2024.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_handler = DataHandler(config)
    
    # Load top pairs
    pairs = load_top_pairs(args.universe_dir, args.top_n)
    print(f"🎯 Testing {len(pairs)} pairs on {args.period_start} to {args.period_end}")
    
    # Backtest each pair
    results = []
    for i, pair in enumerate(pairs, 1):
        print(f"  {i}/{len(pairs)}: {pair}...", end='')
        result = backtest_pair(
            pair, 
            data_handler,
            args.period_start,
            args.period_end,
            config
        )
        
        if result:
            results.append(result)
            print(f" ✓ PnL=${result['total_pnl']:.2f}, Sharpe={result['sharpe_ratio']:.2f}")
        else:
            print(" ✗ No data")
    
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n📊 SUMMARY")
        print("="*50)
        print(f"Pairs tested: {len(results)}")
        print(f"Profitable: {(df_results['total_pnl'] > 0).sum()}")
        print(f"Average PnL: ${df_results['total_pnl'].mean():.2f}")
        print(f"Average Sharpe: {df_results['sharpe_ratio'].mean():.2f}")
        print(f"Average trades: {df_results['num_trades'].mean():.1f}")
        
        print("\n🏆 TOP 5 BY SHARPE")
        top5 = df_results.nlargest(5, 'sharpe_ratio')
        for _, row in top5.iterrows():
            print(f"  {row['pair']:20s} Sharpe={row['sharpe_ratio']:.2f} PnL=${row['total_pnl']:.2f}")
        
        # Save results
        output_file = Path(args.universe_dir) / 'validation_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")


if __name__ == '__main__':
    main()