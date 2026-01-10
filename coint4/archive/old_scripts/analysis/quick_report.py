#!/usr/bin/env python3
"""Quick report generator for paper trading runs."""

import argparse
import pandas as pd
import yaml
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description='Generate quick report from paper run')
    parser.add_argument('--dir', required=True, help='Directory with paper run results')
    parser.add_argument('--output', help='Output CSV path (default: dir/QUICK_SUMMARY.csv)')
    
    args = parser.parse_args()
    
    run_dir = Path(args.dir)
    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        return 1
    
    # Initialize metrics
    metrics = {
        'num_trades': 0,
        'total_pnl': 0.0,
        'sharpe_ratio': 0.0,
        'hit_rate': 0.0,
        'avg_trade_duration_hours': 0.0
    }
    
    # Try to load metrics from various sources
    # 1. Check for metrics.yaml
    metrics_yaml = run_dir / 'metrics.yaml'
    if metrics_yaml.exists():
        with open(metrics_yaml) as f:
            data = yaml.safe_load(f)
            metrics.update({k: v for k, v in data.items() if k in metrics})
    
    # 2. Check for summary.json
    summary_json = run_dir / 'summary.json'
    if summary_json.exists():
        with open(summary_json) as f:
            data = json.load(f)
            metrics.update({k: v for k, v in data.items() if k in metrics})
    
    # 3. Check for trades.csv
    trades_csv = run_dir / 'trades.csv'
    if trades_csv.exists():
        trades_df = pd.read_csv(trades_csv)
        if not trades_df.empty:
            metrics['num_trades'] = len(trades_df)
            if 'pnl' in trades_df.columns:
                metrics['total_pnl'] = trades_df['pnl'].sum()
                metrics['hit_rate'] = (trades_df['pnl'] > 0).mean()
            if 'duration_hours' in trades_df.columns:
                metrics['avg_trade_duration_hours'] = trades_df['duration_hours'].mean()
    
    # Print report
    print(f"\n📊 Quick Report for {run_dir.name}")
    print("=" * 40)
    print(f"Number of trades: {metrics['num_trades']}")
    print(f"Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Hit Rate: {metrics['hit_rate']:.1%}")
    print(f"Avg Trade Duration: {metrics['avg_trade_duration_hours']:.1f} hours")
    
    # Save to CSV
    output_path = Path(args.output) if args.output else run_dir / 'QUICK_SUMMARY.csv'
    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(output_path, index=False)
    print(f"\n💾 Summary saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())