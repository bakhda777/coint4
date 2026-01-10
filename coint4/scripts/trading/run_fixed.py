#!/usr/bin/env python3
"""
Lightweight OOS runner with fixed parameters (no Optuna).
Runs backtests on selected pairs with fixed z-score parameters.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.coint2.core.data_loader import DataHandler
from src.coint2.utils.config import load_config


def load_pairs(pairs_file: str):
    """Load pairs from universe YAML file."""
    with open(pairs_file) as f:
        yml = yaml.safe_load(f)
    
    pairs = []
    for row in yml.get('pairs', []):
        pair_str = row['pair']  # Format: "AAA/BBB"
        symbols = pair_str.split('/')
        if len(symbols) == 2:
            pairs.append({
                'symbol1': symbols[0],
                'symbol2': symbols[1],
                'beta': row.get('metrics', {}).get('beta', 1.0),
                'alpha': row.get('metrics', {}).get('alpha', 0.0)
            })
    return pairs


def run_backtest(prices_df, pair_info, config, max_bars=0):
    """Run simple backtest for a single pair."""
    sym1, sym2 = pair_info['symbol1'], pair_info['symbol2']
    beta = pair_info.get('beta', 1.0)
    alpha = pair_info.get('alpha', 0.0)
    
    # Get price series
    if sym1 not in prices_df.columns or sym2 not in prices_df.columns:
        return None
    
    y = prices_df[sym1].values
    x = prices_df[sym2].values
    
    # Limit to max_bars if specified
    if max_bars > 0 and len(y) > max_bars:
        y = y[-max_bars:]
        x = x[-max_bars:]
    
    # Calculate spread
    spread = y - beta * x - alpha
    
    # Calculate z-score
    lookback = config['backtest'].get('rolling_window', 30)
    z_scores = np.zeros(len(spread))
    
    for i in range(lookback, len(spread)):
        window = spread[i-lookback:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            z_scores[i] = (spread[i] - mean) / std
    
    # Trading signals
    z_enter = config['backtest'].get('zscore_threshold', 1.0)
    z_exit = config['backtest'].get('zscore_exit', 0.0)
    z_stop = config['backtest'].get('stop_loss_multiplier', 3.0)
    time_stop = int(config['backtest'].get('time_stop_multiplier', 2.0) * lookback)
    
    # Simulate trades
    position = 0  # 1=long spread, -1=short spread, 0=flat
    entry_bar = 0
    trades = []
    pnl = []
    
    for i in range(lookback, len(spread)):
        z = z_scores[i]
        
        # Entry logic
        if position == 0:
            if abs(z) >= z_enter:
                position = -1 if z > 0 else 1  # Short if z>0, long if z<0
                entry_bar = i
                entry_price = spread[i]
        # Exit logic
        elif position != 0:
            bars_held = i - entry_bar
            should_exit = (
                abs(z) <= z_exit or  # Normal exit
                abs(z) >= z_stop or  # Stop loss
                bars_held >= time_stop  # Time stop
            )
            
            if should_exit:
                exit_price = spread[i]
                trade_pnl = (exit_price - entry_price) * position
                
                # Apply costs
                commission_pct = config['backtest'].get('commission_pct', 0.0004)
                slippage_pct = config['backtest'].get('slippage_pct', 0.0005)
                cost_pct = 2 * (commission_pct + slippage_pct)  # Both sides
                trade_pnl *= (1 - cost_pct)
                
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'bars_held': bars_held,
                    'entry_z': z_scores[entry_bar],
                    'exit_z': z,
                    'pnl': trade_pnl
                })
                pnl.append(trade_pnl)
                position = 0
    
    return {'trades': trades, 'pnl': pnl, 'pair': f"{sym1}/{sym2}"}


def calculate_metrics(all_trades, all_pnl):
    """Calculate performance metrics."""
    if not all_pnl:
        return {'error': 'No trades executed'}
    
    pnl_array = np.array(all_pnl)
    returns = pnl_array
    
    metrics = {
        'total_pnl': float(np.sum(pnl_array)),
        'num_trades': len(all_trades),
        'win_rate': float(np.mean([1 if t['pnl'] > 0 else 0 for t in all_trades])) if all_trades else 0,
        'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
        'max_drawdown': float(np.min(np.minimum.accumulate(pnl_array) - pnl_array)) if len(pnl_array) > 0 else 0,
        'avg_bars_held': float(np.mean([t['bars_held'] for t in all_trades])) if all_trades else 0
    }
    return metrics


def deep_merge_configs(base, delta):
    """Deep merge delta config into base config."""
    if delta is None:
        return base
    
    result = base.copy() if isinstance(base, dict) else base
    
    if not isinstance(result, dict) or not isinstance(delta, dict):
        return delta
    
    for key, value in delta.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Fixed parameter OOS backtest runner')
    parser.add_argument('--data-root', default='./data_downloaded', help='Data root directory')
    parser.add_argument('--timeframe', default='15T', help='Data timeframe')
    parser.add_argument('--period-start', required=True, help='Period start (YYYY-MM-DD)')
    parser.add_argument('--period-end', required=True, help='Period end (YYYY-MM-DD)')
    parser.add_argument('--pairs-file', required=True, help='Pairs universe YAML file')
    parser.add_argument('--config', required=True, help='Config with fixed parameters')
    parser.add_argument('--config-delta', help='YAML overlay to apply on top of base config')
    parser.add_argument('--out-dir', default='outputs/fixed_run', help='Output directory')
    parser.add_argument('--max-bars', type=int, default=0, help='Max bars to use (0=all)')
    
    args = parser.parse_args()
    
    # Setup output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and apply delta if provided
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Apply config delta (deep merge)
    if args.config_delta:
        with open(args.config_delta) as f:
            delta = yaml.safe_load(f)
        config = deep_merge_configs(config, delta)
        print(f"📝 Applied config delta from {args.config_delta}")
    
    pairs = load_pairs(args.pairs_file)
    print(f"📊 Loaded {len(pairs)} pairs for testing")
    
    # Initialize data handler
    base_config = load_config(args.config)
    data_handler = DataHandler(base_config)
    
    # Load price data
    print(f"📈 Loading data from {args.period_start} to {args.period_end}...")
    prices_df = data_handler.load_all_data_for_period(
        lookback_days=90,
        end_date=pd.Timestamp(args.period_end)
    )
    
    # Run backtests
    all_trades = []
    all_pnl = []
    cumulative_pnl = []
    running_sum = 0
    
    for i, pair_info in enumerate(pairs):
        if i % 10 == 0:
            print(f"⏱️ Processing pair {i+1}/{len(pairs)}...")
        
        result = run_backtest(prices_df, pair_info, config, args.max_bars)
        if result:
            # Add pair info to trades
            for trade in result['trades']:
                trade['pair'] = result['pair']
            all_trades.extend(result['trades'])
            
            # Track cumulative PnL
            for pnl in result['pnl']:
                running_sum += pnl
                cumulative_pnl.append(running_sum)
            all_pnl.extend(result['pnl'])
    
    # Calculate and save metrics
    metrics = calculate_metrics(all_trades, all_pnl)
    
    with open(out_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)
    
    # Save trades CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(out_dir / 'trades.csv', index=False)
    
    # Save equity curve
    if cumulative_pnl:
        equity_df = pd.DataFrame({
            'bar': range(len(cumulative_pnl)),
            'cumulative_pnl': cumulative_pnl
        })
        equity_df.to_csv(out_dir / 'equity.csv', index=False)
    
    print(f"✅ Completed: {metrics.get('num_trades', 0)} trades, Sharpe={metrics.get('sharpe_ratio', 0):.2f}")
    print(f"📁 Results saved to {out_dir}")
if __name__ == '__main__':
    main()