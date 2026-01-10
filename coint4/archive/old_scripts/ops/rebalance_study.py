#!/usr/bin/env python3
"""Study rebalancing policies to optimize PSR while reducing turnover."""

import argparse
import pandas as pd
import numpy as np
import optuna
import yaml
from pathlib import Path
from datetime import datetime
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coint2.utils.config import load_config


def calculate_psr(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """Calculate Probabilistic Sharpe Ratio."""
    if len(returns) < 30:
        return 0.0
    
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    n = len(returns)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
    
    from scipy import stats
    psr = stats.norm.cdf((sharpe - benchmark_sr) / se_sharpe)
    return psr


def calculate_turnover(positions: pd.DataFrame) -> float:
    """Calculate average daily turnover."""
    daily_changes = positions.diff().abs().sum(axis=1)
    avg_turnover = daily_changes.mean()
    return avg_turnover


def simulate_rebalancing(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict
) -> dict:
    """Simulate rebalancing with given policy."""
    
    mode = config['mode']
    frequency = config['frequency']
    drift_threshold = config['drift_threshold']
    hysteresis = config['hysteresis']
    turnover_cap = config['turnover_cap']
    min_hold_bars = config['min_hold_bars']
    
    # Initialize positions
    positions = pd.DataFrame(0, index=signals.index, columns=signals.columns)
    target_weights = signals.copy()
    
    # Track metrics
    holds = pd.DataFrame(0, index=signals.index, columns=signals.columns)
    
    for i in range(1, len(signals)):
        # Check min hold constraint
        mask_can_change = holds.iloc[i-1] >= min_hold_bars
        
        # Determine if we should rebalance
        should_rebalance = False
        
        if mode == 'periodic':
            # Rebalance on schedule
            if frequency == '1D':
                should_rebalance = True
            elif frequency == '2D' and i % 2 == 0:
                should_rebalance = True
            elif frequency == '1W' and i % 5 == 0:
                should_rebalance = True
                
        elif mode == 'threshold':
            # Rebalance if drift exceeds threshold
            drift = abs(positions.iloc[i-1] - target_weights.iloc[i]).sum()
            should_rebalance = drift > drift_threshold
            
        elif mode == 'hybrid':
            # Combine periodic and threshold
            is_periodic = (frequency == '2D' and i % 2 == 0) or (frequency == '1W' and i % 5 == 0)
            drift = abs(positions.iloc[i-1] - target_weights.iloc[i]).sum()
            should_rebalance = is_periodic or (drift > drift_threshold)
        
        if should_rebalance:
            # Apply hysteresis bands
            new_positions = positions.iloc[i-1].copy()
            
            for col in signals.columns:
                if not mask_can_change[col]:
                    continue
                    
                current = positions.iloc[i-1, col]
                target = target_weights.iloc[i, col]
                
                # Enter band
                if abs(target - current) > hysteresis['enter']:
                    new_positions[col] = target
                # Exit band
                elif abs(target) < hysteresis['exit']:
                    new_positions[col] = 0
                    
            # Apply turnover cap
            turnover = abs(new_positions - positions.iloc[i-1]).sum()
            if turnover > turnover_cap:
                scale = turnover_cap / turnover
                new_positions = positions.iloc[i-1] + scale * (new_positions - positions.iloc[i-1])
            
            positions.iloc[i] = new_positions
            
            # Reset hold counters for changed positions
            changed = (positions.iloc[i] != positions.iloc[i-1])
            holds.iloc[i] = holds.iloc[i-1] + 1
            holds.iloc[i][changed] = 0
        else:
            positions.iloc[i] = positions.iloc[i-1]
            holds.iloc[i] = holds.iloc[i-1] + 1
    
    # Calculate returns
    price_returns = prices.pct_change()
    strategy_returns = (positions.shift(1) * price_returns).sum(axis=1)
    
    # Calculate metrics
    psr = calculate_psr(strategy_returns.dropna().values)
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
    turnover = calculate_turnover(positions)
    num_trades = (positions.diff() != 0).sum().sum()
    
    # Calculate costs (simplified)
    commission_pct = 0.001
    slippage_pct = 0.0005
    cost_per_trade = commission_pct + slippage_pct
    total_cost = num_trades * cost_per_trade
    
    return {
        'psr': psr,
        'sharpe': sharpe,
        'turnover': turnover,
        'num_trades': num_trades,
        'total_cost': total_cost,
        'positions': positions
    }


def objective(trial, signals, prices, lambda_turnover=0.5):
    """Optuna objective for rebalancing optimization."""
    
    config = {
        'mode': trial.suggest_categorical('mode', ['periodic', 'threshold', 'hybrid']),
        'frequency': trial.suggest_categorical('frequency', ['1D', '2D', '1W']),
        'drift_threshold': trial.suggest_float('drift_threshold', 0.03, 0.08),
        'hysteresis': {
            'enter': trial.suggest_float('hysteresis_enter', 0.2, 0.3),
            'exit': trial.suggest_float('hysteresis_exit', 0.1, 0.2)
        },
        'turnover_cap': trial.suggest_float('turnover_cap', 0.10, 0.20),
        'min_hold_bars': trial.suggest_int('min_hold_bars', 8, 16)
    }
    
    results = simulate_rebalancing(signals, prices, config)
    
    # Objective: maximize PSR - lambda * turnover
    objective_value = results['psr'] - lambda_turnover * results['turnover']
    
    # Store additional metrics
    trial.set_user_attr('sharpe', results['sharpe'])
    trial.set_user_attr('turnover', results['turnover'])
    trial.set_user_attr('num_trades', results['num_trades'])
    
    return objective_value


def run_grid_search(signals, prices):
    """Run grid search over rebalancing parameters."""
    
    param_grid = {
        'mode': ['periodic', 'threshold', 'hybrid'],
        'frequency': ['1D', '2D', '1W'],
        'drift_threshold': [0.03, 0.05, 0.08],
        'hysteresis_enter': [0.2, 0.3],
        'hysteresis_exit': [0.1, 0.2],
        'turnover_cap': [0.10, 0.20],
        'min_hold_bars': [8, 16]
    }
    
    results = []
    
    for params in product(*param_grid.values()):
        config = {
            'mode': params[0],
            'frequency': params[1],
            'drift_threshold': params[2],
            'hysteresis': {
                'enter': params[3],
                'exit': params[4]
            },
            'turnover_cap': params[5],
            'min_hold_bars': params[6]
        }
        
        result = simulate_rebalancing(signals, prices, config)
        result.update(config)
        results.append(result)
    
    return pd.DataFrame(results)


def generate_pareto_report(results_df, output_path):
    """Generate Pareto front report."""
    
    # Find Pareto optimal points
    pareto_points = []
    
    for i, row in results_df.iterrows():
        is_pareto = True
        for j, other in results_df.iterrows():
            if i != j:
                # Check if 'other' dominates 'row'
                if other['psr'] >= row['psr'] and other['turnover'] <= row['turnover']:
                    if other['psr'] > row['psr'] or other['turnover'] < row['turnover']:
                        is_pareto = False
                        break
        
        if is_pareto:
            pareto_points.append(row)
    
    pareto_df = pd.DataFrame(pareto_points)
    
    # Generate report
    report = f"""# Rebalancing Policy Study

*Generated: {datetime.now().isoformat()}*

## Executive Summary

Studied {len(results_df)} rebalancing configurations to optimize PSR while reducing turnover.

## Pareto Optimal Configurations

Found {len(pareto_df)} Pareto optimal configurations:

| Mode | Frequency | Drift Threshold | PSR | Sharpe | Turnover | Trades |
|------|-----------|----------------|-----|--------|----------|--------|
"""
    
    for _, row in pareto_df.head(10).iterrows():
        report += f"| {row.get('mode', 'N/A')} | {row.get('frequency', 'N/A')} | "
        report += f"{row.get('drift_threshold', 0):.3f} | {row['psr']:.3f} | "
        report += f"{row.get('sharpe', 0):.2f} | {row['turnover']:.3f} | {row.get('num_trades', 0):.0f} |\n"
    
    # Select recommended configuration
    # Filter for PSR >= 0.95
    high_psr = pareto_df[pareto_df['psr'] >= 0.95]
    
    if not high_psr.empty:
        # Among high PSR, select lowest turnover
        recommended = high_psr.loc[high_psr['turnover'].idxmin()]
        
        report += f"""
## Recommended Configuration

Selected configuration with PSR ≥ 0.95 and minimum turnover:

- **Mode**: {recommended.get('mode', 'threshold')}
- **Frequency**: {recommended.get('frequency', '2D')}
- **Drift Threshold**: {recommended.get('drift_threshold', 0.05):.3f}
- **Hysteresis Enter**: {recommended.get('hysteresis', {}).get('enter', 0.3):.2f}
- **Hysteresis Exit**: {recommended.get('hysteresis', {}).get('exit', 0.15):.2f}
- **Turnover Cap**: {recommended.get('turnover_cap', 0.20):.2f}
- **Min Hold Bars**: {recommended.get('min_hold_bars', 8):.0f}

### Performance Metrics
- **PSR**: {recommended['psr']:.3f}
- **Sharpe**: {recommended.get('sharpe', 0):.2f}
- **Daily Turnover**: {recommended['turnover']:.3f}
- **Total Trades**: {recommended.get('num_trades', 0):.0f}

### Cost Impact
- **Estimated Cost/Signal**: {recommended.get('total_cost', 0) / max(abs(recommended.get('sharpe', 1)), 0.01):.2%}
"""
    else:
        report += "\n## Warning\n\nNo configuration achieved PSR ≥ 0.95. Consider relaxing constraints.\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return pareto_df


def main():
    parser = argparse.ArgumentParser(description='Study rebalancing policies')
    parser.add_argument('--method', default='optuna', choices=['grid', 'optuna'],
                       help='Search method')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--lambda-turnover', type=float, default=0.5,
                       help='Turnover penalty weight')
    parser.add_argument('--output-dir', default='artifacts/cost',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔍 Running Rebalancing Policy Study...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data (replace with real data in production)
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Sample signals (z-scores)
    signals = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.3,
        index=dates,
        columns=[f'pair_{i}' for i in range(n_assets)]
    )
    
    # Sample prices
    prices = pd.DataFrame(
        100 * np.exp(np.random.randn(n_days, n_assets).cumsum() * 0.01),
        index=dates,
        columns=[f'pair_{i}' for i in range(n_assets)]
    )
    
    if args.method == 'optuna':
        # Run Optuna optimization
        print(f"Running Optuna with {args.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, signals, prices, args.lambda_turnover),
            n_trials=args.n_trials
        )
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **trial.params,
                'psr': trial.user_attrs.get('psr', trial.value),
                'sharpe': trial.user_attrs.get('sharpe', 0),
                'turnover': trial.user_attrs.get('turnover', 0),
                'num_trades': trial.user_attrs.get('num_trades', 0)
            }
            for trial in study.trials
        ])
        
        print(f"Best trial: PSR={study.best_value:.3f}")
        
    else:
        # Run grid search
        print("Running grid search...")
        results_df = run_grid_search(signals, prices)
    
    # Generate Pareto report
    report_path = output_dir / 'REBALANCE_STUDY.md'
    pareto_df = generate_pareto_report(results_df, report_path)
    
    # Save CSV
    csv_path = output_dir / 'rebalance_pareto.csv'
    pareto_df.to_csv(csv_path, index=False)
    
    print(f"📄 Report saved to {report_path}")
    print(f"📊 Pareto front saved to {csv_path}")
    
    # Print summary
    if not pareto_df.empty:
        best = pareto_df.iloc[0]
        print(f"\n📊 Best Configuration:")
        print(f"  PSR: {best['psr']:.3f}")
        print(f"  Turnover: {best['turnover']:.3f}")
        print(f"  Mode: {best.get('mode', 'N/A')}")


if __name__ == '__main__':
    main()