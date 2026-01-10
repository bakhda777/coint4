#!/usr/bin/env python3
"""Tune execution profiles (POV/TWAP) to minimize slippage."""

import argparse
import pandas as pd
import numpy as np
import optuna
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coint2.execution.simulator import ExecutionSimulator, Order


def simulate_execution_batch(
    orders: List[Order],
    config: dict,
    market_conditions: dict
) -> dict:
    """Simulate batch execution with given config."""
    
    simulator = ExecutionSimulator(config)
    
    total_slippage = 0
    total_commission = 0
    total_latency = 0
    n_partial = 0
    
    for order in orders:
        # Get market data for order
        atr = market_conditions.get('atr', 0.02)
        zscore = market_conditions.get('zscore', np.random.randn())
        volatility = market_conditions.get('volatility', 0.3)
        volume = market_conditions.get('volume', 100000)
        
        fill = simulator.simulate_fill(order, atr, zscore, volatility, volume)
        
        # Calculate costs
        if order.side == 'buy':
            slippage_cost = fill.filled_quantity * (fill.filled_price - order.price)
        else:
            slippage_cost = fill.filled_quantity * (order.price - fill.filled_price)
            
        total_slippage += slippage_cost
        total_commission += fill.commission
        total_latency += fill.latency_ms
        
        if fill.partial:
            n_partial += 1
    
    return {
        'total_slippage': total_slippage,
        'avg_slippage_pct': total_slippage / sum(o.quantity * o.price for o in orders),
        'total_commission': total_commission,
        'avg_latency_ms': total_latency / len(orders),
        'partial_rate': n_partial / len(orders)
    }


def generate_test_orders(n_orders: int = 100, seed: int = 42) -> List[Order]:
    """Generate test orders for simulation."""
    
    np.random.seed(seed)
    
    orders = []
    for i in range(n_orders):
        order = Order(
            pair=f"BTC/USDT",
            side=np.random.choice(['buy', 'sell']),
            quantity=np.random.uniform(0.01, 1.0),  # BTC quantity
            price=50000 * (1 + np.random.uniform(-0.01, 0.01)),  # Around $50k
            timestamp=i * 60.0  # 1 minute apart
        )
        orders.append(order)
    
    return orders


def objective(trial, orders, market_conditions):
    """Optuna objective for execution tuning."""
    
    config = {
        'latency_mean_ms': 10,
        'latency_std_ms': 5,
        'commission_rate': 0.001,
        'slippage_base': 0.0001,
        'slippage_atr_coef': 0.1,
        'slippage_zscore_coef': 0.0002,
        'max_slippage_pct': 0.01,
        'partial_fill_prob': 0.1,
        'partial_fill_ratio': 0.7,
        
        # POV settings to optimize
        'pov': {
            'enabled': trial.suggest_categorical('pov_enabled', [True, False]),
            'participation': trial.suggest_float('pov_participation', 0.05, 0.20)
        },
        
        # TWAP settings to optimize
        'twap': {
            'enabled': trial.suggest_categorical('twap_enabled', [True, False]),
            'slices': trial.suggest_int('twap_slices', 2, 8)
        },
        
        # Size clipping
        'clip_sizing': {
            'enabled': trial.suggest_categorical('clip_enabled', [True, False]),
            'max_adv_pct': trial.suggest_float('max_adv_pct', 0.5, 2.0)
        }
    }
    
    # Simulate execution
    results = simulate_execution_batch(orders, config, market_conditions)
    
    # Objective: minimize slippage + commission
    total_cost = results['avg_slippage_pct'] + config['commission_rate']
    
    # Store metrics
    trial.set_user_attr('slippage_pct', results['avg_slippage_pct'])
    trial.set_user_attr('latency_ms', results['avg_latency_ms'])
    trial.set_user_attr('partial_rate', results['partial_rate'])
    
    return total_cost


def analyze_execution_profiles(study: optuna.Study) -> pd.DataFrame:
    """Analyze different execution profiles from study."""
    
    profiles = []
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        profile = {
            'trial_id': trial.number,
            'total_cost': trial.value,
            'slippage_pct': trial.user_attrs.get('slippage_pct', 0),
            'latency_ms': trial.user_attrs.get('latency_ms', 0),
            'partial_rate': trial.user_attrs.get('partial_rate', 0),
            'pov_enabled': trial.params.get('pov_enabled', False),
            'pov_participation': trial.params.get('pov_participation', 0),
            'twap_enabled': trial.params.get('twap_enabled', False),
            'twap_slices': trial.params.get('twap_slices', 1),
            'clip_enabled': trial.params.get('clip_enabled', False),
            'max_adv_pct': trial.params.get('max_adv_pct', 1.0)
        }
        
        # Classify profile
        if profile['pov_enabled'] and profile['twap_enabled']:
            profile['profile_type'] = 'POV+TWAP'
        elif profile['pov_enabled']:
            profile['profile_type'] = 'POV'
        elif profile['twap_enabled']:
            profile['profile_type'] = 'TWAP'
        else:
            profile['profile_type'] = 'Direct'
            
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def generate_execution_report(profiles_df: pd.DataFrame, output_path: Path):
    """Generate execution tuning report."""
    
    report = f"""# Execution Profile Tuning Report

*Generated: {datetime.now().isoformat()}*

## Executive Summary

Analyzed {len(profiles_df)} execution configurations to minimize trading costs.

## Best Profiles by Type

"""
    
    # Group by profile type
    for profile_type in ['Direct', 'POV', 'TWAP', 'POV+TWAP']:
        type_df = profiles_df[profiles_df['profile_type'] == profile_type]
        
        if type_df.empty:
            continue
            
        best = type_df.loc[type_df['total_cost'].idxmin()]
        
        report += f"""### {profile_type}

- **Total Cost**: {best['total_cost']:.3%}
- **Slippage**: {best['slippage_pct']:.3%}
- **Latency**: {best['latency_ms']:.1f} ms
- **Partial Fill Rate**: {best['partial_rate']:.1%}
"""
        
        if profile_type in ['POV', 'POV+TWAP']:
            report += f"- **POV Participation**: {best['pov_participation']:.1%}\n"
        if profile_type in ['TWAP', 'POV+TWAP']:
            report += f"- **TWAP Slices**: {best['twap_slices']}\n"
        if best['clip_enabled']:
            report += f"- **Max ADV**: {best['max_adv_pct']:.0%}\n"
            
        report += "\n"
    
    # Overall best
    best_overall = profiles_df.loc[profiles_df['total_cost'].idxmin()]
    
    report += f"""## Recommended Configuration

The optimal execution profile achieves **{best_overall['total_cost']:.3%}** total cost:

- **Profile Type**: {best_overall['profile_type']}
- **Slippage**: {best_overall['slippage_pct']:.3%}
- **Commission**: 0.100%
- **Total Cost**: {best_overall['total_cost']:.3%}

### Configuration
```yaml
pov:
  enabled: {str(best_overall['pov_enabled']).lower()}
  participation: {best_overall['pov_participation']:.3f}

twap:
  enabled: {str(best_overall['twap_enabled']).lower()}
  slices: {int(best_overall['twap_slices'])}

clip_sizing:
  enabled: {str(best_overall['clip_enabled']).lower()}
  max_adv_pct: {best_overall['max_adv_pct']:.2f}
```

## Cost Reduction Analysis

"""
    
    # Compare to baseline (Direct execution)
    baseline = profiles_df[profiles_df['profile_type'] == 'Direct']
    if not baseline.empty:
        baseline_cost = baseline['total_cost'].min()
        reduction = (baseline_cost - best_overall['total_cost']) / baseline_cost
        
        report += f"""Compared to direct execution:
- **Baseline Cost**: {baseline_cost:.3%}
- **Optimized Cost**: {best_overall['total_cost']:.3%}
- **Cost Reduction**: {reduction:.1%}

This translates to:
- Annual savings on $1M volume: ${reduction * 10000:.0f}
- Break-even after: {100 / max(reduction * 100, 0.1):.0f} trades
"""
    
    # Market conditions impact
    report += """
## Market Conditions Impact

The optimal profile varies with market conditions:

| Condition | Recommended Profile | Rationale |
|-----------|-------------------|-----------|
| High Volatility | POV+TWAP | Spread execution to reduce impact |
| Low Liquidity | POV | Limit participation rate |
| Urgent Execution | Direct | Accept higher costs for speed |
| Large Orders | TWAP | Split to minimize market impact |

## Implementation Checklist

- [ ] Update `configs/execution.yaml` with recommended settings
- [ ] Test with paper trading for 1 week
- [ ] Monitor actual vs predicted slippage
- [ ] Adjust participation rates based on liquidity
- [ ] Set up alerts for high slippage events
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return best_overall


def main():
    parser = argparse.ArgumentParser(description='Tune execution profiles')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--n-orders', type=int, default=100,
                       help='Number of test orders')
    parser.add_argument('--volatility', type=float, default=0.3,
                       help='Market volatility (annualized)')
    parser.add_argument('--volume', type=float, default=100000,
                       help='Average daily volume')
    parser.add_argument('--output-dir', default='artifacts/cost',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("⚙️ Tuning Execution Profiles...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test orders
    orders = generate_test_orders(args.n_orders)
    
    # Market conditions
    market_conditions = {
        'atr': 0.02,
        'volatility': args.volatility,
        'volume': args.volume
    }
    
    # Run optimization
    print(f"Running {args.n_trials} trials...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, orders, market_conditions),
        n_trials=args.n_trials
    )
    
    print(f"Best cost: {study.best_value:.3%}")
    
    # Analyze profiles
    profiles_df = analyze_execution_profiles(study)
    
    # Save profiles CSV
    csv_path = output_dir / 'execution_profiles.csv'
    profiles_df.to_csv(csv_path, index=False)
    
    # Generate report
    report_path = output_dir / 'EXECUTION_TUNING.md'
    best_config = generate_execution_report(profiles_df, report_path)
    
    # Save best config
    config_path = output_dir / 'best_execution.yaml'
    with open(config_path, 'w') as f:
        yaml.dump({
            'pov': {
                'enabled': bool(best_config['pov_enabled']),
                'participation': float(best_config['pov_participation'])
            },
            'twap': {
                'enabled': bool(best_config['twap_enabled']),
                'slices': int(best_config['twap_slices'])
            },
            'clip_sizing': {
                'enabled': bool(best_config['clip_enabled']),
                'max_adv_pct': float(best_config['max_adv_pct'])
            }
        }, f)
    
    print(f"📄 Report saved to {report_path}")
    print(f"📊 Profiles saved to {csv_path}")
    print(f"⚙️ Best config saved to {config_path}")
    
    # Print summary
    print(f"\n📊 Execution Tuning Summary:")
    print(f"  Profile: {best_config['profile_type']}")
    print(f"  Total Cost: {best_config['total_cost']:.3%}")
    print(f"  Slippage: {best_config['slippage_pct']:.3%}")
    
    # Calculate cost reduction
    baseline = profiles_df[profiles_df['profile_type'] == 'Direct']
    if not baseline.empty:
        baseline_cost = baseline['total_cost'].min()
        reduction = (baseline_cost - best_config['total_cost']) / baseline_cost
        print(f"  Cost Reduction: {reduction:.1%}")


if __name__ == '__main__':
    main()