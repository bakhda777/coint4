#!/usr/bin/env python3
"""Analyze performance under different fee scenarios."""

import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_trades(trades_file: Path) -> pd.DataFrame:
    """Load trades from CSV."""
    
    df = pd.read_csv(trades_file, parse_dates=['entry_time', 'exit_time'])
    
    # Calculate trade metrics
    df['trade_size'] = df['entry_price'] * df['quantity']
    df['raw_pnl'] = df['quantity'] * (df['exit_price'] - df['entry_price'])
    
    return df


def apply_fee_schedule(
    trades: pd.DataFrame,
    fee_schedule: dict
) -> dict:
    """Apply fee schedule to trades and calculate metrics."""
    
    maker_fee = fee_schedule['maker']
    taker_fee = fee_schedule['taker']
    maker_ratio = fee_schedule.get('maker_ratio', 0.3)
    
    # Calculate fees per trade
    n_trades = len(trades)
    n_maker = int(n_trades * maker_ratio)
    n_taker = n_trades - n_maker
    
    # Assign maker/taker randomly
    is_maker = np.zeros(n_trades, dtype=bool)
    is_maker[np.random.choice(n_trades, n_maker, replace=False)] = True
    
    # Calculate fees
    entry_fees = np.where(
        is_maker,
        trades['trade_size'] * maker_fee,
        trades['trade_size'] * taker_fee
    )
    
    exit_fees = np.where(
        is_maker,
        trades['trade_size'] * maker_fee,
        trades['trade_size'] * taker_fee
    )
    
    total_fees = entry_fees + exit_fees
    
    # Calculate net PnL
    net_pnl = trades['raw_pnl'] - total_fees
    
    # Calculate metrics
    total_pnl = net_pnl.sum()
    total_fees_paid = total_fees.sum()
    n_profitable = (net_pnl > 0).sum()
    win_rate = n_profitable / n_trades if n_trades > 0 else 0
    
    # Calculate Sharpe (simplified daily)
    daily_returns = net_pnl.groupby(pd.Grouper(key='entry_time', freq='D')).sum()
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
    
    # Cost/Signal ratio
    signal_pnl = trades['raw_pnl'].sum()
    cost_signal_ratio = total_fees_paid / abs(signal_pnl) if signal_pnl != 0 else float('inf')
    
    return {
        'total_pnl': total_pnl,
        'signal_pnl': signal_pnl,
        'total_fees': total_fees_paid,
        'avg_fee_per_trade': total_fees_paid / n_trades if n_trades > 0 else 0,
        'n_trades': n_trades,
        'n_profitable': n_profitable,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'cost_signal_ratio': cost_signal_ratio,
        'maker_ratio': maker_ratio,
        'effective_fee_rate': total_fees_paid / (2 * trades['trade_size'].sum())
    }


def analyze_fee_scenarios(trades: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    """Analyze multiple fee scenarios."""
    
    results = []
    
    for scenario_name, fee_schedule in scenarios.items():
        print(f"Analyzing {scenario_name}...")
        
        metrics = apply_fee_schedule(trades.copy(), fee_schedule)
        metrics['scenario'] = scenario_name
        metrics['maker_fee_bps'] = fee_schedule['maker'] * 10000
        metrics['taker_fee_bps'] = fee_schedule['taker'] * 10000
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def find_breakeven_fee(trades: pd.DataFrame, target_sharpe: float = 0.5) -> float:
    """Find breakeven fee rate for target Sharpe."""
    
    signal_pnl = trades['raw_pnl'].sum()
    trade_volume = trades['trade_size'].sum() * 2  # Entry + exit
    
    # Binary search for breakeven fee
    low, high = 0.0, 0.01  # 0 to 100 bps
    
    for _ in range(20):  # 20 iterations should be enough
        mid = (low + high) / 2
        
        # Apply uniform fee
        total_fees = trade_volume * mid
        net_pnl = signal_pnl - total_fees
        
        # Calculate Sharpe
        trades['net_pnl'] = trades['raw_pnl'] - trades['trade_size'] * 2 * mid
        daily_returns = trades.groupby(pd.Grouper(key='entry_time', freq='D'))['net_pnl'].sum()
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
        
        if sharpe > target_sharpe:
            low = mid
        else:
            high = mid
    
    return mid


def generate_fee_report(results_df: pd.DataFrame, trades: pd.DataFrame, output_path: Path):
    """Generate fee scenario analysis report."""
    
    report = f"""# Fee Scenario Analysis

*Generated: {datetime.now().isoformat()}*

## Executive Summary

Analyzed {len(results_df)} fee scenarios across {len(trades)} trades.

## Fee Scenarios Comparison

| Scenario | Maker (bps) | Taker (bps) | Total PnL | Sharpe | Win Rate | Cost/Signal |
|----------|------------|-------------|-----------|--------|----------|-------------|
"""
    
    for _, row in results_df.iterrows():
        report += f"| {row['scenario']} | {row['maker_fee_bps']:.1f} | "
        report += f"{row['taker_fee_bps']:.1f} | ${row['total_pnl']:.2f} | "
        report += f"{row['sharpe']:.2f} | {row['win_rate']:.1%} | "
        report += f"{row['cost_signal_ratio']:.2f} |\n"
    
    # Best scenario
    best_idx = results_df['sharpe'].idxmax()
    best = results_df.loc[best_idx]
    
    report += f"""
## Optimal Fee Structure

The **{best['scenario']}** scenario maximizes Sharpe ratio:

- **Sharpe Ratio**: {best['sharpe']:.2f}
- **Total PnL**: ${best['total_pnl']:.2f}
- **Win Rate**: {best['win_rate']:.1%}
- **Cost/Signal**: {best['cost_signal_ratio']:.2f}

### Fee Details
- Maker Fee: {best['maker_fee_bps']:.1f} bps
- Taker Fee: {best['taker_fee_bps']:.1f} bps
- Maker Ratio: {best['maker_ratio']:.0%}
- Effective Rate: {best['effective_fee_rate']*10000:.1f} bps

## Sensitivity Analysis

"""
    
    # Calculate breakeven fees
    breakeven_05 = find_breakeven_fee(trades, 0.5) * 10000
    breakeven_10 = find_breakeven_fee(trades, 1.0) * 10000
    breakeven_15 = find_breakeven_fee(trades, 1.5) * 10000
    
    report += f"""### Breakeven Fee Rates

Maximum fee rates to maintain target Sharpe:

| Target Sharpe | Max Fee Rate (bps) | Annual Cost on $1M |
|--------------|-------------------|-------------------|
| 0.5 | {breakeven_05:.1f} | ${breakeven_05 * 100:.0f} |
| 1.0 | {breakeven_10:.1f} | ${breakeven_10 * 100:.0f} |
| 1.5 | {breakeven_15:.1f} | ${breakeven_15 * 100:.0f} |

## Cost Breakdown

"""
    
    # Analyze cost components
    baseline = results_df[results_df['scenario'] == 'baseline'].iloc[0]
    
    report += f"""For baseline scenario:
- **Signal PnL**: ${baseline['signal_pnl']:.2f}
- **Total Fees**: ${baseline['total_fees']:.2f}
- **Net PnL**: ${baseline['total_pnl']:.2f}

Cost allocation:
- Fees are **{baseline['cost_signal_ratio']:.1f}x** the signal PnL
- Average fee per trade: ${baseline['avg_fee_per_trade']:.2f}
- Effective fee rate: {baseline['effective_fee_rate']*10000:.1f} bps

## Recommendations

"""
    
    # Generate recommendations based on Cost/Signal ratio
    if best['cost_signal_ratio'] > 1.0:
        report += """### ⚠️ High Cost Environment

Current fees exceed signal profits. Consider:
1. **Negotiate VIP rates** - Target < 5 bps taker fees
2. **Increase maker ratio** - Use limit orders when possible
3. **Reduce trade frequency** - Focus on high-conviction signals
4. **Increase position sizes** - Amortize fixed costs
"""
    elif best['cost_signal_ratio'] > 0.5:
        report += """### ⚡ Moderate Cost Environment

Fees are significant but manageable. Optimize by:
1. **Maximize maker orders** - Target 70%+ maker ratio
2. **Use POV execution** - Reduce market impact
3. **Batch small orders** - Reduce number of transactions
"""
    else:
        report += """### ✅ Low Cost Environment

Fees are well-controlled. Maintain by:
1. **Monitor fee changes** - Set alerts for rate increases
2. **Track maker ratio** - Ensure it stays above 50%
3. **Review monthly** - Identify cost creep early
"""
    
    report += """

## Implementation Priorities

1. **Immediate Actions**
   - [ ] Review current fee tier with exchange
   - [ ] Audit maker/taker order ratio
   - [ ] Implement order type optimization

2. **Short-term (1 week)**
   - [ ] Negotiate better rates if volume > $10M/month
   - [ ] Test limit order placement strategy
   - [ ] Set up fee monitoring dashboard

3. **Long-term (1 month)**
   - [ ] Evaluate alternative venues
   - [ ] Implement smart order routing
   - [ ] Build fee-aware position sizing
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return best


def main():
    parser = argparse.ArgumentParser(description='Analyze fee scenarios')
    parser.add_argument('--trades-file', default='artifacts/wfa/trades.csv',
                       help='Path to trades CSV')
    parser.add_argument('--config', default='configs/execution.yaml',
                       help='Execution config with fee schedules')
    parser.add_argument('--output-dir', default='artifacts/cost',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("💰 Analyzing Fee Scenarios...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trades
    trades_file = Path(args.trades_file)
    
    if not trades_file.exists():
        print(f"⚠️ Trades file not found, generating sample trades...")
        # Generate sample trades
        np.random.seed(42)
        n_trades = 100
        dates = pd.date_range('2024-01-01', periods=n_trades, freq='H')
        
        trades = pd.DataFrame({
            'entry_time': dates,
            'exit_time': dates + pd.Timedelta(hours=4),
            'entry_price': 50000 * (1 + np.random.randn(n_trades) * 0.01),
            'exit_price': 50000 * (1 + np.random.randn(n_trades) * 0.01),
            'quantity': np.random.uniform(0.01, 0.1, n_trades)
        })
        
        trades['trade_size'] = trades['entry_price'] * trades['quantity']
        trades['raw_pnl'] = trades['quantity'] * (trades['exit_price'] - trades['entry_price'])
    else:
        trades = load_trades(trades_file)
    
    print(f"Loaded {len(trades)} trades")
    
    # Load fee schedules from config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    fee_scenarios = config.get('fee_schedules', {
        'baseline': {'maker': 0.0002, 'taker': 0.0010, 'maker_ratio': 0.3},
        'aggressive': {'maker': 0.0001, 'taker': 0.0008, 'maker_ratio': 0.5},
        'vip': {'maker': 0.0000, 'taker': 0.0005, 'maker_ratio': 0.7}
    })
    
    # Add additional scenarios
    fee_scenarios.update({
        'retail': {'maker': 0.0005, 'taker': 0.0015, 'maker_ratio': 0.2},
        'zero_maker': {'maker': 0.0000, 'taker': 0.0008, 'maker_ratio': 0.8},
        'flat_5bps': {'maker': 0.0005, 'taker': 0.0005, 'maker_ratio': 0.5}
    })
    
    # Analyze scenarios
    results_df = analyze_fee_scenarios(trades, fee_scenarios)
    
    # Sort by Sharpe
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    # Save results
    csv_path = output_dir / 'fee_scenarios.csv'
    results_df.to_csv(csv_path, index=False)
    
    # Generate report
    report_path = output_dir / 'FEE_SCENARIOS.md'
    best = generate_fee_report(results_df, trades, report_path)
    
    print(f"📄 Report saved to {report_path}")
    print(f"📊 Results saved to {csv_path}")
    
    # Print summary
    print(f"\n📊 Fee Analysis Summary:")
    print(f"  Best Scenario: {best['scenario']}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Cost/Signal: {best['cost_signal_ratio']:.2f}")
    print(f"  Effective Fee: {best['effective_fee_rate']*10000:.1f} bps")
    
    # Print warning if costs too high
    if best['cost_signal_ratio'] > 0.5:
        print(f"\n⚠️ WARNING: Costs are {best['cost_signal_ratio']:.1f}x signal PnL!")
        print(f"  Target: Cost/Signal ≤ 0.5")
        print(f"  Action: Negotiate better rates or reduce frequency")


if __name__ == '__main__':
    main()