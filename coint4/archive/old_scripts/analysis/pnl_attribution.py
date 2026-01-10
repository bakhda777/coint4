#!/usr/bin/env python3
"""Run PnL attribution analysis."""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from coint2.analytics.pnl_attribution import (
    calculate_pnl_attribution,
    generate_pnl_attribution_report,
    analyze_pnl_by_period
)


def load_trades_from_artifacts() -> pd.DataFrame:
    """Load trades from WFA or paper trading artifacts."""
    
    # Try WFA trades first
    wfa_trades_path = Path('artifacts/wfa/trades.csv')
    if wfa_trades_path.exists():
        print(f"Loading trades from {wfa_trades_path}")
        return pd.read_csv(wfa_trades_path)
    
    # Try paper trading trades
    paper_trades_path = Path('artifacts/live/paper_trades.csv')
    if paper_trades_path.exists():
        print(f"Loading trades from {paper_trades_path}")
        return pd.read_csv(paper_trades_path)
    
    # Try to find any trades file
    for trades_file in Path('artifacts').rglob('*trades*.csv'):
        print(f"Loading trades from {trades_file}")
        return pd.read_csv(trades_file)
    
    print("No trades file found, generating sample trades")
    # Generate sample trades for demo
    return generate_sample_trades()


def generate_sample_trades() -> pd.DataFrame:
    """Generate sample trades for demonstration."""
    import numpy as np
    
    np.random.seed(42)
    n_trades = 50
    
    trades = []
    for i in range(n_trades):
        entry_price = 100 + np.random.randn() * 5
        exit_price = entry_price + np.random.randn() * 2
        
        trades.append({
            'pair': f'BTC/ETH',
            'entry_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i*2),
            'exit_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i*2+1),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': 1.0,
            'side': np.random.choice([1, -1]),
            'pnl': (exit_price - entry_price) * np.random.choice([1, -1])
        })
    
    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description='Run PnL attribution analysis')
    parser.add_argument('--trades-file', help='Path to trades CSV file')
    parser.add_argument('--output-dir', default='artifacts/audit', 
                       help='Output directory for reports')
    parser.add_argument('--commission-pct', type=float, default=0.001,
                       help='Commission percentage')
    parser.add_argument('--slippage-pct', type=float, default=0.0005,
                       help='Slippage percentage')
    
    args = parser.parse_args()
    
    print("🔍 Running PnL Attribution Analysis...")
    
    # Load trades
    if args.trades_file:
        trades = pd.read_csv(args.trades_file)
    else:
        trades = load_trades_from_artifacts()
    
    print(f"📊 Loaded {len(trades)} trades")
    
    # Load prices (dummy for now)
    prices = pd.DataFrame()
    
    # Calculate attribution
    attribution = calculate_pnl_attribution(
        trades=trades,
        prices=prices,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct
    )
    
    # Generate report
    output_path = Path(args.output_dir) / 'PNL_ATTRIBUTION.md'
    report = generate_pnl_attribution_report(attribution, str(output_path))
    
    print(f"📄 Report saved to {output_path}")
    
    # Save attribution data as CSV
    csv_path = Path(args.output_dir) / 'pnl_attribution.csv'
    pd.DataFrame([attribution]).to_csv(csv_path, index=False)
    print(f"📊 Attribution data saved to {csv_path}")
    
    # Analyze by period
    period_analysis = analyze_pnl_by_period(trades, 'W')
    if not period_analysis.empty:
        period_path = Path(args.output_dir) / 'pnl_by_period.csv'
        period_analysis.to_csv(period_path, index=False)
        print(f"📅 Period analysis saved to {period_path}")
    
    # Update artifact registry
    update_artifact_registry(output_path, csv_path)
    
    # Print summary
    print("\n📊 PnL Attribution Summary:")
    print(f"  Total PnL: ${attribution['total_pnl']:,.2f}")
    print(f"  Signal PnL: ${attribution['signal_pnl']:,.2f}")
    print(f"  Commission Cost: ${attribution['commission_cost']:,.2f}")
    print(f"  Slippage Cost: ${attribution['slippage_cost']:,.2f}")
    print(f"  Cost Ratio: {attribution['cost_ratio']:.1%}")
    
    return attribution


def update_artifact_registry(report_path: Path, csv_path: Path):
    """Update artifact registry with attribution files."""
    
    registry_path = Path('artifacts/ARTIFACT_INDEX.json')
    
    # Load or create registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {'artifacts': []}
    
    # Add entries
    timestamp = datetime.now().isoformat()
    
    for path in [report_path, csv_path]:
        entry = {
            'type': 'audit',
            'category': 'pnl_attribution',
            'path': str(path),
            'filename': path.name,
            'created': timestamp
        }
        
        # Remove old entry if exists
        registry['artifacts'] = [
            a for a in registry.get('artifacts', [])
            if a.get('path') != str(path)
        ]
        registry['artifacts'].append(entry)
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


if __name__ == '__main__':
    main()