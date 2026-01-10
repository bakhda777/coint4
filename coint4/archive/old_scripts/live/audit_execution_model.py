#!/usr/bin/env python3
"""Audit execution model performance."""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from coint2.execution.simulator import ExecutionSimulator, Order


def audit_execution_model(config_path: str = None) -> dict:
    """Audit execution model with synthetic orders."""
    
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'enable_fill_trace': True,
            'slippage_base': 0.0001,
            'latency_mean_ms': 10,
            'latency_std_ms': 5,
            'partial_fill_prob': 0.1
        }
    
    # Initialize simulator
    sim = ExecutionSimulator(config)
    
    # Generate test orders
    n_orders = 100
    orders = []
    market_data = {}
    
    for i in range(n_orders):
        pair = np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        
        order = Order(
            pair=pair,
            side=np.random.choice(['buy', 'sell']),
            quantity=np.random.uniform(0.1, 10),
            price=100 + np.random.randn() * 5,
            timestamp=datetime.now().timestamp() + i
        )
        orders.append(order)
        
        # Market data
        market_data[pair] = {
            'atr': np.random.uniform(0.01, 0.05),
            'zscore': np.random.uniform(-3, 3),
            'volatility': np.random.uniform(0.1, 0.5)
        }
    
    # Simulate fills
    fills = sim.simulate_batch(orders, market_data)
    
    # Get statistics
    stats = sim.get_fill_statistics()
    
    # Analyze fills
    fill_analysis = analyze_fills(sim.fills_trace)
    
    # Generate report
    report = generate_execution_audit_report(stats, fill_analysis)
    
    return {
        'statistics': stats,
        'analysis': fill_analysis,
        'report': report
    }


def analyze_fills(fills_trace: list) -> dict:
    """Analyze fill trace data."""
    
    if not fills_trace:
        return {}
    
    df = pd.DataFrame(fills_trace)
    
    analysis = {
        'latency': {
            'mean_ms': df['latency_ms'].mean(),
            'median_ms': df['latency_ms'].median(),
            'p95_ms': df['latency_ms'].quantile(0.95),
            'p99_ms': df['latency_ms'].quantile(0.99),
            'max_ms': df['latency_ms'].max()
        },
        'slippage': {
            'mean_pct': df['slippage_pct'].mean() * 100,
            'median_pct': df['slippage_pct'].median() * 100,
            'p95_pct': df['slippage_pct'].quantile(0.95) * 100,
            'max_pct': df['slippage_pct'].max() * 100
        },
        'partial_fills': {
            'rate': df['partial'].mean() * 100,
            'avg_ratio': df[df['partial']]['partial_fill_ratio'].mean() if df['partial'].any() else 1.0
        },
        'costs': {
            'total_commission': df['commission'].sum(),
            'avg_commission': df['commission'].mean(),
            'total_slippage_cost': (df['slippage_pct'] * df['filled_qty'] * df['filled_price']).sum()
        }
    }
    
    # Analyze by side
    for side in ['buy', 'sell']:
        side_df = df[df['side'] == side]
        if not side_df.empty:
            analysis[f'{side}_orders'] = {
                'count': len(side_df),
                'avg_slippage_pct': side_df['slippage_pct'].mean() * 100,
                'avg_latency_ms': side_df['latency_ms'].mean()
            }
    
    return analysis


def generate_execution_audit_report(stats: dict, analysis: dict) -> str:
    """Generate execution audit report."""
    
    report = f"""# Execution Model Audit Report

*Generated: {datetime.now().isoformat()}*

## Summary Statistics

- **Total Fills**: {stats.get('total_fills', 0)}
- **Partial Fills**: {stats.get('partial_fills', 0)} ({stats.get('partial_fill_pct', 0):.1f}%)
- **Average Latency**: {stats.get('avg_latency_ms', 0):.2f} ms
- **Average Slippage**: {stats.get('avg_slippage_pct', 0):.3f}%

## Latency Distribution

| Metric | Value (ms) |
|--------|-----------|
| Mean | {analysis.get('latency', {}).get('mean_ms', 0):.2f} |
| Median | {analysis.get('latency', {}).get('median_ms', 0):.2f} |
| P95 | {analysis.get('latency', {}).get('p95_ms', 0):.2f} |
| P99 | {analysis.get('latency', {}).get('p99_ms', 0):.2f} |
| Max | {analysis.get('latency', {}).get('max_ms', 0):.2f} |

## Slippage Distribution

| Metric | Value (%) |
|--------|----------|
| Mean | {analysis.get('slippage', {}).get('mean_pct', 0):.4f} |
| Median | {analysis.get('slippage', {}).get('median_pct', 0):.4f} |
| P95 | {analysis.get('slippage', {}).get('p95_pct', 0):.4f} |
| Max | {analysis.get('slippage', {}).get('max_pct', 0):.4f} |

## Partial Fills

- **Partial Fill Rate**: {analysis.get('partial_fills', {}).get('rate', 0):.1f}%
- **Average Fill Ratio**: {analysis.get('partial_fills', {}).get('avg_ratio', 1.0):.2%}

## Cost Analysis

- **Total Commission**: ${analysis.get('costs', {}).get('total_commission', 0):.2f}
- **Average Commission**: ${analysis.get('costs', {}).get('avg_commission', 0):.4f}
- **Total Slippage Cost**: ${analysis.get('costs', {}).get('total_slippage_cost', 0):.2f}

"""
    
    # Add side-specific analysis
    for side in ['buy', 'sell']:
        side_key = f'{side}_orders'
        if side_key in analysis:
            side_data = analysis[side_key]
            report += f"""
## {side.capitalize()} Orders

- **Count**: {side_data['count']}
- **Avg Slippage**: {side_data['avg_slippage_pct']:.4f}%
- **Avg Latency**: {side_data['avg_latency_ms']:.2f} ms
"""
    
    # Add recommendations
    if stats.get('avg_latency_ms', 0) > 50:
        report += """
## ⚠️ Recommendations

- High average latency detected (>50ms)
- Consider optimizing order routing
- Review network connectivity
"""
    
    if stats.get('partial_fill_pct', 0) > 20:
        report += """
- High partial fill rate (>20%)
- Consider adjusting order sizing
- Review liquidity assumptions
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Audit execution model')
    parser.add_argument('--config', help='Path to execution config')
    parser.add_argument('--output-dir', default='artifacts/audit',
                       help='Output directory')
    parser.add_argument('--n-orders', type=int, default=100,
                       help='Number of test orders')
    
    args = parser.parse_args()
    
    print("🔍 Auditing Execution Model...")
    
    # Run audit
    result = audit_execution_model(args.config)
    
    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'EXECUTION_AUDIT.md'
    with open(report_path, 'w') as f:
        f.write(result['report'])
    
    print(f"📄 Report saved to {report_path}")
    
    # Save statistics (convert numpy types for JSON)
    stats_path = output_dir / 'execution_stats.json'
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    with open(stats_path, 'w') as f:
        json.dump(convert_to_json_serializable({
            'statistics': result['statistics'],
            'analysis': result['analysis']
        }), f, indent=2)
    
    print(f"📊 Statistics saved to {stats_path}")
    
    # Print summary
    stats = result['statistics']
    print("\n📊 Execution Audit Summary:")
    print(f"  Total Fills: {stats.get('total_fills', 0)}")
    print(f"  Partial Fill Rate: {stats.get('partial_fill_pct', 0):.1f}%")
    print(f"  Avg Latency: {stats.get('avg_latency_ms', 0):.2f} ms")
    print(f"  Avg Slippage: {stats.get('avg_slippage_pct', 0):.3f}%")


if __name__ == '__main__':
    main()