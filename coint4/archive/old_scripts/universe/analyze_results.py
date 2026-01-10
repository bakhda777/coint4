#!/usr/bin/env python3
"""Quick analysis of universe selection results."""

import argparse
import yaml
import pandas as pd
from pathlib import Path

def analyze_results(out_dir: str):
    """Analyze universe selection results."""
    out_path = Path(out_dir)
    
    # Load rejection breakdown
    breakdown_path = out_path / 'REJECTION_BREAKDOWN.yaml'
    if breakdown_path.exists():
        with open(breakdown_path) as f:
            breakdown = yaml.safe_load(f)
        
        print("📊 REJECTION BREAKDOWN")
        print(f"Tested: {breakdown['tested_pairs']}")
        print(f"Passed: {breakdown['passed_pairs']} ({breakdown['passed_pairs']/breakdown['tested_pairs']*100:.1f}%)")
        print("\nTop rejection reasons:")
        for reason, count in sorted(breakdown['reasons'].items(), key=lambda x: x[1], reverse=True):
            pct = count/breakdown['tested_pairs']*100
            print(f"  {reason:12s}: {count:5d} ({pct:5.1f}%)")
    
    # Load pairs
    pairs_path = out_path / 'pairs_universe.yaml'
    if pairs_path.exists():
        with open(pairs_path) as f:
            data = yaml.safe_load(f)
        
        print(f"\n📈 SELECTED PAIRS: {len(data['pairs'])}")
        
        # Analyze base symbols
        base_counts = {}
        for pair_data in data['pairs']:
            base = pair_data['symbol1'].replace('USDT', '').replace('USDC', '')
            base_counts[base] = base_counts.get(base, 0) + 1
        
        print("\nMost common base symbols:")
        for base, count in sorted(base_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {base:10s}: {count} pairs")
        
        # Show top pairs
        print("\nTop 10 pairs by score:")
        for i, pair_data in enumerate(data['pairs'][:10], 1):
            print(f"  {i:2d}. {pair_data['pair']:20s} score={pair_data['score']:.3f} pval={pair_data['metrics']['pvalue']:.4f}")
    
    # Load metrics
    metrics_path = out_path / 'universe_metrics.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        passed = df[df['verdict'] == 'PASS']
        
        print(f"\n📊 METRICS SUMMARY (from {len(passed)} passed pairs)")
        print("\nScore distribution:")
        print(f"  Min:    {passed['score'].min():.3f}")
        print(f"  Q1:     {passed['score'].quantile(0.25):.3f}")
        print(f"  Median: {passed['score'].median():.3f}")
        print(f"  Q3:     {passed['score'].quantile(0.75):.3f}")
        print(f"  Max:    {passed['score'].max():.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze universe selection results')
    parser.add_argument('out_dir', help='Output directory from selection run')
    args = parser.parse_args()
    
    analyze_results(args.out_dir)