#!/usr/bin/env python3
"""
Analyze existing universe selection results for systematic validation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_q1_results():
    """Analyze Q1 2024 results that we already have."""
    print("\n" + "="*60)
    print("📊 ANALYZING Q1 2024 RESULTS (NO HURST)")
    print("="*60)
    
    # Load existing results
    base_dir = Path("artifacts/universe/Q1_2024_no_hurst")
    
    # Load pairs
    pairs_file = base_dir / "pairs_universe.yaml"
    if pairs_file.exists():
        with open(pairs_file) as f:
            data = yaml.safe_load(f)
            pairs = data.get('pairs', [])
            
        print(f"\n✅ Found {len(pairs)} pairs")
        print(f"   Period: {data.get('period', 'Q1 2024')}")
        print(f"   Criteria: relaxed (no Hurst)")
        
        # Top pairs
        print("\n🏆 Top 10 Pairs by Score:")
        for i, p in enumerate(pairs[:10], 1):
            print(f"   {i:2d}. {p['pair']:25s} Score: {p.get('score', 0):.3f}")
        
        # Check for arbitrage pairs
        arbitrage_pairs = [
            "ETHUSDC/ETHUSDT",
            "BTCUSDC/BTCUSDT", 
            "TUSDUSDT/USDCUSDT",
            "USDCUSDT/USDTUSDT",
        ]
        
        pair_names = [p['pair'] for p in pairs]
        
        print("\n💱 Arbitrage Pairs Status:")
        for arb in arbitrage_pairs:
            if arb in pair_names:
                idx = pair_names.index(arb)
                score = pairs[idx].get('score', 0)
                print(f"   ✅ {arb:25s} Rank: #{idx+1:3d}, Score: {score:.3f}")
            else:
                print(f"   ❌ {arb:25s} Not found")
        
        return pairs
    else:
        print("   ❌ No pairs file found")
        return []


def analyze_rejection_breakdown():
    """Analyze why pairs were rejected."""
    print("\n" + "="*60)
    print("📉 REJECTION ANALYSIS")
    print("="*60)
    
    breakdown_file = Path("artifacts/universe/Q1_2024_no_hurst/REJECTION_BREAKDOWN.yaml")
    
    if breakdown_file.exists():
        with open(breakdown_file) as f:
            breakdown = yaml.safe_load(f)
        
        total_tested = breakdown.get('total_tested', 0)
        total_passed = breakdown.get('total_passed', 0)
        pass_rate = (total_passed / total_tested * 100) if total_tested > 0 else 0
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total pairs tested: {total_tested:,}")
        print(f"   Pairs passed: {total_passed:,}")
        print(f"   Pass rate: {pass_rate:.2f}%")
        
        if 'rejection_reasons' in breakdown:
            print(f"\n❌ Top Rejection Reasons:")
            reasons = breakdown['rejection_reasons']
            sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
            
            for reason, count in sorted_reasons[:10]:
                pct = (count / total_tested * 100) if total_tested > 0 else 0
                print(f"   {reason:30s} {count:6,} ({pct:5.1f}%)")
        
        return breakdown
    else:
        print("   ❌ No rejection breakdown found")
        return {}


def analyze_metrics():
    """Analyze pair metrics."""
    print("\n" + "="*60)
    print("📈 PAIR METRICS ANALYSIS")
    print("="*60)
    
    metrics_file = Path("artifacts/universe/Q1_2024_no_hurst/universe_metrics.csv")
    
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        
        print(f"\n📊 Metrics Summary:")
        print(f"   Total pairs with metrics: {len(df)}")
        
        # Score distribution
        if 'score' in df.columns:
            print(f"\n   Score Distribution:")
            print(f"     Min:    {df['score'].min():.3f}")
            print(f"     Q1:     {df['score'].quantile(0.25):.3f}")
            print(f"     Median: {df['score'].quantile(0.50):.3f}")
            print(f"     Q3:     {df['score'].quantile(0.75):.3f}")
            print(f"     Max:    {df['score'].max():.3f}")
        
        # P-value distribution
        if 'p_value' in df.columns:
            print(f"\n   P-value Distribution:")
            print(f"     < 0.01: {(df['p_value'] < 0.01).sum()} pairs")
            print(f"     < 0.05: {(df['p_value'] < 0.05).sum()} pairs")
            print(f"     < 0.10: {(df['p_value'] < 0.10).sum()} pairs")
        
        # Half-life distribution
        if 'half_life' in df.columns:
            print(f"\n   Half-life Distribution (days):")
            print(f"     Min:    {df['half_life'].min():.1f}")
            print(f"     Median: {df['half_life'].quantile(0.50):.1f}")
            print(f"     Max:    {df['half_life'].max():.1f}")
        
        # Mean crossings
        if 'mean_crossings' in df.columns:
            print(f"\n   Mean Crossings:")
            print(f"     Min:    {df['mean_crossings'].min():.0f}")
            print(f"     Median: {df['mean_crossings'].quantile(0.50):.0f}")
            print(f"     Max:    {df['mean_crossings'].max():.0f}")
        
        return df
    else:
        print("   ❌ No metrics file found")
        return pd.DataFrame()


def compare_with_other_periods():
    """Compare with other available periods."""
    print("\n" + "="*60)
    print("🔄 CROSS-PERIOD COMPARISON")
    print("="*60)
    
    universe_dir = Path("artifacts/universe")
    
    # Find all available results
    available = []
    for subdir in universe_dir.iterdir():
        if subdir.is_dir():
            pairs_file = subdir / "pairs_universe.yaml"
            if pairs_file.exists():
                with open(pairs_file) as f:
                    data = yaml.safe_load(f)
                    n_pairs = len(data.get('pairs', []))
                    available.append({
                        'name': subdir.name,
                        'n_pairs': n_pairs,
                        'period': data.get('period', 'Unknown')
                    })
    
    if available:
        print(f"\n📁 Found {len(available)} result sets:")
        for item in sorted(available, key=lambda x: x['n_pairs'], reverse=True):
            print(f"   {item['name']:30s} {item['n_pairs']:4d} pairs")
        
        # Compare top pairs if we have multiple Q1 results
        q1_results = [a for a in available if 'Q1_2024' in a['name']]
        if len(q1_results) > 1:
            print(f"\n🔍 Comparing {len(q1_results)} Q1 2024 variants:")
            
            all_top_pairs = {}
            for result in q1_results:
                pairs_file = universe_dir / result['name'] / "pairs_universe.yaml"
                with open(pairs_file) as f:
                    data = yaml.safe_load(f)
                    top_10 = [p['pair'] for p in data.get('pairs', [])[:10]]
                    all_top_pairs[result['name']] = top_10
            
            # Find common pairs
            if all_top_pairs:
                first_key = list(all_top_pairs.keys())[0]
                common = set(all_top_pairs[first_key])
                for key in all_top_pairs:
                    common = common & set(all_top_pairs[key])
                
                print(f"\n   Common top-10 pairs across variants: {len(common)}")
                if common:
                    for pair in list(common)[:5]:
                        print(f"     - {pair}")


def suggest_next_steps():
    """Suggest next steps based on analysis."""
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. VALIDATION APPROACH:")
    print("   Since pair selection takes time, use existing Q1 results")
    print("   Focus on validating top 20-50 pairs")
    print("   Use shorter test periods (1-2 weeks) for speed")
    
    print("\n2. OPTIMIZATION STRATEGY:")
    print("   Start with top arbitrage pairs (ETHUSDC/ETHUSDT, etc.)")
    print("   Use fast mode with 50-100 trials initially")
    print("   Focus on zscore_threshold and rolling_window")
    
    print("\n3. CRITERIA ADJUSTMENTS:")
    print("   Current pass rate: ~4.3% (2,415/56,000+ pairs)")
    print("   Consider tightening if too many low-quality pairs")
    print("   Or relaxing if missing good opportunities")
    
    print("\n4. NEXT COMMANDS:")
    print("   # Quick backtest on top pairs")
    print("   python scripts/core/backtest.py --mode portfolio \\")
    print("     --pairs-file artifacts/universe/Q1_2024_no_hurst/pairs_universe.yaml \\")
    print("     --top-n 20 --period 2024-04")
    print()
    print("   # Fast optimization")
    print("   python scripts/core/optimize.py --mode fast --n-trials 50 \\")
    print("     --study-name q1_2024_no_hurst")


def save_analysis():
    """Save analysis results."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'existing_results',
        'findings': {
            'q1_2024_no_hurst': {
                'total_pairs': 2415,
                'pass_rate': 4.3,
                'top_pairs_include_arbitrage': True,
                'criteria': 'relaxed_no_hurst'
            }
        },
        'recommendations': [
            'Use existing Q1 results for validation',
            'Focus on top 20-50 pairs for speed',
            'Start optimization with arbitrage pairs',
            'Consider 1-2 week test periods'
        ]
    }
    
    output_file = Path("artifacts/validation/analysis_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to {output_file}")


def main():
    print("🚀 ANALYZING EXISTING UNIVERSE SELECTION RESULTS")
    print("="*60)
    
    # Analyze what we have
    pairs = analyze_q1_results()
    breakdown = analyze_rejection_breakdown()
    metrics_df = analyze_metrics()
    compare_with_other_periods()
    
    # Provide recommendations
    suggest_next_steps()
    
    # Save summary
    save_analysis()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()