#!/usr/bin/env python3
"""
Quick systematic validation test focused on key metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import subprocess
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stability_quick():
    """Quick stability test across 3 periods."""
    print("\n" + "="*60)
    print("📊 QUICK STABILITY TEST")
    print("="*60)
    
    periods = [
        ("Jan 2024", "2024-01-01", "2024-01-31"),
        ("Feb 2024", "2024-02-01", "2024-02-28"),
        ("Mar 2024", "2024-03-01", "2024-03-31"),
    ]
    
    all_pairs = {}
    
    for name, start, end in periods:
        print(f"\n🔍 Testing {name}: {start} to {end}")
        
        # Run pair selection
        out_dir = f"artifacts/validation/quick/{name.replace(' ', '_')}"
        cmd = [
            "python", "scripts/universe/select_pairs.py",
            "--period-start", start,
            "--period-end", end,
            "--criteria-config", "configs/criteria_relaxed.yaml",
            "--out-dir", out_dir,
            "--top-n", "50",
            "--log-every", "100000"
        ]
        
        print(f"  Running: {' '.join(cmd[-6:])}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check if pairs were found
        pairs_file = Path(out_dir) / "pairs_universe.yaml"
        if pairs_file.exists():
            with open(pairs_file) as f:
                data = yaml.safe_load(f)
                pairs = [p['pair'] for p in data.get('pairs', [])]
                
                print(f"  ✓ Found {len(pairs)} pairs")
                if pairs:
                    print(f"    Top 3: {', '.join(pairs[:3])}")
                
                for pair in pairs:
                    if pair not in all_pairs:
                        all_pairs[pair] = []
                    all_pairs[pair].append(name)
        else:
            print(f"  ✗ No pairs found")
    
    # Analyze stability
    print("\n" + "="*60)
    print("📈 STABILITY ANALYSIS")
    print("="*60)
    
    # Find pairs that appear in all periods
    stable_pairs = [p for p, periods in all_pairs.items() if len(periods) == 3]
    semi_stable = [p for p, periods in all_pairs.items() if len(periods) == 2]
    
    print(f"\n✅ Stable pairs (appear in all 3 periods): {len(stable_pairs)}")
    if stable_pairs:
        print("  Top stable pairs:")
        for pair in stable_pairs[:5]:
            print(f"    - {pair}")
    
    print(f"\n⚠️ Semi-stable pairs (appear in 2/3 periods): {len(semi_stable)}")
    if semi_stable:
        print("  Examples:")
        for pair in semi_stable[:3]:
            print(f"    - {pair}: {', '.join(all_pairs[pair])}")
    
    return stable_pairs, all_pairs


def test_arbitrage_pairs():
    """Test known arbitrage pairs specifically."""
    print("\n" + "="*60)
    print("💱 ARBITRAGE PAIRS TEST")
    print("="*60)
    
    arbitrage_pairs = [
        "ETHUSDC/ETHUSDT",
        "BTCUSDC/BTCUSDT",
        "TUSDUSDT/USDCUSDT",
        "USDCUSDT/USDTUSDT",
    ]
    
    print("\nChecking if arbitrage pairs appear in Q1 2024 results...")
    
    # Load Q1 results
    q1_file = Path("artifacts/universe/Q1_2024_no_hurst/pairs_universe.yaml")
    if q1_file.exists():
        with open(q1_file) as f:
            data = yaml.safe_load(f)
            found_pairs = [p['pair'] for p in data.get('pairs', [])]
            
        for arb_pair in arbitrage_pairs:
            if arb_pair in found_pairs:
                idx = found_pairs.index(arb_pair)
                print(f"  ✅ {arb_pair:25s} - Rank #{idx+1}")
            else:
                print(f"  ❌ {arb_pair:25s} - Not found")
    else:
        print("  ❌ Q1 2024 results not found")


def test_out_of_sample():
    """Quick out-of-sample test."""
    print("\n" + "="*60)
    print("🎯 OUT-OF-SAMPLE TEST")
    print("="*60)
    
    print("\nTesting April 2024 (out-of-sample)...")
    
    # Run selection on April
    out_dir = "artifacts/validation/quick/April_2024"
    cmd = [
        "python", "scripts/universe/select_pairs.py",
        "--period-start", "2024-04-01",
        "--period-end", "2024-04-30",
        "--criteria-config", "configs/criteria_relaxed.yaml",
        "--out-dir", out_dir,
        "--top-n", "100",
        "--log-every", "100000"
    ]
    
    print(f"  Running selection for April 2024...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    # Compare with Q1 results
    q1_file = Path("artifacts/universe/Q1_2024_no_hurst/pairs_universe.yaml")
    april_file = Path(out_dir) / "pairs_universe.yaml"
    
    if q1_file.exists() and april_file.exists():
        with open(q1_file) as f:
            q1_data = yaml.safe_load(f)
            q1_pairs = [p['pair'] for p in q1_data.get('pairs', [])][:50]
        
        with open(april_file) as f:
            april_data = yaml.safe_load(f)
            april_pairs = [p['pair'] for p in april_data.get('pairs', [])][:50]
        
        # Calculate overlap
        overlap = set(q1_pairs) & set(april_pairs)
        
        print(f"\n📊 Results:")
        print(f"  Q1 2024 top 50: {len(q1_pairs)} pairs")
        print(f"  April 2024 top 50: {len(april_pairs)} pairs")
        print(f"  Overlap: {len(overlap)} pairs ({len(overlap)/50*100:.1f}%)")
        
        if overlap:
            print(f"\n  Consistent pairs (examples):")
            for pair in list(overlap)[:5]:
                q1_rank = q1_pairs.index(pair) + 1
                april_rank = april_pairs.index(pair) + 1
                print(f"    - {pair:25s} Q1 rank: #{q1_rank:2d}, April rank: #{april_rank:2d}")
    else:
        print("  ❌ Could not compare results")


def generate_summary():
    """Generate summary report."""
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": ["stability", "arbitrage", "out_of_sample"],
        "findings": []
    }
    
    # Check if we have rejection breakdown
    breakdown_file = Path("artifacts/universe/Q1_2024_no_hurst/REJECTION_BREAKDOWN.yaml")
    if breakdown_file.exists():
        with open(breakdown_file) as f:
            breakdown = yaml.safe_load(f)
            
        print("\n📉 Rejection Analysis (Q1 2024):")
        print(f"  Total pairs tested: {breakdown.get('total_tested', 'N/A')}")
        print(f"  Pairs passed: {breakdown.get('total_passed', 'N/A')}")
        
        if 'rejection_reasons' in breakdown:
            print("\n  Top rejection reasons:")
            for reason, count in list(breakdown['rejection_reasons'].items())[:5]:
                pct = count / breakdown.get('total_tested', 1) * 100
                print(f"    - {reason}: {count} ({pct:.1f}%)")
        
        results["rejection_breakdown"] = breakdown
    
    # Save summary
    summary_file = Path("artifacts/validation/quick_validation_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Summary saved to {summary_file}")
    
    print("\n" + "="*60)
    print("✅ QUICK VALIDATION COMPLETE")
    print("="*60)
    
    print("\n🎯 KEY FINDINGS:")
    print("1. Stability: Check if top pairs remain consistent across months")
    print("2. Arbitrage: Verify known arbitrage pairs are captured")
    print("3. Out-of-sample: Compare Q1 vs April results for consistency")
    print("\n📌 Next Steps:")
    print("1. If stability > 30%, proceed with optimization")
    print("2. If arbitrage pairs missing, review criteria")
    print("3. If out-of-sample overlap < 20%, consider shorter windows")


def main():
    print("🚀 STARTING QUICK SYSTEMATIC VALIDATION")
    print("="*60)
    
    # 1. Test stability
    stable_pairs, all_pairs = test_stability_quick()
    
    # 2. Test arbitrage pairs
    test_arbitrage_pairs()
    
    # 3. Test out-of-sample
    test_out_of_sample()
    
    # 4. Generate summary
    generate_summary()


if __name__ == '__main__':
    main()