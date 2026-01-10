#!/usr/bin/env python3
"""Aggregate stable pairs across multiple universe windows."""

import json
import yaml
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np


def load_metrics_files(input_pattern):
    """Load all universe metrics CSV files."""
    files = list(Path(".").glob(input_pattern))
    
    if not files:
        raise FileNotFoundError(f"No metrics files found matching: {input_pattern}")
    
    print(f"📂 Found {len(files)} metrics files")
    
    all_metrics = []
    window_info = {}
    
    for f in files:
        # Extract window label from path
        parts = f.parts
        window_label = None
        for part in parts:
            if part.startswith('win'):
                window_label = part
                break
        
        if not window_label:
            window_label = f.parent.name
        
        try:
            df = pd.read_csv(f)
            df['window'] = window_label
            df['source_file'] = str(f)
            all_metrics.append(df)
            
            # Track window info
            window_info[window_label] = {
                'file': str(f),
                'pairs_count': len(df),
                'passed_count': len(df[df.get('verdict', '') == 'PASS']) if 'verdict' in df.columns else 0
            }
            
            print(f"  ✓ {window_label}: {len(df)} pairs, {window_info[window_label]['passed_count']} passed")
            
        except Exception as e:
            print(f"  ⚠️ Error loading {f}: {e}")
    
    if not all_metrics:
        raise ValueError("No valid metrics files loaded")
    
    combined = pd.concat(all_metrics, ignore_index=True)
    
    return combined, window_info


def aggregate_stable_pairs(df, min_windows, top_n, diversify_by_base, max_per_base):
    """Aggregate pairs that pass in multiple windows."""
    
    # Filter only passed pairs
    if 'verdict' in df.columns:
        df_passed = df[df['verdict'] == 'PASS'].copy()
    else:
        # If no verdict column, assume all are candidates
        df_passed = df.copy()
    
    if df_passed.empty:
        print("⚠️ No pairs passed in any window!")
        return pd.DataFrame()
    
    # Group by pair
    pair_stats = []
    
    for pair in df_passed['pair'].unique():
        pair_data = df_passed[df_passed['pair'] == pair]
        
        windows_passed = len(pair_data['window'].unique())
        
        # Calculate aggregate metrics
        stats = {
            'pair': pair,
            'windows_passed': windows_passed,
            'windows_list': sorted(pair_data['window'].unique().tolist()),
            'median_score': pair_data['score'].median() if 'score' in pair_data.columns else 0,
            'mean_score': pair_data['score'].mean() if 'score' in pair_data.columns else 0,
            'std_score': pair_data['score'].std() if 'score' in pair_data.columns else 0,
            'median_pvalue': pair_data['coint_pvalue'].median() if 'coint_pvalue' in pair_data.columns else 0,
            'median_half_life': pair_data['half_life'].median() if 'half_life' in pair_data.columns else 0,
            'median_hurst': pair_data['hurst'].median() if 'hurst' in pair_data.columns else 0,
            'median_crossings': pair_data['mean_crossings'].median() if 'mean_crossings' in pair_data.columns else 0
        }
        
        # Extract base asset (first symbol)
        if '/' in pair:
            stats['base_asset'] = pair.split('/')[0]
        else:
            stats['base_asset'] = pair.split('_')[0] if '_' in pair else pair
        
        pair_stats.append(stats)
    
    df_agg = pd.DataFrame(pair_stats)
    
    # Filter by minimum windows
    df_filtered = df_agg[df_agg['windows_passed'] >= min_windows].copy()
    
    if df_filtered.empty:
        print(f"⚠️ No pairs passed in >= {min_windows} windows!")
        # Relax to any pairs that passed
        df_filtered = df_agg[df_agg['windows_passed'] > 0].copy()
    
    print(f"📊 {len(df_filtered)} pairs passed in >= {min_windows} windows")
    
    # Sort by score and stability
    df_filtered['stability_score'] = df_filtered['windows_passed'] / df_filtered['windows_passed'].max()
    df_filtered['combined_score'] = (
        0.6 * df_filtered['median_score'] / (df_filtered['median_score'].max() + 1e-6) +
        0.4 * df_filtered['stability_score']
    )
    
    df_sorted = df_filtered.sort_values(
        ['combined_score', 'median_pvalue'],
        ascending=[False, True]
    )
    
    # Apply diversification
    if diversify_by_base and max_per_base > 0:
        selected = []
        base_counts = defaultdict(int)
        
        for _, row in df_sorted.iterrows():
            base = row['base_asset']
            if base_counts[base] < max_per_base:
                selected.append(row)
                base_counts[base] += 1
                
                if len(selected) >= top_n:
                    break
        
        df_final = pd.DataFrame(selected)
        print(f"📊 After diversification: {len(df_final)} pairs")
    else:
        df_final = df_sorted.head(top_n)
    
    return df_final


def normalize_weights(df, method='equal'):
    """Normalize pair weights."""
    if method == 'equal':
        weights = [1.0 / len(df)] * len(df)
    elif method == 'score_proportional':
        scores = df['combined_score'].values
        weights = scores / scores.sum()
    else:
        weights = [1.0] * len(df)
    
    # Normalize to sum to 1.0
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return weights.tolist()


def save_results(df_pairs, out_yaml, out_csv, window_info, args):
    """Save aggregated pairs to files."""
    
    # Prepare pairs list for YAML
    weights = normalize_weights(df_pairs, method='equal')
    
    pairs_list = []
    for i, (_, row) in enumerate(df_pairs.iterrows()):
        pairs_list.append({
            'pair': row['pair'],
            'weight': float(weights[i]),
            'windows_passed': int(row['windows_passed']),
            'median_score': float(row['median_score'])
        })
    
    # Create YAML structure
    yaml_data = {
        'metadata': {
            'built_from_windows': list(window_info.keys()),
            'min_windows_passed': args.min_windows,
            'generated': datetime.utcnow().isoformat(),
            'criteria_profiles': 'mixed',  # Will be determined from window names
            'aggregation_params': {
                'top_n': args.top_n,
                'diversify_by_base': args.diversify_by_base,
                'max_per_base': args.max_per_base
            },
            'total_windows_analyzed': len(window_info),
            'total_pairs_selected': len(pairs_list)
        },
        'pairs': pairs_list
    }
    
    # Save YAML
    with open(out_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Saved pairs to {out_yaml}")
    
    # Save CSV with detailed metrics
    df_pairs.to_csv(out_csv, index=False)
    print(f"✅ Saved metrics to {out_csv}")
    
    # Generate report
    report_path = Path("artifacts/universe/AGGREGATED_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(f"""# Aggregated Universe Report

## Summary
- **Total Windows Analyzed**: {len(window_info)}
- **Minimum Windows Required**: {args.min_windows}
- **Pairs Selected**: {len(pairs_list)}
- **Diversification**: {'Enabled' if args.diversify_by_base else 'Disabled'}

## Window Statistics
{chr(10).join(f"- {w}: {info['pairs_count']} total, {info['passed_count']} passed" 
              for w, info in window_info.items())}

## Top 10 Stable Pairs
{chr(10).join(f"{i+1}. **{row['pair']}**: {row['windows_passed']} windows, score={row['median_score']:.3f}"
              for i, (_, row) in enumerate(df_pairs.head(10).iterrows()))}

## Filtering Analysis
- Pairs in 4+ windows: {len(df_pairs[df_pairs['windows_passed'] >= 4])}
- Pairs in 3+ windows: {len(df_pairs[df_pairs['windows_passed'] >= 3])}
- Pairs in 2+ windows: {len(df_pairs[df_pairs['windows_passed'] >= 2])}
- Pairs in 1 window: {len(df_pairs[df_pairs['windows_passed'] == 1])}

## Base Asset Distribution
{pd.DataFrame(df_pairs['base_asset'].value_counts()).to_markdown() if len(df_pairs) > 0 else 'No pairs selected'}

---
Generated: {datetime.utcnow().isoformat()}
""")
    print(f"✅ Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate stable pairs')
    parser.add_argument('--inputs', 
                       default='artifacts/universe/*/universe_metrics.csv',
                       help='Input metrics files pattern')
    parser.add_argument('--out-yaml', 
                       default='benchmarks/pairs_universe.yaml',
                       help='Output YAML file')
    parser.add_argument('--out-csv',
                       default='artifacts/universe/AGGREGATED_PAIRS.csv',
                       help='Output CSV file')
    parser.add_argument('--min-windows', type=int, default=2,
                       help='Minimum windows for pair to pass')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Top N pairs to select')
    parser.add_argument('--diversify-by-base', 
                       type=lambda x: x.lower() == 'true',
                       default=True,
                       help='Diversify by base asset')
    parser.add_argument('--max-per-base', type=int, default=5,
                       help='Max pairs per base asset')
    
    args = parser.parse_args()
    
    print(f"🔄 Aggregating stable pairs across windows...")
    print(f"📊 Parameters: min_windows={args.min_windows}, top_n={args.top_n}")
    
    try:
        # Load all metrics
        df_combined, window_info = load_metrics_files(args.inputs)
        
        # Aggregate stable pairs
        df_pairs = aggregate_stable_pairs(
            df_combined,
            args.min_windows,
            args.top_n,
            args.diversify_by_base,
            args.max_per_base
        )
        
        if df_pairs.empty:
            print("❌ No pairs to save!")
            return 1
        
        # Save results
        save_results(df_pairs, args.out_yaml, args.out_csv, window_info, args)
        
        print(f"\n✅ Aggregation complete!")
        print(f"📊 Selected {len(df_pairs)} stable pairs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())