#!/usr/bin/env python3
"""
Build universe of cointegrated pairs from available data.
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from coint2.core.data_loader import DataHandler
    from coint2.utils.config import load_config
    from coint2.pipeline.pair_scanner import scan_universe, calculate_pair_score
except ImportError as e:
    import sys
    # Only exit if running as main script, not during import
    if __name__ == '__main__':
        print(f"❌ Import error: {e}")
        print("   Please install the package: pip install -e . from repository root")
        sys.exit(1)
    else:
        # Re-raise during import for proper error handling
        raise


def load_universe_config(config_path: str) -> dict:
    """Load universe configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def build_universe(config_path: str = 'configs/universe.yaml'):
    """Build universe of pairs based on configuration."""
    
    print("🔍 Building universe of cointegrated pairs...")
    
    # Load configurations
    universe_cfg = load_universe_config(config_path)
    app_cfg = load_config('configs/main_2024.yaml')
    
    # Initialize data handler
    handler = DataHandler(app_cfg)
    
    # Set dates
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(
        days=universe_cfg['universe']['train_days'] + 
             universe_cfg['universe']['valid_days']
    )
    
    print(f"📅 Analysis period: {start_date.date()} to {end_date.date()}")
    
    # Scan universe
    print("🔬 Testing cointegration for all pairs...")
    df_results = scan_universe(
        handler,
        universe_cfg['universe']['symbols'],
        start_date,
        end_date,
        universe_cfg
    )
    
    if df_results.empty:
        print("❌ No pairs found in universe")
        return
    
    print(f"📊 Tested {len(df_results)} pairs")
    
    # Calculate scores
    df_results['score'] = df_results.apply(
        lambda row: calculate_pair_score(row.to_dict(), universe_cfg),
        axis=1
    )
    
    # Sort by score
    df_results = df_results.sort_values('score', ascending=False)
    
    # Filter passing pairs
    passing = df_results[df_results['verdict'] == 'PASS']
    print(f"✅ {len(passing)} pairs passed criteria")
    
    # Apply selection rules
    selected = apply_selection_rules(passing, universe_cfg['selection'])
    print(f"🎯 Selected {len(selected)} pairs for universe")
    
    # Save outputs
    save_outputs(selected, df_results, universe_cfg, config_path)
    
    print("✨ Universe building complete!")
    return selected


def apply_selection_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply selection rules to filter pairs."""
    
    # Start with all passing pairs
    selected = df.copy()
    
    # Apply top N
    if 'top_n' in rules:
        selected = selected.head(rules['top_n'])
    
    # Apply diversification
    if rules.get('diversify_by_base', False):
        max_per_base = rules.get('max_per_base', 5)
        
        # Count pairs per base asset
        base_counts = {}
        final_selection = []
        
        for _, row in selected.iterrows():
            # Extract base assets (first part of symbol)
            base1 = row['symbol1'].split('USDT')[0] if 'USDT' in row['symbol1'] else row['symbol1']
            base2 = row['symbol2'].split('USDT')[0] if 'USDT' in row['symbol2'] else row['symbol2']
            
            # Check if we can add this pair
            count1 = base_counts.get(base1, 0)
            count2 = base_counts.get(base2, 0)
            
            if count1 < max_per_base and count2 < max_per_base:
                final_selection.append(row)
                base_counts[base1] = count1 + 1
                base_counts[base2] = count2 + 1
        
        selected = pd.DataFrame(final_selection)
    
    return selected


def save_outputs(selected: pd.DataFrame, all_results: pd.DataFrame, 
                 config: dict, config_path: str):
    """Save all output files."""
    
    # Create output directories
    Path('bench').mkdir(exist_ok=True)
    Path('artifacts/universe').mkdir(parents=True, exist_ok=True)
    
    # 1. Save pairs file
    pairs_file = config['output']['pairs_file']
    pairs_data = {
        'pairs': [
            {
                'symbol1': row['symbol1'],
                'symbol2': row['symbol2'],
                'score': float(row['score']),
                'pvalue': float(row['pvalue']),
                'half_life': float(row['half_life']),
                'hurst': float(row['hurst'])
            }
            for _, row in selected.iterrows()
        ],
        'metadata': {
            'generated': datetime.now().isoformat(),
            'config': config_path,
            'git_hash': get_git_hash(),
            'total_tested': len(all_results),
            'total_passed': len(all_results[all_results['verdict'] == 'PASS']),
            'total_selected': len(selected)
        }
    }
    
    with open(pairs_file, 'w') as f:
        yaml.dump(pairs_data, f, default_flow_style=False)
    print(f"📝 Saved pairs to {pairs_file}")
    
    # 2. Save metrics CSV
    metrics_file = config['output']['metrics_file']
    all_results.to_csv(metrics_file, index=False)
    print(f"📊 Saved metrics to {metrics_file}")
    
    # 3. Generate and save report
    report_file = config['output']['report_file']
    report = generate_report(selected, all_results, config)
    
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"📄 Saved report to {report_file}")
    
    # 4. Update artifact registry
    update_artifact_registry(pairs_file, metrics_file, report_file, config)


def generate_report(selected: pd.DataFrame, all_results: pd.DataFrame, 
                    config: dict) -> str:
    """Generate universe report."""
    
    report = f"""# Universe Selection Report

*Generated: {datetime.now().isoformat()}*
*Git Hash: {get_git_hash()}*

## Summary

- **Total Pairs Tested**: {len(all_results)}
- **Pairs Passing Criteria**: {len(all_results[all_results['verdict'] == 'PASS'])}
- **Pairs Selected**: {len(selected)}

## Selection Criteria

| Criterion | Value |
|-----------|-------|
| Max P-value | {config['criteria']['coint_pvalue_max']} |
| Half-life Range | {config['criteria']['hl_min']}-{config['criteria']['hl_max']} |
| Hurst Range | {config['criteria']['hurst_min']}-{config['criteria']['hurst_max']} |
| Min Crossings | {config['criteria']['min_cross']} |
| Max Beta Drift | {config['criteria']['beta_drift_max']} |

## Selected Pairs

| Rank | Pair | Score | P-value | Half-life | Hurst | Crossings | Beta Drift |
|------|------|-------|---------|-----------|-------|-----------|------------|
"""
    
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        report += f"| {i} | {row['pair']} | {row['score']:.2f} | "
        report += f"{row['pvalue']:.4f} | {row['half_life']:.1f} | "
        report += f"{row['hurst']:.3f} | {row['crossings']} | {row['beta_drift']:.3f} |\n"
    
    # Add distribution analysis
    report += """
## Statistical Distribution

### P-value Distribution
"""
    
    pvalue_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
    hist, _ = np.histogram(all_results['pvalue'], bins=pvalue_bins)
    
    for i in range(len(hist)):
        lower = pvalue_bins[i]
        upper = pvalue_bins[i+1]
        report += f"- [{lower:.2f}, {upper:.2f}): {hist[i]} pairs\n"
    
    report += """
### Half-life Distribution
"""
    
    hl_valid = all_results[all_results['half_life'] < 1000]['half_life']
    if len(hl_valid) > 0:
        report += f"- Mean: {hl_valid.mean():.1f} periods\n"
        report += f"- Median: {hl_valid.median():.1f} periods\n"
        report += f"- Std: {hl_valid.std():.1f} periods\n"
    
    report += """
### Hurst Exponent Distribution
"""
    
    hurst_data = all_results['hurst']
    report += f"- Mean: {hurst_data.mean():.3f}\n"
    report += f"- Median: {hurst_data.median():.3f}\n"
    report += f"- Mean-reverting (<0.5): {len(hurst_data[hurst_data < 0.5])} pairs\n"
    report += f"- Random walk (~0.5): {len(hurst_data[(hurst_data >= 0.45) & (hurst_data <= 0.55)])} pairs\n"
    report += f"- Trending (>0.5): {len(hurst_data[hurst_data > 0.5])} pairs\n"
    
    return report


def update_artifact_registry(pairs_file: str, metrics_file: str, 
                            report_file: str, config: dict):
    """Update artifact registry with universe files."""
    
    registry_path = Path('artifacts/ARTIFACT_INDEX.json')
    
    # Load existing or create new
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {'artifacts': []}
    
    # Add universe entries
    timestamp = datetime.now().isoformat()
    
    entries = [
        {
            'type': 'universe',
            'category': 'universe',
            'path': pairs_file,
            'filename': Path(pairs_file).name,
            'created': timestamp,
            'metadata': {
                'criteria': config['criteria'],
                'top_n': config['selection']['top_n']
            }
        },
        {
            'type': 'universe',
            'category': 'universe',
            'path': metrics_file,
            'filename': Path(metrics_file).name,
            'created': timestamp
        },
        {
            'type': 'universe',
            'category': 'universe',
            'path': report_file,
            'filename': Path(report_file).name,
            'created': timestamp
        }
    ]
    
    # Add or update entries
    for entry in entries:
        # Remove old entry if exists
        registry['artifacts'] = [
            a for a in registry.get('artifacts', [])
            if a.get('path') != entry['path']
        ]
        registry['artifacts'].append(entry)
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Build universe of cointegrated pairs')
    parser.add_argument('--config', default='configs/universe.yaml',
                       help='Path to universe configuration')
    
    args = parser.parse_args()
    
    build_universe(args.config)


if __name__ == '__main__':
    main()