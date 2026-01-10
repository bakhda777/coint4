#!/usr/bin/env python3
"""Generate universe window configurations for multi-period analysis."""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def load_data_scan(scan_file="artifacts/universe/DATA_SCAN.json"):
    """Load data inventory scan results."""
    if not Path(scan_file).exists():
        raise FileNotFoundError(
            f"Data scan not found at {scan_file}. "
            "Please run: python scripts/scan_data_inventory.py"
        )
    
    with open(scan_file, 'r') as f:
        return json.load(f)


def generate_windows(min_date, max_date, train_days, valid_days, num_windows=4):
    """Generate time windows for universe selection.
    
    Returns list of (start, end) tuples.
    """
    min_dt = pd.to_datetime(min_date)
    max_dt = pd.to_datetime(max_date)
    
    total_window = train_days + valid_days
    total_days = (max_dt - min_dt).days
    
    if total_days < total_window * 2:
        print(f"⚠️ Data period too short for {num_windows} windows, reducing to 2")
        num_windows = 2
    
    # Calculate step size
    step_days = max(30, (total_days - total_window) // (num_windows - 1))
    
    windows = []
    for i in range(num_windows):
        start = min_dt + timedelta(days=i * step_days)
        end = start + timedelta(days=total_window)
        
        if end > max_dt:
            end = max_dt
            start = end - timedelta(days=total_window)
        
        if start >= min_dt:
            windows.append({
                'label': f'win{i+1}',
                'start': start.strftime('%Y-%m-%d'),
                'end': end.strftime('%Y-%m-%d')
            })
    
    return windows


def create_criteria_profiles():
    """Create strict and relaxed criteria profiles."""
    return {
        'strict': {
            'coint_pvalue_max': 0.05,
            'hl_min': 10,
            'hl_max': 150,
            'hurst_min': 0.2,
            'hurst_max': 0.5,
            'min_cross': 15,
            'beta_drift_max': 0.10
        },
        'relaxed': {
            'coint_pvalue_max': 0.15,
            'hl_min': 5,
            'hl_max': 300,
            'hurst_min': 0.1,
            'hurst_max': 0.7,
            'min_cross': 8,
            'beta_drift_max': 0.25
        }
    }


def generate_window_config(window, profile_name, criteria, 
                          data_root, timeframe, train_days, valid_days):
    """Generate configuration for a single window."""
    return {
        'data': {
            'root': data_root
        },
        'period': {
            'start': window['start'],
            'end': window['end']
        },
        'universe': {
            'symbols': [],  # Use all available
            'timeframe': timeframe,
            'train_days': train_days,
            'valid_days': valid_days
        },
        'criteria': criteria,
        'selection': {
            'top_n': 30,  # More candidates per window
            'diversify_by_base': True,
            'max_per_base': 7
        },
        'output': {
            'pairs_file': f"artifacts/universe/{window['label']}_{profile_name}/pairs.yaml",
            'report_file': f"artifacts/universe/{window['label']}_{profile_name}/REPORT.md",
            'metrics_file': f"artifacts/universe/{window['label']}_{profile_name}/universe_metrics.csv"
        },
        'metadata': {
            'window_label': window['label'],
            'profile': profile_name,
            'generated': datetime.utcnow().isoformat()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Generate universe window configs')
    parser.add_argument('--data-root', default='./data_downloaded',
                       help='Root directory with data')
    parser.add_argument('--timeframe', default='15T',
                       help='Data timeframe')
    parser.add_argument('--train-days', type=int, default=120,
                       help='Training period days')
    parser.add_argument('--valid-days', type=int, default=30,
                       help='Validation period days')
    parser.add_argument('--num-windows', type=int, default=4,
                       help='Number of windows to generate')
    
    args = parser.parse_args()
    
    print(f"🎯 Generating universe window configurations...")
    print(f"📊 Windows: {args.num_windows}")
    print(f"📅 Train: {args.train_days} days, Valid: {args.valid_days} days")
    
    # Load data scan
    scan = load_data_scan()
    print(f"📁 Data range: {scan['min_ts'][:10]} to {scan['max_ts'][:10]}")
    
    # Generate windows
    windows = generate_windows(
        scan['min_ts'], scan['max_ts'],
        args.train_days, args.valid_days,
        args.num_windows
    )
    
    print(f"✅ Generated {len(windows)} windows")
    
    # Get criteria profiles
    profiles = create_criteria_profiles()
    
    # Create output directory
    out_dir = Path("configs/windows")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configs for each window and profile
    configs_index = []
    
    for window in windows:
        for profile_name, criteria in profiles.items():
            config = generate_window_config(
                window, profile_name, criteria,
                args.data_root, args.timeframe,
                args.train_days, args.valid_days
            )
            
            # Save config
            config_name = f"universe_{window['label']}_{profile_name}.yaml"
            config_path = out_dir / config_name
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"  📝 Created {config_path}")
            
            # Add to index
            configs_index.append({
                'label': f"{window['label']}_{profile_name}",
                'window': window['label'],
                'profile': profile_name,
                'start': window['start'],
                'end': window['end'],
                'config': str(config_path),
                'criteria_summary': {
                    'p_value': criteria['coint_pvalue_max'],
                    'half_life': f"{criteria['hl_min']}-{criteria['hl_max']}",
                    'hurst': f"{criteria['hurst_min']}-{criteria['hurst_max']}"
                }
            })
    
    # Save index
    index_path = out_dir / "index.json"
    with open(index_path, 'w') as f:
        json.dump({
            'windows': windows,
            'profiles': list(profiles.keys()),
            'configs': configs_index,
            'generated': datetime.utcnow().isoformat(),
            'train_days': args.train_days,
            'valid_days': args.valid_days
        }, f, indent=2)
    
    print(f"\n✅ Saved index to {index_path}")
    print(f"📊 Total configs generated: {len(configs_index)}")
    print(f"\n📋 Windows summary:")
    for w in windows:
        print(f"  - {w['label']}: {w['start']} to {w['end']}")
    
    return 0


if __name__ == '__main__':
    exit(main())