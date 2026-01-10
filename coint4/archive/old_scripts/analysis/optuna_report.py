#!/usr/bin/env python3
"""Unified Optuna study analysis and reporting."""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import sqlite3
import json
from datetime import datetime

import optuna
import pandas as pd
import numpy as np


def analyze_study(study_path: str, top_n: int = 10, verbose: bool = False):
    """Analyze single Optuna study."""
    storage = f'sqlite:///{study_path}'
    
    # Get all studies in database
    conn = sqlite3.connect(study_path)
    cursor = conn.cursor()
    cursor.execute("SELECT study_name FROM studies")
    studies = cursor.fetchall()
    conn.close()
    
    if not studies:
        print(f"❌ No studies found in {study_path}")
        return None
    
    results = {}
    for study_name, in studies:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            continue
            
        values = [t.value for t in completed if t.value is not None]
        if not values:
            continue
        
        results[study_name] = {
            'n_trials': len(study.trials),
            'n_completed': len(completed),
            'best_value': study.best_value if hasattr(study, 'best_value') else max(values),
            'best_params': study.best_params if hasattr(study, 'best_params') else {},
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'positive_rate': sum(1 for v in values if v > 0) / len(values),
            'top_trials': sorted(completed, key=lambda t: t.value or -999, reverse=True)[:top_n]
        }
    
    return results


def analyze_negative_sharpe(study_path: str, threshold: float = 0.0):
    """Analyze trials with negative Sharpe ratios."""
    storage = f'sqlite:///{study_path}'
    
    conn = sqlite3.connect(study_path)
    cursor = conn.cursor()
    cursor.execute("SELECT study_name FROM studies")
    studies = cursor.fetchall()
    conn.close()
    
    negative_patterns = {}
    
    for study_name, in studies:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        negative_trials = [
            t for t in study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE 
            and t.value is not None 
            and t.value < threshold
        ]
        
        if negative_trials:
            # Analyze parameter patterns
            param_stats = {}
            for t in negative_trials:
                for param, value in t.params.items():
                    if param not in param_stats:
                        param_stats[param] = []
                    param_stats[param].append(value)
            
            negative_patterns[study_name] = {
                'count': len(negative_trials),
                'worst_value': min(t.value for t in negative_trials),
                'param_ranges': {
                    param: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values)
                    } for param, values in param_stats.items()
                }
            }
    
    return negative_patterns


def monitor_active(study_path: str, refresh_seconds: int = 10):
    """Monitor active optimization in real-time."""
    import time
    
    storage = f'sqlite:///{study_path}'
    
    print(f"📊 Monitoring {study_path}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            conn = sqlite3.connect(study_path)
            cursor = conn.cursor()
            
            # Get latest trials
            cursor.execute("""
                SELECT COUNT(*), MAX(datetime_complete) 
                FROM trials 
                WHERE state = 'COMPLETE'
            """)
            n_completed, last_complete = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 'RUNNING'")
            n_running = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT value 
                FROM trials 
                WHERE state = 'COMPLETE' AND value IS NOT NULL
                ORDER BY value DESC 
                LIMIT 5
            """)
            top_values = [r[0] for r in cursor.fetchall()]
            
            conn.close()
            
            # Clear and print status
            print("\033[2J\033[H")  # Clear screen
            print(f"📊 Optimization Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print(f"Completed trials: {n_completed}")
            print(f"Running trials: {n_running}")
            if last_complete:
                print(f"Last completion: {last_complete}")
            if top_values:
                print(f"\nTop 5 Sharpe ratios:")
                for i, val in enumerate(top_values, 1):
                    print(f"  {i}. {val:.3f}")
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


def compare_studies(study_paths: List[str]):
    """Compare multiple Optuna studies."""
    all_results = {}
    
    for path in study_paths:
        if not Path(path).exists():
            print(f"⚠️ Skipping {path} (not found)")
            continue
        
        results = analyze_study(path, top_n=5)
        if results:
            all_results[Path(path).stem] = results
    
    if not all_results:
        print("❌ No valid studies to compare")
        return
    
    # Create comparison table
    comparison = []
    for db_name, studies in all_results.items():
        for study_name, stats in studies.items():
            comparison.append({
                'Database': db_name,
                'Study': study_name,
                'Trials': stats['n_trials'],
                'Best Sharpe': stats['best_value'],
                'Mean Sharpe': stats['mean_value'],
                'Positive %': stats['positive_rate'] * 100
            })
    
    df = pd.DataFrame(comparison)
    print("\n📊 Study Comparison")
    print("=" * 80)
    print(df.to_string(index=False))


def export_best_params(study_path: str, output_file: str, top_n: int = 1):
    """Export best parameters to YAML."""
    results = analyze_study(study_path, top_n=top_n)
    
    if not results:
        print("❌ No results to export")
        return
    
    # Get best trial from all studies
    best_trial = None
    best_value = -float('inf')
    
    for study_name, stats in results.items():
        if stats['top_trials'] and stats['top_trials'][0].value > best_value:
            best_trial = stats['top_trials'][0]
            best_value = best_trial.value
    
    if not best_trial:
        print("❌ No valid trials found")
        return
    
    # Export to YAML
    import yaml
    
    config = {
        'signals': {},
        'metadata': {
            'sharpe': best_value,
            'study_path': str(study_path),
            'exported_at': datetime.now().isoformat()
        }
    }
    
    # Map parameter names
    param_mapping = {
        'zscore_threshold': 'zscore_threshold',
        'zscore_exit': 'zscore_exit',
        'rolling_window': 'rolling_window',
        'max_holding_days': 'max_holding_days',
        'z_enter': 'zscore_threshold',
        'z_exit': 'zscore_exit'
    }
    
    for param, value in best_trial.params.items():
        mapped_param = param_mapping.get(param, param)
        config['signals'][mapped_param] = value
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Best parameters exported to {output_file}")
    print(f"   Sharpe: {best_value:.3f}")
    for k, v in config['signals'].items():
        print(f"   {k}: {v}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified Optuna study analysis and reporting'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze study')
    analyze_parser.add_argument('study', help='Study database path')
    analyze_parser.add_argument('--top-n', type=int, default=10,
                               help='Number of top trials to show')
    analyze_parser.add_argument('--verbose', action='store_true',
                               help='Verbose output')
    
    # Negative Sharpe analysis
    negative_parser = subparsers.add_parser('negative', help='Analyze negative Sharpe')
    negative_parser.add_argument('study', help='Study database path')
    negative_parser.add_argument('--threshold', type=float, default=0.0,
                                help='Sharpe threshold')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor active optimization')
    monitor_parser.add_argument('study', help='Study database path')
    monitor_parser.add_argument('--refresh', type=int, default=10,
                               help='Refresh interval in seconds')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple studies')
    compare_parser.add_argument('studies', nargs='+', help='Study database paths')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export best parameters')
    export_parser.add_argument('study', help='Study database path')
    export_parser.add_argument('--output', default='configs/best_params.yaml',
                              help='Output YAML file')
    export_parser.add_argument('--top-n', type=int, default=1,
                              help='Number of top configs to consider')
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'analyze':
        results = analyze_study(args.study, args.top_n, args.verbose)
        if results:
            for study_name, stats in results.items():
                print(f"\n📊 Study: {study_name}")
                print("=" * 60)
                print(f"Trials: {stats['n_trials']} ({stats['n_completed']} completed)")
                print(f"Best Sharpe: {stats['best_value']:.3f}")
                print(f"Mean Sharpe: {stats['mean_value']:.3f} ± {stats['std_value']:.3f}")
                print(f"Positive rate: {stats['positive_rate']:.1%}")
                
                if args.verbose and stats['top_trials']:
                    print(f"\nTop {len(stats['top_trials'])} trials:")
                    for i, t in enumerate(stats['top_trials'], 1):
                        print(f"  {i}. Sharpe={t.value:.3f}")
                        for k, v in t.params.items():
                            print(f"     {k}: {v}")
    
    elif args.command == 'negative':
        patterns = analyze_negative_sharpe(args.study, args.threshold)
        if patterns:
            for study_name, analysis in patterns.items():
                print(f"\n❌ Negative Sharpe Analysis: {study_name}")
                print("=" * 60)
                print(f"Negative trials: {analysis['count']}")
                print(f"Worst Sharpe: {analysis['worst_value']:.3f}")
                print("\nParameter patterns in negative trials:")
                for param, stats in analysis['param_ranges'].items():
                    print(f"  {param}:")
                    print(f"    Mean: {stats['mean']:.3f}")
                    print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    elif args.command == 'monitor':
        monitor_active(args.study, args.refresh)
    
    elif args.command == 'compare':
        compare_studies(args.studies)
    
    elif args.command == 'export':
        export_best_params(args.study, args.output, args.top_n)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())