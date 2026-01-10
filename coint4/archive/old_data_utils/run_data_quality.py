#!/usr/bin/env python3
"""Run data quality checks and drift detection."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

from coint2.data.quality import DataQualityChecker, generate_quality_report


def load_sample_data() -> pd.DataFrame:
    """Load sample data for quality checking."""
    # Generate synthetic data for demonstration
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-03-01',
        freq='15min'
    )
    
    n = len(dates)
    
    # Create data with some quality issues
    data = {
        'BTCUSDT': np.random.randn(n).cumsum() + 50000,
        'ETHUSDT': np.random.randn(n).cumsum() + 3000,
        'BNBUSDT': np.random.randn(n).cumsum() + 500,
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce quality issues
    # 1. Missing data
    df.loc['2024-01-15':'2024-01-16', 'BTCUSDT'] = np.nan
    
    # 2. Outliers
    df.loc['2024-02-01 12:00', 'ETHUSDT'] *= 10
    
    # 3. Duplicates (simulate)
    # df = df.append(df.iloc[100:105])  # Would create duplicates
    
    return df


def run_quality_checks(
    data_path: str = None,
    output_dir: str = "artifacts/data"
):
    """Run comprehensive data quality checks."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    if data_path and Path(data_path).exists():
        df = pd.read_parquet(data_path)
    else:
        print("Using synthetic data for demonstration")
        df = load_sample_data()
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Initialize checker
    config = {
        'max_missing_pct': 0.05,
        'outlier_zscore': 5.0,
        'max_duplicate_pct': 0.01,
        'drift_pvalue_threshold': 0.05,
        'psi_threshold': 0.1
    }
    
    checker = DataQualityChecker(config)
    
    # Run quality checks
    print("\nRunning data quality checks...")
    issues = checker.check_data_quality(df)
    
    print(f"Found {len(issues)} issues")
    
    # Save issues to CSV
    if issues:
        issues_data = []
        for issue in issues:
            issues_data.append({
                'severity': issue.severity,
                'type': issue.issue_type,
                'description': issue.description,
                'start_time': issue.affected_range[0],
                'end_time': issue.affected_range[1],
                'symbols': ','.join(issue.affected_symbols),
                **issue.metrics
            })
        
        issues_df = pd.DataFrame(issues_data)
        issues_df.to_csv(f"{output_dir}/quality_issues.csv", index=False)
        print(f"Saved issues to {output_dir}/quality_issues.csv")
    
    # Drift detection (compare first vs last month)
    print("\nRunning drift detection...")
    
    mid_point = df.index[len(df)//2]
    reference_df = df[df.index < mid_point]
    current_df = df[df.index >= mid_point]
    
    drift_results = checker.detect_drift(reference_df, current_df)
    
    # Count drifted features
    n_drifted = sum(1 for r in drift_results.values() if r['drift_detected'])
    print(f"Drift detected in {n_drifted}/{len(drift_results)} features")
    
    # Save drift results
    drift_df = pd.DataFrame(drift_results).T
    drift_df.to_csv(f"{output_dir}/drift_results.csv")
    print(f"Saved drift results to {output_dir}/drift_results.csv")
    
    # Generate report
    report = generate_quality_report(issues, drift_results)
    
    report_path = f"{output_dir}/DATA_QUALITY_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nGenerated quality report: {report_path}")
    
    # Generate summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': list(df.shape),
        'date_range': [str(df.index.min()), str(df.index.max())],
        'total_issues': len(issues),
        'issues_by_severity': {
            'critical': sum(1 for i in issues if i.severity == 'critical'),
            'error': sum(1 for i in issues if i.severity == 'error'),
            'warning': sum(1 for i in issues if i.severity == 'warning')
        },
        'drift_detected': n_drifted > 0,
        'drifted_features': n_drifted,
        'recommendation': 'Fix critical issues' if any(i.severity == 'critical' for i in issues) else 'Review warnings'
    }
    
    with open(f"{output_dir}/quality_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nQuality Check Summary:")
    print(f"  Critical issues: {summary['issues_by_severity']['critical']}")
    print(f"  Errors: {summary['issues_by_severity']['error']}")
    print(f"  Warnings: {summary['issues_by_severity']['warning']}")
    print(f"  Drift detected: {'Yes' if summary['drift_detected'] else 'No'}")
    print(f"  Recommendation: {summary['recommendation']}")
    
    return issues, drift_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data quality checker")
    parser.add_argument("--data", help="Path to parquet data file")
    parser.add_argument("--output", default="artifacts/data", help="Output directory")
    
    args = parser.parse_args()
    
    run_quality_checks(args.data, args.output)