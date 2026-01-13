#!/usr/bin/env python3
"""Unified multi-stage pair selection pipeline."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.coint2.pipeline.pair_scanner import (
    scan_universe,
    calculate_pair_score,
    evaluate_pair
)
from src.coint2.core.data_loader import DataHandler
from src.coint2.utils.config import load_config


def load_criteria(criteria_path: Optional[str]) -> Dict:
    """Load selection criteria from YAML or use defaults."""
    if criteria_path and Path(criteria_path).exists():
        with open(criteria_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default criteria
    return {
        'coint_pvalue_max': 0.05,
        'hl_min': 5,
        'hl_max': 200,
        'min_cross': 10,
        'beta_drift_max': 0.15
    }


def load_symbols(symbols_file: Optional[str], data_root: str) -> List[str]:
    """Load symbols from file or auto-detect from data."""
    # Special case: ALL means use all detected symbols
    if symbols_file == "ALL":
        from scripts.universe.scan_data import scan_data_inventory
        inventory = scan_data_inventory(data_root)
        symbols = inventory.get('symbols_sample', [])
        print(f"📋 Using ALL detected symbols: {len(symbols)} total")
        return symbols
    
    if symbols_file and Path(symbols_file).exists():
        with open(symbols_file, 'r') as f:
            if symbols_file.endswith('.yaml'):
                data = yaml.safe_load(f)
                return data.get('symbols', [])
            else:
                return [line.strip() for line in f if line.strip()]
    
    # Auto-detect using scan_data
    from scripts.universe.scan_data import scan_data_inventory
    inventory = scan_data_inventory(data_root)
    
    # Save detected symbols
    symbols_path = Path('artifacts/universe') / f"SYMBOLS_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"
    symbols_path.parent.mkdir(parents=True, exist_ok=True)
    
    symbols = inventory.get('symbols_sample', [])
    with open(symbols_path, 'w') as f:
        for sym in symbols:
            f.write(f"{sym}\n")
    
    print(f"📝 Auto-detected {len(symbols)} symbols, saved to {symbols_path}")
    return symbols


def select_top_pairs(results_df: pd.DataFrame, 
                     top_n: int = 50,
                     diversify_by_base: bool = True,
                     max_per_base: int = 5) -> pd.DataFrame:
    """Select top pairs with optional diversification."""
    # Filter passed pairs
    passed = results_df[results_df['verdict'] == 'PASS'].copy()
    
    if passed.empty:
        return pd.DataFrame()
    
    # Sort by score
    passed = passed.sort_values('score', ascending=False)
    
    if not diversify_by_base:
        return passed.head(top_n)
    
    # Diversify by base symbol
    selected = []
    base_counts = {}
    
    for _, row in passed.iterrows():
        base = row['symbol1']
        if base_counts.get(base, 0) < max_per_base:
            selected.append(row)
            base_counts[base] = base_counts.get(base, 0) + 1
            
        if len(selected) >= top_n:
            break
    
    return pd.DataFrame(selected)


def generate_report(results_df: pd.DataFrame, 
                    selected_df: pd.DataFrame,
                    criteria: Dict,
                    period: Dict,
                    out_dir: Path = None) -> str:
    """Generate markdown report."""
    passed = results_df[results_df['verdict'] == 'PASS']
    
    report = f"""# Universe Selection Report

## Summary
- **Period**: {period['start']} to {period['end']}
- **Total pairs tested**: {len(results_df)}
- **Pairs passed criteria**: {len(passed)}
- **Pairs selected**: {len(selected_df)}
- **Selection rate**: {len(passed)/len(results_df)*100:.1f}%

## Criteria Used
```yaml
{yaml.dump(criteria, default_flow_style=False)}
```

## Distributions

### P-value Distribution
- Min: {results_df['pvalue'].min():.4f}
- Median: {results_df['pvalue'].median():.4f}
- Max: {results_df['pvalue'].max():.4f}
- Passed (<{criteria['coint_pvalue_max']}): {(results_df['pvalue'] < criteria['coint_pvalue_max']).sum()}

### Half-life Distribution
- Min: {results_df['half_life'].min():.1f}
- Median: {results_df['half_life'].median():.1f}
- Max: {results_df['half_life'].max():.1f}
- In range ({criteria['hl_min']}-{criteria['hl_max']}): {((results_df['half_life'] >= criteria['hl_min']) & (results_df['half_life'] <= criteria['hl_max'])).sum()}

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
"""
    
    for i, row in selected_df.head(20).iterrows():
        report += f"| {row['pair']} | {row['score']:.3f} | {row['pvalue']:.4f} | {row['half_life']:.1f} | {row['crossings']} | {row['beta_drift']:.3f} |\n"
    
    report += f"""

---
Generated: {datetime.now(timezone.utc).isoformat()}
"""
    
    # Add rejection breakdown if available
    if out_dir:
        breakdown_path = out_dir / 'REJECTION_BREAKDOWN.yaml'
        if breakdown_path.exists():
            with open(breakdown_path, 'r') as f:
                breakdown = yaml.safe_load(f)
            
            report += f"""

## Rejection Breakdown

**Tested**: {breakdown['tested_pairs']} pairs  
**Passed**: {breakdown['passed_pairs']} pairs

### Top Rejection Reasons:
"""
            if breakdown.get('reasons'):
                sorted_reasons = sorted(breakdown['reasons'].items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons:
                    report += f"- **{reason}**: {count} pairs ({count/breakdown['tested_pairs']*100:.1f}%)\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Multi-stage pair selection pipeline')
    parser.add_argument('--data-root', default='./data_downloaded', help='Data root directory')
    parser.add_argument('--timeframe', default='15min', help='Data timeframe')
    parser.add_argument('--period-start', required=True, help='Period start (YYYY-MM-DD)')
    parser.add_argument('--period-end', required=True, help='Period end (YYYY-MM-DD)')
    parser.add_argument('--criteria-config', help='YAML file with selection criteria')
    parser.add_argument('--universe-symbols', help='YAML/CSV file with symbols to test (or "ALL" for all detected)')
    parser.add_argument('--limit-pairs', type=int, default=1000, help='Max pairs to test')
    parser.add_argument('--out-dir', default='artifacts/universe', help='Output directory')
    parser.add_argument('--top-n', type=int, default=50, help='Top N pairs to select')
    parser.add_argument('--diversify-by-base', action='store_true', help='Diversify by base symbol')
    parser.add_argument('--max-per-base', type=int, default=5, help='Max pairs per base symbol')
    parser.add_argument('--min-trades', type=int, default=0, help='Min trades in validation (TODO)')
    parser.add_argument('--out-pairs', help='Output path for pairs YAML (default: out-dir/pairs_universe.yaml)')
    parser.add_argument('--log-every', type=int, default=1000,
                       help='Progress heartbeat frequency in pairs')
    
    args = parser.parse_args()
    
    # Set environment variable for heartbeat logging
    import os
    os.environ['COINT_LOG_EVERY'] = str(max(1, args.log_every))
    print(f"🪵 Progress heartbeat every {args.log_every} pairs")
    
    # Setup paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load criteria
    criteria = load_criteria(args.criteria_config)
    print(f"📊 Loaded criteria: {list(criteria.keys())}")
    
    # Load symbols
    symbols = load_symbols(args.universe_symbols, args.data_root)
    print(f"📋 Testing {len(symbols)} symbols")
    
    # Convert dates
    start_date = pd.Timestamp(args.period_start)
    end_date = pd.Timestamp(args.period_end)
    
    # Setup config for scanner
    config = {
        'criteria': criteria,
        'train_days': 60,
        'valid_days': 30,
        'data_root': args.data_root,
        'timeframe': args.timeframe
    }
    
    # Initialize data handler
    base_config = load_config('configs/main_2024.yaml')
    
    def _set_data_cfg(cfg, root, timeframe):
        """
        Make base_config point to the requested data root/timeframe.
        Works for both dict-like configs and AppConfig (pydantic/dataclass).
        """
        # dict-like branch
        if isinstance(cfg, dict):
            cfg.setdefault('data', {})
            cfg['data']['path'] = root
            if timeframe:
                cfg['data']['timeframe'] = timeframe
            return

        # object-like branch (AppConfig)
        data = getattr(cfg, 'data', None)
        if data is None:
            return

        # set root/path field if present
        if hasattr(data, 'path'):
            setattr(data, 'path', root)
        elif hasattr(data, 'root'):
            setattr(data, 'root', root)
        elif hasattr(data, 'data_root'):
            setattr(data, 'data_root', root)

        # set timeframe if provided
        if timeframe:
            if hasattr(data, 'timeframe'):
                setattr(data, 'timeframe', timeframe)
            elif hasattr(data, 'tf'):
                setattr(data, 'tf', timeframe)
    
    _set_data_cfg(base_config, args.data_root, args.timeframe)
    data_handler = DataHandler(base_config)
    
    print(f"🔍 Scanning universe from {start_date} to {end_date}...")
    
    # Run multi-stage selection
    results_df = scan_universe(
        data_handler,
        symbols[:args.limit_pairs] if args.limit_pairs else symbols,
        start_date,
        end_date,
        config
    )
    
    # Save rejection breakdown if available
    if hasattr(results_df, '_rejection_breakdown'):
        breakdown_path = out_dir / 'REJECTION_BREAKDOWN.yaml'
        with open(breakdown_path, 'w') as f:
            yaml.dump(results_df._rejection_breakdown, f, default_flow_style=False)
        print(f"📊 Saved rejection breakdown to {breakdown_path}")
    
    # Add scores
    results_df['score'] = results_df.apply(
        lambda row: calculate_pair_score(row.to_dict(), config), 
        axis=1
    )
    
    # Select top pairs
    selected_df = select_top_pairs(
        results_df, 
        top_n=args.top_n,
        diversify_by_base=args.diversify_by_base,
        max_per_base=args.max_per_base
    )
    
    print(f"✅ Selected {len(selected_df)} pairs from {len(results_df)} tested")
    
    # Save outputs
    # 1. Pairs universe YAML
    pairs_yaml = {
        'metadata': {
            'generated': datetime.now(timezone.utc).isoformat(),
            'period': {'start': args.period_start, 'end': args.period_end},
            'criteria': criteria,
            'selection': {
                'top_n': args.top_n,
                'diversify_by_base': args.diversify_by_base,
                'max_per_base': args.max_per_base
            }
        },
        'pairs': [
            {
                'pair': row['pair'],
                'symbol1': row['symbol1'],
                'symbol2': row['symbol2'],
                'score': float(row['score']),
                'metrics': {
                    'pvalue': float(row['pvalue']),
                    'half_life': float(row['half_life']),
                    'crossings': int(row['crossings']),
                    'beta_drift': float(row['beta_drift'])
                }
            }
            for _, row in selected_df.iterrows()
        ]
    }
    
    # Determine pairs output path
    if args.out_pairs:
        pairs_path = Path(args.out_pairs)
    else:
        pairs_path = out_dir / 'pairs_universe.yaml'
    
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, 'w') as f:
        yaml.dump(pairs_yaml, f, default_flow_style=False)
    print(f"📝 Saved pairs to {pairs_path}")
    
    # 2. Metrics CSV
    metrics_path = out_dir / 'universe_metrics.csv'
    results_df.to_csv(metrics_path, index=False)
    print(f"📊 Saved metrics to {metrics_path}")
    
    # 3. Report MD
    report = generate_report(
        results_df, 
        selected_df,
        criteria,
        {'start': args.period_start, 'end': args.period_end},
        out_dir
    )
    
    report_path = out_dir / 'UNIVERSE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Saved report to {report_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
