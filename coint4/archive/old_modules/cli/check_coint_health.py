#!/usr/bin/env python3
"""
Check health of cointegration for current universe pairs.
"""

import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

try:
    from coint2.core.data_loader import DataHandler
    from coint2.utils.config import load_config
    from coint2.pipeline.pair_scanner import test_cointegration, estimate_half_life
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


def check_coint_health(pairs_file: str = 'bench/pairs_universe.yaml'):
    """Check health of cointegration for universe pairs."""
    
    print("🏥 Checking cointegration health...")
    
    # Load pairs
    if not Path(pairs_file).exists():
        print(f"❌ Pairs file not found: {pairs_file}")
        print("   Run: python -m coint2.cli.build_universe first")
        return
    
    with open(pairs_file) as f:
        pairs_data = yaml.safe_load(f)
    
    pairs = pairs_data['pairs']
    print(f"📊 Checking {len(pairs)} pairs")
    
    # Load app config and data
    app_cfg = load_config('configs/main_2024.yaml')
    handler = DataHandler(app_cfg)
    
    # Check period: last 30 days
    end_date = pd.Timestamp.now()
    lookback_days = 30
    
    df = handler.load_all_data_for_period(
        lookback_days=lookback_days,
        end_date=end_date
    )
    
    # Check each pair
    health_results = []
    
    for pair_info in pairs:
        sym1 = pair_info['symbol1']
        sym2 = pair_info['symbol2']
        
        if sym1 not in df.columns or sym2 not in df.columns:
            health_results.append({
                'pair': f"{sym1}/{sym2}",
                'status': 'MISSING',
                'issue': 'Symbol not found in data'
            })
            continue
        
        # Get aligned data
        y = df[sym1].dropna()
        x = df[sym2].dropna()
        aligned = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(aligned) < 50:
            health_results.append({
                'pair': f"{sym1}/{sym2}",
                'status': 'INSUFFICIENT',
                'issue': f'Only {len(aligned)} data points'
            })
            continue
        
        # Quick cointegration test
        health = check_pair_health(
            aligned['y'].values,
            aligned['x'].values,
            pair_info
        )
        
        health['pair'] = f"{sym1}/{sym2}"
        health_results.append(health)
    
    # Generate report
    report = generate_health_report(health_results, pairs_data)
    
    # Save report
    report_path = Path('artifacts/universe/COINT_HEALTH.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Health report saved to {report_path}")
    
    # Print summary
    ok_count = sum(1 for h in health_results if h['status'] == 'OK')
    warn_count = sum(1 for h in health_results if h['status'] == 'WARN')
    fail_count = sum(1 for h in health_results if h['status'] == 'FAIL')
    
    print(f"\n📊 Health Summary:")
    print(f"  ✅ OK: {ok_count}")
    print(f"  ⚠️ WARN: {warn_count}")
    print(f"  ❌ FAIL: {fail_count}")
    
    # Suggest replacements if needed
    if fail_count > 0:
        print(f"\n💡 Recommendation: Re-run universe selection to replace {fail_count} failing pairs")
    
    return health_results


def check_pair_health(y: np.ndarray, x: np.ndarray, original_metrics: dict) -> dict:
    """Check health of a single pair."""
    
    result = {'status': 'OK', 'issue': None}
    
    # 1. ADF test on spread
    X = np.column_stack([np.ones(len(x)), x])
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = betas[0], betas[1]
    spread = y - beta * x - alpha
    
    try:
        adf_result = adfuller(spread, maxlag=10, autolag='BIC')
        pvalue = adf_result[1]
        result['pvalue_current'] = pvalue
        result['pvalue_original'] = original_metrics.get('pvalue', 0)
        
        # Check degradation
        if pvalue > 0.10:
            result['status'] = 'FAIL'
            result['issue'] = f'P-value degraded to {pvalue:.3f}'
        elif pvalue > 0.05:
            result['status'] = 'WARN'
            result['issue'] = f'P-value weakened to {pvalue:.3f}'
    except:
        result['status'] = 'FAIL'
        result['issue'] = 'ADF test failed'
        result['pvalue_current'] = 1.0
    
    # 2. Half-life check
    half_life = estimate_half_life(spread)
    result['half_life_current'] = half_life
    result['half_life_original'] = original_metrics.get('half_life', 0)
    
    if half_life > 500:
        if result['status'] == 'OK':
            result['status'] = 'WARN'
            result['issue'] = f'Half-life increased to {half_life:.0f}'
    
    # 3. Beta stability
    mid = len(y) // 2
    beta1 = np.linalg.lstsq(
        np.column_stack([np.ones(mid), x[:mid]]),
        y[:mid],
        rcond=None
    )[0][1]
    
    beta2 = np.linalg.lstsq(
        np.column_stack([np.ones(len(y)-mid), x[mid:]]),
        y[mid:],
        rcond=None
    )[0][1]
    
    beta_drift = abs(beta2 - beta1) / abs(beta1) if beta1 != 0 else 0
    result['beta_drift_current'] = beta_drift
    
    if beta_drift > 0.30:
        if result['status'] != 'FAIL':
            result['status'] = 'WARN'
            if result['issue'] is None:
                result['issue'] = f'Beta drift {beta_drift:.2%}'
    
    return result


def generate_health_report(health_results: list, pairs_data: dict) -> str:
    """Generate health report markdown."""
    
    report = f"""# Cointegration Health Report

*Generated: {datetime.now().isoformat()}*
*Universe Generated: {pairs_data['metadata']['generated']}*

## Summary

- **Total Pairs**: {len(health_results)}
- **Healthy (OK)**: {sum(1 for h in health_results if h['status'] == 'OK')}
- **Warning**: {sum(1 for h in health_results if h['status'] == 'WARN')}
- **Failed**: {sum(1 for h in health_results if h['status'] == 'FAIL')}

## Detailed Status

| Pair | Status | P-value (Now/Orig) | Half-life (Now/Orig) | Issue |
|------|--------|-------------------|---------------------|--------|
"""
    
    # Sort by status (FAIL first, then WARN, then OK)
    status_order = {'FAIL': 0, 'WARN': 1, 'OK': 2, 'MISSING': 3, 'INSUFFICIENT': 4}
    health_results.sort(key=lambda x: status_order.get(x['status'], 5))
    
    for result in health_results:
        status_emoji = {
            'OK': '✅',
            'WARN': '⚠️',
            'FAIL': '❌',
            'MISSING': '❓',
            'INSUFFICIENT': '📉'
        }.get(result['status'], '❓')
        
        report += f"| {result['pair']} | {status_emoji} {result['status']} | "
        
        if 'pvalue_current' in result:
            report += f"{result['pvalue_current']:.3f}/{result.get('pvalue_original', 0):.3f} | "
            report += f"{result.get('half_life_current', 0):.0f}/{result.get('half_life_original', 0):.0f} | "
        else:
            report += "N/A | N/A | "
        
        report += f"{result.get('issue', '')} |\n"
    
    # Add recommendations
    report += """
## Recommendations

"""
    
    failed = [h for h in health_results if h['status'] == 'FAIL']
    warned = [h for h in health_results if h['status'] == 'WARN']
    
    if failed:
        report += f"""### Immediate Action Required

The following {len(failed)} pairs have failed cointegration tests and should be replaced:
"""
        for h in failed[:5]:  # Show top 5
            report += f"- {h['pair']}: {h.get('issue', 'Unknown issue')}\n"
        
        if len(failed) > 5:
            report += f"- ... and {len(failed) - 5} more\n"
        
        report += """
**Action**: Run `python -m coint2.cli.build_universe` to rebuild universe.
"""
    
    if warned:
        report += f"""### Monitor Closely

The following {len(warned)} pairs show warning signs:
"""
        for h in warned[:5]:  # Show top 5
            report += f"- {h['pair']}: {h.get('issue', 'Unknown issue')}\n"
        
        if len(warned) > 5:
            report += f"- ... and {len(warned) - 5} more\n"
    
    if not failed and not warned:
        report += """### All Systems Healthy

All pairs are showing healthy cointegration. No action required.
"""
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check cointegration health')
    parser.add_argument('--pairs-file', default='bench/pairs_universe.yaml',
                       help='Path to pairs file')
    
    args = parser.parse_args()
    
    check_coint_health(args.pairs_file)


if __name__ == '__main__':
    main()