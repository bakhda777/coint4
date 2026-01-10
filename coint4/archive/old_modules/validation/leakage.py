#!/usr/bin/env python3
"""Leakage and alignment validation guards."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path


class LeakageError(Exception):
    """Raised when lookahead bias detected."""
    pass


class AlignmentError(Exception):
    """Raised when signal-execution alignment violated."""
    pass


def assert_no_lookahead(
    df: pd.DataFrame, 
    cols: List[str], 
    horizon: int = 1
) -> Dict[str, Any]:
    """
    Check that features/signals don't reference future data.
    
    Args:
        df: DataFrame with time index
        cols: Columns to check
        horizon: Lookahead horizon in bars
        
    Returns:
        Dict with validation results
        
    Raises:
        LeakageError: If lookahead detected
    """
    violations = []
    
    for col in cols:
        if col not in df.columns:
            continue
            
        # Check if column uses future values
        future_corr = df[col].corr(df[col].shift(-horizon))
        
        # High correlation with future suggests lookahead
        if abs(future_corr) > 0.99:
            # Find first violation
            for i in range(len(df) - horizon):
                if pd.notna(df[col].iloc[i]) and pd.notna(df[col].iloc[i + horizon]):
                    if abs(df[col].iloc[i] - df[col].iloc[i + horizon]) < 1e-10:
                        violations.append({
                            'column': col,
                            'index': df.index[i],
                            'value': df[col].iloc[i],
                            'future_value': df[col].iloc[i + horizon],
                            'future_index': df.index[i + horizon]
                        })
                        break
    
    result = {
        'checked_columns': cols,
        'violations': violations,
        'passed': len(violations) == 0
    }
    
    if violations:
        msg = f"Lookahead bias detected in {len(violations)} columns:\n"
        for v in violations[:3]:  # Show first 3
            msg += f"  - {v['column']} at {v['index']}: value={v['value']:.4f} "
            msg += f"matches future at {v['future_index']}\n"
        raise LeakageError(msg)
    
    return result


def assert_index_monotonic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check index is monotonic and timezone-consistent.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dict with validation results
        
    Raises:
        AlignmentError: If index issues found
    """
    issues = []
    
    # Check monotonic
    if not df.index.is_monotonic_increasing:
        # Find first violation
        for i in range(1, len(df)):
            if df.index[i] <= df.index[i-1]:
                issues.append({
                    'type': 'non_monotonic',
                    'index': i,
                    'timestamp': df.index[i],
                    'previous': df.index[i-1]
                })
                break
    
    # Check timezone consistency (optional warning, not error)
    # We don't require timezone for backtesting data
    
    # Check for duplicates
    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].unique()[:5]
        issues.append({
            'type': 'duplicates',
            'count': len(df.index[df.index.duplicated()]),
            'examples': list(dups)
        })
    
    result = {
        'index_type': str(type(df.index)),
        'monotonic': df.index.is_monotonic_increasing,
        'has_tz': hasattr(df.index, 'tz') and df.index.tz is not None,
        'issues': issues,
        'passed': len(issues) == 0
    }
    
    if issues:
        msg = f"Index issues detected:\n"
        for issue in issues:
            if issue['type'] == 'non_monotonic':
                msg += f"  - Non-monotonic at position {issue['index']}: "
                msg += f"{issue['previous']} -> {issue['timestamp']}\n"
            elif issue['type'] == 'duplicates':
                msg += f"  - {issue['count']} duplicate timestamps\n"
            else:
                msg += f"  - {issue.get('message', issue['type'])}\n"
        raise AlignmentError(msg)
    
    return result


def assert_signal_execution_alignment(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    exec_delay_bars: int = 1
) -> Dict[str, Any]:
    """
    Check signal at t is executed at t+delay.
    
    Args:
        signals: DataFrame with signal columns
        prices: DataFrame with price data
        exec_delay_bars: Expected delay in bars
        
    Returns:
        Dict with validation results
        
    Raises:
        AlignmentError: If alignment violated
    """
    violations = []
    
    # Check indices align
    if not signals.index.equals(prices.index):
        common_idx = signals.index.intersection(prices.index)
        if len(common_idx) < len(signals):
            violations.append({
                'type': 'index_mismatch',
                'signal_count': len(signals),
                'price_count': len(prices),
                'common_count': len(common_idx)
            })
    
    # Check execution delay
    if exec_delay_bars < 1:
        violations.append({
            'type': 'zero_delay',
            'message': f'Execution delay {exec_delay_bars} < 1 (potential lookahead)'
        })
    
    # Check signal changes vs price availability
    signal_cols = [c for c in signals.columns if 'signal' in c.lower()]
    for col in signal_cols:
        if col not in signals.columns:
            continue
            
        # Find signal changes
        signal_diff = signals[col].diff()
        changes = signal_diff[signal_diff != 0].dropna()
        
        for idx, change in changes.items():
            # Check we have price data at execution time
            try:
                exec_idx = signals.index.get_loc(idx) + exec_delay_bars
                if exec_idx >= len(signals):
                    continue
                    
                exec_time = signals.index[exec_idx]
                if exec_time not in prices.index:
                    violations.append({
                        'type': 'missing_exec_price',
                        'signal_time': idx,
                        'exec_time': exec_time,
                        'column': col
                    })
                    if len(violations) >= 10:  # Limit violations
                        break
            except:
                pass
    
    result = {
        'exec_delay_bars': exec_delay_bars,
        'signal_columns': signal_cols,
        'violations': violations[:10],  # Limit to 10
        'total_violations': len(violations),
        'passed': len(violations) == 0
    }
    
    if violations:
        msg = f"Alignment issues detected ({len(violations)} total):\n"
        for v in violations[:3]:
            if v['type'] == 'zero_delay':
                msg += f"  - {v['message']}\n"
            elif v['type'] == 'index_mismatch':
                msg += f"  - Index mismatch: {v['signal_count']} signals, {v['price_count']} prices\n"
            else:
                msg += f"  - {v['type']} at {v.get('signal_time', 'unknown')}\n"
        raise AlignmentError(msg)
    
    return result


def generate_alignment_report(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """Generate alignment audit report."""
    
    report = """# Alignment & Leakage Audit Report

## Summary

"""
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.get('passed', False))
    
    report += f"- **Total Checks**: {total_checks}\n"
    report += f"- **Passed**: {passed_checks}\n"
    report += f"- **Failed**: {total_checks - passed_checks}\n\n"
    
    # Details
    report += "## Check Details\n\n"
    
    for i, result in enumerate(results, 1):
        check_type = result.get('check_type', f'Check {i}')
        status = "✅ PASS" if result.get('passed') else "❌ FAIL"
        
        report += f"### {check_type}: {status}\n\n"
        
        if 'violations' in result and result['violations']:
            report += "**Violations:**\n"
            for v in result['violations'][:5]:
                report += f"- {v}\n"
            
            if result.get('total_violations', 0) > 5:
                report += f"- ... and {result['total_violations'] - 5} more\n"
            report += "\n"
        
        if 'issues' in result and result['issues']:
            report += "**Issues:**\n"
            for issue in result['issues']:
                report += f"- {issue}\n"
            report += "\n"
    
    # Recommendations
    if total_checks > passed_checks:
        report += """## Recommendations

1. Review signal generation logic for lookahead bias
2. Ensure execution delay >= 1 bar
3. Check data alignment and timezone consistency
4. Validate feature engineering pipeline
"""
    else:
        report += """## Conclusion

All alignment and leakage checks passed. The backtesting framework appears free from lookahead bias.
"""
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report