#!/usr/bin/env python3
"""Test leakage and alignment validation guards."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

leakage_module = pytest.importorskip(
    "coint2.validation.leakage",
    reason="Legacy leakage validation module not available."
)
assert_no_lookahead = leakage_module.assert_no_lookahead
assert_index_monotonic = leakage_module.assert_index_monotonic
assert_signal_execution_alignment = leakage_module.assert_signal_execution_alignment
generate_alignment_report = leakage_module.generate_alignment_report
LeakageError = leakage_module.LeakageError
AlignmentError = leakage_module.AlignmentError


def test_no_lookahead_clean_data():
    """Test no lookahead with clean data."""
    # Create clean data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'signal': np.random.randn(100),
        'feature': np.random.randn(100)
    }, index=dates)
    
    # Should pass
    result = assert_no_lookahead(df, ['signal', 'feature'])
    assert result['passed']
    assert len(result['violations']) == 0


def test_no_lookahead_with_future_data():
    """Test detection of lookahead bias."""
    # Create data with lookahead
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    # Create feature that is identical to shifted future values
    feature = np.arange(100, dtype=float)
    # Copy exact future values (perfect lookahead)
    feature_with_lookahead = np.concatenate([feature[1:], [feature[-1]]])
    
    df = pd.DataFrame({
        'signal': feature,
        'feature': feature_with_lookahead
    }, index=dates)
    
    # Should detect lookahead
    with pytest.raises(LeakageError) as exc_info:
        assert_no_lookahead(df, ['feature'])
    
    assert 'Lookahead bias detected' in str(exc_info.value)


def test_index_monotonic_valid():
    """Test monotonic index validation."""
    # Valid monotonic index
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
    
    result = assert_index_monotonic(df)
    assert result['passed']
    assert result['monotonic']


def test_index_non_monotonic():
    """Test non-monotonic index detection."""
    # Create non-monotonic index
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    # Convert to list to make it mutable
    dates_list = list(dates)
    dates_list[50], dates_list[51] = dates_list[51], dates_list[50]
    dates_shuffled = pd.DatetimeIndex(dates_list)
    
    df = pd.DataFrame({'value': np.random.randn(100)}, index=dates_shuffled)
    
    with pytest.raises(AlignmentError) as exc_info:
        assert_index_monotonic(df)
    
    assert 'Non-monotonic' in str(exc_info.value)


def test_index_with_duplicates():
    """Test duplicate index detection."""
    # Create index with duplicates
    dates = pd.date_range('2024-01-01', periods=99, freq='15min')
    dates_with_dup = dates.append(pd.DatetimeIndex([dates[50]]))
    
    df = pd.DataFrame({'value': np.random.randn(100)}, index=dates_with_dup)
    
    with pytest.raises(AlignmentError) as exc_info:
        assert_index_monotonic(df)
    
    assert 'duplicate' in str(exc_info.value).lower()


def test_signal_execution_alignment_valid():
    """Test valid signal-execution alignment."""
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], 100)
    }, index=dates)
    
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum()
    }, index=dates)
    
    # Should pass with delay >= 1
    result = assert_signal_execution_alignment(signals, prices, exec_delay_bars=1)
    assert result['passed']


def test_signal_execution_zero_delay():
    """Test zero execution delay detection."""
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], 100)
    }, index=dates)
    
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum()
    }, index=dates)
    
    # Should fail with zero delay
    with pytest.raises(AlignmentError) as exc_info:
        assert_signal_execution_alignment(signals, prices, exec_delay_bars=0)
    
    assert 'zero_delay' in str(exc_info.value).lower() or 'potential lookahead' in str(exc_info.value).lower()


def test_signal_execution_mismatched_indices():
    """Test mismatched signal-price indices."""
    signal_dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    price_dates = pd.date_range('2024-01-01', periods=50, freq='15min')
    
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], 100)
    }, index=signal_dates)
    
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(50).cumsum()
    }, index=price_dates)
    
    # Should detect mismatch
    with pytest.raises(AlignmentError) as exc_info:
        assert_signal_execution_alignment(signals, prices, exec_delay_bars=1)
    
    assert 'mismatch' in str(exc_info.value).lower()


def test_generate_alignment_report():
    """Test alignment report generation."""
    results = [
        {
            'check_type': 'Lookahead Check',
            'passed': True,
            'violations': []
        },
        {
            'check_type': 'Index Monotonic',
            'passed': False,
            'issues': [{'type': 'non_monotonic', 'index': 50}]
        }
    ]
    
    report = generate_alignment_report(results)
    
    assert '# Alignment & Leakage Audit Report' in report
    assert 'Total Checks' in report
    assert 'Passed' in report
    assert 'Failed' in report
    assert '❌ FAIL' in report


def test_generate_alignment_report_with_file():
    """Test alignment report file generation."""
    import tempfile
    
    results = [
        {
            'check_type': 'Test Check',
            'passed': True,
            'violations': []
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        report = generate_alignment_report(results, output_path=f.name)
        
        # Check file was created
        with open(f.name, 'r') as rf:
            content = rf.read()
            assert '# Alignment & Leakage Audit Report' in content
            assert 'All alignment and leakage checks passed' in content
