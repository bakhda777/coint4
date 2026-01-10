#!/usr/bin/env python3
"""Unit tests for drift monitor."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

import monitor_drift


@pytest.fixture
def temp_config():
    """Temporary drift monitor configuration."""
    config_data = {
        'windows': {
            'short_days': 7,
            'long_days': 30,
            'min_observations': 5
        },
        'thresholds': {
            'psr_p5_min': 0.80,
            'sharpe_p5_min': 0.50,
            'sharpe_drop_tol': 0.25,
            'psr_drop_tol': 0.15
        },
        'actions': {
            'derisk_scale': [0.75, 0.5],
            'rebuild_portfolio': False  # Disable for tests
        },
        'data_sources': {
            'confidence_file': 'test_confidence.csv',
            'wfa_results': 'test_wfa.csv'
        },
        'outputs': {
            'dashboard': 'test_dashboard.md',
            'actions_log': 'test_actions.md',
            'drift_data': 'test_drift.csv'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def mock_confidence_data():
    """Mock confidence data."""
    return pd.DataFrame({
        'pair': ['PORTFOLIO', 'PORTFOLIO', 'PORTFOLIO'],
        'metric': ['sharpe', 'psr', 'dsr'],
        'p05': [0.3, 0.7, 0.6],
        'p50': [0.6, 0.8, 0.75],
        'p95': [1.0, 0.9, 0.85],
        'observed': [0.7, 0.82, 0.78]
    })


@pytest.fixture
def mock_performance_data():
    """Mock performance data showing degradation."""
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    
    # Simulate degrading performance
    sharpe_values = 1.0 - 0.5 * np.arange(60) / 60 + np.random.normal(0, 0.1, 60)
    psr_values = 0.9 - 0.3 * np.arange(60) / 60 + np.random.normal(0, 0.05, 60)
    
    return pd.DataFrame({
        'date': dates,
        'sharpe': sharpe_values,
        'psr': psr_values,
        'trades': np.random.poisson(20, 60),
        'pnl': np.random.normal(0.01, 0.3, 60)
    })


def test_drift_monitor_initialization(temp_config):
    """Test drift monitor initialization."""
    monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
    
    assert monitor.config is not None
    assert monitor.drift_status == "OK"
    assert monitor.degradation_level == 0


def test_drift_metrics_calculation(temp_config, mock_performance_data):
    """Test drift metrics calculation."""
    monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
    
    drift_metrics = monitor.calculate_drift_metrics(mock_performance_data)
    
    assert 'sharpe_short' in drift_metrics
    assert 'sharpe_long' in drift_metrics
    assert 'sharpe_drop' in drift_metrics
    assert 'psr_short' in drift_metrics
    assert 'psr_long' in drift_metrics


def test_drift_status_assessment_ok(temp_config, mock_confidence_data):
    """Test drift assessment with OK status."""
    monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
    
    # Mock good drift metrics
    drift_metrics = {
        'sharpe_drop': 0.1,  # Low drop
        'psr_drop': 0.05     # Low drop
    }
    
    # High confidence bounds - above all thresholds
    mock_confidence_data.loc[mock_confidence_data['metric'] == 'sharpe', 'p05'] = 0.9
    mock_confidence_data.loc[mock_confidence_data['metric'] == 'psr', 'p05'] = 0.95
    
    status, level = monitor.assess_drift_status(mock_confidence_data, drift_metrics)
    
    assert status == "OK"
    assert level == 0


def test_drift_status_assessment_fail(temp_config, mock_confidence_data):
    """Test drift assessment with FAIL status."""
    monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
    
    # Mock bad drift metrics
    drift_metrics = {
        'sharpe_drop': 0.5,  # High drop
        'psr_drop': 0.3      # High drop
    }
    
    # Low confidence bounds
    mock_confidence_data.loc[mock_confidence_data['metric'] == 'sharpe', 'p05'] = 0.2
    mock_confidence_data.loc[mock_confidence_data['metric'] == 'psr', 'p05'] = 0.4
    
    status, level = monitor.assess_drift_status(mock_confidence_data, drift_metrics)
    
    assert status in ["WARN", "FAIL"]
    assert level >= 1


def test_dashboard_generation(temp_config, mock_confidence_data):
    """Test dashboard generation."""
    monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
    
    drift_metrics = {
        'sharpe_short': 0.5,
        'sharpe_long': 0.8,
        'sharpe_drop': 0.3,
        'psr_short': 0.6,
        'psr_long': 0.8,
        'psr_drop': 0.2
    }
    
    dashboard = monitor.generate_dashboard(
        mock_confidence_data, drift_metrics, "WARN", 1, ["Test action"]
    )
    
    assert "# Drift Monitoring Dashboard" in dashboard
    assert "WARN" in dashboard
    assert "Test action" in dashboard
    assert "0.300" in dashboard  # Sharpe drop should be formatted


@pytest.mark.smoke
def test_full_monitoring_cycle(temp_config):
    """Smoke test for full monitoring cycle."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)
            
            monitor = monitor_drift.DriftMonitor(temp_config, verbose=False)
            
            # Should complete without error even with missing data
            result = monitor.run_monitoring()
            
            assert 'status' in result
            assert 'level' in result
            assert 'actions' in result
            
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])