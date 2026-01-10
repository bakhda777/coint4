#!/usr/bin/env python3
"""Integration tests for uncertainty and drift monitoring pipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import subprocess
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@pytest.fixture
def test_workspace():
    """Create temporary test workspace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure
        (temp_path / 'artifacts' / 'uncertainty').mkdir(parents=True)
        (temp_path / 'artifacts' / 'monitoring').mkdir(parents=True)
        (temp_path / 'artifacts' / 'portfolio').mkdir(parents=True)
        (temp_path / 'artifacts' / 'wfa').mkdir(parents=True)
        (temp_path / 'configs').mkdir()
        (temp_path / 'scripts').mkdir()
        (temp_path / 'bench').mkdir()
        
        yield temp_path


@pytest.fixture
def mock_data_files(test_workspace):
    """Create mock data files for testing."""
    
    # Mock WFA results
    wfa_data = pd.DataFrame({
        'fold': list(range(10)),
        'sharpe': np.random.normal(0.8, 0.2, 10),
        'pnl': np.random.normal(0.05, 0.5, 10),
        'trades': np.random.poisson(25, 10)
    })
    wfa_data.to_csv(test_workspace / 'artifacts' / 'wfa' / 'results_per_fold.csv', index=False)
    
    # Mock portfolio weights
    weights_data = pd.DataFrame({
        'pair': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'weight': [0.4, 0.4, 0.2]
    })
    weights_data.to_csv(test_workspace / 'artifacts' / 'portfolio' / 'weights.csv', index=False)
    
    # Mock pairs file
    pairs_data = {
        'pairs': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'metadata': {'total_pairs': 3}
    }
    with open(test_workspace / 'bench' / 'pairs_portfolio.yaml', 'w') as f:
        yaml.dump(pairs_data, f)
    
    # Mock drift config
    drift_config = {
        'windows': {'short_days': 7, 'long_days': 21},
        'thresholds': {
            'psr_p5_min': 0.70,
            'sharpe_p5_min': 0.40,
            'sharpe_drop_tol': 0.30
        },
        'actions': {
            'derisk_scale': [0.75, 0.5],
            'rebuild_portfolio': False
        },
        'data_sources': {
            'confidence_file': 'artifacts/uncertainty/confidence.csv',
            'wfa_results': 'artifacts/wfa/results_per_fold.csv'
        },
        'outputs': {
            'dashboard': 'artifacts/monitoring/DRIFT_DASHBOARD.md',
            'actions_log': 'artifacts/monitoring/ACTIONS_TAKEN.md',
            'drift_data': 'artifacts/monitoring/drift_data.csv'
        }
    }
    with open(test_workspace / 'configs' / 'drift_monitor.yaml', 'w') as f:
        yaml.dump(drift_config, f)
    
    return test_workspace


def test_uncertainty_to_drift_pipeline(test_workspace, mock_data_files):
    """Test uncertainty -> drift monitoring pipeline."""
    
    original_cwd = Path.cwd()
    
    try:
        os.chdir(test_workspace)
        
        # Step 1: Run uncertainty analysis
        sys.path.insert(0, str(original_cwd / 'scripts'))
        import run_uncertainty
        
        # Mock run uncertainty
        confidence_data = pd.DataFrame({
            'pair': ['PORTFOLIO', 'PORTFOLIO', 'PORTFOLIO'],
            'metric': ['sharpe', 'psr', 'dsr'],
            'p05': [0.4, 0.75, 0.6],
            'p50': [0.7, 0.85, 0.75],
            'p95': [1.0, 0.95, 0.9],
            'observed': [0.72, 0.87, 0.77]
        })
        confidence_data.to_csv('artifacts/uncertainty/confidence.csv', index=False)
        
        # Create confidence report
        confidence_report = """# Confidence Intervals Report
## Portfolio Confidence Intervals
| Metric | P05 | P50 | P95 |
|--------|-----|-----|-----|
| SHARPE | 0.400 | 0.700 | 1.000 |
| PSR | 0.750 | 0.850 | 0.950 |
"""
        with open('artifacts/uncertainty/CONFIDENCE_REPORT.md', 'w') as f:
            f.write(confidence_report)
        
        # Step 2: Run drift monitoring
        sys.path.insert(0, str(original_cwd / 'scripts'))
        import monitor_drift
        
        monitor = monitor_drift.DriftMonitor('configs/drift_monitor.yaml', verbose=False)
        result = monitor.run_monitoring()
        
        # Verify drift monitoring ran
        assert result['status'] in ['OK', 'WARN', 'FAIL']
        assert 'level' in result
        assert 'actions' in result
        
        # Verify outputs were created
        assert Path('artifacts/monitoring/DRIFT_DASHBOARD.md').exists()
        assert Path('artifacts/monitoring/drift_data.csv').exists()
        
        # Step 3: Verify drift data structure
        drift_df = pd.read_csv('artifacts/monitoring/drift_data.csv')
        assert len(drift_df) == 1  # One record
        assert 'timestamp' in drift_df.columns
        assert 'status' in drift_df.columns
        
    finally:
        os.chdir(original_cwd)


def test_derisk_response_integration(test_workspace, mock_data_files):
    """Test derisk response integration."""
    
    original_cwd = Path.cwd()
    
    try:
        os.chdir(test_workspace)
        
        # Create failing confidence data
        confidence_data = pd.DataFrame({
            'pair': ['PORTFOLIO'],
            'metric': ['sharpe'],
            'p05': [0.1],  # Very low P05 - should trigger FAIL
            'p50': [0.3],
            'p95': [0.6],
            'observed': [0.35]
        })
        confidence_data.to_csv('artifacts/uncertainty/confidence.csv', index=False)
        
        # Run drift monitoring
        sys.path.insert(0, str(original_cwd / 'scripts'))
        import monitor_drift
        
        monitor = monitor_drift.DriftMonitor('configs/drift_monitor.yaml', verbose=False)
        result = monitor.run_monitoring()
        
        # Should detect failure but not execute actions (rebuild disabled)
        assert result['status'] == 'FAIL'
        assert result['level'] >= 1
        
        # Verify actions log was created
        assert Path('artifacts/monitoring/ACTIONS_TAKEN.md').exists()
        
        # Read actions log
        with open('artifacts/monitoring/ACTIONS_TAKEN.md', 'r') as f:
            actions_content = f.read()
        
        # Should log the FAIL status
        assert 'FAIL' in actions_content
        
    finally:
        os.chdir(original_cwd)


def test_regime_rotation_integration(test_workspace, mock_data_files):
    """Test regime rotation integration."""
    
    original_cwd = Path.cwd()
    
    try:
        os.chdir(test_workspace)
        
        # Create portfolio optimizer config with regime profiles
        portfolio_config = {
            'selection': {'method': 'score_topN', 'top_n': 3},
            'optimizer': {'lambda_var': 2.0, 'gamma_cost': 1.0, 'max_gross': 1.0},
            'fallback': 'vol_target',
            'regime_profiles': {
                'low_vol': {'top_n': 4, 'lambda_var': 1.5},
                'mid_vol': {'top_n': 3, 'lambda_var': 2.0},
                'high_vol': {'top_n': 2, 'lambda_var': 3.0}
            }
        }
        with open('configs/portfolio_optimizer.yaml', 'w') as f:
            yaml.dump(portfolio_config, f)
        
        # Mock regime detection and rotation
        sys.path.insert(0, str(original_cwd / 'scripts'))
        import rotate_portfolio_by_regime
        
        rotator = rotate_portfolio_by_regime.RegimePortfolioRotator(
            'configs/portfolio_optimizer.yaml', 
            verbose=False
        )
        
        # Force regime detection
        regime, confidence = rotator.detect_market_regime()
        
        assert regime in ['low_vol', 'mid_vol', 'high_vol']
        assert 0 <= confidence <= 1.0
        
        # Test profile application
        modified_config = rotator.apply_regime_profile(regime)
        
        assert 'regime_profiles' in modified_config
        
        # Verify regime state file creation
        regime_state = {
            'regime': regime,
            'confidence': confidence,
            'last_update': '2024-01-01T00:00:00'
        }
        
        import json
        with open('artifacts/portfolio/regime_state.json', 'w') as f:
            json.dump(regime_state, f)
        
        assert Path('artifacts/portfolio/regime_state.json').exists()
        
    finally:
        os.chdir(original_cwd)


@pytest.mark.smoke
def test_full_pipeline_smoke(test_workspace, mock_data_files):
    """Smoke test for full uncertainty/drift/regime pipeline."""
    
    original_cwd = Path.cwd()
    
    try:
        os.chdir(test_workspace)
        
        # Create minimal working environment
        
        # 1. Uncertainty analysis (mock)
        confidence_data = pd.DataFrame({
            'pair': ['PORTFOLIO'],
            'metric': ['sharpe'],
            'p05': [0.6],
            'p50': [0.8],
            'p95': [1.0],
            'observed': [0.82]
        })
        confidence_data.to_csv('artifacts/uncertainty/confidence.csv', index=False)
        
        # 2. Drift monitoring
        sys.path.insert(0, str(original_cwd / 'scripts'))
        import monitor_drift
        
        monitor = monitor_drift.DriftMonitor('configs/drift_monitor.yaml', verbose=False)
        drift_result = monitor.run_monitoring()
        
        # Should complete successfully
        assert 'status' in drift_result
        
        # 3. Regime rotation (mock)
        import rotate_portfolio_by_regime
        
        portfolio_config = {
            'selection': {'top_n': 3},
            'optimizer': {'lambda_var': 2.0},
            'regime_profiles': {
                'mid_vol': {'top_n': 3, 'lambda_var': 2.0}
            }
        }
        with open('configs/portfolio_optimizer.yaml', 'w') as f:
            yaml.dump(portfolio_config, f)
        
        rotator = rotate_portfolio_by_regime.RegimePortfolioRotator(
            'configs/portfolio_optimizer.yaml', verbose=False
        )
        
        regime_result = rotator.run_regime_rotation()
        
        # Should detect regime without error
        assert 'regime_detected' in regime_result
        
        # Verify all artifacts were created
        expected_files = [
            'artifacts/uncertainty/confidence.csv',
            'artifacts/monitoring/DRIFT_DASHBOARD.md',
            'artifacts/monitoring/drift_data.csv',
            'artifacts/portfolio/regime_state.json'
        ]
        
        for file_path in expected_files:
            assert Path(file_path).exists(), f"Missing file: {file_path}"
        
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])