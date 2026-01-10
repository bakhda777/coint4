"""Quick smoke test for universe selector."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


@pytest.mark.smoke
def test_selector_pipeline_smoke():
    """Test that selector can process mock data and produce outputs."""
    
    # Create mock price data
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
    
    # Generate cointegrated-like series
    np.random.seed(42)
    base = np.cumsum(np.random.randn(100))
    
    data = {}
    for i, sym in enumerate(symbols):
        noise = np.random.randn(100) * 0.1
        data[sym] = base + noise + i * 0.5
    
    df = pd.DataFrame(data, index=dates)
    
    # Mock config
    config = {
        'criteria': {
            'coint_pvalue_max': 0.1,
            'hl_min': 5,
            'hl_max': 200,
            'hurst_min': 0.2,
            'hurst_max': 0.6,
            'min_cross': 5,
            'beta_drift_max': 0.2
        }
    }
    
    # Test basic functionality without running full pipeline
    from src.coint2.pipeline.pair_scanner import (
        test_cointegration,
        evaluate_pair,
        calculate_pair_score
    )
    
    # Test cointegration
    y = df['BTC'].values
    x = df['ETH'].values
    
    result = test_cointegration(y, x, config)
    
    assert 'pvalue' in result
    assert 'half_life' in result
    assert 'hurst' in result
    assert 'crossings' in result
    assert 'beta_drift' in result
    
    # Test evaluation
    verdict = evaluate_pair(result, config)
    assert verdict in ['PASS', 'FAIL']
    
    # Test scoring
    score = calculate_pair_score(result, config)
    assert isinstance(score, float)
    
    # Test can write outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / 'test_universe.yaml'
        
        test_data = {
            'metadata': {'test': True},
            'pairs': [
                {
                    'pair': 'BTC/ETH',
                    'score': float(score),
                    'metrics': {k: float(v) if isinstance(v, (np.number, float)) else v 
                             for k, v in result.items()}
                }
            ]
        }
        
        with open(out_path, 'w') as f:
            yaml.dump(test_data, f)
        
        assert out_path.exists()
        
        # Verify can read back
        with open(out_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded['metadata']['test'] is True
        assert len(loaded['pairs']) == 1


@pytest.mark.smoke
def test_jobs_config_valid():
    """Test that universe_jobs.yaml is valid."""
    config_path = Path('configs/universe_jobs.yaml')
    
    if not config_path.exists():
        pytest.skip("universe_jobs.yaml not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'data' in config
    assert 'jobs' in config
    assert len(config['jobs']) > 0
    
    # Check first job structure
    job = config['jobs'][0]
    assert 'name' in job
    assert 'period' in job
    assert 'criteria' in job
    assert 'top_n' in job
    
    # Check criteria
    criteria = job['criteria']
    assert 'coint_pvalue_max' in criteria
    assert 'hl_min' in criteria
    assert 'hl_max' in criteria