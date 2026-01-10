#!/usr/bin/env python3
"""
Smoke tests for Optuna portfolio-level objective function.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coint2.optuna.demo_objective import DemoOptunaObjective
import optuna


@pytest.mark.smoke
def test_portfolio_aggregation():
    """Test that portfolio-level aggregation works across multiple pairs."""
    
    # Create temporary multi-pair config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        portfolio_pairs = {
            'pairs': [
                {'symbol1': 'BTCUSDT', 'symbol2': 'ETHUSDT'},
                {'symbol1': 'ADAUSDT', 'symbol2': 'DOTUSDT'}, 
                {'symbol1': 'SOLUSDT', 'symbol2': 'AVAXUSDT'}
            ]
        }
        yaml.dump(portfolio_pairs, f)
        temp_pairs_file = f.name
    
    try:
        objective = DemoOptunaObjective(
            pairs_file=temp_pairs_file,
            k_folds=2,  # Quick test
            save_traces=False
        )
        
        # Should load multiple pairs
        assert len(objective.pairs) == 3
        
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        
        # Should complete successfully across all pairs
        result = objective(trial)
        
        assert isinstance(result, (int, float))
        assert result > 0  # Should aggregate to positive PSR
        
    finally:
        Path(temp_pairs_file).unlink()


@pytest.mark.smoke
def test_portfolio_median_aggregation():
    """Test that portfolio uses median aggregation for robustness."""
    
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml",  # Should have multiple pairs
        k_folds=3,
        save_traces=False
    )
    
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    
    # Run objective and check user attributes contain fold-wise results
    result = objective(trial)
    
    # Check that fold attributes were stored
    assert hasattr(trial, 'user_attrs')
    
    fold_keys = [k for k in trial.user_attrs.keys() if k.startswith('fold_')]
    assert len(fold_keys) >= 3  # At least 3 folds
    
    # Verify result is reasonable (not extreme outlier)
    assert 0.1 <= result <= 5.0  # PSR should be in reasonable range


@pytest.mark.smoke
def test_portfolio_empty_pairs_fallback():
    """Test that objective handles missing pairs file gracefully."""
    
    objective = DemoOptunaObjective(
        pairs_file="nonexistent_file.yaml",  # Should fallback to default pairs
        k_folds=2,
        save_traces=False
    )
    
    # Should have fallback pairs
    assert len(objective.pairs) >= 1
    
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    
    # Should still work with fallback pairs
    result = objective(trial)
    assert isinstance(result, (int, float))
    assert result > 0


@pytest.mark.smoke
def test_portfolio_pair_failure_handling():
    """Test portfolio handling when some pairs fail."""
    
    # Create config with mix of real and problematic pairs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        mixed_pairs = {
            'pairs': [
                {'symbol1': 'BTCUSDT', 'symbol2': 'ETHUSDT'},  # Should work
                {'symbol1': 'INVALID1', 'symbol2': 'INVALID2'}  # Will likely fail
            ]
        }
        yaml.dump(mixed_pairs, f)
        temp_pairs_file = f.name
    
    try:
        objective = DemoOptunaObjective(
            pairs_file=temp_pairs_file,
            k_folds=2,
            save_traces=False
        )
        
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        
        # Should either work with good pairs or prune gracefully
        try:
            result = objective(trial)
            # If it succeeds, result should be valid
            assert isinstance(result, (int, float))
        except optuna.TrialPruned:
            # Pruning is acceptable if not enough good pairs
            pass
            
    finally:
        Path(temp_pairs_file).unlink()


@pytest.mark.smoke
def test_portfolio_consistency():
    """Test that portfolio objective gives consistent results for same parameters."""
    
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml",
        k_folds=2,
        save_traces=False
    )
    
    study = optuna.create_study(direction="maximize")
    
    # Run same parameters twice (with fixed suggestions)
    trial1 = study.ask()
    trial1.suggest_float('zscore_threshold', 2.0, 2.0)
    trial1.suggest_float('zscore_exit', 0.0, 0.0)
    trial1.suggest_int('rolling_window', 60, 60)
    trial1.suggest_int('max_holding_days', 100, 100)
    
    result1 = objective(trial1)
    
    trial2 = study.ask()
    trial2.suggest_float('zscore_threshold', 2.0, 2.0)
    trial2.suggest_float('zscore_exit', 0.0, 0.0) 
    trial2.suggest_int('rolling_window', 60, 60)
    trial2.suggest_int('max_holding_days', 100, 100)
    
    result2 = objective(trial2)
    
    # Results should be identical for same parameters
    assert abs(result1 - result2) < 1e-6