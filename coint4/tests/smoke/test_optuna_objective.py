#!/usr/bin/env python3
"""
Smoke tests for Optuna objective function - ensures basic functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

demo_objective = pytest.importorskip(
    "coint2.optuna.demo_objective",
    reason="Legacy Optuna demo objective not available."
)
DemoOptunaObjective = demo_objective.DemoOptunaObjective
import optuna
from optuna.exceptions import TrialPruned


@pytest.mark.smoke
def test_demo_objective_creation():
    """Test that demo objective can be created."""
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml",
        k_folds=2,  # Quick test
        save_traces=False
    )
    
    assert objective.k_folds == 2
    assert len(objective.pairs) >= 1  # At least fallback pairs


@pytest.mark.smoke 
def test_objective_returns_numeric():
    """Test that objective function returns a numeric value."""
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml", 
        k_folds=2,
        save_traces=False
    )
    
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    
    # Should complete successfully and return numeric PSR
    result = objective(trial)
    
    assert isinstance(result, (int, float))
    assert result > 0  # PSR should be positive for good parameters


@pytest.mark.smoke
def test_objective_prunes_invalid_params():
    """Test that invalid parameters get pruned."""
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml",
        k_folds=2,
        save_traces=False  
    )
    
    study = optuna.create_study(direction="maximize")
    
    # Create trial with invalid hysteresis (zscore_exit >= zscore_threshold)
    trial = study.ask()
    trial.suggest_float('zscore_threshold', 1.5, 1.5)  # Fixed low value
    trial.suggest_float('zscore_exit', 2.0, 2.0)       # Fixed high value (invalid)
    trial.suggest_int('rolling_window', 50, 50)
    trial.suggest_int('max_holding_days', 100, 100) 
    
    # Should raise TrialPruned for invalid hysteresis
    with pytest.raises(TrialPruned, match="Invalid band"):
        objective(trial)


@pytest.mark.smoke
def test_objective_with_minimal_folds():
    """Test objective works with minimal fold count."""
    objective = DemoOptunaObjective(
        pairs_file="bench/pairs_canary.yaml",
        k_folds=1,  # Minimal
        save_traces=False
    )
    
    study = optuna.create_study(direction="maximize") 
    trial = study.ask()
    
    result = objective(trial)
    
    assert isinstance(result, (int, float))
    assert result >= 0.1  # Should have some positive PSR
