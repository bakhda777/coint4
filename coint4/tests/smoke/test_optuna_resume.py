#!/usr/bin/env python3
"""
Smoke tests for Optuna resume functionality.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coint2.optuna.demo_objective import DemoOptunaObjective
import optuna


@pytest.mark.smoke
def test_optuna_resume_functionality():
    """Test that Optuna studies can be created, resumed, and extended."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup temporary database
        db_path = Path(tmp_dir) / "test_study.db"
        storage = f"sqlite:///{db_path}"
        study_name = "test_resume_study"
        
        objective = DemoOptunaObjective(
            pairs_file="bench/pairs_canary.yaml",
            k_folds=2,
            save_traces=False
        )
        
        # Phase 1: Create new study and run 3 trials
        study1 = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize"
        )
        
        study1.optimize(objective, n_trials=3)
        
        initial_trials = len(study1.trials)
        assert initial_trials == 3
        assert db_path.exists()
        
        # Phase 2: Load existing study and run 2 more trials
        study2 = optuna.load_study(
            study_name=study_name, 
            storage=storage
        )
        
        # Verify we loaded the same study
        assert len(study2.trials) == initial_trials
        
        # Add more trials
        study2.optimize(objective, n_trials=2)
        
        # Verify total trials
        final_trials = len(study2.trials)
        assert final_trials == 5
        
        # Verify database persistence
        study3 = optuna.load_study(study_name=study_name, storage=storage)
        assert len(study3.trials) == final_trials


@pytest.mark.smoke
def test_study_database_structure():
    """Test that study database has expected structure."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_structure.db"
        storage = f"sqlite:///{db_path}"
        study_name = "test_structure_study"
        
        objective = DemoOptunaObjective(
            pairs_file="bench/pairs_canary.yaml",
            k_folds=2,
            save_traces=False
        )
        
        # Create and run study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage, 
            direction="maximize"
        )
        
        study.optimize(objective, n_trials=2)
        
        # Check database structure
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check that expected tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['studies', 'trials', 'trial_params', 'trial_user_attributes']
        for table in expected_tables:
            assert table in tables, f"Missing table: {table}"
        
        # Check we have trials with expected data
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        assert trial_count == 2
        
        # Check user attributes exist (fold metrics)
        cursor.execute("SELECT COUNT(*) FROM trial_user_attributes WHERE key LIKE 'fold_%'")
        fold_attrs = cursor.fetchone()[0]
        assert fold_attrs > 0, "No fold attributes found"
        
        conn.close()


@pytest.mark.smoke  
def test_resume_with_different_objective():
    """Test that resuming with different objective handles gracefully."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_different_obj.db" 
        storage = f"sqlite:///{db_path}"
        study_name = "test_different_obj_study"
        
        # First objective - 2 folds
        objective1 = DemoOptunaObjective(
            pairs_file="bench/pairs_canary.yaml",
            k_folds=2,
            save_traces=False
        )
        
        study1 = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize"  
        )
        
        study1.optimize(objective1, n_trials=2)
        initial_count = len(study1.trials)
        
        # Second objective - 3 folds (different)
        objective2 = DemoOptunaObjective(
            pairs_file="bench/pairs_canary.yaml", 
            k_folds=3,  # Different k_folds
            save_traces=False
        )
        
        # Resume study with different objective
        study2 = optuna.load_study(study_name=study_name, storage=storage)
        study2.optimize(objective2, n_trials=1)
        
        # Should still work and add trials
        assert len(study2.trials) == initial_count + 1