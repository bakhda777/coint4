#!/usr/bin/env python3
"""Tests for WFA checkpoint/resume functionality."""

import json
import tempfile
from pathlib import Path
import time
import pytest
import sys
sys.path.insert(0, 'src')


def test_wfa_checkpoint_save(tmp_path):
    """Test that WFA saves checkpoints correctly."""
    from scripts.run_walk_forward_with_resume import save_state
    
    state_path = tmp_path / "state.json"
    test_state = {
        'completed_folds': [0, 1],
        'last_completed_fold': 1,
        'elapsed_seconds': 120.5,
        'results': [
            {'fold': 0, 'sharpe': 1.2, 'pnl': 1000},
            {'fold': 1, 'sharpe': 1.5, 'pnl': 1200}
        ]
    }
    
    # Save state
    save_state(str(state_path), test_state)
    
    # Verify file exists
    assert state_path.exists()
    
    # Load and verify contents
    with open(state_path, 'r') as f:
        loaded_state = json.load(f)
    
    assert loaded_state == test_state
    assert len(loaded_state['completed_folds']) == 2
    assert loaded_state['last_completed_fold'] == 1


def test_wfa_resume_from_checkpoint(tmp_path):
    """Test resuming WFA from saved checkpoint."""
    from scripts.run_walk_forward_with_resume import run_wfa, save_state
    
    state_path = tmp_path / "state.json"
    
    # Create initial state (simulating interruption after fold 2)
    initial_state = {
        'completed_folds': [0, 1, 2],
        'last_completed_fold': 2,
        'elapsed_seconds': 180.0,
        'results': [
            {'fold': 0, 'sharpe': 1.2, 'pnl': 1000, 'trades': 25},
            {'fold': 1, 'sharpe': 1.5, 'pnl': 1200, 'trades': 30},
            {'fold': 2, 'sharpe': 1.3, 'pnl': 1100, 'trades': 28}
        ]
    }
    save_state(str(state_path), initial_state)
    
    # Resume WFA
    final_state = run_wfa(resume=True, state_path=str(state_path))
    
    # Verify completion
    assert len(final_state['completed_folds']) == 5
    assert final_state['last_completed_fold'] == 4
    assert len(final_state['results']) == 5
    
    # Verify initial results preserved
    assert final_state['results'][0]['fold'] == 0
    assert final_state['results'][1]['fold'] == 1
    assert final_state['results'][2]['fold'] == 2
    
    # Verify new results added
    assert final_state['results'][3]['fold'] == 3
    assert final_state['results'][4]['fold'] == 4


def test_wfa_interrupt_simulation(tmp_path):
    """Simulate WFA interruption and successful resume."""
    from scripts.run_walk_forward_with_resume import save_state, load_state, run_wfa
    
    state_path = tmp_path / "state.json"
    
    # Simulate partial run (interrupted after 2 folds)
    interrupted_state = {
        'completed_folds': [0, 1],
        'last_completed_fold': 1,
        'elapsed_seconds': 120.0,
        'results': [
            {'fold': 0, 'sharpe': 1.1, 'pnl': 950, 'trades': 20},
            {'fold': 1, 'sharpe': 1.4, 'pnl': 1150, 'trades': 27}
        ]
    }
    save_state(str(state_path), interrupted_state)
    
    # Simulate restart and resume
    loaded_state = load_state(str(state_path))
    assert loaded_state is not None
    assert loaded_state['last_completed_fold'] == 1
    
    # Complete the run
    final_state = run_wfa(resume=True, state_path=str(state_path))
    
    # Verify full completion
    assert final_state['last_completed_fold'] == 4
    assert len(final_state['results']) == 5
    
    # Check results file
    results_path = Path("artifacts/wfa/results_per_fold.csv")
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)
        assert len(df) == 5
        assert list(df['fold']) == [0, 1, 2, 3, 4]


def test_wfa_fresh_start(tmp_path):
    """Test WFA fresh start without resume."""
    from scripts.run_walk_forward_with_resume import run_wfa
    
    state_path = tmp_path / "fresh_state.json"
    
    # Run fresh WFA
    state = run_wfa(resume=False, state_path=str(state_path))
    
    # Verify complete run
    assert len(state['completed_folds']) == 5
    assert state['last_completed_fold'] == 4
    assert len(state['results']) == 5
    
    # Verify all folds processed
    for i in range(5):
        assert i in state['completed_folds']
        assert state['results'][i]['fold'] == i


def test_atomic_state_save(tmp_path):
    """Test atomic file operations for state saving."""
    from scripts.run_walk_forward_with_resume import save_state
    
    state_path = tmp_path / "atomic_state.json"
    
    # Save multiple times rapidly
    for i in range(10):
        state = {
            'iteration': i,
            'completed_folds': list(range(i + 1)),
            'last_completed_fold': i,
            'elapsed_seconds': i * 10.5,
            'results': [{'fold': j, 'value': j * 100} for j in range(i + 1)]
        }
        save_state(str(state_path), state)
    
    # Verify final state is consistent
    with open(state_path, 'r') as f:
        final_state = json.load(f)
    
    assert final_state['iteration'] == 9
    assert len(final_state['completed_folds']) == 10
    assert final_state['last_completed_fold'] == 9
    
    # Verify no temp files left
    temp_files = list(tmp_path.glob("*.tmp"))
    assert len(temp_files) == 0


@pytest.mark.parametrize("interrupt_after", [0, 1, 2, 3])
def test_wfa_multiple_interrupts(tmp_path, interrupt_after):
    """Test WFA with interrupts at different points."""
    from scripts.run_walk_forward_with_resume import save_state, run_wfa
    
    state_path = tmp_path / f"interrupt_{interrupt_after}.json"
    
    # Create interrupted state
    if interrupt_after > 0:
        interrupted_state = {
            'completed_folds': list(range(interrupt_after)),
            'last_completed_fold': interrupt_after - 1,
            'elapsed_seconds': interrupt_after * 60.0,
            'results': [
                {'fold': i, 'sharpe': 1.0 + i * 0.1, 'pnl': 1000 + i * 100, 'trades': 20 + i * 5}
                for i in range(interrupt_after)
            ]
        }
        save_state(str(state_path), interrupted_state)
        
        # Resume from interrupt
        final_state = run_wfa(resume=True, state_path=str(state_path))
    else:
        # Fresh start
        final_state = run_wfa(resume=False, state_path=str(state_path))
    
    # Verify completion regardless of interrupt point
    assert len(final_state['completed_folds']) == 5
    assert final_state['last_completed_fold'] == 4
    assert all(i in final_state['completed_folds'] for i in range(5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])