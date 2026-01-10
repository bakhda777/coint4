"""Tests for parameter rollback system."""

import pytest
from pathlib import Path
import yaml
import json
from datetime import datetime
from unittest.mock import patch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestParameterRollback:
    """Test parameter rollback functionality."""
    
    @pytest.fixture
    def temp_rollback_structure(self, tmp_path):
        """Create temporary structure with multiple parameter versions."""
        # Create directories
        locked_dir = tmp_path / "artifacts/production/locked"
        locked_dir.mkdir(parents=True)
        
        # Create multiple versions
        versions = [
            ('2025W01', 1.2, 0.18),
            ('2025W02', 1.4, 0.15),
            ('2025W03', 1.6, 0.12),  # Best
            ('2025W04', 1.3, 0.20),  # Current but worse
        ]
        
        for week, sharpe, dd in versions:
            params = {
                'parameters': {
                    'zscore_threshold': 2.0 + (sharpe - 1.2) * 0.5,
                    'zscore_exit': 0.5,
                    'rolling_window': 60
                },
                'metadata': {
                    'sharpe': sharpe,
                    'max_dd': dd,
                    'week': week,
                    'promoted_at': datetime.now().isoformat()
                }
            }
            
            with open(locked_dir / f"params_BTCETH_H1_{week}.yaml", 'w') as f:
                yaml.dump(params, f)
        
        # Create current symlink pointing to W04
        current = locked_dir / "params_BTCETH_H1_current.yaml"
        if current.exists():
            current.unlink()
        current.symlink_to(locked_dir / "params_BTCETH_H1_2025W04.yaml")
        
        # Create catalog
        catalog = {
            'promotions': [
                {'week': week, 'sharpe': sharpe, 'max_dd': dd, 
                 'pair': 'BTCETH', 'timeframe': 'H1',
                 'timestamp': datetime.now().isoformat()}
                for week, sharpe, dd in versions
            ]
        }
        
        with open(tmp_path / "artifacts/production/PARAMS_CATALOG.json", 'w') as f:
            json.dump(catalog, f)
        
        return tmp_path
    
    def test_rollback_to_previous_week(self, temp_rollback_structure):
        """Test rolling back to previous week's parameters."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Rollback to W03
        success = rollback.rollback_to_week('BTCETH', 'H1', '2025W03')
        assert success == True
        
        # Check symlink points to W03
        current = temp_rollback_structure / "artifacts/production/locked/params_BTCETH_H1_current.yaml"
        assert current.is_symlink()
        target = current.resolve()
        assert "2025W03" in str(target)
    
    def test_rollback_to_best_performance(self, temp_rollback_structure):
        """Test rolling back to best performing parameters."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Find and rollback to best
        best_week = rollback.find_best_week('BTCETH', 'H1')
        assert best_week == '2025W03'  # Has highest Sharpe
        
        success = rollback.rollback_to_best('BTCETH', 'H1')
        assert success == True
        
        # Verify symlink
        current = temp_rollback_structure / "artifacts/production/locked/params_BTCETH_H1_current.yaml"
        target = current.resolve()
        assert "2025W03" in str(target)
    
    def test_rollback_updates_catalog(self, temp_rollback_structure):
        """Test that rollback updates the catalog."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Rollback
        rollback.rollback_to_week('BTCETH', 'H1', '2025W02')
        
        # Check catalog has rollback entry
        catalog_file = temp_rollback_structure / "artifacts/production/PARAMS_CATALOG.json"
        with open(catalog_file) as f:
            catalog = json.load(f)
        
        # Should have rollback entry
        assert 'rollbacks' in catalog
        assert len(catalog['rollbacks']) > 0
        assert catalog['rollbacks'][-1]['to_week'] == '2025W02'
        assert catalog['rollbacks'][-1]['from_week'] == '2025W04'
    
    def test_rollback_with_missing_week(self, temp_rollback_structure):
        """Test rollback fails gracefully with missing week."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Try to rollback to non-existent week
        success = rollback.rollback_to_week('BTCETH', 'H1', '2025W99')
        assert success == False
    
    def test_rollback_history(self, temp_rollback_structure):
        """Test viewing rollback history."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Perform multiple rollbacks
        rollback.rollback_to_week('BTCETH', 'H1', '2025W03')
        rollback.rollback_to_week('BTCETH', 'H1', '2025W02')
        rollback.rollback_to_best('BTCETH', 'H1')
        
        # Get history
        history = rollback.get_rollback_history('BTCETH', 'H1')
        assert len(history) == 3
        assert history[0]['to_week'] == '2025W03'
        assert history[1]['to_week'] == '2025W02'
        assert history[2]['to_week'] == '2025W03'  # Best
    
    def test_atomic_rollback(self, temp_rollback_structure):
        """Test that rollback is atomic."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Save current target
        current = temp_rollback_structure / "artifacts/production/locked/params_BTCETH_H1_current.yaml"
        original_target = current.resolve()
        
        # Simulate failure during rollback
        with patch('pathlib.Path.unlink', side_effect=IOError("Simulated failure")):
            success = rollback.rollback_to_week('BTCETH', 'H1', '2025W02')
            assert success == False
        
        # Symlink should still point to original
        assert current.resolve() == original_target
    
    def test_compare_weeks(self, temp_rollback_structure):
        """Test comparing performance between weeks."""
        from scripts.rollback_params import ParameterRollback
        rollback = ParameterRollback(base_dir=str(temp_rollback_structure))
        
        # Compare two weeks
        comparison = rollback.compare_weeks('BTCETH', 'H1', '2025W03', '2025W04')
        
        assert comparison['week1'] == '2025W03'
        assert comparison['week2'] == '2025W04'
        assert comparison['sharpe_diff'] == 0.3  # 1.6 - 1.3
        assert comparison['dd_diff'] == -0.08  # 0.12 - 0.20
        assert comparison['better'] == '2025W03'