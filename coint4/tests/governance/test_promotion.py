"""Tests for parameter promotion system."""

import pytest
from pathlib import Path
import yaml
import json
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestParameterPromotion:
    """Test parameter promotion functionality."""
    
    @pytest.fixture
    def temp_structure(self, tmp_path):
        """Create temporary directory structure."""
        # Create directories
        (tmp_path / "artifacts/wfa").mkdir(parents=True)
        (tmp_path / "artifacts/optuna").mkdir(parents=True)
        (tmp_path / "artifacts/production/locked").mkdir(parents=True)
        
        # Create sample WFA results
        wfa_results = {
            'sharpe': 1.45,
            'max_dd': 0.15,
            'trades': 150,
            'psr': 2.8
        }
        with open(tmp_path / "artifacts/wfa/performance_summary.json", 'w') as f:
            json.dump(wfa_results, f)
        
        # Create sample Optuna results
        optuna_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'rolling_window': 60
        }
        with open(tmp_path / "artifacts/optuna/best_params.json", 'w') as f:
            json.dump(optuna_params, f)
        
        return tmp_path
    
    def test_promotion_validation(self, temp_structure):
        """Test parameter validation before promotion."""
        # Mock the promoter with temp paths
        with patch('scripts.promote_params.Path') as mock_path:
            mock_path.return_value = temp_structure
            
            from scripts.promote_params import ParameterPromoter
            promoter = ParameterPromoter(base_dir=str(temp_structure))
            
            # Validate should pass with good metrics
            validation = promoter.validate_params()
            assert validation['passed'] == True
            assert validation['sharpe'] == 1.45
            assert validation['max_dd'] == 0.15
    
    def test_promotion_creates_locked_config(self, temp_structure):
        """Test that promotion creates locked config file."""
        from scripts.promote_params import ParameterPromoter
        promoter = ParameterPromoter(base_dir=str(temp_structure))
        
        # Promote parameters
        success = promoter.promote(pair='BTCETH', timeframe='H1', week='2025W02')
        assert success == True
        
        # Check locked file exists
        locked_file = temp_structure / "artifacts/production/locked/params_BTCETH_H1_2025W02.yaml"
        assert locked_file.exists()
        
        # Verify content
        with open(locked_file) as f:
            locked_data = yaml.safe_load(f)
        
        assert locked_data['parameters']['zscore_threshold'] == 2.0
        assert locked_data['metadata']['sharpe'] == 1.45
        assert locked_data['metadata']['promoted_at'] is not None
    
    def test_promotion_updates_catalog(self, temp_structure):
        """Test that promotion updates the catalog."""
        from scripts.promote_params import ParameterPromoter
        promoter = ParameterPromoter(base_dir=str(temp_structure))
        
        # Promote parameters
        promoter.promote(pair='BTCETH', timeframe='H1', week='2025W02')
        
        # Check catalog
        catalog_file = temp_structure / "artifacts/production/PARAMS_CATALOG.json"
        assert catalog_file.exists()
        
        with open(catalog_file) as f:
            catalog = json.load(f)
        
        assert len(catalog['promotions']) == 1
        assert catalog['promotions'][0]['pair'] == 'BTCETH'
        assert catalog['promotions'][0]['week'] == '2025W02'
    
    def test_promotion_fails_with_low_sharpe(self, temp_structure):
        """Test that promotion fails with low Sharpe ratio."""
        # Update WFA results with low Sharpe
        wfa_results = {
            'sharpe': 0.5,  # Below threshold
            'max_dd': 0.15,
            'trades': 150,
            'psr': 1.0
        }
        with open(temp_structure / "artifacts/wfa/performance_summary.json", 'w') as f:
            json.dump(wfa_results, f)
        
        from scripts.promote_params import ParameterPromoter
        promoter = ParameterPromoter(base_dir=str(temp_structure))
        
        # Validation should fail
        validation = promoter.validate_params()
        assert validation['passed'] == False
        assert 'Sharpe too low' in validation['reason']
        
        # Promotion should fail
        success = promoter.promote(pair='BTCETH', timeframe='H1')
        assert success == False
    
    def test_promotion_with_existing_params(self, temp_structure):
        """Test promotion when parameters already exist for the week."""
        from scripts.promote_params import ParameterPromoter
        promoter = ParameterPromoter(base_dir=str(temp_structure))
        
        # First promotion
        success1 = promoter.promote(pair='BTCETH', timeframe='H1', week='2025W02')
        assert success1 == True
        
        # Second promotion for same week should update
        # Update metrics first
        wfa_results = {
            'sharpe': 1.6,  # Better Sharpe
            'max_dd': 0.12,
            'trades': 160,
            'psr': 3.0
        }
        with open(temp_structure / "artifacts/wfa/performance_summary.json", 'w') as f:
            json.dump(wfa_results, f)
        
        success2 = promoter.promote(pair='BTCETH', timeframe='H1', week='2025W02')
        assert success2 == True
        
        # Check that file was updated
        locked_file = temp_structure / "artifacts/production/locked/params_BTCETH_H1_2025W02.yaml"
        with open(locked_file) as f:
            locked_data = yaml.safe_load(f)
        
        assert locked_data['metadata']['sharpe'] == 1.6
    
    def test_atomic_promotion(self, temp_structure):
        """Test that promotion is atomic (all or nothing)."""
        from scripts.promote_params import ParameterPromoter
        promoter = ParameterPromoter(base_dir=str(temp_structure))
        
        # Simulate failure during catalog update
        with patch('builtins.open', side_effect=IOError("Simulated failure")):
            success = promoter.promote(pair='BTCETH', timeframe='H1', week='2025W02')
            assert success == False
        
        # Check that no partial files were created
        locked_file = temp_structure / "artifacts/production/locked/params_BTCETH_H1_2025W02.yaml"
        assert not locked_file.exists()