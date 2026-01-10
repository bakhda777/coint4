"""Tests for determinism and repeatability."""

import pytest
import numpy as np
import random
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coint2.utils.determinism import (
    set_global_seed,
    get_optuna_sampler,
    log_seed_to_artifact,
    validate_reproducibility,
    DeterministicContext
)


class TestDeterminism:
    """Test deterministic execution utilities."""
    
    def test_set_global_seed(self):
        """Test that global seed sets all RNGs."""
        # Set seed
        set_global_seed(42)
        
        # Python random
        py_val1 = random.random()
        
        # NumPy random
        np_val1 = np.random.randn()
        
        # Reset seed
        set_global_seed(42)
        
        # Should get same values
        py_val2 = random.random()
        np_val2 = np.random.randn()
        
        assert py_val1 == py_val2
        assert np_val1 == np_val2
    
    def test_optuna_sampler_determinism(self):
        """Test that Optuna sampler is deterministic."""
        # Mock optuna import
        with patch('coint2.utils.determinism.optuna') as mock_optuna:
            mock_tpe = MagicMock()
            mock_optuna.samplers.TPESampler.return_value = mock_tpe
            
            # Get sampler with seed
            sampler1 = get_optuna_sampler(seed=123)
            
            # Should have called with seed
            mock_optuna.samplers.TPESampler.assert_called_with(seed=123)
            assert sampler1 == mock_tpe
    
    def test_log_seed_to_artifact(self, tmp_path):
        """Test logging seed to artifact file."""
        artifact_path = tmp_path / "test_artifact.json"
        
        # Log seed
        log_seed_to_artifact(42, str(artifact_path))
        
        # Check file exists and contains seed
        assert artifact_path.exists()
        
        import json
        with open(artifact_path) as f:
            data = json.load(f)
        
        assert data['determinism']['seed'] == 42
        assert 'numpy_version' in data['determinism']
        assert 'python_version' in data['determinism']
    
    def test_validate_reproducibility_scalar(self):
        """Test reproducibility validation for scalar functions."""
        def test_func():
            return random.random() + np.random.randn()
        
        # Should be reproducible with same seed
        is_reproducible = validate_reproducibility(test_func, seed=42)
        assert is_reproducible == True
    
    def test_validate_reproducibility_array(self):
        """Test reproducibility validation for array functions."""
        def test_func():
            return np.random.randn(10, 5)
        
        # Should be reproducible
        is_reproducible = validate_reproducibility(test_func, seed=42)
        assert is_reproducible == True
    
    def test_validate_reproducibility_dict(self):
        """Test reproducibility validation for dict functions."""
        def test_func():
            return {
                'metric1': random.random(),
                'metric2': np.random.randn(),
                'metric3': np.random.randint(0, 100)
            }
        
        # Should be reproducible
        is_reproducible = validate_reproducibility(test_func, seed=42)
        assert is_reproducible == True
    
    def test_deterministic_context_manager(self):
        """Test DeterministicContext context manager."""
        # Get initial random values
        initial_py = random.random()
        initial_np = np.random.randn()
        
        # Use deterministic context
        with DeterministicContext(seed=42):
            det_py1 = random.random()
            det_np1 = np.random.randn()
        
        # Use same context again
        with DeterministicContext(seed=42):
            det_py2 = random.random()
            det_np2 = np.random.randn()
        
        # Values in context should match
        assert det_py1 == det_py2
        assert det_np1 == det_np2
        
        # After context, should get different values
        post_py = random.random()
        post_np = np.random.randn()
        
        assert post_py != det_py1
        assert post_np != det_np1
    
    def test_nested_deterministic_contexts(self):
        """Test nested deterministic contexts."""
        values = []
        
        with DeterministicContext(seed=42):
            values.append(random.random())
            
            with DeterministicContext(seed=123):
                values.append(random.random())
            
            values.append(random.random())
        
        # Repeat with same seeds
        values2 = []
        
        with DeterministicContext(seed=42):
            values2.append(random.random())
            
            with DeterministicContext(seed=123):
                values2.append(random.random())
            
            values2.append(random.random())
        
        # All values should match
        assert values == values2
    
    def test_determinism_with_numpy_operations(self):
        """Test determinism with complex NumPy operations."""
        def complex_operation():
            # Generate random data
            data = np.random.randn(100, 50)
            
            # Apply transformations
            normalized = (data - data.mean()) / data.std()
            cov = np.cov(normalized.T)
            eigenvalues = np.linalg.eigvals(cov)
            
            return {
                'mean': normalized.mean(),
                'std': normalized.std(),
                'max_eigenvalue': np.max(eigenvalues.real)
            }
        
        # Test reproducibility
        is_reproducible = validate_reproducibility(
            complex_operation, 
            seed=42, 
            tolerance=1e-10
        )
        assert is_reproducible == True
    
    def test_determinism_persistence(self, tmp_path):
        """Test that seed persists across function calls."""
        set_global_seed(42)
        
        # Generate sequence
        sequence1 = [random.random() for _ in range(5)]
        
        # Reset and regenerate
        set_global_seed(42)
        sequence2 = [random.random() for _ in range(5)]
        
        assert sequence1 == sequence2
    
    def test_determinism_with_shuffling(self):
        """Test determinism with data shuffling."""
        def shuffle_operation():
            data = list(range(100))
            random.shuffle(data)
            return data[:10]  # Return first 10
        
        # Test reproducibility
        set_global_seed(42)
        result1 = shuffle_operation()
        
        set_global_seed(42)
        result2 = shuffle_operation()
        
        assert result1 == result2