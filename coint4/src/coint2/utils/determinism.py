"""
Determinism utilities for reproducible results.
"""

import random
import numpy as np
import os
from typing import Optional


def set_global_seed(seed: int = 42):
    """Set seeds for all random number generators.
    
    Args:
        seed: Random seed to use
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Log seed setting
    print(f"🎲 Global seed set to: {seed}")
    
    return seed


def get_optuna_sampler(seed: int = 42):
    """Get deterministic Optuna sampler.
    
    Args:
        seed: Random seed for sampler
        
    Returns:
        TPESampler with fixed seed
    """
    try:
        import optuna
        return optuna.samplers.TPESampler(seed=seed)
    except ImportError:
        return None


def log_seed_to_artifact(seed: int, artifact_path: str):
    """Log seed to artifact file.
    
    Args:
        seed: Seed value used
        artifact_path: Path to artifact file
    """
    import json
    from pathlib import Path
    
    path = Path(artifact_path)
    
    # Read existing or create new
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {}
    
    # Add seed info
    data['determinism'] = {
        'seed': seed,
        'timestamp': str(Path.cwd()),
        'numpy_version': np.__version__,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
    }
    
    # Write back
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_reproducibility(run_func, seed: int = 42, tolerance: float = 1e-6):
    """Validate that a function produces reproducible results.
    
    Args:
        run_func: Function to test
        seed: Seed to use
        tolerance: Numerical tolerance for comparison
        
    Returns:
        bool: True if results are reproducible
    """
    # First run
    set_global_seed(seed)
    result1 = run_func()
    
    # Second run with same seed
    set_global_seed(seed)
    result2 = run_func()
    
    # Compare results
    if isinstance(result1, (int, float)):
        return abs(result1 - result2) < tolerance
    elif isinstance(result1, np.ndarray):
        return np.allclose(result1, result2, atol=tolerance)
    elif isinstance(result1, dict):
        for key in result1:
            if isinstance(result1[key], (int, float)):
                if abs(result1[key] - result2[key]) > tolerance:
                    return False
        return True
    else:
        return result1 == result2


class DeterministicContext:
    """Context manager for deterministic execution."""
    
    def __init__(self, seed: int = 42):
        """Initialize deterministic context.
        
        Args:
            seed: Random seed to use
        """
        self.seed = seed
        self.old_python_seed = None
        self.old_numpy_seed = None
        
    def __enter__(self):
        """Enter deterministic context."""
        # Save current state
        self.old_python_seed = random.getstate()
        self.old_numpy_seed = np.random.get_state()
        
        # Set deterministic seeds
        set_global_seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic context and restore previous state."""
        # Restore previous state
        if self.old_python_seed:
            random.setstate(self.old_python_seed)
        if self.old_numpy_seed:
            np.random.set_state(self.old_numpy_seed)