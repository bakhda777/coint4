#!/usr/bin/env python3
"""Unified optimization script with multiple modes and options."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_optimization(
    mode: str = "balanced",
    n_trials: int = 100,
    config: str = "configs/main.yaml",
    search_space: str = "configs/search_spaces/default.yaml",
    resume: bool = False,
    traces: bool = False,
    real_data: bool = True,
    parallel: bool = False,
    n_jobs: int = 4,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    **kwargs
):
    """Run optimization with specified mode and options.
    
    Modes:
    - fast: Quick optimization with fewer trials
    - balanced: Standard optimization (default)
    - strict: Conservative parameters, tighter filters
    - relaxed: Looser parameters, more pairs
    - parallel: Parallel optimization with multiple workers
    - iterative: Iterative refinement of parameters
    - large: Large-scale optimization with many trials
    """
    
    # Import here to avoid circular imports
    from src.optimiser.run_optimization import run_optimization as optuna_main
    
    # Configure based on mode
    mode_configs = {
        'fast': {
            'n_trials': min(n_trials, 50),
            'config': config,
            'search_space': 'configs/search_spaces/fast.yaml'
        },
        'strict': {
            'config': 'configs/base/main_strict.yaml' if Path('configs/base/main_strict.yaml').exists() else config,
            'search_space': 'configs/search_spaces/strict.yaml' if Path('configs/search_spaces/strict.yaml').exists() else search_space
        },
        'relaxed': {
            'config': 'configs/base/main_relaxed.yaml' if Path('configs/base/main_relaxed.yaml').exists() else config,
            'search_space': 'configs/search_spaces/relaxed.yaml' if Path('configs/search_spaces/relaxed.yaml').exists() else search_space
        },
        'large': {
            'n_trials': max(n_trials, 500),
            'n_jobs': max(n_jobs, 8) if parallel else n_jobs
        },
        'parallel': {
            'n_jobs': max(n_jobs, 4),
            'parallel': True
        },
        'iterative': {
            'iterative': True,
            'n_iterations': 5
        }
    }
    
    # Apply mode-specific settings
    if mode in mode_configs:
        settings = mode_configs[mode]
        for key, value in settings.items():
            if key == 'n_trials':
                n_trials = value
            elif key == 'config':
                config = value
            elif key == 'search_space':
                search_space = value
            elif key == 'n_jobs':
                n_jobs = value
            elif key == 'parallel':
                parallel = value
    
    print(f"🚀 Starting optimization")
    print(f"   Mode: {mode}")
    print(f"   Trials: {n_trials}")
    print(f"   Config: {config}")
    print(f"   Search space: {search_space}")
    print(f"   Resume: {resume}")
    print(f"   Parallel: {parallel} (jobs: {n_jobs})")
    
    # Build arguments for optuna
    args = [
        '--n-trials', str(n_trials),
        '--base-config', config,
        '--search-space', search_space
    ]
    
    if study_name:
        args.extend(['--study-name', study_name])
    
    if storage:
        args.extend(['--storage', storage])
    elif real_data:
        # Default storage for real data
        args.extend(['--storage', 'sqlite:///outputs/optuna/study.db'])
    
    if resume:
        args.append('--resume')
    
    if traces:
        args.append('--save-traces')
    
    if parallel:
        args.extend(['--n-jobs', str(n_jobs)])
    
    # Additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            args.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Run optimization
    try:
        # Call the actual optimization function directly with keyword arguments
        if not study_name:
            study_name = f"{mode}_optimization"
        
        if not storage:
            storage = f"outputs/studies/{study_name}.db"
        
        return optuna_main(
            n_trials=n_trials,
            study_name=study_name,
            storage_path=storage,
            base_config_path=config,
            search_space_path=search_space,
            n_jobs=n_jobs if parallel else 1,
            seed=42
        )
    except ImportError:
        # Fallback to script execution
        import subprocess
        cmd = [sys.executable, 'scripts/optimize.py'] + args
        return subprocess.call(cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified optimization script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  %(prog)s --mode fast --n-trials 25
  
  # Strict optimization with resume
  %(prog)s --mode strict --n-trials 200 --resume
  
  # Large parallel optimization
  %(prog)s --mode large --n-trials 1000 --parallel --n-jobs 8
  
  # Custom configuration
  %(prog)s --config configs/custom.yaml --search-space configs/search_custom.yaml
"""
    )
    
    parser.add_argument('--mode', 
                       choices=['fast', 'balanced', 'strict', 'relaxed', 
                               'parallel', 'iterative', 'large'],
                       default='balanced',
                       help='Optimization mode')
    
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    
    parser.add_argument('--config', default='configs/main.yaml',
                       help='Base configuration file')
    
    parser.add_argument('--search-space', default='configs/search_spaces/default.yaml',
                       help='Search space configuration')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing study')
    
    parser.add_argument('--traces', action='store_true',
                       help='Save optimization traces')
    
    parser.add_argument('--real-data', action='store_true', default=True,
                       help='Use real data (default: True)')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel optimization')
    
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel jobs')
    
    parser.add_argument('--study-name', 
                       help='Optuna study name')
    
    parser.add_argument('--storage',
                       help='Optuna storage URL')
    
    # Parse known args to allow additional passthrough
    args, unknown = parser.parse_known_args(argv)
    
    # Convert unknown args to kwargs
    kwargs = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:].replace('-', '_')
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                kwargs[key] = unknown[i + 1]
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    return run_optimization(**vars(args), **kwargs)


if __name__ == '__main__':
    sys.exit(main())