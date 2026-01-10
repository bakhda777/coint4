#!/usr/bin/env python3
"""Unified walk-forward analysis script."""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_walk_forward(
    config: str = "configs/main.yaml",
    pairs_file: str = "benchmarks/pairs_universe.yaml",
    resume: bool = False,
    with_portfolio: bool = False,
    scale: str = "medium",
    stub: bool = False,
    train_days: int = 120,
    test_days: int = 30,
    roll_step: int = None,
    n_windows: int = 6,
    output_dir: str = "outputs/wfa",
    **kwargs
):
    """Run walk-forward analysis with specified options.
    
    Options:
    - resume: Resume from last checkpoint
    - with_portfolio: Include portfolio optimization
    - scale: Scale of analysis (small/medium/large)
    - stub: Run stub version for testing
    - roll_step: Rolling step in days (default: test_days)
    """
    
    # Import here to avoid circular imports
    import pandas as pd
    from datetime import datetime
    
    # Set roll_step to test_days if not provided
    if roll_step is None:
        roll_step = test_days
    
    try:
        from src.coint2.pipeline.walk_forward_orchestrator import WalkForwardOrchestrator
        
        # Configure based on scale
        scale_configs = {
            'small': {
                'train_days': 60,
                'test_days': 15,
                'n_windows': 3
            },
            'medium': {
                'train_days': 120,
                'test_days': 30,
                'n_windows': 6
            },
            'large': {
                'train_days': 180,
                'test_days': 60,
                'n_windows': 12
            }
        }
        
        if scale in scale_configs:
            settings = scale_configs[scale]
            train_days = settings['train_days']
            test_days = settings['test_days']
            n_windows = settings['n_windows']
        
        print(f"🚀 Starting walk-forward analysis")
        print(f"   Config: {config}")
        print(f"   Pairs: {pairs_file}")
        print(f"   Scale: {scale} ({n_windows} windows)")
        print(f"   Train/Test/Roll: {train_days}/{test_days}/{roll_step} days")
        print(f"   Resume: {resume}")
        print(f"   Portfolio: {with_portfolio}")
        
        if stub:
            print("   ⚠️ Running in STUB mode (limited data)")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"wfa_{timestamp}"
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create orchestrator
        orchestrator = WalkForwardOrchestrator(
            config_path=config,
            pairs_file=pairs_file,
            train_days=train_days,
            test_days=test_days,
            n_windows=n_windows,
            output_dir=str(run_dir),
            resume=resume,
            with_portfolio=with_portfolio,
            stub_mode=stub,
            roll_step=roll_step
        )
        
        # Run analysis
        results = orchestrator.run()
        
        # Save summary reports
        summary_path = run_dir / "summary.csv"
        by_split_path = run_dir / "by_split.csv"
        
        # Create summary dataframe
        summary_data = {
            'metric': ['total_return', 'sharpe_ratio', 'num_trades', 'win_rate'],
            'value': [
                results.get('total_return', 0),
                results.get('sharpe_ratio', 0),
                results.get('num_trades', 0),
                results.get('win_rate', 0)
            ]
        }
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        
        # Save by_split data if available
        if 'splits' in results:
            pd.DataFrame(results['splits']).to_csv(by_split_path, index=False)
        
        print(f"✅ Walk-forward analysis complete")
        print(f"   Results saved to: {run_dir}")
        print(f"   Summary: {summary_path}")
        print(f"   By split: {by_split_path}")
        
        return 0
        
    except ImportError as e:
        print(f"⚠️ Module not found: {e}")
        print("Falling back to script execution...")
        
        # Fallback to script execution
        import subprocess
        
        script = "scripts/run_walk_forward.py"
        if resume:
            script = "scripts/run_walk_forward_with_resume.py"
        elif with_portfolio:
            script = "scripts/run_wfa_with_portfolio.py"
        elif stub:
            script = "scripts/run_walk_forward_stub.py"
        
        if not Path(script).exists():
            script = "scripts/run_walk_forward.py"
        
        cmd = [
            sys.executable, script,
            '--config', config,
            '--pairs-file', pairs_file,
            '--train-days', str(train_days),
            '--test-days', str(test_days),
            '--roll-step', str(roll_step),
            '--n-windows', str(n_windows),
            '--output-dir', output_dir
        ]
        
        if resume:
            cmd.append('--resume')
        if with_portfolio:
            cmd.append('--with-portfolio')
        
        return subprocess.call(cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified walk-forward analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard walk-forward
  %(prog)s --config configs/main.yaml
  
  # Resume with portfolio
  %(prog)s --resume --with-portfolio
  
  # Large scale analysis
  %(prog)s --scale large --n-windows 12
  
  # Quick test with stub
  %(prog)s --stub --scale small
"""
    )
    
    parser.add_argument('--config', default='configs/main.yaml',
                       help='Configuration file')
    
    parser.add_argument('--pairs-file', default='benchmarks/pairs_universe.yaml',
                       help='Pairs file for analysis')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    parser.add_argument('--with-portfolio', action='store_true',
                       help='Include portfolio optimization')
    
    parser.add_argument('--scale', choices=['small', 'medium', 'large'],
                       default='medium',
                       help='Scale of analysis')
    
    parser.add_argument('--stub', action='store_true',
                       help='Run stub version for testing')
    
    parser.add_argument('--train-days', type=int, default=120,
                       help='Training period days')
    
    parser.add_argument('--test-days', type=int, default=30,
                       help='Test period days')
    
    parser.add_argument('--roll-step', type=int, default=None,
                       help='Rolling step in days (default: test-days)')
    
    parser.add_argument('--n-windows', type=int, default=6,
                       help='Number of windows')
    
    parser.add_argument('--output-dir', '--out-dir', default='outputs/wfa',
                       help='Output directory')
    
    args = parser.parse_args(argv)
    
    # Set roll-step to test-days if not provided
    if args.roll_step is None:
        args.roll_step = args.test_days
    
    return run_walk_forward(**vars(args))


if __name__ == '__main__':
    sys.exit(main())