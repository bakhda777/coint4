#!/usr/bin/env python3
"""
Parameter promotion system for moving validated parameters to production.
Tracks history and enables rollback.
"""

import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class ParameterPromoter:
    """Manages parameter promotion from research to production."""
    
    def __init__(self, min_sharpe: float = 1.0, min_psr: float = 0.95, min_trades: int = 20):
        """Initialize promoter with validation thresholds."""
        self.min_sharpe = min_sharpe
        self.min_psr = min_psr
        self.min_trades = min_trades
        self.locked_dir = Path('configs/locked')
        self.current_dir = self.locked_dir / 'current'
        self.catalog_path = Path('artifacts/governance/PARAMS_CATALOG.md')
        
        # Create directories
        self.locked_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        
    def get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return 'unknown'
    
    def validate_params(self, source: str = 'wfa') -> Dict:
        """Validate parameters meet promotion criteria."""
        validation = {
            'passed': False,
            'sharpe': 0.0,
            'psr': 0.0,
            'trades': 0,
            'stability': 0.0,
            'errors': []
        }
        
        # Read WFA results
        wfa_results = Path('artifacts/wfa/results_per_fold.csv')
        if not wfa_results.exists():
            validation['errors'].append(f"WFA results not found: {wfa_results}")
            return validation
        
        df = pd.read_csv(wfa_results)
        
        # Calculate metrics
        validation['sharpe'] = df['sharpe'].mean() if 'sharpe' in df else 0
        validation['trades'] = df['trades'].sum() if 'trades' in df else 0
        
        # Calculate PSR (simplified)
        if 'sharpe' in df and len(df) > 1:
            sharpe_std = df['sharpe'].std()
            n_samples = len(df)
            validation['psr'] = validation['sharpe'] / (sharpe_std + 1e-8) * np.sqrt(n_samples)
        else:
            validation['psr'] = validation['sharpe']
        
        # Check stability
        stability_file = Path('artifacts/wfa/params_stability.csv')
        if stability_file.exists():
            stab_df = pd.read_csv(stability_file)
            if 'std' in stab_df:
                validation['stability'] = stab_df['std'].mean()
        
        # Validation checks
        if validation['sharpe'] < self.min_sharpe:
            validation['errors'].append(f"Sharpe {validation['sharpe']:.2f} < {self.min_sharpe}")
        
        if validation['psr'] < self.min_psr:
            validation['errors'].append(f"PSR {validation['psr']:.2f} < {self.min_psr}")
        
        if validation['trades'] < self.min_trades:
            validation['errors'].append(f"Trades {validation['trades']} < {self.min_trades}")
        
        validation['passed'] = len(validation['errors']) == 0
        
        return validation
    
    def load_best_params(self) -> Dict:
        """Load best parameters from Optuna or WFA."""
        # Try Optuna first
        optuna_params = Path('artifacts/optuna/best_params.json')
        if optuna_params.exists():
            with open(optuna_params) as f:
                return json.load(f)
        
        # Fallback to default
        return {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'rolling_window': 60,
            'max_holding_days': 100
        }
    
    def promote(self, pair: str = 'BTCETH', timeframe: str = 'H1', 
                week: Optional[str] = None) -> bool:
        """Promote parameters to production."""
        
        # Validate parameters
        validation = self.validate_params()
        if not validation['passed']:
            print(f"❌ Validation failed:")
            for error in validation['errors']:
                print(f"   {error}")
            return False
        
        # Load parameters
        params = self.load_best_params()
        
        # Add metadata
        if week is None:
            week = datetime.now().strftime('%Y_W%U')
        
        git_hash = self.get_git_hash()
        
        metadata = {
            'promoted_at': datetime.now().isoformat(),
            'git_hash': git_hash,
            'week': week,
            'validation': {
                'sharpe': validation['sharpe'],
                'psr': validation['psr'],
                'trades': validation['trades'],
                'stability': validation['stability']
            }
        }
        
        # Create locked config
        locked_config = {
            'parameters': params,
            'metadata': metadata
        }
        
        # Save locked file
        locked_file = self.locked_dir / f"params_{pair}_{timeframe}_{week}.yaml"
        with open(locked_file, 'w') as f:
            yaml.dump(locked_config, f, default_flow_style=False)
        
        # Update symlink
        current_link = self.current_dir / f"{pair}_{timeframe}.yaml"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        
        # Create relative symlink
        relative_path = Path('..') / locked_file.name
        current_link.symlink_to(relative_path)
        
        # Update catalog
        self.update_catalog(pair, timeframe, week, validation, git_hash)
        
        print(f"✅ Parameters promoted successfully!")
        print(f"   Locked file: {locked_file}")
        print(f"   Current link: {current_link}")
        print(f"   Sharpe: {validation['sharpe']:.2f}")
        print(f"   PSR: {validation['psr']:.2f}")
        print(f"   Trades: {validation['trades']}")
        
        return True
    
    def update_catalog(self, pair: str, timeframe: str, week: str, 
                       validation: Dict, git_hash: str):
        """Update parameter promotion catalog."""
        
        # Read existing catalog or create new
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                content = f.read()
        else:
            content = """# Parameter Promotion Catalog

Track history of parameter promotions to production.

| Date | Pair | TF | Week | Git Hash | Sharpe | PSR | Trades | Status |
|------|------|----|------|----------|--------|-----|--------|--------|
"""
        
        # Add new entry
        date = datetime.now().strftime('%Y-%m-%d %H:%M')
        new_entry = f"| {date} | {pair} | {timeframe} | {week} | {git_hash} | "
        new_entry += f"{validation['sharpe']:.2f} | {validation['psr']:.2f} | "
        new_entry += f"{validation['trades']} | ✅ Promoted |\n"
        
        # Append to table
        content += new_entry
        
        # Save catalog
        with open(self.catalog_path, 'w') as f:
            f.write(content)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Promote parameters to production')
    parser.add_argument('--pair', default='BTCETH', help='Trading pair')
    parser.add_argument('--timeframe', default='H1', help='Timeframe')
    parser.add_argument('--week', help='Week identifier (default: current)')
    parser.add_argument('--min-sharpe', type=float, default=1.0, help='Minimum Sharpe ratio')
    parser.add_argument('--min-psr', type=float, default=0.95, help='Minimum PSR')
    parser.add_argument('--min-trades', type=int, default=20, help='Minimum trades')
    
    args = parser.parse_args()
    
    promoter = ParameterPromoter(
        min_sharpe=args.min_sharpe,
        min_psr=args.min_psr,
        min_trades=args.min_trades
    )
    
    success = promoter.promote(
        pair=args.pair,
        timeframe=args.timeframe,
        week=args.week
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()