#!/usr/bin/env python3
"""
Web-based optimization backend for Streamlit UI.
Provides a simplified interface for running Optuna optimization.
"""

import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna
from optuna.trial import Trial
import pandas as pd
import numpy as np

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebOptimizer:
    """Streamlit-friendly optimizer wrapper."""
    
    def __init__(
        self,
        base_config_path: str = "configs/main_2024.yaml",
        search_space_path: str = "configs/search_spaces/web_ui.yaml",
        output_dir: str = "outputs/optimization"
    ):
        """Initialize the optimizer.
        
        Args:
            base_config_path: Path to base configuration
            search_space_path: Path to search space configuration
            output_dir: Directory for saving results
        """
        self.base_config_path = base_config_path
        self.search_space_path = search_space_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load search space
        with open(search_space_path, 'r') as f:
            self.search_space = yaml.safe_load(f)
        
        # Initialize objective
        self.objective = None
        self.study = None
        self.callback = None
        
    def create_objective(self, param_ranges: Dict[str, tuple]) -> Callable:
        """Create Optuna objective function with given parameter ranges.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
        
        Returns:
            Objective function for Optuna
        """
        # Initialize FastWalkForwardObjective
        self.objective = FastWalkForwardObjective(
            base_config_path=self.base_config_path,
            search_space_path=self.search_space_path
        )
        
        def objective(trial: Trial) -> float:
            """Optuna objective function."""
            
            # Sample parameters from ranges
            params = {}
            
            # Signals
            if 'zscore_threshold' in param_ranges:
                params['zscore_threshold'] = trial.suggest_float(
                    'zscore_threshold',
                    param_ranges['zscore_threshold'][0],
                    param_ranges['zscore_threshold'][1],
                    step=0.1
                )
            
            if 'zscore_exit' in param_ranges:
                params['zscore_exit'] = trial.suggest_float(
                    'zscore_exit',
                    param_ranges['zscore_exit'][0],
                    param_ranges['zscore_exit'][1],
                    step=0.1
                )
            
            if 'rolling_window' in param_ranges:
                params['rolling_window'] = trial.suggest_int(
                    'rolling_window',
                    int(param_ranges['rolling_window'][0]),
                    int(param_ranges['rolling_window'][1]),
                    step=5
                )
            
            # Risk management
            if 'stop_loss_multiplier' in param_ranges:
                params['stop_loss_multiplier'] = trial.suggest_float(
                    'stop_loss_multiplier',
                    param_ranges['stop_loss_multiplier'][0],
                    param_ranges['stop_loss_multiplier'][1],
                    step=0.5
                )
            
            if 'time_stop_multiplier' in param_ranges:
                params['time_stop_multiplier'] = trial.suggest_float(
                    'time_stop_multiplier',
                    param_ranges['time_stop_multiplier'][0],
                    param_ranges['time_stop_multiplier'][1],
                    step=0.5
                )
            
            # Position sizing
            if 'max_position_size_pct' in param_ranges:
                params['max_position_size_pct'] = trial.suggest_float(
                    'max_position_size_pct',
                    param_ranges['max_position_size_pct'][0],
                    param_ranges['max_position_size_pct'][1],
                    step=0.005
                )
            
            # Filters
            if 'coint_pvalue_threshold' in param_ranges:
                params['coint_pvalue_threshold'] = trial.suggest_float(
                    'coint_pvalue_threshold',
                    param_ranges['coint_pvalue_threshold'][0],
                    param_ranges['coint_pvalue_threshold'][1],
                    step=0.01
                )
            
            if 'max_hurst_exponent' in param_ranges:
                params['max_hurst_exponent'] = trial.suggest_float(
                    'max_hurst_exponent',
                    param_ranges['max_hurst_exponent'][0],
                    param_ranges['max_hurst_exponent'][1],
                    step=0.05
                )
            
            # Costs
            if 'commission_pct' in param_ranges:
                params['commission_pct'] = trial.suggest_float(
                    'commission_pct',
                    param_ranges['commission_pct'][0],
                    param_ranges['commission_pct'][1],
                    step=0.0001
                )
            
            if 'slippage_pct' in param_ranges:
                params['slippage_pct'] = trial.suggest_float(
                    'slippage_pct',
                    param_ranges['slippage_pct'][0],
                    param_ranges['slippage_pct'][1],
                    step=0.0001
                )
            
            # Call actual objective
            try:
                result = self.objective(trial)
                
                # Report intermediate values
                if self.callback:
                    self.callback(trial.number, result, params)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                return -999.0  # Return very bad value for failed trials
        
        return objective
    
    def optimize(
        self,
        n_trials: int = 50,
        param_ranges: Dict[str, tuple] = None,
        target_metric: str = "sharpe_ratio",
        sampler: str = "TPE",
        pruner: bool = True,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run optimization.
        
        Args:
            n_trials: Number of optimization trials
            param_ranges: Parameter ranges for optimization
            target_metric: Metric to optimize (sharpe_ratio, total_pnl, win_rate, calmar)
            sampler: Sampler type (TPE, Random, Grid)
            pruner: Whether to use pruning
            progress_callback: Callback function for progress updates
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with optimization results
        """
        self.callback = progress_callback
        
        # Create sampler
        if sampler == "TPE":
            sampler_obj = optuna.samplers.TPESampler(seed=42)
        elif sampler == "Random":
            sampler_obj = optuna.samplers.RandomSampler(seed=42)
        else:  # Grid
            sampler_obj = optuna.samplers.GridSampler(param_ranges)
        
        # Create pruner
        if pruner:
            pruner_obj = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        else:
            pruner_obj = optuna.pruners.NopPruner()
        
        # Create study
        study_name = f"web_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler_obj,
            pruner=pruner_obj
        )
        
        # Create objective
        objective = self.create_objective(param_ranges or {})
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False  # We use custom callback
        )
        
        # Get results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Save results
        results = {
            "study_name": study_name,
            "n_trials": n_trials,
            "best_value": best_value,
            "best_params": best_params,
            "target_metric": target_metric,
            "timestamp": datetime.now().isoformat(),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state)
                }
                for t in self.study.trials
            ]
        }
        
        # Save to file
        results_file = self.output_dir / f"{study_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best params as YAML
        params_file = self.output_dir / f"{study_name}_best_params.yaml"
        with open(params_file, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        
        logger.info(f"Optimization complete. Best {target_metric}: {best_value:.3f}")
        logger.info(f"Results saved to {results_file}")
        
        return results
    
    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance from the study.
        
        Returns:
            Dictionary of parameter names to importance scores
        """
        if self.study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.error(f"Could not calculate importance: {e}")
            return {}
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame.
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            return pd.DataFrame()
        
        history = []
        for trial in self.study.trials:
            row = {
                'trial': trial.number,
                'value': trial.value,
                'state': str(trial.state),
                **trial.params
            }
            history.append(row)
        
        return pd.DataFrame(history)
    
    def visualize_optimization(self) -> Dict[str, Any]:
        """Create visualization data for the optimization.
        
        Returns:
            Dictionary with plot data
        """
        if self.study is None:
            return {}
        
        # Get optimization history
        history_df = self.get_optimization_history()
        
        # Create plots data
        plots = {
            'optimization_history': {
                'x': history_df['trial'].tolist(),
                'y': history_df['value'].tolist(),
                'type': 'scatter',
                'title': 'Optimization History'
            },
            'parameter_importance': self.get_importance(),
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
        
        return plots


def main():
    """Test the optimizer."""
    optimizer = WebOptimizer()
    
    # Define parameter ranges
    param_ranges = {
        'zscore_threshold': (0.5, 2.0),
        'zscore_exit': (-0.2, 0.2),
        'rolling_window': (20, 50),
        'stop_loss_multiplier': (2.5, 3.5),
        'commission_pct': (0.0003, 0.0005),
        'slippage_pct': (0.0004, 0.0006)
    }
    
    # Progress callback
    def progress_callback(trial_num, value, params):
        print(f"Trial {trial_num}: {value:.3f}")
    
    # Run optimization
    results = optimizer.optimize(
        n_trials=10,
        param_ranges=param_ranges,
        progress_callback=progress_callback
    )
    
    print(f"\nBest value: {results['best_value']:.3f}")
    print(f"Best params: {results['best_params']}")


if __name__ == "__main__":
    main()