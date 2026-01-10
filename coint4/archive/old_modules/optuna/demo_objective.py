#!/usr/bin/env python3
"""
Demo Optuna Objective - простая демонстрационная версия для v0.2.4.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import optuna
from optuna.exceptions import TrialPruned

from ..core.deterministic import DeterministicContext


class DemoOptunaObjective:
    """Demo PSR objective для демонстрации Optuna workflow."""
    
    def __init__(self, 
                 pairs_file: str = "bench/pairs_canary.yaml", 
                 k_folds: int = 3,
                 save_traces: bool = True,
                 data_root: str = "./data_downloaded",
                 period_start: str = "2024-01-01",
                 period_end: str = "2024-03-31",
                 timeframe: str = "15T"):
        """Initialize demo objective function."""
        
        self.pairs_file = pairs_file
        self.k_folds = k_folds
        self.save_traces = save_traces
        self.data_root = data_root
        self.period_start = pd.Timestamp(period_start)
        self.period_end = pd.Timestamp(period_end)
        self.timeframe = timeframe
        
        print(f"📊 Objective initialized with data from: {data_root}")
        print(f"   Period: {period_start} to {period_end}")
        
        # Load pairs list
        self.pairs = self._load_pairs_list()
        
        # Prepare traces directory
        self.traces_dir = Path("artifacts/traces/optuna")
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_pairs_list(self) -> List[Tuple[str, str]]:
        """Load pairs from YAML file."""
        import yaml
        
        pairs_path = Path(self.pairs_file)
        if not pairs_path.exists():
            # Fallback pairs
            return [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "BNBUSDT"), ("ETHUSDT", "ADAUSDT")]
            
        with open(pairs_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Extract pairs from different possible formats
        if 'pairs' in data:
            pairs_list = data['pairs']
        elif 'portfolio' in data:
            pairs_list = data['portfolio']
        else:
            pairs_list = data
            
        # Convert to list of tuples
        pairs = []
        for item in pairs_list:
            if isinstance(item, dict):
                # Format: {symbol1: "BTC", symbol2: "ETH", weight: 0.125}
                pairs.append((item.get('symbol1', 'BTC'), item.get('symbol2', 'ETH')))
            elif isinstance(item, list) and len(item) >= 2:
                # Format: ["BTC", "ETH"]
                pairs.append((item[0], item[1]))
            elif isinstance(item, str) and '/' in item:
                # Format: "BTC/ETH"
                s1, s2 = item.split('/', 1)
                pairs.append((s1.strip(), s2.strip()))
        
        if not pairs:
            # Fallback pairs
            return [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "BNBUSDT")]
            
        return pairs
    
    def _run_single_fold(self, trial: optuna.Trial, fold_idx: int) -> Dict[str, float]:
        """Run demo backtest for single fold."""
        
        # Sample hyperparameters
        zscore_threshold = trial.suggest_float('zscore_threshold', 1.5, 3.5)
        zscore_exit = trial.suggest_float('zscore_exit', -0.5, 0.5)
        rolling_window = trial.suggest_int('rolling_window', 30, 120)
        max_holding_days = trial.suggest_int('max_holding_days', 20, 300)
        
        # Ensure hysteresis: zscore_exit < zscore_threshold
        if zscore_exit >= zscore_threshold:
            raise TrialPruned("Invalid band: zscore_exit >= zscore_threshold")
        
        # Get seed for this trial
        seed = trial.suggest_int('seed', 1, 999999) if fold_idx == 0 else trial.user_attrs.get('seed', 42)
        
        fold_results = []
        
        # Simulate each pair
        with DeterministicContext(seed + fold_idx):
            
            for pair_idx, (symbol1, symbol2) in enumerate(self.pairs):
                
                # Create realistic performance based on parameters
                # Good parameters should generally give better results
                
                # Base performance (inversely related to zscore_threshold)
                base_perf = max(0.5, 3.0 - zscore_threshold) 
                
                # Exit efficiency (smaller absolute zscore_exit is better)
                exit_factor = max(0.8, 1.2 - abs(zscore_exit))
                
                # Window efficiency (moderate windows are better)
                window_factor = 1.0 + 0.3 * np.exp(-0.5 * ((rolling_window - 60) / 30) ** 2)
                
                # Add pair-specific bias
                pair_bias = 0.1 if "BTC" in symbol1 or "ETH" in symbol1 else 0.0
                
                # Add deterministic noise
                np.random.seed(seed + pair_idx + fold_idx * 100)
                noise = np.random.normal(0, 0.2)
                
                # Calculate simulated metrics
                simulated_sharpe = base_perf * exit_factor * window_factor + pair_bias + noise
                simulated_psr = max(0.1, simulated_sharpe * 0.85 + np.random.normal(0, 0.05))
                simulated_dsr = max(0.1, simulated_psr * 0.9 + np.random.normal(0, 0.05))
                
                # Simulate trade count (related to rolling window)
                base_trades = int(rolling_window * 0.8 + np.random.poisson(25))
                
                fold_results.append({
                    'pair': f"{symbol1}/{symbol2}",
                    'trades': base_trades,
                    'sharpe': simulated_sharpe,
                    'psr': simulated_psr,
                    'dsr': simulated_dsr,
                    'total_return': simulated_sharpe * 0.08,
                    'max_drawdown': abs(simulated_sharpe) * 0.06
                })
                
                # Save demo traces
                if self.save_traces:
                    trace_file = self.traces_dir / f"trial_{trial.number}_fold_{fold_idx}_{symbol1}_{symbol2}.csv"
                    # Create demo trace data
                    n_points = max(50, rolling_window)
                    demo_trace = pd.DataFrame({
                        'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='4H'),
                        'signal': np.random.choice([-1, 0, 1], n_points, p=[0.15, 0.7, 0.15]),
                        'position': np.random.choice([-1, 0, 1], n_points, p=[0.2, 0.6, 0.2]),
                        'pnl': np.random.normal(simulated_sharpe/100, 0.02, n_points).cumsum(),
                        'zscore': np.random.normal(0, 1, n_points),
                        'threshold': zscore_threshold,
                        'exit_level': zscore_exit
                    })
                    demo_trace.to_csv(trace_file, index=False)
        
        # Guard: check minimum trades
        total_trades = sum(r['trades'] for r in fold_results)
        if total_trades < 30:  # Reasonable minimum
            raise TrialPruned(f"Too few trades in fold {fold_idx}: {total_trades}")
        
        # Portfolio-level aggregation using median
        fold_sharpe = np.median([r['sharpe'] for r in fold_results])
        fold_psr = np.median([r['psr'] for r in fold_results])
        fold_dsr = np.median([r['dsr'] for r in fold_results])
        
        return {
            'fold_sharpe': fold_sharpe,
            'fold_psr': fold_psr,
            'fold_dsr': fold_dsr,
            'fold_trades': total_trades,
            'pairs_count': len(fold_results)
        }
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Main objective function - returns median PSR across K folds."""
        
        fold_results = []
        
        # Run each fold
        for fold_idx in range(self.k_folds):
            
            try:
                fold_result = self._run_single_fold(trial, fold_idx)
                fold_results.append(fold_result)
                
                # Store per-fold metrics in user attributes
                trial.set_user_attr(f'fold_{fold_idx}_psr', fold_result['fold_psr'])
                trial.set_user_attr(f'fold_{fold_idx}_sharpe', fold_result['fold_sharpe'])
                trial.set_user_attr(f'fold_{fold_idx}_dsr', fold_result['fold_dsr'])
                trial.set_user_attr(f'fold_{fold_idx}_trades', fold_result['fold_trades'])
                
                # Intermediate pruning check
                if fold_idx >= 1:  # Have at least 2 folds
                    current_median_psr = np.median([r['fold_psr'] for r in fold_results])
                    trial.report(current_median_psr, fold_idx)
                    
                    # Let pruner decide
                    if trial.should_prune():
                        raise TrialPruned()
                        
            except TrialPruned:
                raise
            except Exception as e:
                # Log error but continue with other folds if possible
                trial.set_user_attr(f'fold_{fold_idx}_error', str(e))
                continue
        
        # Final validation
        if not fold_results:
            raise TrialPruned("No successful folds")
        
        if len(fold_results) < max(1, self.k_folds // 2):
            raise TrialPruned(f"Too few successful folds: {len(fold_results)}/{self.k_folds}")
        
        # Calculate final OOS metrics
        fold_psrs = [r['fold_psr'] for r in fold_results]
        fold_sharpes = [r['fold_sharpe'] for r in fold_results]
        fold_dsrs = [r['fold_dsr'] for r in fold_results]
        
        # Use median for robustness
        oos_psr = np.median(fold_psrs)
        oos_sharpe = np.median(fold_sharpes)
        oos_dsr = np.median(fold_dsrs)
        
        # Store final metrics
        trial.set_user_attr('oos_psr', oos_psr)
        trial.set_user_attr('oos_sharpe', oos_sharpe)
        trial.set_user_attr('oos_dsr', oos_dsr)
        trial.set_user_attr('successful_folds', len(fold_results))
        trial.set_user_attr('seed', trial.params.get('seed', 42))
        
        # Store traces path for best result
        if self.save_traces:
            trial.set_user_attr('traces_dir', str(self.traces_dir))
        
        return oos_psr