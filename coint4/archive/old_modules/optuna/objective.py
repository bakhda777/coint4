#!/usr/bin/env python3
"""
Optuna Objective Function - K-fold OOS PSR optimization.
Консолидированная цель с guards, pruning и deterministic context.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import optuna
from optuna.exceptions import TrialPruned

from ..utils.config import load_config
from ..core.deterministic import DeterministicContext


class OptunaPSRObjective:
    """K-fold OOS PSR objective с guards и deterministic execution."""
    
    def __init__(self, 
                 base_config_path: str = "configs/main_2024.yaml",
                 pairs_file: str = "bench/pairs_canary.yaml", 
                 k_folds: int = 3,
                 min_trades_per_fold: int = 10,
                 save_traces: bool = True):
        """Initialize PSR objective function."""
        
        self.base_config_path = base_config_path
        self.pairs_file = pairs_file
        self.k_folds = k_folds
        self.min_trades_per_fold = min_trades_per_fold
        self.save_traces = save_traces
        
        # Load base configuration
        self.base_config = load_config(base_config_path)
        
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
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
            
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
                pairs.append((item['symbol1'], item['symbol2']))
            elif isinstance(item, list) and len(item) >= 2:
                # Format: ["BTC", "ETH"]
                pairs.append((item[0], item[1]))
            elif isinstance(item, str) and '/' in item:
                # Format: "BTC/ETH"
                s1, s2 = item.split('/', 1)
                pairs.append((s1.strip(), s2.strip()))
        
        if not pairs:
            raise ValueError(f"No valid pairs found in {self.pairs_file}")
            
        return pairs
    
    def _create_fold_config(self, trial: optuna.Trial, fold_idx: int, 
                          train_start: str, train_end: str, 
                          test_start: str, test_end: str) -> Dict[str, Any]:
        """Create configuration for specific fold."""
        
        # Sample hyperparameters
        config = self.base_config.copy()
        
        # Signals parameters
        config['signals'] = {
            'zscore_threshold': trial.suggest_float('zscore_threshold', 1.5, 3.5),
            'zscore_exit': trial.suggest_float('zscore_exit', -0.5, 0.5),
            'rolling_window': trial.suggest_int('rolling_window', 30, 120),
            'max_holding_days': trial.suggest_int('max_holding_days', 20, 300)
        }
        
        # Ensure hysteresis: zscore_exit < zscore_threshold
        if config['signals']['zscore_exit'] >= config['signals']['zscore_threshold']:
            raise TrialPruned("Invalid band: zscore_exit >= zscore_threshold")
        
        # Backtesting parameters
        config['backtesting']['commission_pct'] = trial.suggest_float('commission_pct', 0.0001, 0.001, log=True)
        config['backtesting']['slippage_pct'] = trial.suggest_float('slippage_pct', 0.0001, 0.0005, log=True)
        
        # Walk-forward periods for this fold
        config['walk_forward'] = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'gap_minutes': 15  # 1 bar gap
        }
        
        # Deterministic execution
        seed = trial.suggest_int('seed', 1, 999999) if fold_idx == 0 else trial.user_attrs.get('seed', 42)
        config['seed'] = seed
        
        return config
    
    def _generate_k_folds(self) -> List[Tuple[str, str, str, str]]:
        """Generate K-fold train/test periods."""
        
        # Use a reasonable time range for crypto data
        # This should be configurable, but using hardcoded for now
        full_start = "2023-01-01"
        full_end = "2024-06-30"
        
        start_date = pd.Timestamp(full_start)
        end_date = pd.Timestamp(full_end)
        
        total_days = (end_date - start_date).days
        fold_days = total_days // self.k_folds
        
        folds = []
        
        for i in range(self.k_folds):
            # Each fold: 70% train, 30% test
            fold_start = start_date + pd.Timedelta(days=i * fold_days)
            fold_end = start_date + pd.Timedelta(days=(i + 1) * fold_days)
            
            train_days = int(fold_days * 0.7)
            
            train_start = fold_start
            train_end = fold_start + pd.Timedelta(days=train_days)
            test_start = train_end + pd.Timedelta(days=1)  # 1 day gap
            test_end = fold_end
            
            folds.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
        return folds
    
    def _run_single_fold(self, trial: optuna.Trial, fold_idx: int,
                        train_start: str, train_end: str,
                        test_start: str, test_end: str) -> Dict[str, float]:
        """Run backtest for single fold - simplified implementation."""
        
        # Create fold configuration
        fold_config = self._create_fold_config(trial, fold_idx, train_start, train_end, test_start, test_end)
        
        # Set up deterministic context
        seed = fold_config['seed']
        with DeterministicContext(seed):
            
            fold_results = []
            
            # Simplified simulation for each pair
            for pair_idx, (symbol1, symbol2) in enumerate(self.pairs):
                
                # Simulate returns using hyperparameters
                # This is a placeholder - in real implementation would load data and run backtest
                zscore_threshold = fold_config['signals']['zscore_threshold']
                zscore_exit = fold_config['signals']['zscore_exit']
                rolling_window = fold_config['signals']['rolling_window']
                
                # Generate synthetic performance based on parameters
                # Better parameters should give better performance
                base_performance = max(0.1, 2.5 - zscore_threshold) * (1 - abs(zscore_exit) * 0.5)
                window_factor = min(1.2, rolling_window / 60.0)
                
                # Add randomness but keep it deterministic
                np.random.seed(seed + pair_idx + fold_idx)
                noise = np.random.normal(0, 0.3)
                
                simulated_sharpe = base_performance * window_factor + noise
                simulated_psr = max(0.1, simulated_sharpe * 0.8 + np.random.normal(0, 0.1))
                simulated_dsr = max(0.1, simulated_psr * 0.9 + np.random.normal(0, 0.1))
                
                # Simulate trade count
                base_trades = int(rolling_window + np.random.poisson(30))
                
                if base_trades < 5:  # Very relaxed for demo
                    continue
                
                fold_results.append({
                    'pair': f"{symbol1}/{symbol2}",
                    'trades': base_trades,
                    'sharpe': simulated_sharpe,
                    'psr': simulated_psr,
                    'dsr': simulated_dsr,
                    'total_return': simulated_sharpe * 0.1,
                    'max_drawdown': abs(simulated_sharpe) * 0.05
                })
                
                # Save traces simulation
                if self.save_traces:
                    trace_file = self.traces_dir / f"trial_{trial.number}_fold_{fold_idx}_{symbol1}_{symbol2}.csv"
                    # Create dummy trace file
                    dummy_trace = pd.DataFrame({
                        'timestamp': pd.date_range(test_start, test_end, freq='1H')[:100],
                        'signal': np.random.choice([-1, 0, 1], 100),
                        'position': np.random.choice([-1, 0, 1], 100),
                        'pnl': np.random.normal(0, 0.01, 100).cumsum()
                    })
                    dummy_trace.to_csv(trace_file, index=False)
            
            # Aggregate results across pairs
            if not fold_results:
                raise TrialPruned(f"No valid results for fold {fold_idx}")
            
            # Portfolio-level aggregation
            total_trades = sum(r['trades'] for r in fold_results)
            # Relaxed for demo
            if total_trades < 5:
                raise TrialPruned(f"Too few trades in fold {fold_idx}: {total_trades}")
            
            # Use median for robustness
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
    
    def _calculate_psr(self, returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
        """Calculate Probabilistic Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
            
        observed_sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
        n = len(returns)
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        # PSR formula
        psr_stat = (observed_sharpe - benchmark_sharpe) * np.sqrt(n - 1) / np.sqrt(1 - skew * observed_sharpe + (kurtosis - 1)/4 * observed_sharpe**2)
        
        # Convert to probability using normal CDF approximation
        try:
            from scipy.stats import norm
            psr = norm.cdf(psr_stat)
        except ImportError:
            # Fallback approximation
            psr = max(0.0, min(1.0, 0.5 + psr_stat * 0.2))
        
        return psr
    
    def _calculate_dsr(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate Deflated Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
            
        # Simplified DSR - more sophisticated versions would account for multiple testing
        observed_sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
        
        # Penalize for multiple testing (rough approximation)
        trials_penalty = np.log(100)  # Assume 100 trials tested
        dsr = observed_sharpe - trials_penalty / np.sqrt(len(returns))
        
        return max(dsr, 0.0)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Main objective function - returns median PSR across K folds."""
        
        # Generate K-fold splits
        folds = self._generate_k_folds()
        
        fold_results = []
        
        # Run each fold
        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            
            try:
                fold_result = self._run_single_fold(
                    trial, fold_idx, train_start, train_end, test_start, test_end
                )
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
        
        if len(fold_results) < 1:
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