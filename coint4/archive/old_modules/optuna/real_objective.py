#!/usr/bin/env python3
"""Real Optuna objective with data loading and walk-forward analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import optuna
from optuna.exceptions import TrialPruned
from pathlib import Path

from ..core.data_loader import DataHandler
from ..utils.config import load_config
from ..engine.optimized_backtest_engine import OptimizedPairBacktester


class RealOptunaObjective:
    """Real objective with data loading and PSR calculation."""
    
    def __init__(self,
                 pairs_file: str = "bench/pairs_universe.yaml",
                 k_folds: int = 3,
                 data_root: str = "./data_downloaded",
                 period_start: str = "2023-01-01",
                 period_end: str = "2024-06-30",
                 timeframe: str = "15T",
                 save_traces: bool = False):
        """Initialize objective with data parameters."""
        
        self.pairs_file = pairs_file
        self.k_folds = k_folds
        self.data_root = data_root
        self.period_start = pd.Timestamp(period_start)
        self.period_end = pd.Timestamp(period_end)
        self.timeframe = timeframe
        self.save_traces = save_traces
        
        # Load config and data handler
        self.config = load_config("configs/main_2024.yaml")
        self.data_handler = DataHandler(self.config, root=data_root)
        
        # Load pairs
        self.pairs = self._load_pairs()
        
        # Prepare K-fold splits
        self.fold_splits = self._prepare_kfold_splits()
        
        print(f"📊 RealObjective initialized:")
        print(f"   Data: {data_root}")
        print(f"   Period: {period_start} to {period_end}")
        print(f"   Pairs: {len(self.pairs)}")
        
    def _load_pairs(self) -> List[Tuple[str, str]]:
        """Load pairs from file."""
        import yaml
        
        pairs_path = Path(self.pairs_file)
        if not pairs_path.exists():
            # Default pairs
            return [("BTCUSDT", "ETHUSDT"), ("ETHUSDT", "BNBUSDT")]
            
        with open(pairs_path, 'r') as f:
            data = yaml.safe_load(f)
            
        pairs = []
        for item in data.get('pairs', []):
            if isinstance(item, dict) and 'pair' in item:
                # Format: {pair: "BTC/ETH", weight: 0.5}
                pair_str = item['pair']
                if '/' in pair_str:
                    s1, s2 = pair_str.split('/', 1)
                    pairs.append((s1.strip(), s2.strip()))
            elif isinstance(item, str) and '/' in item:
                # Format: "BTC/ETH"
                s1, s2 = item.split('/', 1)
                pairs.append((s1.strip(), s2.strip()))
                
        return pairs if pairs else [("BTCUSDT", "ETHUSDT")]
    
    def _prepare_kfold_splits(self) -> List[Dict]:
        """Prepare K-fold time series splits."""
        total_days = (self.period_end - self.period_start).days
        fold_days = total_days // (self.k_folds + 1)  # Leave space for test
        
        splits = []
        for fold in range(self.k_folds):
            train_start = self.period_start + pd.Timedelta(days=fold * fold_days // 2)
            train_end = train_start + pd.Timedelta(days=fold_days)
            test_start = train_end + pd.Timedelta(hours=1)  # Small gap
            test_end = test_start + pd.Timedelta(days=fold_days // 3)
            
            splits.append({
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
        return splits
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        
        # Sample hyperparameters
        zscore_threshold = trial.suggest_float('zscore_threshold', 1.5, 3.5)
        zscore_exit = trial.suggest_float('zscore_exit', -0.5, 0.5)
        rolling_window = trial.suggest_int('rolling_window', 30, 150)
        max_holding_days = trial.suggest_int('max_holding_days', 10, 100)
        
        # Ensure hysteresis
        if zscore_exit >= zscore_threshold:
            raise TrialPruned("Invalid: zscore_exit >= zscore_threshold")
        
        fold_results = []
        
        for split in self.fold_splits:
            try:
                # Load train data
                train_df = self.data_handler.load_all_data_for_period(
                    lookback_days=rolling_window // 96 + 1,  # Convert to days
                    end_date=split['train_end']
                )
                
                # Load test data
                test_df = self.data_handler.load_all_data_for_period(
                    lookback_days=(split['test_end'] - split['test_start']).days + 1,
                    end_date=split['test_end']
                )
                
                fold_sharpe = []
                fold_trades = 0
                
                for symbol1, symbol2 in self.pairs[:3]:  # Limit pairs for speed
                    if symbol1 in train_df.columns and symbol2 in train_df.columns:
                        # Prepare pair data
                        pair_train = pd.DataFrame({
                            'symbol1': train_df[symbol1].values,
                            'symbol2': train_df[symbol2].values
                        }, index=train_df.index)
                        
                        pair_test = pd.DataFrame({
                            'symbol1': test_df[symbol1].values,
                            'symbol2': test_df[symbol2].values
                        }, index=test_df.index)
                        
                        # Run backtest on test
                        backtester = OptimizedPairBacktester(
                            pair_data=pair_test,
                            rolling_window=rolling_window,
                            z_threshold=zscore_threshold,
                            z_exit=zscore_exit,
                            commission_pct=0.001,
                            slippage_pct=0.0005
                        )
                        
                        results = backtester.run()
                        
                        if hasattr(results, 'trades') and results.trades:
                            returns = [t.get('pnl', 0) for t in results.trades]
                            if len(returns) > 1:
                                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
                                fold_sharpe.append(sharpe)
                                fold_trades += len(returns)
                
                if fold_sharpe:
                    fold_mean_sharpe = np.mean(fold_sharpe)
                else:
                    fold_mean_sharpe = -1.0
                    
                fold_results.append({
                    'sharpe': fold_mean_sharpe,
                    'trades': fold_trades
                })
                
                # Store in user attrs
                trial.set_user_attr(f'fold_{split["fold"]}_sharpe', fold_mean_sharpe)
                trial.set_user_attr(f'fold_{split["fold"]}_trades', fold_trades)
                
            except Exception as e:
                print(f"Error in fold {split['fold']}: {e}")
                fold_results.append({'sharpe': -1.0, 'trades': 0})
        
        # Calculate PSR (Probabilistic Sharpe Ratio)
        sharpes = [r['sharpe'] for r in fold_results if r['sharpe'] > -1]
        if sharpes:
            mean_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 0.5
            psr = mean_sharpe / (std_sharpe + 0.1)  # Add small constant for stability
        else:
            psr = -1.0
            
        # Store aggregate metrics
        trial.set_user_attr('oos_sharpe', mean_sharpe if sharpes else -1)
        trial.set_user_attr('oos_psr', psr)
        trial.set_user_attr('total_trades', sum(r['trades'] for r in fold_results))
        
        # Report for pruning
        if trial.number % 10 == 0:
            trial.report(psr, trial.number)
            if trial.should_prune():
                raise TrialPruned()
        
        return psr