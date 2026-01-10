"""Signal churn control to reduce unnecessary trading."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChurnConfig:
    """Configuration for churn control."""
    
    # Stability filters
    min_signal_duration: int = 4  # Minimum bars to hold signal
    signal_smoothing: float = 0.2  # EMA smoothing factor
    
    # Change thresholds
    min_change_threshold: float = 0.1  # Minimum change to act
    stability_window: int = 10  # Lookback for stability
    
    # Momentum filters
    require_momentum: bool = True  # Require signal momentum
    momentum_threshold: float = 0.05  # Min momentum to act
    
    # Volume filters
    use_volume_filter: bool = False  # Filter on volume
    min_volume_ratio: float = 0.5  # Min vs average volume


class SignalChurnController:
    """Control signal churn to reduce turnover."""
    
    def __init__(self, config: ChurnConfig):
        self.config = config
        self.signal_history = {}
        self.last_actions = {}
        
    def process_signals(
        self,
        raw_signals: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Process raw signals to reduce churn."""
        
        # Apply smoothing
        smoothed = self._apply_smoothing(raw_signals)
        
        # Apply stability filter
        stable = self._apply_stability_filter(smoothed)
        
        # Apply change threshold
        significant = self._apply_change_threshold(stable)
        
        # Apply momentum filter
        if self.config.require_momentum:
            significant = self._apply_momentum_filter(significant)
        
        # Apply volume filter
        if self.config.use_volume_filter and volumes is not None:
            significant = self._apply_volume_filter(significant, volumes)
        
        # Store history
        self._update_history(significant)
        
        return significant
    
    def _apply_smoothing(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply EMA smoothing to reduce noise."""
        
        if self.config.signal_smoothing <= 0:
            return signals
            
        smoothed = signals.copy()
        alpha = self.config.signal_smoothing
        
        # Apply EMA smoothing
        for col in signals.columns:
            smoothed[col] = signals[col].ewm(alpha=alpha).mean()
            
        return smoothed
    
    def _apply_stability_filter(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter out unstable signals that change too frequently."""
        
        filtered = signals.copy()
        
        for col in signals.columns:
            signal = signals[col]
            
            # Calculate signal direction changes
            signal_direction = np.sign(signal)
            direction_changes = (signal_direction != signal_direction.shift(1))
            
            # Count recent changes
            recent_changes = direction_changes.rolling(
                window=self.config.stability_window,
                min_periods=1
            ).sum()
            
            # Filter out unstable periods (too many changes)
            unstable = recent_changes > (self.config.stability_window / 3)
            
            # Set unstable signals to 0
            filtered.loc[unstable, col] = 0
            
        return filtered
    
    def _apply_change_threshold(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Only act on signals above minimum change threshold."""
        
        filtered = signals.copy()
        threshold = self.config.min_change_threshold
        
        for col in signals.columns:
            signal = signals[col]
            prev_signal = signal.shift(1).fillna(0)
            
            # Calculate change magnitude
            change_magnitude = abs(signal - prev_signal)
            
            # Only change if above threshold
            small_changes = change_magnitude < threshold
            filtered.loc[small_changes, col] = prev_signal.loc[small_changes]
            
        return filtered
    
    def _apply_momentum_filter(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Require signal momentum to act."""
        
        filtered = signals.copy()
        threshold = self.config.momentum_threshold
        
        for col in signals.columns:
            signal = signals[col]
            
            # Calculate signal momentum (rate of change)
            momentum = signal.diff().rolling(3).mean().abs()
            
            # Require minimum momentum
            low_momentum = momentum < threshold
            
            # Keep previous signal if momentum too low
            prev_signal = signal.shift(1).fillna(0)
            filtered.loc[low_momentum, col] = prev_signal.loc[low_momentum]
            
        return filtered
    
    def _apply_volume_filter(
        self,
        signals: pd.DataFrame,
        volumes: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter signals based on volume conditions."""
        
        filtered = signals.copy()
        min_ratio = self.config.min_volume_ratio
        
        # Calculate average volume
        avg_volumes = volumes.rolling(20).mean()
        current_ratio = volumes / avg_volumes
        
        # Filter low volume periods
        low_volume = current_ratio < min_ratio
        
        # Set signals to 0 during low volume
        filtered[low_volume] = 0
        
        return filtered
    
    def _update_history(self, signals: pd.DataFrame):
        """Update signal history for tracking."""
        
        for col in signals.columns:
            if col not in self.signal_history:
                self.signal_history[col] = []
            
            # Keep last 100 observations
            self.signal_history[col].append(signals[col].iloc[-1])
            if len(self.signal_history[col]) > 100:
                self.signal_history[col].pop(0)
    
    def get_churn_statistics(self) -> Dict:
        """Calculate churn reduction statistics."""
        
        stats = {}
        
        for pair, history in self.signal_history.items():
            if len(history) < 10:
                continue
                
            series = pd.Series(history)
            
            # Calculate signal changes
            changes = (series != series.shift(1)).sum()
            change_rate = changes / len(history)
            
            # Calculate average holding period
            signal_runs = []
            current_run = 1
            
            for i in range(1, len(history)):
                if history[i] == history[i-1]:
                    current_run += 1
                else:
                    signal_runs.append(current_run)
                    current_run = 1
            
            avg_holding_period = np.mean(signal_runs) if signal_runs else 1
            
            stats[pair] = {
                'change_rate': change_rate,
                'avg_holding_period': avg_holding_period,
                'total_observations': len(history),
                'active_signals': (series != 0).sum()
            }
        
        return stats


def estimate_churn_reduction(
    original_signals: pd.DataFrame,
    controlled_signals: pd.DataFrame
) -> Dict:
    """Estimate churn reduction from control."""
    
    # Calculate position changes
    original_changes = (original_signals != original_signals.shift(1)).sum().sum()
    controlled_changes = (controlled_signals != controlled_signals.shift(1)).sum().sum()
    
    # Calculate turnover reduction
    turnover_reduction = (original_changes - controlled_changes) / original_changes
    
    # Calculate average holding periods
    def calc_avg_holding(signals):
        holding_periods = []
        for col in signals.columns:
            signal = signals[col]
            changes = signal != signal.shift(1)
            runs = []
            current_run = 1
            
            for change in changes:
                if not change:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            
            holding_periods.extend(runs)
        
        return np.mean(holding_periods) if holding_periods else 1
    
    original_holding = calc_avg_holding(original_signals)
    controlled_holding = calc_avg_holding(controlled_signals)
    
    return {
        'turnover_reduction': turnover_reduction,
        'original_changes': original_changes,
        'controlled_changes': controlled_changes,
        'original_avg_holding': original_holding,
        'controlled_avg_holding': controlled_holding,
        'holding_improvement': controlled_holding / original_holding
    }


def optimize_churn_config(
    signals: pd.DataFrame,
    target_turnover_reduction: float = 0.3,
    volumes: Optional[pd.DataFrame] = None
) -> Tuple[ChurnConfig, Dict]:
    """Find optimal churn control configuration."""
    
    best_config = None
    best_score = -float('inf')
    best_stats = None
    
    # Parameter grid
    param_grid = {
        'signal_smoothing': [0.0, 0.1, 0.2, 0.3],
        'min_change_threshold': [0.05, 0.1, 0.15, 0.2],
        'min_signal_duration': [2, 4, 6, 8],
        'momentum_threshold': [0.02, 0.05, 0.08, 0.1]
    }
    
    # Grid search
    from itertools import product
    
    for smoothing, change_thresh, min_duration, momentum_thresh in product(
        param_grid['signal_smoothing'],
        param_grid['min_change_threshold'],
        param_grid['min_signal_duration'],
        param_grid['momentum_threshold']
    ):
        config = ChurnConfig(
            signal_smoothing=smoothing,
            min_change_threshold=change_thresh,
            min_signal_duration=min_duration,
            momentum_threshold=momentum_thresh
        )
        
        controller = SignalChurnController(config)
        controlled = controller.process_signals(signals, volumes)
        
        # Calculate metrics
        reduction_stats = estimate_churn_reduction(signals, controlled)
        turnover_reduction = reduction_stats['turnover_reduction']
        
        # Score based on how close to target
        score = 1.0 - abs(turnover_reduction - target_turnover_reduction)
        
        if score > best_score:
            best_score = score
            best_config = config
            best_stats = reduction_stats
    
    return best_config, best_stats