#!/usr/bin/env python3
"""
Volatility regime detection for adaptive parameter selection.
Identifies market regimes and selects appropriate strategy parameters.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class VolatilityRegimeDetector:
    """Detects volatility regimes and suggests parameter adjustments."""
    
    def __init__(self, n_regimes: int = 3):
        """Initialize regime detector.
        
        Args:
            n_regimes: Number of volatility regimes to detect
        """
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.regime_params = self._initialize_regime_params()
        self.current_regime = None
        self.regime_history = []
        
    def _initialize_regime_params(self) -> Dict:
        """Initialize parameter sets for each regime."""
        return {
            0: {  # Low volatility regime
                'name': 'low_volatility',
                'zscore_threshold': 1.5,
                'zscore_exit': 0.0,
                'rolling_window': 90,
                'max_holding_days': 150,
                'position_size_multiplier': 1.5,
                'description': 'Stable market, tighter entry signals'
            },
            1: {  # Normal volatility regime
                'name': 'normal_volatility',
                'zscore_threshold': 2.0,
                'zscore_exit': 0.0,
                'rolling_window': 60,
                'max_holding_days': 100,
                'position_size_multiplier': 1.0,
                'description': 'Standard market conditions'
            },
            2: {  # High volatility regime
                'name': 'high_volatility',
                'zscore_threshold': 2.5,
                'zscore_exit': 0.5,
                'rolling_window': 30,
                'max_holding_days': 50,
                'position_size_multiplier': 0.5,
                'description': 'Volatile market, wider entry signals, faster exits'
            }
        }
    
    def calculate_volatility_features(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate volatility-related features for regime detection."""
        features = pd.DataFrame(index=returns.index)
        
        # Realized volatility (multiple windows)
        for window in [5, 10, 20, 60]:
            features[f'vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatility of volatility
        features['vol_of_vol'] = features['vol_20d'].rolling(20).std()
        
        # High-low range proxy
        features['abs_returns'] = returns.abs()
        features['avg_abs_returns'] = features['abs_returns'].rolling(20).mean()
        
        # GARCH-like features
        features['squared_returns'] = returns ** 2
        features['ewm_variance'] = features['squared_returns'].ewm(span=20).mean()
        
        # Tail risk measures - handle empty series
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 20:
            features['downside_vol'] = downside_returns.rolling(20, min_periods=1).std() * np.sqrt(252)
        else:
            features['downside_vol'] = 0
            
        upside_returns = returns[returns > 0]
        if len(upside_returns) > 20:
            features['upside_vol'] = upside_returns.rolling(20, min_periods=1).std() * np.sqrt(252)
        else:
            features['upside_vol'] = 0
        
        # Fill NaN values with forward fill then backward fill
        features = features.ffill().bfill()
        
        return features.dropna()
    
    def fit_regimes(self, returns: pd.Series) -> None:
        """Fit regime model on historical returns."""
        features = self.calculate_volatility_features(returns)
        
        # Use primary volatility measure for regime detection
        X = features[['vol_20d', 'vol_of_vol', 'ewm_variance']].values
        
        # Fit Gaussian Mixture Model
        self.gmm.fit(X)
        
        # Sort regimes by average volatility (low to high)
        regime_labels = self.gmm.predict(X)
        regime_vols = []
        for i in range(self.n_regimes):
            mask = regime_labels == i
            avg_vol = features.loc[mask, 'vol_20d'].mean()
            regime_vols.append((i, avg_vol))
        
        # Reorder regimes from low to high volatility
        sorted_regimes = sorted(regime_vols, key=lambda x: x[1])
        self.regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
    def detect_current_regime(self, recent_returns: pd.Series) -> int:
        """Detect current market regime."""
        features = self.calculate_volatility_features(recent_returns)
        
        if len(features) == 0:
            return 1  # Default to normal regime
        
        # Get latest features
        X = features[['vol_20d', 'vol_of_vol', 'ewm_variance']].iloc[-1:].values
        
        # Predict regime
        regime_raw = self.gmm.predict(X)[0]
        regime = self.regime_mapping.get(regime_raw, regime_raw)
        
        self.current_regime = regime
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'volatility': features['vol_20d'].iloc[-1]
        })
        
        return regime
    
    def get_regime_parameters(self, regime: Optional[int] = None) -> Dict:
        """Get recommended parameters for given regime."""
        if regime is None:
            regime = self.current_regime
        
        if regime is None:
            regime = 1  # Default to normal
        
        return self.regime_params.get(regime, self.regime_params[1])
    
    def analyze_regime_transitions(self, returns: pd.Series, 
                                  window: int = 252) -> pd.DataFrame:
        """Analyze regime transitions over time."""
        features = self.calculate_volatility_features(returns)
        X = features[['vol_20d', 'vol_of_vol', 'ewm_variance']].values
        
        # Predict regimes for entire history
        regimes_raw = self.gmm.predict(X)
        regimes = [self.regime_mapping.get(r, r) for r in regimes_raw]
        
        # Create regime dataframe
        regime_df = pd.DataFrame({
            'date': features.index,
            'regime': regimes,
            'volatility': features['vol_20d'].values,
            'regime_name': [self.regime_params[r]['name'] for r in regimes]
        })
        
        # Calculate regime statistics
        regime_stats = []
        for regime in range(self.n_regimes):
            mask = regime_df['regime'] == regime
            if mask.sum() > 0:
                stats = {
                    'regime': regime,
                    'name': self.regime_params[regime]['name'],
                    'frequency': mask.mean(),
                    'avg_volatility': regime_df.loc[mask, 'volatility'].mean(),
                    'avg_duration': self._calculate_avg_duration(regime_df['regime'].values, regime)
                }
                regime_stats.append(stats)
        
        return pd.DataFrame(regime_stats)
    
    def _calculate_avg_duration(self, regimes: np.ndarray, target_regime: int) -> float:
        """Calculate average duration of a regime."""
        durations = []
        current_duration = 0
        
        for regime in regimes:
            if regime == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def generate_regime_report(self, returns: pd.Series) -> Dict:
        """Generate comprehensive regime analysis report."""
        # Detect current regime
        current_regime = self.detect_current_regime(returns)
        current_params = self.get_regime_parameters(current_regime)
        
        # Analyze transitions
        regime_stats = self.analyze_regime_transitions(returns)
        
        # Build report
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_regime': {
                'id': current_regime,
                'name': current_params['name'],
                'description': current_params['description'],
                'recommended_parameters': {
                    'zscore_threshold': current_params['zscore_threshold'],
                    'zscore_exit': current_params['zscore_exit'],
                    'rolling_window': current_params['rolling_window'],
                    'max_holding_days': current_params['max_holding_days']
                }
            },
            'regime_statistics': regime_stats.to_dict('records'),
            'transition_matrix': self._calculate_transition_matrix(returns),
            'recommendations': self._generate_recommendations(current_regime, returns)
        }
        
        return report
    
    def _calculate_transition_matrix(self, returns: pd.Series) -> Dict:
        """Calculate regime transition probability matrix."""
        features = self.calculate_volatility_features(returns)
        X = features[['vol_20d', 'vol_of_vol', 'ewm_variance']].values
        regimes = self.gmm.predict(X)
        
        # Map to sorted regimes
        regimes = [self.regime_mapping.get(r, r) for r in regimes]
        
        # Calculate transitions
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        # Convert to dict
        matrix_dict = {}
        for i in range(self.n_regimes):
            matrix_dict[self.regime_params[i]['name']] = {
                self.regime_params[j]['name']: float(transition_matrix[i, j])
                for j in range(self.n_regimes)
            }
        
        return matrix_dict
    
    def _generate_recommendations(self, current_regime: int, returns: pd.Series) -> List[str]:
        """Generate actionable recommendations based on regime."""
        recommendations = []
        
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        if current_regime == 0:  # Low volatility
            recommendations.append("✅ Low volatility regime - consider increasing position sizes")
            recommendations.append("📊 Use tighter entry thresholds (z-score ~1.5)")
            recommendations.append("⏱️ Extend holding periods (up to 150 days)")
            
        elif current_regime == 2:  # High volatility
            recommendations.append("⚠️ High volatility regime - reduce position sizes")
            recommendations.append("📊 Use wider entry thresholds (z-score ~2.5)")
            recommendations.append("⏱️ Shorten holding periods (max 50 days)")
            recommendations.append("🛡️ Consider increasing exit thresholds for faster stops")
            
        else:  # Normal
            recommendations.append("📊 Normal volatility regime - use standard parameters")
            
        # Add volatility-specific advice
        if current_vol > 0.30:
            recommendations.append("🚨 Extreme volatility detected - consider reducing overall exposure")
        elif current_vol < 0.10:
            recommendations.append("💤 Very low volatility - mean reversion signals may be weak")
        
        return recommendations


def main():
    """Run regime detection demonstration."""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='D')
    
    # Create returns with regime changes
    n_days = len(dates)
    returns = []
    
    # Low vol period
    returns.extend(np.random.normal(0.0005, 0.005, n_days // 3))
    # High vol period  
    returns.extend(np.random.normal(-0.001, 0.02, n_days // 3))
    # Normal vol period
    returns.extend(np.random.normal(0.0003, 0.01, n_days - 2 * (n_days // 3)))
    
    returns_series = pd.Series(returns, index=dates)
    
    # Initialize detector
    detector = VolatilityRegimeDetector(n_regimes=3)
    
    # Fit regimes
    print("🔍 Fitting volatility regimes...")
    detector.fit_regimes(returns_series)
    
    # Generate report
    report = detector.generate_regime_report(returns_series)
    
    # Display results
    print("\n" + "=" * 60)
    print("VOLATILITY REGIME ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 Current Regime: {report['current_regime']['name']}")
    print(f"   {report['current_regime']['description']}")
    
    print("\n🎯 Recommended Parameters:")
    for param, value in report['current_regime']['recommended_parameters'].items():
        print(f"   {param}: {value}")
    
    print("\n📈 Regime Statistics:")
    for stat in report['regime_statistics']:
        print(f"\n   {stat['name']}:")
        print(f"     Frequency: {stat['frequency']:.1%}")
        print(f"     Avg Volatility: {stat['avg_volatility']:.1%}")
        print(f"     Avg Duration: {stat['avg_duration']:.0f} days")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    # Save report
    output_dir = Path('artifacts/regimes')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'REGIME_ANALYSIS.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Regime analysis saved to {report_path}")


if __name__ == "__main__":
    main()