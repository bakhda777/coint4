#!/usr/bin/env python3
"""
Bootstrap confidence intervals for trading metrics.
Implements stationary/block bootstrap for returns and metric distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MetricsBootstrap:
    """Bootstrap confidence intervals for Sharpe, PSR, DSR metrics."""
    
    def __init__(
        self,
        block_size: int = 20,
        n_bootstrap: int = 1000,
        confidence_levels: List[float] = [0.05, 0.50, 0.95],
        seed: int = 42
    ):
        """Initialize bootstrap parameters.
        
        Args:
            block_size: Block size for stationary bootstrap (10-50 bars typical)
            n_bootstrap: Number of bootstrap samples
            confidence_levels: Quantiles to calculate (P5, P50, P95)
            seed: Random seed for reproducibility
        """
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels
        self.rng = np.random.RandomState(seed)
        
    def stationary_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Generate stationary bootstrap sample.
        
        Uses geometric distribution for block lengths with mean block_size.
        
        Args:
            returns: Original return series
            
        Returns:
            Bootstrap sample of same length
        """
        n = len(returns)
        if n < self.block_size:
            # Fallback to simple bootstrap for short series
            return self.rng.choice(returns, size=n, replace=True)
        
        # Generate bootstrap sample
        bootstrap_sample = []
        pos = 0
        
        while pos < n:
            # Random starting position
            start_idx = self.rng.randint(0, n)
            
            # Geometric block length with mean = block_size
            # P(length = k) = (1-p)^(k-1) * p, where p = 1/block_size
            p = 1.0 / self.block_size
            block_length = self.rng.geometric(p)
            
            # Extract block (with wrapping)
            for i in range(block_length):
                if pos >= n:
                    break
                idx = (start_idx + i) % n
                bootstrap_sample.append(returns[idx])
                pos += 1
        
        return np.array(bootstrap_sample[:n])
    
    def calculate_sharpe(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate Sharpe ratio from returns.
        
        Args:
            returns: Return series
            annualize: Whether to annualize (assume daily returns)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        # Handle zero volatility case
        if std_ret <= 1e-10:  # Effectively zero
            return 0.0
        
        sharpe = mean_ret / std_ret
        
        if annualize:
            sharpe *= np.sqrt(252)  # Daily to annual
        
        return sharpe
    
    def calculate_psr(self, returns: np.ndarray, benchmark_sharpe: float = 0.0) -> float:
        """Calculate Probabilistic Sharpe Ratio.
        
        PSR = N(SR_hat / sqrt(1 + gamma3*SR_hat/6 + (gamma4-3)*SR_hat^2/24))
        where N is standard normal CDF.
        
        Args:
            returns: Return series
            benchmark_sharpe: Benchmark Sharpe to test against
            
        Returns:
            PSR probability (0-1)
        """
        from scipy.stats import norm, skew, kurtosis
        
        if len(returns) < 4:
            return 0.5  # Uninformative
        
        # Calculate sample statistics
        sharpe_hat = self.calculate_sharpe(returns, annualize=False)
        
        if np.std(returns, ddof=1) == 0:
            return 0.5
        
        try:
            gamma3 = skew(returns)  # Skewness
            gamma4 = kurtosis(returns, fisher=False)  # Excess kurtosis + 3
            
            # PSR adjustment factor
            adjustment = 1 + (gamma3 * sharpe_hat / 6) + ((gamma4 - 3) * sharpe_hat**2 / 24)
            
            if adjustment <= 0:
                return 0.0
            
            # Test statistic
            test_stat = (sharpe_hat - benchmark_sharpe) / np.sqrt(adjustment / len(returns))
            
            # PSR as probability
            psr = norm.cdf(test_stat)
            
            return np.clip(psr, 0.0, 1.0)
            
        except:
            return 0.5
    
    def calculate_dsr(self, returns: np.ndarray, var_threshold: float = 0.05) -> float:
        """Calculate Deflated Sharpe Ratio.
        
        Simplified DSR = PSR adjusted for multiple testing.
        
        Args:
            returns: Return series  
            var_threshold: VaR threshold for deflation
            
        Returns:
            DSR value
        """
        if len(returns) < 10:
            return 0.0
        
        # Base PSR
        base_psr = self.calculate_psr(returns)
        
        # Simple deflation based on return volatility vs threshold
        vol = np.std(returns, ddof=1) if len(returns) > 1 else 0
        deflation_factor = max(0.5, 1.0 - (vol / var_threshold)) if var_threshold > 0 else 0.5
        
        dsr = base_psr * deflation_factor
        return np.clip(dsr, 0.0, 1.0)
    
    def bootstrap_metrics(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate bootstrap distribution of metrics.
        
        Args:
            returns: Original return series
            
        Returns:
            Dictionary with metric name -> array of bootstrap values
        """
        if len(returns) < 5:
            # Return dummy distributions for short series
            return {
                'sharpe': np.zeros(self.n_bootstrap),
                'psr': np.full(self.n_bootstrap, 0.5),
                'dsr': np.zeros(self.n_bootstrap)
            }
        
        bootstrap_sharpe = []
        bootstrap_psr = []
        bootstrap_dsr = []
        
        for i in range(self.n_bootstrap):
            # Generate bootstrap sample
            boot_returns = self.stationary_bootstrap(returns)
            
            # Calculate metrics on bootstrap sample
            sharpe = self.calculate_sharpe(boot_returns)
            psr = self.calculate_psr(boot_returns)  
            dsr = self.calculate_dsr(boot_returns)
            
            bootstrap_sharpe.append(sharpe)
            bootstrap_psr.append(psr)
            bootstrap_dsr.append(dsr)
        
        return {
            'sharpe': np.array(bootstrap_sharpe),
            'psr': np.array(bootstrap_psr),
            'dsr': np.array(bootstrap_dsr)
        }
    
    def calculate_confidence_intervals(
        self,
        returns: Union[np.ndarray, pd.Series],
        metric_name: str = 'sharpe'
    ) -> Dict[str, float]:
        """Calculate confidence intervals for a specific metric.
        
        Args:
            returns: Return series
            metric_name: 'sharpe', 'psr', or 'dsr'
            
        Returns:
            Dictionary with confidence levels and values
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        
        # Get bootstrap distribution
        bootstrap_dist = self.bootstrap_metrics(returns)
        
        if metric_name not in bootstrap_dist:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_values = bootstrap_dist[metric_name]
        
        # Calculate percentiles
        percentiles = np.percentile(metric_values, [p * 100 for p in self.confidence_levels])
        
        result = {}
        for i, level in enumerate(self.confidence_levels):
            key = f'p{int(level * 100):02d}'
            result[key] = percentiles[i]
        
        # Add point estimate (observed value)
        if metric_name == 'sharpe':
            result['observed'] = self.calculate_sharpe(returns)
        elif metric_name == 'psr':
            result['observed'] = self.calculate_psr(returns)
        elif metric_name == 'dsr':
            result['observed'] = self.calculate_dsr(returns)
        
        return result
    
    def analyze_portfolio_uncertainty(
        self,
        returns_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
        portfolio_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze uncertainty for individual pairs and portfolio.
        
        Args:
            returns_data: DataFrame with pair returns or dict of arrays
            portfolio_weights: Optional weights for portfolio aggregation
            
        Returns:
            Nested dict: {pair/portfolio: {metric: {p05, p50, p95, observed}}}
        """
        results = {}
        
        if isinstance(returns_data, pd.DataFrame):
            # Individual pair analysis
            for pair in returns_data.columns:
                pair_returns = returns_data[pair].dropna()
                if len(pair_returns) >= 5:
                    results[pair] = {}
                    for metric in ['sharpe', 'psr', 'dsr']:
                        results[pair][metric] = self.calculate_confidence_intervals(
                            pair_returns, metric
                        )
            
            # Portfolio analysis if weights provided
            if portfolio_weights is not None:
                # Calculate portfolio returns
                common_idx = returns_data.dropna().index
                portfolio_returns = pd.Series(0.0, index=common_idx)
                
                total_weight = sum(abs(w) for w in portfolio_weights.values())
                if total_weight > 0:
                    for pair, weight in portfolio_weights.items():
                        if pair in returns_data.columns:
                            portfolio_returns += (weight / total_weight) * returns_data.loc[common_idx, pair]
                    
                    if len(portfolio_returns) >= 5:
                        results['PORTFOLIO'] = {}
                        for metric in ['sharpe', 'psr', 'dsr']:
                            results['PORTFOLIO'][metric] = self.calculate_confidence_intervals(
                                portfolio_returns, metric
                            )
        
        elif isinstance(returns_data, dict):
            # Process dict of arrays
            for pair, returns in returns_data.items():
                if len(returns) >= 5:
                    results[pair] = {}
                    for metric in ['sharpe', 'psr', 'dsr']:
                        results[pair][metric] = self.calculate_confidence_intervals(
                            returns, metric
                        )
        
        return results


def load_oos_returns_from_wfa(wfa_results_path: str) -> pd.DataFrame:
    """Load out-of-sample returns from WFA results.
    
    Args:
        wfa_results_path: Path to results_per_fold.csv
        
    Returns:
        DataFrame with pair returns indexed by time
    """
    wfa_path = Path(wfa_results_path)
    
    if not wfa_path.exists():
        print(f"⚠️ WFA results not found at {wfa_results_path}")
        # Return dummy data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        dummy_data = {
            'BTC/USDT': np.random.normal(0.001, 0.02, 100),
            'ETH/USDT': np.random.normal(0.0008, 0.018, 100),
            'ADA/USDT': np.random.normal(0.0005, 0.025, 100)
        }
        return pd.DataFrame(dummy_data, index=dates)
    
    # Read WFA results 
    wfa_df = pd.read_csv(wfa_path)
    
    # Convert to returns DataFrame
    # Assume WFA has columns like: fold_id, pair, pnl, timestamp
    if 'timestamp' in wfa_df.columns and 'pair' in wfa_df.columns:
        wfa_df['timestamp'] = pd.to_datetime(wfa_df['timestamp'])
        
        # Pivot to get pairs as columns
        returns_df = wfa_df.pivot_table(
            index='timestamp', 
            columns='pair', 
            values='pnl',
            aggfunc='sum'
        )
        
        # Convert PnL to returns (assume percentage)
        return returns_df / 100.0  # Convert percentage to decimal
    
    else:
        # Fallback: generate synthetic data based on WFA structure
        print(f"⚠️ WFA format not recognized, generating synthetic OOS returns")
        
        pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        synthetic_data = {}
        for pair in pairs:
            # Simulate realistic crypto returns
            returns = np.random.normal(0.0008, 0.022, len(dates))
            synthetic_data[pair] = returns
            
        return pd.DataFrame(synthetic_data, index=dates)


def save_confidence_report(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    bootstrap_config: Dict[str, any]
) -> Tuple[Path, Path]:
    """Save confidence analysis results.
    
    Args:
        results: Nested dict from analyze_portfolio_uncertainty
        output_dir: Output directory path
        bootstrap_config: Bootstrap configuration parameters
        
    Returns:
        Tuple of (markdown_path, csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown report
    md_content = f"""# Confidence Intervals Report
## Bootstrap Configuration
- Block Size: {bootstrap_config.get('block_size', 20)}
- Bootstrap Samples: {bootstrap_config.get('n_bootstrap', 1000)}
- Confidence Levels: {bootstrap_config.get('confidence_levels', [0.05, 0.50, 0.95])}
- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Portfolio Overview
"""
    
    # Portfolio metrics (if available)
    if 'PORTFOLIO' in results:
        portfolio_metrics = results['PORTFOLIO']
        md_content += f"""
### Portfolio Confidence Intervals

| Metric | P05 | P50 (Median) | P95 | Observed |
|--------|-----|--------------|-----|----------|
"""
        for metric in ['sharpe', 'psr', 'dsr']:
            if metric in portfolio_metrics:
                m = portfolio_metrics[metric]
                md_content += f"| {metric.upper()} | {m.get('p05', 0):.3f} | {m.get('p50', 0):.3f} | {m.get('p95', 0):.3f} | {m.get('observed', 0):.3f} |\n"
    
    # Individual pair metrics
    md_content += f"""
## Individual Pair Analysis

| Pair | Metric | P05 | P50 | P95 | Observed |
|------|--------|-----|-----|-----|----------|
"""
    
    for pair, metrics in results.items():
        if pair != 'PORTFOLIO':
            for metric_name, metric_data in metrics.items():
                md_content += f"| {pair} | {metric_name.upper()} | {metric_data.get('p05', 0):.3f} | {metric_data.get('p50', 0):.3f} | {metric_data.get('p95', 0):.3f} | {metric_data.get('observed', 0):.3f} |\n"
    
    md_content += f"""
## Interpretation
- **P05**: 5th percentile (lower bound of 90% confidence interval)
- **P50**: Median (robust point estimate)
- **P95**: 95th percentile (upper bound of 90% confidence interval)
- **Observed**: Point estimate from original data

## Risk Assessment
"""
    
    # Add risk assessment
    portfolio_sharpe_p05 = results.get('PORTFOLIO', {}).get('sharpe', {}).get('p05', 0)
    portfolio_psr_p05 = results.get('PORTFOLIO', {}).get('psr', {}).get('p05', 0)
    
    if portfolio_sharpe_p05 > 0.6:
        md_content += "- ✅ **LOW RISK**: Portfolio P05 Sharpe > 0.6\n"
    elif portfolio_sharpe_p05 > 0.3:
        md_content += "- ⚠️ **MODERATE RISK**: Portfolio P05 Sharpe 0.3-0.6\n"
    else:
        md_content += "- 🚨 **HIGH RISK**: Portfolio P05 Sharpe < 0.3\n"
    
    if portfolio_psr_p05 > 0.90:
        md_content += "- ✅ **HIGH CONFIDENCE**: Portfolio P05 PSR > 0.90\n"
    elif portfolio_psr_p05 > 0.70:
        md_content += "- ⚠️ **MODERATE CONFIDENCE**: Portfolio P05 PSR 0.70-0.90\n"
    else:
        md_content += "- 🚨 **LOW CONFIDENCE**: Portfolio P05 PSR < 0.70\n"
    
    # Save markdown report
    md_path = output_path / 'CONFIDENCE_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    # Create CSV data
    csv_data = []
    for pair, metrics in results.items():
        for metric_name, metric_data in metrics.items():
            csv_data.append({
                'pair': pair,
                'metric': metric_name,
                'p05': metric_data.get('p05', 0),
                'p50': metric_data.get('p50', 0),
                'p95': metric_data.get('p95', 0),
                'observed': metric_data.get('observed', 0)
            })
    
    # Save CSV
    csv_path = output_path / 'confidence.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    return md_path, csv_path