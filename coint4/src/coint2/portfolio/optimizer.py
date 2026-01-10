"""Cost-aware portfolio optimizer with risk and capacity constraints."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio optimization configuration."""
    
    # Selection parameters
    method: str = "score_topN"  # score_topN | greedy_diversify
    top_n: int = 12
    min_pairs: int = 5
    diversify_by_base: bool = True
    max_per_base: int = 4
    
    # Scoring weights
    alpha_fee: float = 0.5
    beta_slip: float = 0.5
    
    # Optimizer parameters
    lambda_var: float = 2.0  # Risk aversion
    gamma_cost: float = 1.0  # Cost aversion
    max_gross: float = 1.0   # Sum of absolute weights
    net_target: float = 0.0  # Sum of weights (usually 0 for market neutral)
    max_weight_per_pair: float = 0.15
    max_adv_pct: float = 1.0  # Max % of ADV
    turnover_budget_per_day: float = 0.35
    
    # Fallback and other settings
    fallback: str = "vol_target"
    seed: int = 42


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    
    weights: pd.Series
    selected_pairs: List[str]
    diagnostics: Dict
    method_used: str
    success: bool
    message: str


class PortfolioOptimizer:
    """Cost-aware portfolio optimizer."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        np.random.seed(config.seed)
    
    def optimize_portfolio(
        self, 
        metrics_df: pd.DataFrame
    ) -> OptimizationResult:
        """
        Optimize portfolio with cost awareness.
        
        Expected columns in metrics_df:
        - exp_return: Expected return
        - vol: Volatility  
        - psr: Probabilistic Sharpe Ratio
        - est_fee_per_turnover: Fee cost per unit turnover
        - est_slippage_per_turnover: Slippage cost per unit turnover
        - turnover_baseline: Baseline turnover estimate
        - adv_proxy: Average daily volume proxy
        - cap_per_pair: Individual pair capacity cap
        """
        
        logger.info(f"Starting portfolio optimization with {len(metrics_df)} pairs")
        
        try:
            # Step 1: Select pairs
            selected_pairs = self._select_pairs(metrics_df)
            
            if len(selected_pairs) < self.config.min_pairs:
                return self._fallback_allocation(metrics_df, 
                    f"Only {len(selected_pairs)} pairs selected, need ≥ {self.config.min_pairs}")
            
            # Step 2: Filter to selected pairs
            selected_metrics = metrics_df.loc[selected_pairs].copy()
            
            # Step 3: Optimize weights
            if HAS_CVXPY:
                result = self._optimize_cvx(selected_metrics)
            else:
                result = self._optimize_numpy(selected_metrics)
                
            if not result.success:
                return self._fallback_allocation(metrics_df, result.message)
                
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._fallback_allocation(metrics_df, f"Error: {e}")
    
    def _select_pairs(self, metrics_df: pd.DataFrame) -> List[str]:
        """Select subset of pairs based on scoring."""
        
        if self.config.method == "score_topN":
            return self._score_top_n(metrics_df)
        elif self.config.method == "greedy_diversify":
            return self._greedy_diversify(metrics_df)
        else:
            logger.warning(f"Unknown method {self.config.method}, using score_topN")
            return self._score_top_n(metrics_df)
    
    def _score_top_n(self, metrics_df: pd.DataFrame) -> List[str]:
        """Score pairs and select top N."""
        
        df = metrics_df.copy()
        
        # Calculate composite score
        # score = psr - alpha*fee_cost - beta*slippage_cost
        df['fee_cost'] = (df.get('est_fee_per_turnover', 0) * 
                          df.get('turnover_baseline', 0.1))
        df['slip_cost'] = (df.get('est_slippage_per_turnover', 0) * 
                           df.get('turnover_baseline', 0.1))
        
        df['score'] = (df.get('psr', 0) - 
                       self.config.alpha_fee * df['fee_cost'] - 
                       self.config.beta_slip * df['slip_cost'])
        
        # Sort by score
        df_sorted = df.sort_values('score', ascending=False)
        
        # Apply diversification rules
        if self.config.diversify_by_base:
            selected = self._apply_base_diversification(df_sorted)
        else:
            selected = df_sorted.head(self.config.top_n).index.tolist()
        
        logger.info(f"Selected {len(selected)} pairs with scores {df.loc[selected, 'score'].values[:5]}")
        return selected
    
    def _apply_base_diversification(self, df_sorted: pd.DataFrame) -> List[str]:
        """Apply base currency diversification rules."""
        
        selected = []
        base_counts = {}
        
        for pair in df_sorted.index:
            # Extract base currency (assumes format like 'BTC/USDT')
            if '/' in pair:
                base = pair.split('/')[0]
            elif 'USDT' in pair:
                base = pair.replace('USDT', '')
            else:
                base = pair[:3]  # Default fallback
            
            current_count = base_counts.get(base, 0)
            
            if (current_count < self.config.max_per_base and 
                len(selected) < self.config.top_n):
                selected.append(pair)
                base_counts[base] = current_count + 1
        
        return selected
    
    def _greedy_diversify(self, metrics_df: pd.DataFrame) -> List[str]:
        """Greedy diversification selection."""
        # Simplified implementation - can be enhanced with correlation matrix
        return self._score_top_n(metrics_df)  # Fallback for now
    
    def _optimize_cvx(self, metrics_df: pd.DataFrame) -> OptimizationResult:
        """Optimize using CVXPY with enhanced capacity constraints."""
        
        n = len(metrics_df)
        pairs = metrics_df.index.tolist()
        
        # Extract parameters
        mu = metrics_df.get('exp_return', 0).values
        vol = metrics_df.get('vol', 0.15).values
        
        # Simple diagonal covariance (can be enhanced with correlation matrix)
        Sigma = np.diag(vol ** 2)
        
        # Cost estimates
        fee_cost = (metrics_df.get('est_fee_per_turnover', 0.001) * 
                    metrics_df.get('turnover_baseline', 0.1)).values
        slip_cost = (metrics_df.get('est_slippage_per_turnover', 0.0005) * 
                     metrics_df.get('turnover_baseline', 0.1)).values
        total_cost = fee_cost + slip_cost
        
        # Decision variables
        w = cp.Variable(n)
        
        # Objective function: maximize return - risk penalty - cost penalty
        expected_return = mu.T @ w
        risk_penalty = self.config.lambda_var * cp.quad_form(w, Sigma)
        cost_penalty = self.config.gamma_cost * (total_cost.T @ cp.abs(w))
        
        objective = cp.Maximize(expected_return - risk_penalty - cost_penalty)
        
        # Constraints
        constraints = []
        
        # Gross exposure constraint
        constraints.append(cp.sum(cp.abs(w)) <= self.config.max_gross)
        
        # Net exposure constraint
        constraints.append(cp.sum(w) == self.config.net_target)
        
        # Individual weight constraints
        constraints.append(cp.abs(w) <= self.config.max_weight_per_pair)
        
        # Enhanced capacity constraints (v0.2.1)
        capacity_violations = []
        
        # ADV constraints (if available)
        if 'adv_proxy' in metrics_df.columns:
            adv_limits = metrics_df['adv_proxy'] * self.config.max_adv_pct / 100000  # Convert to weight units
            adv_constraint = cp.abs(w) <= adv_limits.values
            constraints.append(adv_constraint)
            
            # Track potential violations for diagnostics
            for i, pair in enumerate(pairs):
                if self.config.max_weight_per_pair > adv_limits.iloc[i]:
                    capacity_violations.append({
                        'pair': pair,
                        'type': 'ADV',
                        'limit': float(adv_limits.iloc[i]),
                        'requested': self.config.max_weight_per_pair
                    })
        
        # Notional position size constraints (v0.2.1)
        if hasattr(self.config, 'min_position_usd') and hasattr(self.config, 'max_position_usd'):
            # Assume $100k total portfolio for conversion
            portfolio_value = 100000
            min_weight = self.config.min_position_usd / portfolio_value
            max_weight = self.config.max_position_usd / portfolio_value
            
            # Apply only to non-zero positions
            is_active = cp.abs(w) >= min_weight
            constraints.append(cp.abs(w) <= max_weight)
            
            # Log notional constraints applied
            logger.info(f"Applied notional limits: ${self.config.min_position_usd}-${self.config.max_position_usd}")
        
        # Turnover budget constraint (simplified)
        if hasattr(self.config, 'turnover_budget_per_day'):
            # Proxy: limit number of active positions
            max_active = int(self.config.turnover_budget_per_day * n)
            # This is approximate - full implementation would need previous weights
            
        # Solve problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights_array = w.value
                if weights_array is not None:
                    weights = pd.Series(weights_array, index=pairs)
                    
                    # Enhanced diagnostics with capacity analysis (v0.2.1)
                    capacity_analysis = self._analyze_capacity_usage(weights, metrics_df)
                    
                    diagnostics = {
                        'objective_value': problem.value,
                        'expected_return': float(mu.T @ weights_array),
                        'expected_risk': float(np.sqrt(weights_array.T @ Sigma @ weights_array)),
                        'expected_cost': float(total_cost.T @ np.abs(weights_array)),
                        'gross_exposure': float(np.sum(np.abs(weights_array))),
                        'net_exposure': float(np.sum(weights_array)),
                        'max_weight': float(np.max(np.abs(weights_array))),
                        'active_pairs': int(np.sum(np.abs(weights_array) > 1e-4)),
                        'solver_status': problem.status,
                        'capacity_violations': capacity_violations,
                        'capacity_analysis': capacity_analysis
                    }
                    
                    return OptimizationResult(
                        weights=weights,
                        selected_pairs=pairs,
                        diagnostics=diagnostics,
                        method_used="cvxpy",
                        success=True,
                        message="CVXPY optimization successful"
                    )
            
            return OptimizationResult(
                weights=pd.Series(),
                selected_pairs=pairs,
                diagnostics={'solver_status': problem.status},
                method_used="cvxpy",
                success=False,
                message=f"CVXPY solver failed: {problem.status}"
            )
            
        except Exception as e:
            logger.error(f"CVXPY optimization failed: {e}")
            return OptimizationResult(
                weights=pd.Series(),
                selected_pairs=pairs,
                diagnostics={},
                method_used="cvxpy",
                success=False,
                message=f"CVXPY error: {e}"
            )

    
    def _analyze_capacity_usage(self, weights: pd.Series, metrics_df: pd.DataFrame) -> Dict:
        """Analyze capacity usage for portfolio positions (v0.2.1)."""
        
        capacity_analysis = {
            'total_pairs': len(weights),
            'active_pairs': len(weights[weights.abs() > 1e-4]),
            'capacity_warnings': [],
            'utilization_by_pair': {}
        }
        
        for pair in weights.index:
            weight = abs(weights[pair])
            if weight < 1e-4:
                continue
                
            pair_analysis = {'weight': weight}
            
            # ADV utilization analysis
            if 'adv_proxy' in metrics_df.columns:
                adv_proxy = metrics_df.loc[pair, 'adv_proxy']
                # Assume weight represents fraction of $100k portfolio
                position_value = weight * 100000
                adv_utilization = position_value / adv_proxy if adv_proxy > 0 else 0
                
                pair_analysis['adv_proxy'] = adv_proxy
                pair_analysis['position_value_usd'] = position_value
                pair_analysis['adv_utilization_pct'] = adv_utilization * 100
                
                # Flag high utilization
                if adv_utilization > 0.01:  # >1% of ADV
                    capacity_analysis['capacity_warnings'].append({
                        'pair': pair,
                        'type': 'HIGH_ADV_UTILIZATION',
                        'utilization_pct': adv_utilization * 100,
                        'recommendation': 'Consider reducing position size'
                    })
            
            # Weight concentration analysis
            if weight > self.config.max_weight_per_pair * 0.8:  # >80% of max
                capacity_analysis['capacity_warnings'].append({
                    'pair': pair,
                    'type': 'HIGH_CONCENTRATION',
                    'weight': weight,
                    'max_allowed': self.config.max_weight_per_pair,
                    'recommendation': 'Near maximum weight limit'
                })
            
            capacity_analysis['utilization_by_pair'][pair] = pair_analysis
        
        return capacity_analysis
    
    def _optimize_numpy(self, metrics_df: pd.DataFrame) -> OptimizationResult:
        """Fallback optimization using NumPy."""
        
        pairs = metrics_df.index.tolist()
        n = len(pairs)
        
        # Simple equal weight with constraints
        base_weight = self.config.max_gross / n
        weights_array = np.full(n, base_weight)
        
        # Apply individual caps
        max_individual = min(self.config.max_weight_per_pair, 
                           self.config.max_gross / self.config.min_pairs)
        weights_array = np.clip(weights_array, 0, max_individual)
        
        # Renormalize to target gross exposure
        current_gross = np.sum(np.abs(weights_array))
        if current_gross > 0:
            weights_array = weights_array * (self.config.max_gross / current_gross)
        
        weights = pd.Series(weights_array, index=pairs)
        
        # Simple diagnostics
        diagnostics = {
            'expected_return': 0.0,  # Not computed in fallback
            'expected_risk': 0.0,
            'expected_cost': 0.0,
            'gross_exposure': float(np.sum(np.abs(weights_array))),
            'net_exposure': float(np.sum(weights_array)),
            'max_weight': float(np.max(np.abs(weights_array))),
            'active_pairs': int(np.sum(np.abs(weights_array) > 1e-4)),
            'solver_status': 'numpy_fallback'
        }
        
        return OptimizationResult(
            weights=weights,
            selected_pairs=pairs,
            diagnostics=diagnostics,
            method_used="numpy_fallback",
            success=True,
            message="NumPy fallback optimization successful"
        )
    
    def _fallback_allocation(
        self, 
        metrics_df: pd.DataFrame, 
        reason: str
    ) -> OptimizationResult:
        """Fallback to existing allocator methods."""
        
        logger.warning(f"Using fallback allocation: {reason}")
        
        # Import existing allocator
        from .allocator import create_allocator
        
        allocator = create_allocator(self.config.fallback, {
            'target_vol': 0.10,
            'max_weight': self.config.max_weight_per_pair
        })
        
        # Simple signals (use PSR as proxy)
        signals = []
        for pair in metrics_df.index:
            from .allocator import Signal
            signal = Signal(
                pair=pair,
                timestamp=pd.Timestamp.now(),
                value=metrics_df.loc[pair].get('psr', 0.5),
                strength=1.0
            )
            signals.append(signal)
        
        # Get allocation
        allocation = allocator.allocate(signals)
        
        weights = pd.Series(allocation.weights, index=allocation.pairs)
        
        diagnostics = {
            'fallback_reason': reason,
            'fallback_method': self.config.fallback,
            'gross_exposure': float(np.sum(np.abs(weights.values))),
            'net_exposure': float(np.sum(weights.values)),
            'active_pairs': len(allocation.pairs)
        }
        
        return OptimizationResult(
            weights=weights,
            selected_pairs=allocation.pairs,
            diagnostics=diagnostics,
            method_used=f"fallback_{self.config.fallback}",
            success=True,
            message=f"Fallback to {self.config.fallback}: {reason}"
        )


def load_config(config_path: Union[str, Path]) -> PortfolioConfig:
    """Load portfolio configuration from YAML."""
    
    import yaml
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract sections
    selection = data.get('selection', {})
    scoring = data.get('scoring', {})
    optimizer = data.get('optimizer', {})
    
    return PortfolioConfig(
        # Selection
        method=selection.get('method', 'score_topN'),
        top_n=selection.get('top_n', 12),
        min_pairs=selection.get('min_pairs', 5),
        diversify_by_base=selection.get('diversify_by_base', True),
        max_per_base=selection.get('max_per_base', 4),
        
        # Scoring
        alpha_fee=scoring.get('alpha_fee', 0.5),
        beta_slip=scoring.get('beta_slip', 0.5),
        
        # Optimizer
        lambda_var=optimizer.get('lambda_var', 2.0),
        gamma_cost=optimizer.get('gamma_cost', 1.0),
        max_gross=optimizer.get('max_gross', 1.0),
        net_target=optimizer.get('net_target', 0.0),
        max_weight_per_pair=optimizer.get('max_weight_per_pair', 0.15),
        max_adv_pct=optimizer.get('max_adv_pct', 1.0),
        turnover_budget_per_day=optimizer.get('turnover_budget_per_day', 0.35),
        
        # Other
        fallback=data.get('fallback', 'vol_target'),
        seed=data.get('seed', 42)
    )


# Main entry point
def optimize_portfolio(metrics_df: pd.DataFrame, cfg: PortfolioConfig) -> Dict:
    """
    Main portfolio optimization function.
    
    Args:
        metrics_df: DataFrame with pair metrics
        cfg: Portfolio configuration
        
    Returns:
        Dictionary with weights, selected_pairs, and diagnostics
    """
    
    optimizer = PortfolioOptimizer(cfg)
    result = optimizer.optimize_portfolio(metrics_df)
    
    return {
        'weights': result.weights,
        'selected_pairs': result.selected_pairs,
        'diagnostics': result.diagnostics
    }