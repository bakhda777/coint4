"""Portfolio allocation strategies."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    """Trading signal for a pair."""
    pair: str
    position: float  # -1 to 1
    confidence: float  # 0 to 1
    volatility: float  # annualized vol
    sharpe: float
    current_weight: float = 0.0


@dataclass 
class AllocationResult:
    """Result of portfolio allocation."""
    weights: Dict[str, float]
    gross_exposure: float
    net_exposure: float
    n_positions: int
    method: str


class PortfolioAllocator:
    """Base portfolio allocator."""
    
    def __init__(self, config: dict):
        self.max_weight_per_pair = config.get('max_weight_per_pair', 0.10)
        self.max_gross_exposure = config.get('max_gross', 1.0)
        self.max_net_exposure = config.get('max_net', 0.4)
        
    def allocate(self, signals: List[Signal]) -> AllocationResult:
        """Allocate portfolio weights to signals."""
        raise NotImplementedError
        
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position and exposure constraints."""
        # Cap individual weights
        for pair in weights:
            if abs(weights[pair]) > self.max_weight_per_pair:
                weights[pair] = np.sign(weights[pair]) * self.max_weight_per_pair
        
        # Check gross exposure
        gross = sum(abs(w) for w in weights.values())
        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            weights = {k: v * scale for k, v in weights.items()}
        
        # Check net exposure  
        net = sum(weights.values())
        if abs(net) > self.max_net_exposure:
            # Reduce positions proportionally
            if net > 0:
                longs = {k: v for k, v in weights.items() if v > 0}
                scale = (self.max_net_exposure / net)
                for k in longs:
                    weights[k] *= scale
            else:
                shorts = {k: v for k, v in weights.items() if v < 0}
                scale = (self.max_net_exposure / abs(net))
                for k in shorts:
                    weights[k] *= scale
                    
        return weights


class EqualWeightAllocator(PortfolioAllocator):
    """Equal weight allocation."""
    
    def allocate(self, signals: List[Signal]) -> AllocationResult:
        """Allocate equal weights to all signals."""
        if not signals:
            return AllocationResult({}, 0, 0, 0, 'equal_weight')
            
        # Filter active signals
        active = [s for s in signals if abs(s.position) > 0.01]
        if not active:
            return AllocationResult({}, 0, 0, 0, 'equal_weight')
        
        # Equal weight
        weight_per_signal = 1.0 / len(active)
        weights = {s.pair: weight_per_signal * np.sign(s.position) for s in active}
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return AllocationResult(
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            n_positions=len(weights),
            method='equal_weight'
        )


class VolTargetAllocator(PortfolioAllocator):
    """Volatility targeting allocation."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_vol = config.get('target_vol', 0.10)
        
    def allocate(self, signals: List[Signal]) -> AllocationResult:
        """Allocate weights to target volatility."""
        if not signals:
            return AllocationResult({}, 0, 0, 0, 'vol_target')
            
        # Filter active signals
        active = [s for s in signals if abs(s.position) > 0.01]
        if not active:
            return AllocationResult({}, 0, 0, 0, 'vol_target')
        
        # Inverse vol weighting
        weights = {}
        for signal in active:
            if signal.volatility > 0:
                # Weight inversely proportional to volatility
                raw_weight = (self.target_vol / signal.volatility) * signal.position
                weights[signal.pair] = raw_weight
        
        if not weights:
            return AllocationResult({}, 0, 0, 0, 'vol_target')
        
        # Normalize to sum to 1
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return AllocationResult(
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            n_positions=len(weights),
            method='vol_target'
        )


class RiskParityAllocator(PortfolioAllocator):
    """Simple risk parity allocation (no external dependencies)."""
    
    def allocate(self, signals: List[Signal]) -> AllocationResult:
        """Allocate weights using simplified risk parity."""
        if not signals:
            return AllocationResult({}, 0, 0, 0, 'risk_parity')
            
        # Filter active signals
        active = [s for s in signals if abs(s.position) > 0.01]
        if not active:
            return AllocationResult({}, 0, 0, 0, 'risk_parity')
        
        # Simple HRP: weight by inverse vol * sharpe
        weights = {}
        risk_scores = []
        
        for signal in active:
            if signal.volatility > 0:
                # Risk-adjusted score
                risk_score = abs(signal.sharpe) / signal.volatility
                risk_scores.append(risk_score)
                weights[signal.pair] = risk_score * np.sign(signal.position)
            else:
                risk_scores.append(0)
                weights[signal.pair] = 0
        
        # Normalize 
        total_score = sum(abs(w) for w in weights.values())
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return AllocationResult(
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            n_positions=len([w for w in weights.values() if abs(w) > 0.001]),
            method='risk_parity'
        )


class CapPerPairAllocator(PortfolioAllocator):
    """Allocation with strict per-pair caps."""
    
    def allocate(self, signals: List[Signal]) -> AllocationResult:
        """Allocate with strict position caps."""
        if not signals:
            return AllocationResult({}, 0, 0, 0, 'cap_per_pair')
            
        # Filter and sort by confidence/sharpe
        active = [s for s in signals if abs(s.position) > 0.01]
        if not active:
            return AllocationResult({}, 0, 0, 0, 'cap_per_pair')
        
        # Sort by abs(sharpe) * confidence
        active.sort(key=lambda s: abs(s.sharpe) * s.confidence, reverse=True)
        
        weights = {}
        remaining_exposure = self.max_gross_exposure
        
        for signal in active:
            # Allocate up to cap
            desired_weight = min(
                self.max_weight_per_pair,
                remaining_exposure
            ) * np.sign(signal.position)
            
            weights[signal.pair] = desired_weight
            remaining_exposure -= abs(desired_weight)
            
            if remaining_exposure <= 0:
                break
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return AllocationResult(
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            n_positions=len(weights),
            method='cap_per_pair'
        )


def create_allocator(config: dict) -> PortfolioAllocator:
    """Factory function to create allocator."""
    method = config.get('method', 'equal_weight')
    
    if method == 'equal_weight':
        return EqualWeightAllocator(config)
    elif method == 'vol_target':
        return VolTargetAllocator(config)
    elif method == 'risk_parity':
        return RiskParityAllocator(config)
    elif method == 'cap_per_pair':
        return CapPerPairAllocator(config)
    else:
        raise ValueError(f"Unknown allocation method: {method}")