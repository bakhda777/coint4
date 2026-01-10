"""Realistic execution simulator with slippage and latency."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class Order:
    """Trading order."""
    pair: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: float
    order_type: str = 'market'
    
    
@dataclass
class Fill:
    """Order fill/execution."""
    order: Order
    filled_quantity: float
    filled_price: float
    slippage: float
    commission: float
    timestamp: float
    latency_ms: float
    partial: bool = False


class ExecutionSimulator:
    """Simulate realistic order execution."""
    
    def __init__(self, config: dict):
        # Slippage model: slippage = a + b*ATR + c*|z-score|
        self.slippage_base = config.get('slippage_base', 0.0001)  # 1 bp base
        self.slippage_atr_coef = config.get('slippage_atr_coef', 0.1)
        self.slippage_zscore_coef = config.get('slippage_zscore_coef', 0.0002)
        
        # Latency
        self.latency_mean_ms = config.get('latency_mean_ms', 10)
        self.latency_std_ms = config.get('latency_std_ms', 5)
        
        # Partial fills
        self.partial_fill_prob = config.get('partial_fill_prob', 0.1)
        self.partial_fill_ratio = config.get('partial_fill_ratio', 0.7)
        
        # Price limits
        self.max_slippage_pct = config.get('max_slippage_pct', 0.01)  # 1%
        
        # Commission
        self.commission_rate = config.get('commission_rate', 0.001)
        
        # Fill logging
        self.enable_fill_trace = config.get('enable_fill_trace', False)
        self.fills_trace = []
        
        # POV settings
        self.pov_enabled = config.get('pov', {}).get('enabled', False)
        self.pov_participation = config.get('pov', {}).get('participation', 0.10)
        
        # TWAP settings
        self.twap_enabled = config.get('twap', {}).get('enabled', False)
        self.twap_slices = config.get('twap', {}).get('slices', 4)
        
        # Size clipping
        self.clip_enabled = config.get('clip_sizing', {}).get('enabled', False)
        self.max_adv_pct = config.get('clip_sizing', {}).get('max_adv_pct', 1.0)
        
    def simulate_fill(
        self, 
        order: Order,
        atr: float = 0.02,
        zscore: float = 0.0,
        volatility: float = 0.3,
        volume: float = 100000
    ) -> Fill:
        """Simulate order execution with slippage and POV/TWAP."""
        
        # Calculate latency
        latency_ms = max(1, np.random.normal(
            self.latency_mean_ms, 
            self.latency_std_ms
        ))
        
        # Apply POV constraint if enabled
        effective_quantity = order.quantity
        if self.pov_enabled:
            max_pov_quantity = volume * self.pov_participation
            if effective_quantity > max_pov_quantity:
                effective_quantity = max_pov_quantity
        
        # Apply size clipping if enabled
        if self.clip_enabled:
            max_clip_quantity = volume * self.max_adv_pct
            if effective_quantity > max_clip_quantity:
                effective_quantity = max_clip_quantity
        
        # Calculate base slippage
        slippage_pct = (
            self.slippage_base + 
            self.slippage_atr_coef * atr +
            self.slippage_zscore_coef * abs(zscore)
        )
        
        # Reduce slippage with POV/TWAP
        if self.pov_enabled:
            # POV reduces market impact
            participation_rate = effective_quantity / volume
            slippage_pct *= (1 + participation_rate * 0.5)  # Lower impact
        
        if self.twap_enabled and self.twap_slices > 1:
            # TWAP reduces slippage by spreading execution
            slippage_pct *= (1.0 / np.sqrt(self.twap_slices))
        
        # Add volatility-based randomness
        slippage_pct += np.random.normal(0, volatility * 0.001)
        
        # Cap slippage
        slippage_pct = min(slippage_pct, self.max_slippage_pct)
        slippage_pct = max(slippage_pct, 0)
        
        # Apply slippage based on side
        if order.side == 'buy':
            filled_price = order.price * (1 + slippage_pct)
        else:
            filled_price = order.price * (1 - slippage_pct)
            
        # Determine if partial fill (using effective quantity)
        partial = np.random.random() < self.partial_fill_prob
        if partial:
            filled_quantity = effective_quantity * self.partial_fill_ratio
        else:
            filled_quantity = effective_quantity
            
        # Calculate commission
        commission = filled_quantity * filled_price * self.commission_rate
        
        fill = Fill(
            order=order,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            slippage=slippage_pct,
            commission=commission,
            timestamp=order.timestamp + latency_ms / 1000,
            latency_ms=latency_ms,
            partial=partial
        )
        
        # Log fill if enabled
        if self.enable_fill_trace:
            self.fills_trace.append({
                'pair': order.pair,
                'side': order.side,
                'order_qty': order.quantity,
                'filled_qty': filled_quantity,
                'order_price': order.price,
                'filled_price': filled_price,
                'slippage_pct': slippage_pct,
                'commission': commission,
                'latency_ms': latency_ms,
                'partial': partial,
                'timestamp': fill.timestamp,
                'time_to_fill': latency_ms,
                'partial_fill_ratio': filled_quantity / order.quantity
            })
        
        return fill
    
    def simulate_batch(
        self,
        orders: List[Order],
        market_data: Dict[str, dict]
    ) -> List[Fill]:
        """Simulate batch of orders."""
        fills = []
        
        for order in orders:
            # Get market data for pair
            pair_data = market_data.get(order.pair, {})
            atr = pair_data.get('atr', 0.02)
            zscore = pair_data.get('zscore', 0.0)
            volatility = pair_data.get('volatility', 0.3)
            
            fill = self.simulate_fill(order, atr, zscore, volatility)
            fills.append(fill)
            
            # Handle partial fills
            remaining = order.quantity - fill.filled_quantity
            while fill.partial and remaining > 0:
                # Create follow-up order
                follow_up = Order(
                    pair=order.pair,
                    side=order.side,
                    quantity=remaining,
                    price=fill.filled_price,  # Use last fill price
                    timestamp=fill.timestamp,
                    order_type=order.order_type
                )
                
                # Simulate follow-up fill
                next_fill = self.simulate_fill(follow_up, atr, zscore, volatility)
                fills.append(next_fill)
                
                remaining -= next_fill.filled_quantity
                if not next_fill.partial:
                    break
                    
        return fills
    
    def get_fill_statistics(self) -> Dict[str, any]:
        """Get statistics from fill trace."""
        if not self.fills_trace:
            return {}
        
        import pandas as pd
        df = pd.DataFrame(self.fills_trace)
        
        return {
            'total_fills': len(df),
            'partial_fills': df['partial'].sum(),
            'partial_fill_pct': df['partial'].mean() * 100,
            'avg_latency_ms': df['latency_ms'].mean(),
            'median_latency_ms': df['latency_ms'].median(),
            'avg_slippage_pct': df['slippage_pct'].mean() * 100,
            'avg_partial_ratio': df[df['partial']]['partial_fill_ratio'].mean() if df['partial'].any() else 1.0,
            'total_commission': df['commission'].sum()
        }
    
    def calculate_execution_cost(self, fills: List[Fill]) -> dict:
        """Calculate total execution costs."""
        total_slippage_cost = 0
        total_commission = 0
        total_latency_ms = 0
        n_partial = 0
        
        for fill in fills:
            # Slippage cost
            theoretical_value = fill.order.quantity * fill.order.price
            actual_value = fill.filled_quantity * fill.filled_price
            if fill.order.side == 'buy':
                slippage_cost = actual_value - theoretical_value
            else:
                slippage_cost = theoretical_value - actual_value
                
            total_slippage_cost += slippage_cost
            total_commission += fill.commission
            total_latency_ms += fill.latency_ms
            
            if fill.partial:
                n_partial += 1
                
        return {
            'total_slippage_cost': total_slippage_cost,
            'total_commission': total_commission,
            'avg_latency_ms': total_latency_ms / len(fills) if fills else 0,
            'n_fills': len(fills),
            'n_partial': n_partial,
            'partial_rate': n_partial / len(fills) if fills else 0
        }


class ExecutionModel:
    """Advanced execution model with market impact."""
    
    def __init__(self, config: dict):
        self.simulator = ExecutionSimulator(config)
        self.impact_model = config.get('impact_model', 'linear')
        self.impact_coef = config.get('impact_coef', 0.0001)
        
    def estimate_market_impact(
        self,
        order_size: float,
        adv: float,  # Average daily volume
        volatility: float
    ) -> float:
        """Estimate market impact of order."""
        participation_rate = order_size / adv
        
        if self.impact_model == 'linear':
            impact = self.impact_coef * participation_rate
        elif self.impact_model == 'sqrt':
            impact = self.impact_coef * np.sqrt(participation_rate)
        elif self.impact_model == 'power':
            impact = self.impact_coef * (participation_rate ** 0.6)
        else:
            impact = 0
            
        # Scale by volatility
        impact *= volatility
        
        return impact
    
    def optimize_execution(
        self,
        order: Order,
        urgency: float = 0.5,  # 0 = patient, 1 = aggressive
        market_conditions: dict = None
    ) -> List[Order]:
        """Split order for optimal execution."""
        if market_conditions is None:
            market_conditions = {}
            
        volatility = market_conditions.get('volatility', 0.3)
        adv = market_conditions.get('adv', 1000000)
        
        # Estimate impact
        impact = self.estimate_market_impact(order.quantity, adv, volatility)
        
        # Determine splitting strategy based on impact and urgency
        if impact > 0.001 and urgency < 0.7:
            # Split into smaller orders
            n_slices = max(2, int(impact * 1000))
            slice_size = order.quantity / n_slices
            
            slices = []
            for i in range(n_slices):
                slice_order = Order(
                    pair=order.pair,
                    side=order.side,
                    quantity=slice_size,
                    price=order.price,
                    timestamp=order.timestamp + i * 60,  # Space out by 1 minute
                    order_type=order.order_type
                )
                slices.append(slice_order)
                
            return slices
        else:
            # Execute as single order
            return [order]


def create_execution_simulator(config: dict) -> ExecutionSimulator:
    """Factory function for execution simulator."""
    return ExecutionSimulator(config)