#!/usr/bin/env python3
"""
Portfolio risk overlay manager.
Implements vol-targeting, drawdown de-risking, and turnover constraints.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class PortfolioRiskManager:
    """Manages portfolio risk overlays and position adjustments."""
    
    def __init__(self, config_path: str = "configs/risk_overlays.yaml"):
        """Initialize with risk configuration."""
        self.config = self._load_config(config_path)
        self.risk_state = {
            'current_vol': None,
            'target_vol': self.config.get('vol_target', 0.15),
            'max_drawdown': self.config.get('max_drawdown', 0.20),
            'current_drawdown': 0,
            'turnover_limit': self.config.get('turnover_limit_daily', 2.0),
            'current_turnover': 0,
            'risk_scalar': 1.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load risk overlay configuration."""
        if Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'vol_target': 0.15,
                'vol_lookback_days': 20,
                'max_drawdown': 0.20,
                'drawdown_reduction_rate': 0.5,
                'turnover_limit_daily': 2.0,
                'min_risk_scalar': 0.1,
                'max_risk_scalar': 2.0
            }
    
    def calculate_volatility(self, returns: pd.Series, 
                           lookback_days: Optional[int] = None) -> float:
        """Calculate realized volatility."""
        if lookback_days is None:
            lookback_days = self.config.get('vol_lookback_days', 20)
        
        # Annualized volatility
        daily_vol = returns.tail(lookback_days).std()
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate current drawdown from peak."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        return abs(current_dd)
    
    def calculate_turnover(self, positions_current: Dict[str, float],
                          positions_previous: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        turnover = 0
        all_assets = set(positions_current.keys()) | set(positions_previous.keys())
        
        for asset in all_assets:
            curr_pos = positions_current.get(asset, 0)
            prev_pos = positions_previous.get(asset, 0)
            turnover += abs(curr_pos - prev_pos)
        
        return turnover
    
    def apply_vol_targeting(self, positions: Dict[str, float],
                           current_vol: float) -> Tuple[Dict[str, float], float]:
        """Apply volatility targeting to positions."""
        if current_vol <= 0:
            return positions, 1.0
        
        target_vol = self.risk_state['target_vol']
        vol_scalar = min(target_vol / current_vol, 
                        self.config.get('max_risk_scalar', 2.0))
        vol_scalar = max(vol_scalar, 
                        self.config.get('min_risk_scalar', 0.1))
        
        # Scale all positions
        scaled_positions = {
            asset: pos * vol_scalar 
            for asset, pos in positions.items()
        }
        
        self.risk_state['current_vol'] = current_vol
        
        return scaled_positions, vol_scalar
    
    def apply_drawdown_control(self, positions: Dict[str, float],
                              current_drawdown: float) -> Tuple[Dict[str, float], float]:
        """Apply drawdown-based position reduction."""
        max_dd = self.risk_state['max_drawdown']
        
        if current_drawdown > max_dd * 0.5:  # Start reducing at 50% of max DD
            # Linear reduction
            reduction_rate = self.config.get('drawdown_reduction_rate', 0.5)
            dd_ratio = current_drawdown / max_dd
            dd_scalar = 1.0 - (reduction_rate * min(dd_ratio, 1.0))
            dd_scalar = max(dd_scalar, self.config.get('min_risk_scalar', 0.1))
            
            scaled_positions = {
                asset: pos * dd_scalar 
                for asset, pos in positions.items()
            }
            
            self.risk_state['current_drawdown'] = current_drawdown
            
            return scaled_positions, dd_scalar
        
        return positions, 1.0
    
    def apply_turnover_limit(self, positions_target: Dict[str, float],
                            positions_current: Dict[str, float]) -> Dict[str, float]:
        """Apply turnover constraints to limit trading."""
        turnover = self.calculate_turnover(positions_target, positions_current)
        limit = self.risk_state['turnover_limit']
        
        if turnover > limit:
            # Scale down changes proportionally
            scale_factor = limit / turnover
            
            adjusted_positions = {}
            for asset in set(positions_target.keys()) | set(positions_current.keys()):
                target = positions_target.get(asset, 0)
                current = positions_current.get(asset, 0)
                change = target - current
                
                # Scale down the change
                adjusted_change = change * scale_factor
                adjusted_positions[asset] = current + adjusted_change
            
            self.risk_state['current_turnover'] = limit
            return adjusted_positions
        
        self.risk_state['current_turnover'] = turnover
        return positions_target
    
    def apply_all_overlays(self, positions: Dict[str, float],
                          market_data: Dict) -> Dict[str, float]:
        """Apply all risk overlays in sequence."""
        adjusted_positions = positions.copy()
        total_scalar = 1.0
        
        # 1. Volatility targeting
        if 'returns' in market_data:
            current_vol = self.calculate_volatility(market_data['returns'])
            adjusted_positions, vol_scalar = self.apply_vol_targeting(
                adjusted_positions, current_vol
            )
            total_scalar *= vol_scalar
        
        # 2. Drawdown control
        if 'equity_curve' in market_data:
            current_dd = self.calculate_drawdown(market_data['equity_curve'])
            adjusted_positions, dd_scalar = self.apply_drawdown_control(
                adjusted_positions, current_dd
            )
            total_scalar *= dd_scalar
        
        # 3. Turnover limit
        if 'current_positions' in market_data:
            adjusted_positions = self.apply_turnover_limit(
                adjusted_positions, market_data['current_positions']
            )
        
        self.risk_state['risk_scalar'] = total_scalar
        
        return adjusted_positions
    
    def get_risk_report(self) -> Dict:
        """Generate risk overlay report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_state': self.risk_state.copy(),
            'active_overlays': {
                'vol_targeting': self.risk_state['current_vol'] is not None,
                'drawdown_control': self.risk_state['current_drawdown'] > 0,
                'turnover_limit': self.risk_state['current_turnover'] > 0
            },
            'risk_scalar': self.risk_state['risk_scalar'],
            'warnings': self._generate_warnings()
        }
    
    def _generate_warnings(self) -> List[str]:
        """Generate risk warnings based on current state."""
        warnings = []
        
        if self.risk_state['current_vol'] and \
           self.risk_state['current_vol'] > self.risk_state['target_vol'] * 1.5:
            warnings.append(f"⚠️ High volatility: {self.risk_state['current_vol']:.1%} vs target {self.risk_state['target_vol']:.1%}")
        
        if self.risk_state['current_drawdown'] > self.risk_state['max_drawdown'] * 0.8:
            warnings.append(f"⚠️ Approaching max drawdown: {self.risk_state['current_drawdown']:.1%}")
        
        if self.risk_state['current_turnover'] >= self.risk_state['turnover_limit']:
            warnings.append(f"⚠️ Turnover limit reached: {self.risk_state['current_turnover']:.1f}x")
        
        if self.risk_state['risk_scalar'] < 0.5:
            warnings.append(f"⚠️ Significant risk reduction applied: {self.risk_state['risk_scalar']:.1%}")
        
        return warnings


def run_risk_overlay_demo():
    """Demonstrate risk overlay functionality."""
    
    # Initialize manager
    manager = PortfolioRiskManager()
    
    # Simulate market data
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
    equity_curve = (1 + returns).cumprod() * 100000
    
    # Add a drawdown
    equity_curve.iloc[100:120] *= 0.85
    
    # Sample positions
    positions = {
        'BTC-ETH': 0.3,
        'BTC-BNB': 0.2,
        'ETH-BNB': 0.25,
        'BTC-SOL': 0.25
    }
    
    current_positions = {
        'BTC-ETH': 0.25,
        'BTC-BNB': 0.15,
        'ETH-BNB': 0.20,
        'BTC-SOL': 0.20
    }
    
    # Apply overlays
    market_data = {
        'returns': returns,
        'equity_curve': equity_curve,
        'current_positions': current_positions
    }
    
    adjusted_positions = manager.apply_all_overlays(positions, market_data)
    
    # Generate report
    report = manager.get_risk_report()
    
    # Display results
    print("=" * 60)
    print("PORTFOLIO RISK OVERLAY DEMONSTRATION")
    print("=" * 60)
    
    print("\n📊 Original Positions:")
    for asset, weight in positions.items():
        print(f"  {asset}: {weight:.1%}")
    
    print("\n🛡️ Adjusted Positions (after overlays):")
    for asset, weight in adjusted_positions.items():
        print(f"  {asset}: {weight:.1%}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"  Current Vol: {report['risk_state']['current_vol']:.1%}" if report['risk_state']['current_vol'] else "  Current Vol: N/A")
    print(f"  Target Vol: {report['risk_state']['target_vol']:.1%}")
    print(f"  Current DD: {report['risk_state']['current_drawdown']:.1%}")
    print(f"  Max DD: {report['risk_state']['max_drawdown']:.1%}")
    print(f"  Turnover: {report['risk_state']['current_turnover']:.2f}x")
    print(f"  Risk Scalar: {report['risk_state']['risk_scalar']:.1%}")
    
    if report['warnings']:
        print("\n⚠️ Risk Warnings:")
        for warning in report['warnings']:
            print(f"  {warning}")
    
    # Save report
    output_dir = Path('artifacts/portfolio')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'RISK_OVERLAY_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Risk overlay report saved to {report_path}")


if __name__ == "__main__":
    run_risk_overlay_demo()