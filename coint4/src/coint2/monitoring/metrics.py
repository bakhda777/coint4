"""Trading metrics and monitoring."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Container for trading metrics."""
    
    # Performance metrics
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    exposure: float = 0.0  # Total position value / capital
    
    # Activity metrics
    trade_count: int = 0
    active_positions: int = 0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    avg_trade_duration: float = 0.0  # hours
    
    # System metrics
    latency_ms: float = 0.0
    uptime_hours: float = 0.0
    
    # Timestamps
    last_trade_time: Optional[str] = None
    last_update_time: str = None
    

class MetricsCalculator:
    """Calculates and tracks trading metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = TradingMetrics()
        self.trade_history = []
        self.pnl_history = []
        self.start_time = time.time()
        self.last_update = time.time()
        
    def update_metrics(self, positions: Dict, trades: List[Dict], balance: Dict[str, float]):
        """Update all metrics based on current state.
        
        Args:
            positions: Current positions
            trades: Trade history
            balance: Current balance
        """
        start_time = time.time()
        
        # Update trade history
        self.trade_history = trades
        
        # Calculate performance metrics
        self._calculate_pnl_metrics(trades, balance)
        self._calculate_risk_metrics(positions, balance)
        self._calculate_activity_metrics(positions, trades)
        self._calculate_performance_metrics()
        
        # Update system metrics
        self.metrics.uptime_hours = (time.time() - self.start_time) / 3600
        self.metrics.latency_ms = (time.time() - start_time) * 1000
        self.metrics.last_update_time = datetime.now().isoformat()
        
        self.last_update = time.time()
        
    def _calculate_pnl_metrics(self, trades: List[Dict], balance: Dict[str, float]):
        """Calculate PnL-related metrics."""
        if not trades:
            return
            
        # Calculate realized PnL from closed trades
        closed_trades = [t for t in trades if 'pnl' in t and t['pnl'] is not None]
        
        if closed_trades:
            total_pnl = sum(t['pnl'] for t in closed_trades)
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            
            self.metrics.total_pnl = total_pnl
            self.metrics.win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # Track PnL for drawdown calculation
        current_balance = balance.get('USDT', 100000)
        self.pnl_history.append({
            'timestamp': datetime.now(),
            'balance': current_balance
        })
        
        # Keep only recent history for calculations
        cutoff = datetime.now() - timedelta(days=30)
        self.pnl_history = [p for p in self.pnl_history if p['timestamp'] > cutoff]
        
    def _calculate_risk_metrics(self, positions: Dict, balance: Dict[str, float]):
        """Calculate risk-related metrics."""
        # Calculate drawdown
        if self.pnl_history:
            balances = [p['balance'] for p in self.pnl_history]
            peak_balance = max(balances)
            current_balance = balances[-1]
            
            self.metrics.current_drawdown = (peak_balance - current_balance) / peak_balance
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown,
                self.metrics.current_drawdown
            )
        
        # Calculate exposure
        total_capital = balance.get('USDT', 100000)
        position_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        self.metrics.exposure = position_value / total_capital if total_capital > 0 else 0
        
    def _calculate_activity_metrics(self, positions: Dict, trades: List[Dict]):
        """Calculate activity-related metrics."""
        self.metrics.active_positions = len([p for p in positions.values() if p.get('amount', 0) != 0])
        self.metrics.trade_count = len(trades)
        
        # Last trade time
        if trades:
            last_trade = max(trades, key=lambda t: t.get('timestamp', ''))
            self.metrics.last_trade_time = last_trade.get('timestamp')
        
        # Average trade duration
        closed_trades = [t for t in trades if t.get('exit_time')]
        if closed_trades:
            durations = []
            for trade in closed_trades:
                try:
                    entry_time = datetime.fromisoformat(trade['timestamp'])
                    exit_time = datetime.fromisoformat(trade['exit_time'])
                    duration = (exit_time - entry_time).total_seconds() / 3600
                    durations.append(duration)
                except:
                    continue
            
            self.metrics.avg_trade_duration = np.mean(durations) if durations else 0
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics like Sharpe ratio."""
        if len(self.pnl_history) < 2:
            return
        
        # Calculate returns from PnL history
        returns = []
        for i in range(1, len(self.pnl_history)):
            prev_balance = self.pnl_history[i-1]['balance']
            curr_balance = self.pnl_history[i]['balance']
            if prev_balance > 0:
                returns.append((curr_balance - prev_balance) / prev_balance)
        
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized Sharpe ratio (assuming returns are per update)
            if std_return > 0:
                # Assuming updates every 15 minutes = 96 updates per day
                updates_per_year = 96 * 365
                self.metrics.sharpe_ratio = np.sqrt(updates_per_year) * mean_return / std_return
    
    def get_metrics(self) -> TradingMetrics:
        """Get current metrics."""
        return self.metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_pnl': self.metrics.total_pnl,
            'unrealized_pnl': self.metrics.unrealized_pnl,
            'win_rate': self.metrics.win_rate,
            'max_drawdown': self.metrics.max_drawdown,
            'current_drawdown': self.metrics.current_drawdown,
            'exposure': self.metrics.exposure,
            'trade_count': self.metrics.trade_count,
            'active_positions': self.metrics.active_positions,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'avg_trade_duration_hours': self.metrics.avg_trade_duration,
            'latency_ms': self.metrics.latency_ms,
            'uptime_hours': self.metrics.uptime_hours,
            'last_trade_time': self.metrics.last_trade_time,
            'last_update_time': self.metrics.last_update_time
        }


class DashboardGenerator:
    """Generates live trading dashboard."""
    
    def __init__(self, output_path: str = "artifacts/live/LIVE_DASHBOARD.md"):
        """Initialize dashboard generator.
        
        Args:
            output_path: Path to save dashboard markdown
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard(self, metrics: TradingMetrics, alerts: List[str] = None):
        """Generate dashboard markdown.
        
        Args:
            metrics: Current trading metrics
            alerts: List of active alerts
        """
        dashboard_content = []
        
        # Header
        dashboard_content.extend([
            "# Live Trading Dashboard",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ""
        ])
        
        # Status indicators
        status_emoji = "🟢" if metrics.current_drawdown < 0.02 else "🟡" if metrics.current_drawdown < 0.05 else "🔴"
        dashboard_content.extend([
            f"## Status: {status_emoji} {'HEALTHY' if status_emoji == '🟢' else 'WARNING' if status_emoji == '🟡' else 'CRITICAL'}",
            ""
        ])
        
        # Key metrics
        dashboard_content.extend([
            "## Key Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total PnL | ${metrics.total_pnl:.2f} |",
            f"| Win Rate | {metrics.win_rate:.1%} |",
            f"| Max Drawdown | {metrics.max_drawdown:.1%} |",
            f"| Current Drawdown | {metrics.current_drawdown:.1%} |",
            f"| Exposure | {metrics.exposure:.1%} |",
            f"| Active Positions | {metrics.active_positions} |",
            f"| Total Trades | {metrics.trade_count} |",
            f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |",
            ""
        ])
        
        # Performance chart (ASCII)
        dashboard_content.extend([
            "## Performance",
            "",
            "```",
            self._generate_ascii_chart(metrics),
            "```",
            ""
        ])
        
        # System health
        dashboard_content.extend([
            "## System Health",
            "",
            f"- Uptime: {metrics.uptime_hours:.1f} hours",
            f"- Latency: {metrics.latency_ms:.1f}ms",
            f"- Last Trade: {metrics.last_trade_time or 'Never'}",
            f"- Avg Trade Duration: {metrics.avg_trade_duration:.1f} hours",
            ""
        ])
        
        # Alerts
        if alerts:
            dashboard_content.extend([
                "## 🚨 Active Alerts",
                ""
            ])
            for alert in alerts:
                dashboard_content.append(f"- ⚠️ {alert}")
            dashboard_content.append("")
        
        # Save dashboard
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(dashboard_content))
        
        logger.debug(f"Updated dashboard at {self.output_path}")
    
    def _generate_ascii_chart(self, metrics: TradingMetrics) -> str:
        """Generate simple ASCII chart for PnL."""
        # Simple placeholder chart
        pnl = metrics.total_pnl
        
        if pnl > 0:
            bars = "█" * min(20, int(pnl / 100))
            return f"PnL: +${pnl:.2f} {bars}"
        elif pnl < 0:
            bars = "█" * min(20, int(abs(pnl) / 100))
            return f"PnL: -${abs(pnl):.2f} {bars} (negative)"
        else:
            return "PnL: $0.00 (flat)"