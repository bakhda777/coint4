"""Safety systems and guardrails for live trading."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"  
    CRITICAL = "critical"


@dataclass
class Alert:
    """Trading alert."""
    level: AlertLevel
    message: str
    timestamp: str
    source: str
    data: Dict[str, Any] = None


class SafetyGuard:
    """Base class for safety guards."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.last_check = time.time()
        
    def check(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        """Check for violations and return alerts."""
        if not self.enabled:
            return []
        
        self.last_check = time.time()
        return self._check_impl(metrics, positions, config)
    
    def _check_impl(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        """Implementation of safety check."""
        raise NotImplementedError
    
    def enable(self):
        """Enable this guard."""
        self.enabled = True
        logger.info(f"Enabled safety guard: {self.name}")
    
    def disable(self):
        """Disable this guard."""
        self.enabled = False
        logger.warning(f"Disabled safety guard: {self.name}")


class DailyLossGuard(SafetyGuard):
    """Guards against excessive daily losses."""
    
    def __init__(self, max_loss_pct: float = 0.02):
        super().__init__("DailyLoss")
        self.max_loss_pct = max_loss_pct
        self.daily_start_balance = None
        self.last_reset = datetime.now().date()
        
    def _check_impl(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        alerts = []
        
        # Reset daily tracking
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_start_balance = None
            self.last_reset = today
        
        current_balance = metrics.get('total_balance', 100000)
        
        if self.daily_start_balance is None:
            self.daily_start_balance = current_balance
            return alerts
        
        # Calculate daily loss
        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
        
        if daily_loss >= self.max_loss_pct:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Daily loss limit exceeded: {daily_loss:.1%} > {self.max_loss_pct:.1%}",
                timestamp=datetime.now().isoformat(),
                source=self.name,
                data={"daily_loss": daily_loss, "limit": self.max_loss_pct}
            ))
        elif daily_loss >= self.max_loss_pct * 0.75:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Approaching daily loss limit: {daily_loss:.1%}",
                timestamp=datetime.now().isoformat(),
                source=self.name,
                data={"daily_loss": daily_loss, "limit": self.max_loss_pct}
            ))
        
        return alerts


class DrawdownGuard(SafetyGuard):
    """Guards against excessive drawdown."""
    
    def __init__(self, max_drawdown_pct: float = 0.05):
        super().__init__("Drawdown")
        self.max_drawdown_pct = max_drawdown_pct
        
    def _check_impl(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        alerts = []
        
        current_dd = metrics.get('current_drawdown', 0)
        max_dd = metrics.get('max_drawdown', 0)
        
        if current_dd > self.max_drawdown_pct:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Max drawdown exceeded: {current_dd:.1%} > {self.max_drawdown_pct:.1%}",
                timestamp=datetime.now().isoformat(),
                source=self.name,
                data={"current_drawdown": current_dd, "limit": self.max_drawdown_pct}
            ))
        elif current_dd > self.max_drawdown_pct * 0.8:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Approaching max drawdown: {current_dd:.1%}",
                timestamp=datetime.now().isoformat(),
                source=self.name
            ))
        
        return alerts


class NoDataGuard(SafetyGuard):
    """Guards against missing data."""
    
    def __init__(self, max_no_data_minutes: int = 5):
        super().__init__("NoData")
        self.max_no_data_minutes = max_no_data_minutes
        self.last_data_time = time.time()
        
    def update_data_time(self):
        """Update last data received time."""
        self.last_data_time = time.time()
        
    def _check_impl(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        alerts = []
        
        minutes_since_data = (time.time() - self.last_data_time) / 60
        
        if minutes_since_data > self.max_no_data_minutes:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"No data for {minutes_since_data:.1f} minutes",
                timestamp=datetime.now().isoformat(),
                source=self.name,
                data={"minutes_no_data": minutes_since_data}
            ))
        elif minutes_since_data > self.max_no_data_minutes * 0.8:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Data delay: {minutes_since_data:.1f} minutes",
                timestamp=datetime.now().isoformat(),
                source=self.name
            ))
        
        return alerts


class NoTradeGuard(SafetyGuard):
    """Guards against lack of trading activity."""
    
    def __init__(self, max_no_trade_hours: int = 24):
        super().__init__("NoTrade")
        self.max_no_trade_hours = max_no_trade_hours
        
    def _check_impl(self, metrics: Dict[str, Any], positions: Dict, config: Dict) -> List[Alert]:
        alerts = []
        
        last_trade_time_str = metrics.get('last_trade_time')
        if not last_trade_time_str:
            return alerts  # No trades yet, normal
        
        try:
            last_trade_time = datetime.fromisoformat(last_trade_time_str)
            hours_since_trade = (datetime.now() - last_trade_time).total_seconds() / 3600
            
            if hours_since_trade > self.max_no_trade_hours:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"No trades for {hours_since_trade:.1f} hours",
                    timestamp=datetime.now().isoformat(),
                    source=self.name,
                    data={"hours_no_trade": hours_since_trade}
                ))
        except Exception as e:
            logger.error(f"Error parsing last trade time: {e}")
        
        return alerts


class SafetySystem:
    """Coordinates all safety guards and emergency responses."""
    
    def __init__(self, config: Dict[str, Any], alerts_log: str = "artifacts/live/ALERTS.log"):
        """Initialize safety system.
        
        Args:
            config: Configuration dictionary
            alerts_log: Path to alerts log file
        """
        self.config = config
        self.alerts_log = Path(alerts_log)
        self.alerts_log.parent.mkdir(parents=True, exist_ok=True)
        
        self.guards = []
        self.active_alerts = []
        self.emergency_callbacks = []
        self.failsafe_active = False
        self.failsafe_log = Path("artifacts/live/FAILSAFE.log")
        
        # Initialize guards
        self._init_guards()
        
    def _init_guards(self):
        """Initialize safety guards from config."""
        risk_config = self.config.get('risk', {})
        monitoring_config = self.config.get('monitoring', {})
        
        # Daily loss guard
        if risk_config.get('max_daily_loss_pct'):
            self.guards.append(DailyLossGuard(risk_config['max_daily_loss_pct']))
        
        # Drawdown guard  
        if risk_config.get('max_drawdown_pct'):
            self.guards.append(DrawdownGuard(risk_config['max_drawdown_pct']))
        
        # No data guard
        if monitoring_config.get('no_data_alert_minutes'):
            self.guards.append(NoDataGuard(monitoring_config['no_data_alert_minutes']))
        
        # No trade guard
        if risk_config.get('min_trades_per_day'):
            # Convert to hours (24 hours / min trades per day)
            max_hours = 24 / max(1, risk_config['min_trades_per_day'])
            self.guards.append(NoTradeGuard(int(max_hours)))
        
        logger.info(f"Initialized {len(self.guards)} safety guards")
    
    def add_guard(self, guard: SafetyGuard):
        """Add a custom safety guard."""
        self.guards.append(guard)
        logger.info(f"Added safety guard: {guard.name}")
    
    def add_emergency_callback(self, callback: Callable):
        """Add emergency response callback."""
        self.emergency_callbacks.append(callback)
    
    def check_all(self, metrics: Dict[str, Any], positions: Dict) -> List[Alert]:
        """Run all safety checks.
        
        Args:
            metrics: Current trading metrics
            positions: Current positions
            
        Returns:
            List of active alerts
        """
        all_alerts = []
        
        # Run all guards
        for guard in self.guards:
            try:
                alerts = guard.check(metrics, positions, self.config)
                all_alerts.extend(alerts)
            except Exception as e:
                logger.error(f"Error in safety guard {guard.name}: {e}")
                # Create alert for guard failure
                all_alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"Safety guard {guard.name} failed: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    source="SafetySystem"
                ))
        
        # Update active alerts
        self.active_alerts = all_alerts
        
        # Log alerts
        self._log_alerts(all_alerts)
        
        # Check for emergency conditions
        critical_alerts = [a for a in all_alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            self._handle_emergency(critical_alerts)
        
        return all_alerts
    
    def _log_alerts(self, alerts: List[Alert]):
        """Log alerts to file."""
        if not alerts:
            return
        
        try:
            with open(self.alerts_log, 'a') as f:
                for alert in alerts:
                    f.write(f"{alert.timestamp} [{alert.level.value.upper()}] {alert.source}: {alert.message}\n")
        except Exception as e:
            logger.error(f"Failed to log alerts: {e}")
    
    def _handle_emergency(self, critical_alerts: List[Alert]):
        """Handle emergency conditions."""
        logger.critical(f"EMERGENCY: {len(critical_alerts)} critical alerts active")
        
        # Activate failsafe if configured
        if self.config.get('guards', {}).get('use_reference_on_error', False) and not self.failsafe_active:
            self._activate_failsafe(critical_alerts)
        
        # Execute emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(critical_alerts)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def _activate_failsafe(self, critical_alerts: List[Alert]):
        """Activate failsafe mode."""
        self.failsafe_active = True
        
        failsafe_msg = f"FAILSAFE ACTIVATED at {datetime.now().isoformat()}\n"
        failsafe_msg += f"Reason: {len(critical_alerts)} critical alerts\n"
        for alert in critical_alerts:
            failsafe_msg += f"  - {alert.message}\n"
        failsafe_msg += "Switching to reference engine until manual reset.\n\n"
        
        try:
            with open(self.failsafe_log, 'a') as f:
                f.write(failsafe_msg)
            
            logger.critical("FAILSAFE ACTIVATED - Check FAILSAFE.log for details")
            
        except Exception as e:
            logger.error(f"Failed to write failsafe log: {e}")
    
    def reset_failsafe(self):
        """Reset failsafe mode (manual operation)."""
        if self.failsafe_active:
            self.failsafe_active = False
            logger.info("Failsafe mode reset")
            
            reset_msg = f"FAILSAFE RESET at {datetime.now().isoformat()}\n\n"
            try:
                with open(self.failsafe_log, 'a') as f:
                    f.write(reset_msg)
            except Exception as e:
                logger.error(f"Failed to write failsafe reset: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get safety system status."""
        return {
            "guards_active": len([g for g in self.guards if g.enabled]),
            "guards_total": len(self.guards),
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts if a.level == AlertLevel.CRITICAL]),
            "failsafe_active": self.failsafe_active,
            "last_check": max([g.last_check for g in self.guards]) if self.guards else 0
        }
    
    def update_data_received(self):
        """Update data received timestamp for NoData guard."""
        for guard in self.guards:
            if isinstance(guard, NoDataGuard):
                guard.update_data_time()
                break
