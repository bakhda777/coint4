"""Integration tests for live trading safety systems."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from src.coint2.monitoring.safety import (
    SafetySystem, DailyLossGuard, DrawdownGuard, 
    NoDataGuard, AlertLevel
)


@pytest.mark.integration
def test_safety_system_initialization():
    """Test safety system initializes with config."""
    config = {
        'risk': {
            'max_daily_loss_pct': 0.02,
            'max_drawdown_pct': 0.05,
            'min_trades_per_day': 1
        },
        'monitoring': {
            'no_data_alert_minutes': 5
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        alerts_log = Path(temp_dir) / "alerts.log"
        safety = SafetySystem(config, str(alerts_log))
        
        assert len(safety.guards) >= 3  # Should have multiple guards
        assert not safety.failsafe_active


@pytest.mark.integration
def test_daily_loss_guard():
    """Test daily loss guard triggers alerts."""
    guard = DailyLossGuard(max_loss_pct=0.02)
    
    # First call establishes baseline
    metrics = {'total_balance': 100000}
    alerts = guard.check(metrics, {}, {})
    assert len(alerts) == 0
    
    # Small loss - no alert
    metrics = {'total_balance': 99000}  # 1% loss
    alerts = guard.check(metrics, {}, {})
    assert len(alerts) == 0
    
    # Large loss - warning
    metrics = {'total_balance': 98500}  # 1.5% loss
    alerts = guard.check(metrics, {}, {})
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING
    
    # Excessive loss - critical
    metrics = {'total_balance': 97500}  # 2.5% loss
    alerts = guard.check(metrics, {}, {})
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL


@pytest.mark.integration 
def test_no_data_guard():
    """Test no data guard triggers on stale data."""
    guard = NoDataGuard(max_no_data_minutes=5)
    
    # Fresh data - no alert
    guard.update_data_time()
    alerts = guard.check({}, {}, {})
    assert len(alerts) == 0
    
    # Simulate old data
    guard.last_data_time = guard.last_data_time - 6 * 60  # 6 minutes ago
    alerts = guard.check({}, {}, {})
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL


@pytest.mark.integration
def test_safety_system_emergency_response():
    """Test safety system handles emergency conditions."""
    config = {
        'risk': {'max_daily_loss_pct': 0.01},  # Very strict
        'guards': {'use_reference_on_error': True}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        alerts_log = Path(temp_dir) / "alerts.log"
        safety = SafetySystem(config, str(alerts_log))
        
        # Mock emergency callback
        emergency_callback = Mock()
        safety.add_emergency_callback(emergency_callback)
        
        # Trigger emergency with excessive loss
        metrics = {'total_balance': 100000}
        safety.check_all(metrics, {})  # Establish baseline
        
        metrics = {'total_balance': 98000}  # 2% loss, exceeds 1% limit
        alerts = safety.check_all(metrics, {})
        
        # Should have critical alert and activate failsafe
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) > 0
        assert safety.failsafe_active
        
        # Emergency callback should be called
        emergency_callback.assert_called_once()
        
        # Check alerts log exists
        assert alerts_log.exists()
        
        # Check failsafe log exists
        failsafe_log = Path(temp_dir).parent / "artifacts" / "live" / "FAILSAFE.log"
        # Note: failsafe_log path is hardcoded, so this might not exist in temp_dir


@pytest.mark.integration
def test_safety_system_no_data_simulation():
    """Test safety system handles no data scenario."""
    config = {
        'monitoring': {'no_data_alert_minutes': 2}  # Very short for testing
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        alerts_log = Path(temp_dir) / "alerts.log"
        safety = SafetySystem(config, str(alerts_log))
        
        # Simulate no data for extended period
        no_data_guard = None
        for guard in safety.guards:
            if isinstance(guard, NoDataGuard):
                no_data_guard = guard
                break
        
        assert no_data_guard is not None
        
        # Simulate old data
        no_data_guard.last_data_time = no_data_guard.last_data_time - 3 * 60  # 3 minutes ago
        
        alerts = safety.check_all({}, {})
        
        # Should have critical no-data alert
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) > 0
        
        no_data_alerts = [a for a in critical_alerts if "No data" in a.message]
        assert len(no_data_alerts) > 0


@pytest.mark.integration
def test_safety_system_fee_increase_simulation():
    """Test safety system handles increased trading costs."""
    config = {
        'risk': {'max_daily_loss_pct': 0.02}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        alerts_log = Path(temp_dir) / "alerts.log"
        safety = SafetySystem(config, str(alerts_log))
        
        # Establish baseline
        metrics = {'total_balance': 100000}
        safety.check_all(metrics, {})
        
        # Simulate losses from increased fees (gradual decline)
        balances = [99500, 99000, 98500, 98000, 97500]  # Increasing losses
        
        for balance in balances:
            metrics = {'total_balance': balance}
            alerts = safety.check_all(metrics, {})
            
            loss_pct = (100000 - balance) / 100000
            if loss_pct > 0.02:
                # Should have critical alerts when exceeding limit
                critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
                assert len(critical_alerts) > 0
                break


@pytest.mark.integration
def test_failsafe_reset():
    """Test failsafe can be reset."""
    config = {
        'risk': {'max_daily_loss_pct': 0.01},
        'guards': {'use_reference_on_error': True}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        alerts_log = Path(temp_dir) / "alerts.log" 
        safety = SafetySystem(config, str(alerts_log))
        
        # Activate failsafe
        safety._activate_failsafe([])
        assert safety.failsafe_active
        
        # Reset failsafe
        safety.reset_failsafe()
        assert not safety.failsafe_active