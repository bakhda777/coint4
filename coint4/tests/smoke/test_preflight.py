"""Smoke tests for preflight checks."""

import pytest
import subprocess
from pathlib import Path
import sys


@pytest.mark.smoke
def test_preflight_script_exists():
    """Test that preflight script exists and is executable."""
    script_path = Path("scripts/run_preflight.py")
    assert script_path.exists(), "Preflight script not found"
    assert script_path.is_file(), "Preflight script is not a file"


@pytest.mark.smoke
def test_preflight_returns_exit_code():
    """Test that preflight script returns proper exit code."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_preflight.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should return either 0 (success) or 1 (failure)
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}"
        
        # Should produce some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0, "No output produced"
        
    except subprocess.TimeoutExpired:
        pytest.fail("Preflight script timed out")
    except Exception as e:
        pytest.fail(f"Failed to run preflight script: {e}")


@pytest.mark.smoke
def test_preflight_creates_log_file():
    """Test that preflight creates log file."""
    log_path = Path("artifacts/live/PREFLIGHT.log")
    
    # Run preflight
    try:
        subprocess.run(
            [sys.executable, "scripts/run_preflight.py"],
            capture_output=True,
            timeout=30
        )
    except:
        pass  # Don't fail on script errors, just check if log was created
    
    # Check log file exists and has content
    assert log_path.exists(), "Preflight log file not created"
    
    if log_path.stat().st_size > 0:
        # Check for key log messages
        log_content = log_path.read_text()
        assert "preflight" in log_content.lower(), "Log missing preflight references"


@pytest.mark.smoke  
def test_preflight_creates_report():
    """Test that preflight creates report file."""
    report_path = Path("artifacts/live/PREFLIGHT_REPORT.md")
    
    # Run preflight
    try:
        subprocess.run(
            [sys.executable, "scripts/run_preflight.py"],
            capture_output=True,
            timeout=30
        )
    except:
        pass  # Don't fail on script errors, just check if report was created
    
    # Check report file exists
    assert report_path.exists(), "Preflight report not created"
    
    if report_path.stat().st_size > 0:
        # Check for key report sections
        report_content = report_path.read_text()
        assert "Preflight Report" in report_content, "Report missing title"
        assert "Overall Status" in report_content, "Report missing status section"


@pytest.mark.smoke
def test_risk_config_exists():
    """Test that risk.yaml config exists and is readable."""
    risk_config_path = Path("configs/risk.yaml")
    assert risk_config_path.exists(), "Risk config not found"
    
    # Try to read it
    import yaml
    with open(risk_config_path, 'r') as f:
        risk_config = yaml.safe_load(f)
    
    # Check for key parameters
    required_params = [
        'max_daily_loss_pct',
        'max_drawdown_pct',
        'max_no_data_minutes', 
        'min_trade_count_per_day',
        'position_size_usd'
    ]
    
    for param in required_params:
        assert param in risk_config, f"Risk config missing parameter: {param}"
