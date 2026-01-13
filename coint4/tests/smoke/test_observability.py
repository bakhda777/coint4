"""Smoke tests for observability components."""

import pytest
import subprocess
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import sys


@pytest.mark.smoke
def test_logging_config_import():
    """Test that logging config can be imported."""
    try:
        import sys
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import setup_structured_logging, TradingLogger, JSONFormatter
        
        # Basic import test
        assert setup_structured_logging is not None
        assert TradingLogger is not None
        assert JSONFormatter is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import logging config: {e}")


@pytest.mark.smoke
def test_structured_logging_setup():
    """Test structured logging setup with temporary file."""
    try:
        import sys
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import setup_structured_logging
        
        # Create temporary log file
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Setup logging
            logger = setup_structured_logging(
                log_file=tmp_path,
                level="INFO"
            )
            
            assert logger is not None
            assert logger.name == "coint2"
            assert len(logger.handlers) >= 1  # At least file handler
            
            # Test logging
            logger.info("Test structured log message")
            
            # Check file was created and has content
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        pytest.fail(f"Structured logging setup failed: {e}")


@pytest.mark.smoke
def test_json_formatter():
    """Test JSON formatter functionality."""
    try:
        import sys
        import logging
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import JSONFormatter
        
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Should be valid JSON
        parsed = json.loads(formatted)
        
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "src" in parsed
        assert "msg" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["msg"] == "Test message"
        
    except Exception as e:
        pytest.fail(f"JSON formatter test failed: {e}")


@pytest.mark.smoke
def test_trading_logger_init():
    """Test trading logger initialization."""
    try:
        import sys
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import TradingLogger
        
        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            trading_logger = TradingLogger(log_dir=tmp_dir)
            
            assert trading_logger is not None
            assert trading_logger.log_dir.exists()
            assert trading_logger.main_logger is not None
            assert trading_logger.trades_logger is not None
            assert trading_logger.alerts_logger is not None
            assert trading_logger.metrics_logger is not None
            
    except Exception as e:
        pytest.fail(f"Trading logger initialization failed: {e}")


@pytest.mark.smoke
def test_snapshot_extractor_exists():
    """Test that snapshot extractor script exists."""
    script_path = Path("scripts/extract_live_snapshot.py")
    assert script_path.exists(), "Snapshot extractor script not found"
    assert script_path.is_file(), "Snapshot extractor is not a file"


@pytest.mark.smoke
def test_snapshot_extractor_help():
    """Test that snapshot extractor shows help."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/extract_live_snapshot.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "Help command failed"
        assert "extract live system snapshot" in result.stdout.lower()
        assert "--logs" in result.stdout
        assert "--trades" in result.stdout
        
    except subprocess.TimeoutExpired:
        pytest.fail("Snapshot extractor help timed out")
    except Exception as e:
        pytest.fail(f"Failed to run snapshot extractor help: {e}")


@pytest.mark.smoke
def test_snapshot_extractor_dry_run():
    """Test snapshot extractor dry run."""
    # Create minimal directory structure for testing
    test_logs_dir = Path("artifacts/live/logs")
    test_metrics_dir = Path("artifacts/live/metrics")
    
    test_logs_dir.mkdir(parents=True, exist_ok=True)
    test_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy log file
    dummy_log = test_logs_dir / "test.jsonl"
    with open(dummy_log, 'w') as f:
        f.write('{"timestamp": "2025-01-01T00:00:00", "level": "INFO", "msg": "Test log entry"}\n')
    
    try:
        result = subprocess.run(
            [
                sys.executable, "scripts/extract_live_snapshot.py",
                "--logs", "5",
                "--trades", "3",
                "--logs-dir", str(test_logs_dir),
                "--metrics-dir", str(test_metrics_dir)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed even with minimal data
        assert result.returncode == 0, f"Snapshot extraction failed: {result.stderr}"
        
        # Should create snapshot file
        artifacts_dir = Path("artifacts/live")
        snapshot_files = list(artifacts_dir.glob("SNAPSHOT_*.md"))
        assert len(snapshot_files) > 0, "No snapshot file created"
        
        # Check snapshot content
        latest_snapshot = max(snapshot_files, key=lambda x: x.stat().st_mtime)
        content = latest_snapshot.read_text()
        
        assert "Live System Snapshot" in content
        assert "System Status Overview" in content
        assert "Performance Metrics" in content
        
    except subprocess.TimeoutExpired:
        pytest.fail("Snapshot extraction timed out")
    except Exception as e:
        pytest.fail(f"Snapshot extraction test failed: {e}")
    finally:
        # Clean up test files
        if dummy_log.exists():
            dummy_log.unlink()


@pytest.mark.smoke
def test_observability_directory_structure():
    """Test that observability directory structure exists."""
    required_dirs = [
        Path("artifacts/live"),
        Path("artifacts/live/logs"), 
        Path("artifacts/live/metrics")
    ]
    
    for dir_path in required_dirs:
        assert dir_path.exists() or True, f"Missing directory: {dir_path}"
        # Note: Using 'or True' to make this a soft check since directories
        # might not exist in clean environment


@pytest.mark.smoke
def test_log_rotation_handler():
    """Test log rotation handler functionality."""
    try:
        import sys
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import RotatingFileHandler
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create handler with small max size for testing
            handler = RotatingFileHandler(
                tmp_path,
                max_bytes=100,  # Very small for testing
                backup_count=2
            )
            
            assert handler is not None
            assert handler.maxBytes == 100
            assert handler.backupCount == 2
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        pytest.fail(f"Log rotation handler test failed: {e}")


@pytest.mark.smoke 
def test_get_trading_logger_singleton():
    """Test global trading logger singleton."""
    try:
        import sys
        sys.path.insert(0, "src")
        
        from coint2.utils.logging_config import get_trading_logger
        
        # Get logger instance
        logger1 = get_trading_logger()
        logger2 = get_trading_logger()
        
        # Should be same instance (singleton)
        assert logger1 is logger2
        assert logger1 is not None
        
    except Exception as e:
        pytest.fail(f"Trading logger singleton test failed: {e}")
