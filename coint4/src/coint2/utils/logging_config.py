"""Structured logging configuration for coint2."""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "src": record.name,
            "msg": record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'fields') and record.fields:
            log_entry.update(record.fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with better error handling."""
    
    def __init__(self, filename: str, max_bytes: int = 10 * 1024 * 1024, 
                 backup_count: int = 5, **kwargs):
        """Initialize with auto-created directories."""
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, **kwargs)


def setup_structured_logging(
    log_file: str = "artifacts/live/logs/coint2.jsonl",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """Setup structured logging configuration.
    
    Args:
        log_file: Path to main log file
        level: Logging level
        max_bytes: Max file size before rotation (10MB default)
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger("coint2")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with JSON formatting
    file_handler = RotatingFileHandler(
        log_file, 
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger


def log_structured(logger: logging.Logger, level: str, message: str, **fields):
    """Log structured message with extra fields.
    
    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **fields: Additional fields to include in log
    """
    
    # Create log record with extra fields
    log_method = getattr(logger, level.lower(), logger.info)
    
    # Add fields as extra attribute
    extra = {"fields": fields} if fields else {}
    
    log_method(message, extra=extra)


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, log_dir: str = "artifacts/live/logs"):
        """Initialize trading logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different log streams
        self.main_logger = setup_structured_logging(
            log_file=str(self.log_dir / "main.jsonl")
        )
        
        self.trades_logger = setup_structured_logging(
            log_file=str(self.log_dir / "trades.jsonl"),
            level="INFO"
        )
        
        self.alerts_logger = setup_structured_logging(
            log_file=str(self.log_dir / "alerts.jsonl"),
            level="WARNING"
        )
        
        self.metrics_logger = setup_structured_logging(
            log_file=str(self.log_dir / "metrics.jsonl"),
            level="INFO"
        )
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution."""
        log_structured(
            self.trades_logger,
            "info",
            "Trade executed",
            **trade_data
        )
    
    def log_metric(self, metric_name: str, value: float, **metadata):
        """Log metric value."""
        log_structured(
            self.metrics_logger,
            "info",
            f"Metric: {metric_name}",
            metric=metric_name,
            value=value,
            **metadata
        )
    
    def log_alert(self, alert_type: str, message: str, severity: str = "warning", **data):
        """Log alert/warning."""
        log_structured(
            self.alerts_logger,
            severity,
            f"ALERT [{alert_type}]: {message}",
            alert_type=alert_type,
            **data
        )
    
    def log_system(self, message: str, level: str = "info", **data):
        """Log system event."""
        log_structured(
            self.main_logger,
            level,
            message,
            **data
        )


# Global trading logger instance
_trading_logger = None


def get_trading_logger() -> TradingLogger:
    """Get global trading logger instance."""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger