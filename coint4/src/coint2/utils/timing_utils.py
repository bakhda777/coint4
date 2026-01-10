"""Timing utilities for performance profiling."""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Generator


def logged_time(tag: str, logger_name: str = "timing") -> callable:
    """
    Decorator for timing function execution.
    
    Args:
        tag: Description of what's being timed
        logger_name: Name of logger to use
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            t0 = time.perf_counter()
            logger.info(f"⇢ {tag} — start")
            try:
                result = func(*args, **kwargs)
                dt = time.perf_counter() - t0
                logger.info(f"⇠ {tag} — done in {dt:,.2f}s")
                return result
            except Exception as e:
                dt = time.perf_counter() - t0
                logger.error(f"⇠ {tag} — failed after {dt:,.2f}s: {e}")
                raise
        return wrapper
    return decorator


@contextmanager
def time_block(tag: str, logger_name: str = "timing") -> Generator[None, None, None]:
    """
    Context manager for timing code blocks.
    
    Usage:
        with time_block("Loading data"):
            data = load_data()
    """
    logger = logging.getLogger(logger_name)
    t0 = time.perf_counter()
    logger.info(f"⇢ {tag} — start")
    try:
        yield
        dt = time.perf_counter() - t0
        logger.info(f"⇠ {tag} — done in {dt:,.2f}s")
    except Exception as e:
        dt = time.perf_counter() - t0
        logger.error(f"⇠ {tag} — failed after {dt:,.2f}s: {e}")
        raise


def log_progress(current: int, total: int, step: int = 500, tag: str = "Progress", logger_name: str = "timing") -> None:
    """
    Log progress for long-running loops.
    
    Args:
        current: Current iteration (1-based)
        total: Total iterations
        step: Log every N iterations
        tag: Description of the process
        logger_name: Logger name
    """
    if current % step == 0 or current == total:
        logger = logging.getLogger(logger_name)
        pct = (current / total) * 100
        logger.info(f"  {tag}: {current:,} / {total:,} done ({pct:.1f}%)")


class ProgressTracker:
    """Helper class for tracking progress in loops with timing."""
    
    def __init__(self, total: int, tag: str, step: int = 500, logger_name: str = "timing"):
        self.total = total
        self.tag = tag
        self.step = step
        self.logger = logging.getLogger(logger_name)
        self.start_time = time.perf_counter()
        self.current = 0
        
        self.logger.info(f"⇢ {self.tag} — starting {total:,} items")
    
    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment
        if self.current % self.step == 0 or self.current == self.total:
            elapsed = time.perf_counter() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            pct = (self.current / self.total) * 100
            
            self.logger.info(
                f"  {self.tag}: {self.current:,} / {self.total:,} "
                f"({pct:.1f}%) | {rate:.1f} items/s | ETA: {eta:.1f}s"
            )
    
    def finish(self) -> None:
        """Mark completion."""
        elapsed = time.perf_counter() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        self.logger.info(f"⇠ {self.tag} — completed {self.total:,} items in {elapsed:.2f}s ({rate:.1f} items/s)")


def setup_timing_logger(level: int = logging.INFO) -> None:
    """Setup timing logger with proper formatting."""
    timing_logger = logging.getLogger("timing")
    timing_logger.setLevel(level)
    
    # Only add handler if not already present
    if not timing_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [TIMING] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        timing_logger.addHandler(handler)
        timing_logger.propagate = False  # Don't propagate to root logger 