"""
Coint2 - Cointegration Backtest Framework.

A framework for backtesting cointegration-based trading strategies
with walk-forward optimization and Optuna parameter tuning.
"""

__version__ = "0.1.0"

# Экспорт основных компонентов для удобного импорта
from coint2.utils.config import load_config, AppConfig
from coint2.core.data_loader import DataHandler

__all__ = [
    "__version__",
    "load_config",
    "AppConfig",
    "DataHandler",
]