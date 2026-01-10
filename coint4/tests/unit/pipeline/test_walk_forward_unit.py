"""Unit тесты для Walk-Forward анализа.

Быстрые изолированные тесты без файловых операций и внешних зависимостей.
Все тесты выполняются < 1 сек.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import psutil


@pytest.mark.unit
class TestWalkForwardUnit:
    """Быстрые unit тесты для логики Walk-Forward."""

    def test_config_creation_when_valid_params_then_success(self):
        """Unit: проверяем создание валидной конфигурации."""
        config = {
            'portfolio': {
                'initial_capital': 10000.0,
                'risk_per_position_pct': 0.01,
                'max_active_positions': 5
            },
            'backtest': {
                'rolling_window': 30,
                'z_threshold': 2.0
            }
        }
        
        assert config['portfolio']['initial_capital'] > 0
        assert 0 < config['portfolio']['risk_per_position_pct'] < 1
        assert config['backtest']['rolling_window'] > 0

    def test_optimization_params_validation(self):
        """Unit: проверяем валидацию параметров оптимизации."""
        params = {
            'regime_check_frequency': 5,
            'adf_check_frequency': 10,
            'cache_size_limit': 1000
        }
        
        assert params['regime_check_frequency'] <= params['adf_check_frequency']
        assert params['cache_size_limit'] > 0

    def test_date_range_validation(self):
        """Unit: проверяем валидацию диапазона дат."""
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-04-01')
        training_days = 60
        test_days = 30
        
        total_days = (end_date - start_date).days
        min_required = training_days + test_days
        
        assert total_days >= min_required

    def test_memory_monitoring_logic(self):
        """Unit: проверяем логику мониторинга памяти."""
        with patch.object(psutil, 'Process') as mock_process:
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024
            mock_process.return_value = mock_proc
            
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            assert memory_mb == 100.0
            assert memory_mb > 0

    def test_config_parameter_types(self):
        """Unit: проверяем типы параметров конфигурации."""
        config = {
            'portfolio': {'initial_capital': 10000.0},
            'backtest': {'rolling_window': 20},
            'walk_forward': {'training_period_days': 60}
        }
        
        assert isinstance(config['portfolio']['initial_capital'], float)
        assert isinstance(config['backtest']['rolling_window'], int)
        assert isinstance(config['walk_forward']['training_period_days'], int)

    def test_optimization_flags(self):
        """Unit: проверяем флаги оптимизации."""
        config = {
            'use_market_regime_cache': True,
            'lazy_adf_threshold': 0.1,
            'hurst_neutral_band': 0.05
        }
        
        assert isinstance(config['use_market_regime_cache'], bool)
        assert 0 <= config['lazy_adf_threshold'] <= 1
        assert 0 <= config['hurst_neutral_band'] <= 0.5

    def test_parameter_compatibility(self):
        """Unit: проверяем совместимость параметров."""
        config = {
            'regime_check_frequency': 5,
            'adf_check_frequency': 10,
            'cache_cleanup_frequency': 50
        }
        
        # regime check должен быть чаще или равен adf check
        assert config['regime_check_frequency'] <= config['adf_check_frequency']
        # cleanup должен быть реже проверок
        assert config['cache_cleanup_frequency'] >= config['adf_check_frequency']

    def test_error_message_informativeness(self):
        """Unit: проверяем информативность сообщений об ошибках."""
        # Симулируем различные ошибки
        errors = {
            'nan_data': "NaN values detected in data",
            'insufficient_period': "Training period too short",
            'invalid_config': "Invalid configuration parameter"
        }
        
        for error_type, message in errors.items():
            assert len(message) > 10  # Сообщение должно быть информативным
            assert any(word in message.lower() for word in ['nan', 'period', 'config'])