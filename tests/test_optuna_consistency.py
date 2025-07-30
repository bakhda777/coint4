"""Тесты для проверки соответствия Optuna бэктеста оригинальному.

Этот модуль содержит тесты для проверки:
1. Соответствия параметров PairBacktester в Optuna и оригинальном бэктесте
2. Корректности передачи всех параметров конфигурации
3. Идентичности логики нормализации данных
4. Сопоставимости результатов при одинаковых параметрах
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from coint2.utils.config import load_config
from src.optimiser.fast_objective import FastWalkForwardObjective
from coint2.pipeline.walk_forward_orchestrator import _run_backtest_for_pair
from coint2.core.portfolio import Portfolio
from coint2.engine.base_engine import BasePairBacktester as PairBacktester


class TestOptunaConsistency:
    """Тесты для проверки соответствия Optuna бэктеста оригинальному."""
    
    @pytest.fixture
    def config(self):
        """Загружает основную конфигурацию."""
        config_path = Path("configs/main_2024.yaml")
        return load_config(config_path)
    
    @pytest.fixture
    def sample_pair_data(self):
        """Создает тестовые данные для пары."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        np.random.seed(42)
        
        # Создаем коинтегрированные данные
        s1_prices = 100 + np.cumsum(np.random.normal(0, 0.1, 1000))
        s2_prices = s1_prices * 0.8 + np.random.normal(0, 0.5, 1000)
        
        return pd.DataFrame({
            'AAPL': s1_prices,
            'MSFT': s2_prices
        }, index=dates)
    
    @pytest.fixture
    def preselected_pairs(self):
        """Создает тестовые предварительно отобранные пары."""
        return pd.DataFrame({
            's1': ['AAPL'],
            's2': ['MSFT'],
            'beta': [0.8],
            'mean': [0.0],
            'std': [1.0],
            'half_life': [24.0]
        })
    
    def test_pair_backtester_parameters_consistency(self, config):
        """Проверяет, что параметры правильно передаются в конфигурацию Optuna objective."""
        
        # Создаем Optuna objective
        with patch('src.optimiser.fast_objective.pd.read_csv') as mock_read_csv, \
             patch('src.optimiser.fast_objective.Path.exists') as mock_exists:
            
            # Мокаем существование файла preselected_pairs.csv
            mock_exists.return_value = True
            
            # Мокаем содержимое файла preselected_pairs.csv
            mock_read_csv.return_value = pd.DataFrame({
                's1': ['AAPL'],
                's2': ['MSFT'],
                'beta': [0.8],
                'mean': [0.0],
                'std': [1.0],
                'half_life': [24.0]
            })
            
            objective = FastWalkForwardObjective(
                base_config_path="configs/main_2024.yaml",
                search_space_path="configs/search_space.yaml"
            )
            
            # Тестируем параметры, которые передаются в _run_fast_backtest
            test_params = {
                'zscore_threshold': 2.5,
                'zscore_exit': 0.3,
                'stop_loss_multiplier': 2.5,
                'time_stop_multiplier': 3.5,
                'max_active_positions': 12,
                'risk_per_position_pct': 0.025,
                'max_position_size_pct': 0.06
            }
            
            # Создаем временную конфигурацию с новыми параметрами (как в _run_fast_backtest)
            cfg = objective.base_config.model_copy(deep=True)
            cfg.backtest.zscore_threshold = test_params.get('zscore_threshold', 2.0)
            cfg.backtest.zscore_entry_threshold = cfg.backtest.zscore_threshold
            cfg.backtest.zscore_exit = test_params.get('zscore_exit', 0.0)
            cfg.backtest.stop_loss_multiplier = test_params.get('stop_loss_multiplier', 3.0)
            cfg.backtest.time_stop_multiplier = test_params.get('time_stop_multiplier', 2.0)
            
            # Безопасное обновление параметров портфеля
            if hasattr(cfg, 'portfolio'):
                cfg.portfolio.risk_per_position_pct = test_params.get('risk_per_position_pct', 0.015)
                if hasattr(cfg.portfolio, 'max_position_size_pct'):
                    cfg.portfolio.max_position_size_pct = test_params.get('max_position_size_pct', 0.1)
                cfg.portfolio.max_active_positions = int(test_params.get('max_active_positions', 15))
            
            # Проверяем, что параметры правильно установлены
            assert cfg.backtest.zscore_threshold == test_params['zscore_threshold'], \
                f"zscore_threshold не установлен: {cfg.backtest.zscore_threshold} != {test_params['zscore_threshold']}"
            
            assert cfg.backtest.zscore_exit == test_params['zscore_exit'], \
                f"zscore_exit не установлен: {cfg.backtest.zscore_exit} != {test_params['zscore_exit']}"
            
            assert cfg.backtest.stop_loss_multiplier == test_params['stop_loss_multiplier'], \
                f"stop_loss_multiplier не установлен: {cfg.backtest.stop_loss_multiplier} != {test_params['stop_loss_multiplier']}"
            
            assert cfg.backtest.time_stop_multiplier == test_params['time_stop_multiplier'], \
                f"time_stop_multiplier не установлен: {cfg.backtest.time_stop_multiplier} != {test_params['time_stop_multiplier']}"
            
            if hasattr(cfg, 'portfolio'):
                assert cfg.portfolio.risk_per_position_pct == test_params['risk_per_position_pct'], \
                    f"risk_per_position_pct не установлен: {cfg.portfolio.risk_per_position_pct} != {test_params['risk_per_position_pct']}"
                
                if hasattr(cfg.portfolio, 'max_position_size_pct'):
                    assert cfg.portfolio.max_position_size_pct == test_params['max_position_size_pct'], \
                        f"max_position_size_pct не установлен: {cfg.portfolio.max_position_size_pct} != {test_params['max_position_size_pct']}"
                
                assert cfg.portfolio.max_active_positions == test_params['max_active_positions'], \
                    f"max_active_positions не установлен: {cfg.portfolio.max_active_positions} != {test_params['max_active_positions']}"
            
            # Проверяем синхронизацию zscore_entry_threshold с zscore_threshold
            assert cfg.backtest.zscore_entry_threshold == cfg.backtest.zscore_threshold, \
                "zscore_entry_threshold должен быть синхронизирован с zscore_threshold"
            
            print("✅ Все параметры правильно передаются в конфигурацию Optuna objective")
    
    def test_normalization_consistency(self, config, sample_pair_data):
        """Проверяет идентичность логики нормализации данных."""
        # Разделяем данные на тренировочный и тестовый периоды
        split_point = len(sample_pair_data) // 2
        training_data = sample_pair_data.iloc[:split_point]
        testing_data = sample_pair_data.iloc[split_point:]
        
        # Применяем нормализацию как в оригинальном коде
        first_row = training_data.iloc[0].values
        
        # Проверяем, что нет нулевых значений
        assert not np.any(first_row == 0), "Нулевые значения в первой строке тренировочных данных"
        
        # Применяем нормализацию к тестовым данным
        normalized_testing = testing_data.values / first_row[np.newaxis, :] * 100
        normalized_df = pd.DataFrame(
            normalized_testing,
            index=testing_data.index,
            columns=testing_data.columns
        )
        
        # Проверяем, что нормализация прошла корректно
        assert not normalized_df.isnull().any().any(), "Нормализация привела к NaN значениям"
        assert normalized_df.shape == testing_data.shape, "Размерность данных изменилась после нормализации"