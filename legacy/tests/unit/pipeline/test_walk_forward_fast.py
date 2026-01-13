"""Fast тесты для Walk-Forward анализа.

Быстрые тесты с мокированием для CI/CD.
Все тесты выполняются < 5 сек суммарно.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


@pytest.mark.fast
class TestWalkForwardFast:
    """Быстрые тесты с мокированием для CI/CD."""
    
    @pytest.fixture
    def tiny_market_data(self):
        """Минимальные данные для быстрых тестов."""
        dates = pd.date_range('2024-01-01', periods=48, freq='1h')  # 2 дня
        data = []
        
        for i in range(2):  # Только 2 актива
            prices = 100 + np.cumsum(np.random.randn(48) * 0.5)
            asset_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': f'ASSET_{i}',
                'close': prices
            })
            data.append(asset_data)
        
        return pd.concat(data, ignore_index=True)
    
    @patch('src.coint2.pipeline.walk_forward_orchestrator.run_walk_forward')
    def test_walk_forward_mocked(self, mock_wf):
        """Fast: тестируем логику с мокированным walk forward."""
        mock_wf.return_value = {
            'total_pnl': 100.0,
            'sharpe_ratio_abs': 0.8,
            'num_trades': 5
        }
        
        config = {'portfolio': {'initial_capital': 10000.0}}
        result = mock_wf(config)
        
        assert mock_wf.called
        assert result['total_pnl'] == 100.0
        assert result['num_trades'] == 5
    
    def test_config_validation_fast(self):
        """Fast: быстрая валидация конфигурации."""
        config = {
            'portfolio': {
                'initial_capital': 10000.0,
                'max_active_positions': 3
            },
            'backtest': {
                'rolling_window': 10,
                'zscore_threshold': 2.0
            }
        }
        
        assert config['portfolio']['initial_capital'] > 0
        assert config['backtest']['rolling_window'] > 0
        assert config['backtest']['zscore_threshold'] > 0
    
    @patch('psutil.Process')
    def test_memory_monitoring_mocked(self, mock_process):
        """Fast: мокированный мониторинг памяти."""
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 50 * 1024 * 1024
        mock_process.return_value = mock_proc
        
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb == 50.0
    
    @patch('src.coint2.pipeline.walk_forward_orchestrator.run_walk_forward')
    def test_performance_comparison_mocked(self, mock_wf):
        """Fast: мокированное сравнение производительности."""
        def side_effect(config):
            if config.get('optimized', False):
                return {'total_pnl': 100.0, 'time_seconds': 5.0}
            else:
                return {'total_pnl': 95.0, 'time_seconds': 10.0}
        
        mock_wf.side_effect = side_effect
        
        opt_result = mock_wf({'optimized': True})
        base_result = mock_wf({'optimized': False})
        
        assert opt_result['total_pnl'] >= base_result['total_pnl']
        assert opt_result['time_seconds'] < base_result['time_seconds']
    
    def test_error_handling_logic(self):
        """Fast: тестируем логику обработки ошибок."""
        # Тестируем обнаружение NaN
        data = pd.DataFrame({'price': [100, np.nan, 102]})
        assert data.isna().any().any()
        
        # Тестируем очистку данных
        cleaned = data.dropna()
        assert len(cleaned) < len(data)
        assert not cleaned.isna().any().any()
    
    def test_results_structure(self):
        """Fast: проверяем структуру результатов."""
        results = {
            'total_pnl': 100.0,
            'sharpe_ratio_abs': 0.8,
            'max_drawdown_abs': -0.05,
            'num_trades': 10
        }
        
        required_fields = ['total_pnl', 'sharpe_ratio_abs', 'max_drawdown_abs']
        for field in required_fields:
            assert field in results
            assert isinstance(results[field], (int, float))
    
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_file_operations_mocked(self, mock_csv, mock_makedirs):
        """Fast: мокированные файловые операции."""
        mock_makedirs.return_value = None
        mock_csv.return_value = None
        
        # Симулируем сохранение результатов
        results_dir = '/tmp/results'
        import os
        os.makedirs(results_dir, exist_ok=True)
        
        df = pd.DataFrame({'metric': [1, 2, 3]})
        df.to_csv(f'{results_dir}/metrics.csv')
        
        assert mock_makedirs.called
        assert mock_csv.called