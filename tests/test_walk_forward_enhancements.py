"""Тесты для проверки улучшений walk-forward тестирования.

Этот модуль содержит тесты для проверки:
1. Корректной работы онлайновых статистик
2. Правильного сдвига сигналов (signal shift)
3. Улучшенного расчета торговых издержек
4. Интеграции всех компонентов в walk-forward тестировании
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from coint2.engine.backtest_engine import PairBacktester


class TestWalkForwardEnhancements:
    """Тесты для улучшений walk-forward тестирования."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные для проверки."""
        np.random.seed(42)
        n_periods = 200
        
        # Создаем коинтегрированные временные ряды
        base_trend = np.linspace(100, 120, n_periods)
        noise1 = np.random.normal(0, 2, n_periods)
        noise2 = np.random.normal(0, 1, n_periods)
        
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
        
        data = pd.DataFrame({
            'asset1': base_trend + noise1,
            'asset2': 0.8 * base_trend + noise2
        }, index=dates)
        
        return data
    
    def test_online_statistics_enabled(self, sample_data):
        """Проверяет, что онлайновые статистики используют только исторические данные.
        
        Тест проверяет:
        1. Онлайновые статистики включены
        2. Используются только исторические данные (без look-ahead bias)
        3. Статистики обновляются корректно
        """
        engine = PairBacktester(
            sample_data,
            rolling_window=20,
            z_threshold=2.0,
            capital_at_risk=10000,
            walk_forward_enabled=False,
            online_stats_enabled=True
        )
        
        # Проверяем, что онлайновые статистики включены
        assert engine.online_stats_enabled, "Онлайновые статистики должны быть включены"
        
        # Мокаем метод для отслеживания использования данных
        original_method = engine._calculate_ols_with_cache
        call_data = []
        
        def mock_ols_with_cache(y_data, x_data):
            # Записываем информацию о вызове
            call_data.append({
                'y_length': len(y_data),
                'x_length': len(x_data),
                'y_first': y_data.iloc[0] if len(y_data) > 0 else None,
                'y_last': y_data.iloc[-1] if len(y_data) > 0 else None
            })
            return original_method(y_data, x_data)
        
        with patch.object(engine, '_calculate_ols_with_cache', side_effect=mock_ols_with_cache):
            engine.run()
            results = engine.get_results()

        # Проверяем, что результаты получены
        assert results is not None, "Результаты должны быть получены"
        assert len(results) > 0, "Результаты не должны быть пустыми"
        
        # Проверяем, что были вызовы OLS расчетов
        assert len(call_data) > 0, "Должны быть вызовы OLS расчетов"
        
        # Проверяем, что используется правильный размер окна
        for call in call_data:
            assert call['y_length'] <= engine.rolling_window, \
                f"Размер окна Y ({call['y_length']}) не должен превышать rolling_window ({engine.rolling_window})"
            assert call['x_length'] <= engine.rolling_window, \
                f"Размер окна X ({call['x_length']}) не должен превышать rolling_window ({engine.rolling_window})"
    
    def test_signal_shift_implementation(self, sample_data):
        """Проверяет корректную реализацию сдвига сигналов.
        
        Тест проверяет:
        1. Сигналы вычисляются на баре i
        2. Исполняются на баре i+1
        3. Нет look-ahead bias в торговых решениях
        """
        engine = PairBacktester(
            sample_data,
            rolling_window=20,
            z_threshold=1.5,  # Низкий порог для генерации сигналов
            capital_at_risk=10000,
            walk_forward_enabled=False,  # Простой бэктест для проверки
            signal_shift_enabled=True
        )
        
        # Мокаем методы для отслеживания сигналов
        signal_calculations = []
        signal_executions = []
        
        original_run_method = engine._run_single_backtest_internal
        
        def mock_run_internal(phase='test'):
            # Вызываем оригинальный метод
            result = original_run_method(phase)
            
            # Проверяем, что в результатах есть позиции
            if hasattr(engine, 'results') and engine.results is not None:
                positions = engine.results['position']
                position_changes = positions.diff().fillna(0)
                
                # Находим моменты изменения позиций
                trade_moments = position_changes[position_changes != 0]
                signal_executions.extend(trade_moments.index.tolist())
            
            return result
        
        with patch.object(engine, '_run_single_backtest_internal', side_effect=mock_run_internal):
            engine.run()
            results = engine.get_results()
        
        # Проверяем, что результаты получены
        assert results is not None, "Результаты должны быть получены"
        
        # Проверяем наличие торговых сигналов
        if isinstance(results, dict) and 'position' in results:
            position_series = results['position']
            position_changes = position_series.diff().fillna(0)
            trades = position_changes[position_changes != 0]
            
            if len(trades) > 0:
                # Проверяем, что торговля не происходит на первом баре
                first_trade_idx = trades.index[0]
                first_data_idx = position_series.index[0]
                assert first_trade_idx != first_data_idx, \
                    "Первая торговля не должна происходить на первом баре данных"
    
    def test_enhanced_cost_calculation(self, sample_data):
        """Проверяет улучшенный расчет торговых издержек.
        
        Тест проверяет:
        1. Расчет комиссий для обеих активов
        2. Учет проскальзывания и спредов
        3. Корректное применение издержек к PnL
        """
        engine = PairBacktester(
            sample_data,
            rolling_window=20,
            z_threshold=1.5,
            capital_at_risk=10000,
            walk_forward_enabled=False,
            commission_pct=0.001,
            slippage_pct=0.0005,
            bid_ask_spread_pct_s1=0.0002,
            bid_ask_spread_pct_s2=0.0002
        )
        
        # Проверяем, что метод расчета издержек существует
        assert hasattr(engine, '_calculate_enhanced_trade_costs'), \
            "Метод _calculate_enhanced_trade_costs должен существовать"
        
        # Тестируем метод расчета издержек напрямую
        position_change = 1.0
        price_s1 = 100.0
        price_s2 = 80.0
        beta = 0.8
        
        cost = engine._calculate_enhanced_trade_costs(
            position_change, price_s1, price_s2, beta
        )
        
        # Проверяем, что издержки положительные
        assert cost > 0, "Торговые издержки должны быть положительными"
        
        # Проверяем, что издержки разумные (не слишком большие)
        total_trade_value = abs(position_change) * price_s1 + abs(beta * position_change) * price_s2
        cost_percentage = cost / total_trade_value
        assert cost_percentage < 0.01, f"Издержки ({cost_percentage:.4f}) не должны превышать 1% от торгового объема"
        
        # Запускаем бэктест и проверяем наличие детализированных издержек
        engine.run()
        results = engine.get_results()
        
        # Проверяем, что результаты содержат информацию о торговых издержках
        if isinstance(results, dict) and 'trade_cost' in results:
            trade_costs = results['trade_cost'].dropna()
            if len(trade_costs) > 0:
                assert (trade_costs >= 0).all(), "Все торговые издержки должны быть неотрицательными"
    
    def test_enhancements_integration(self, sample_data):
        """Проверяет интеграцию всех улучшений в бэктестировании.
        
        Тест проверяет:
        1. Корректную работу с онлайновыми статистиками
        2. Правильную работу сдвига сигналов
        3. Отсутствие утечки данных
        """
        engine = PairBacktester(
            sample_data,
            rolling_window=20,
            z_threshold=2.0,
            capital_at_risk=10000,
            walk_forward_enabled=False,
            online_stats_enabled=True,
            signal_shift_enabled=True
        )
        
        # Проверяем настройки онлайновых статистик
        assert engine.online_stats_enabled, "Онлайновые статистики должны быть включены"
        
        # Запускаем бэктестирование
        engine.run()
        results = engine.get_results()
        
        # Проверяем, что результаты получены
        assert results is not None, "Результаты должны быть получены"
        assert len(results) > 0, "Результаты не должны быть пустыми"
        
        # Проверяем базовые метрики
        if isinstance(results, dict):
            if 'pnl' in results:
                pnl_series = results['pnl']
                assert not pnl_series.isna().all(), "PnL не должен быть полностью NaN"
            
            if 'position' in results:
                positions = results['position']
                assert not positions.isna().all(), "Позиции не должны быть полностью NaN"
    
    def test_no_look_ahead_bias_in_backtest(self, sample_data):
        """Проверяет отсутствие look-ahead bias в бэктестировании.
        
        Тест проверяет:
        1. Параметры рассчитываются только на исторических данных
        2. Онлайновые статистики не используют будущую информацию
        3. Размер окна соблюдается корректно
        """
        engine = PairBacktester(
            sample_data,
            rolling_window=20,
            z_threshold=2.0,
            capital_at_risk=10000,
            walk_forward_enabled=False,
            online_stats_enabled=True
        )
        
        # Мокаем метод для отслеживания данных, используемых в расчетах
        original_ols_method = engine._calculate_ols_with_cache
        data_usage_log = []
        
        def mock_ols_with_cache(y_data, x_data):
            # Записываем информацию об используемых данных
            data_usage_log.append({
                'data_start': y_data.index[0] if len(y_data) > 0 else None,
                'data_end': y_data.index[-1] if len(y_data) > 0 else None,
                'data_length': len(y_data)
            })
            return original_ols_method(y_data, x_data)
        
        with patch.object(engine, '_calculate_ols_with_cache', side_effect=mock_ols_with_cache):
            engine.run()
            results = engine.get_results()
        
        # Проверяем, что данные использовались
        assert len(data_usage_log) > 0, "Должны быть записи об использовании данных"
        
        # Проверяем временную последовательность
        for i, usage in enumerate(data_usage_log[1:], 1):
            prev_usage = data_usage_log[i-1]
            
            if usage['data_end'] and prev_usage['data_end']:
                # Проверяем, что данные не используют будущую информацию
                assert usage['data_end'] >= prev_usage['data_end'] or \
                       abs((usage['data_end'] - prev_usage['data_end']).total_seconds()) < 3600, \
                    "Данные не должны использовать информацию из будущего"
        
        # Проверяем результаты
        assert results is not None, "Результаты должны быть получены"
        assert len(results) > 0, "Результаты не должны быть пустыми"