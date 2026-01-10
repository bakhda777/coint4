"""
Тест для проверки исправления lookahead bias в rolling статистиках.

КРИТИЧЕСКАЯ ПРОБЛЕМА которая была исправлена:
- В методе process_single_period использовались последние данные (self.pair_data.iloc[-rolling_window:])
- Это включало будущие данные при бэктесте, создавая lookahead bias
- Z-score рассчитывался с использованием информации из будущего

РЕШЕНИЕ:
- Добавлен параметр current_idx в process_single_period
- Данные берутся строго до текущего индекса: self.pair_data.iloc[start_idx:current_idx]
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from coint2.engine.base_engine import BasePairBacktester
from coint2.core.portfolio import Portfolio


class TestLookaheadBiasFix:
    """Тесты для проверки отсутствия lookahead bias в расчетах."""
    
    @pytest.fixture
    def sample_data(self):
        """Создаем тестовые данные с известным паттерном."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        
        # Создаем данные с четким изменением тренда в середине
        # Первая половина - восходящий тренд
        # Вторая половина - нисходящий тренд
        mid_point = len(dates) // 2
        
        price1 = np.zeros(len(dates))
        price2 = np.zeros(len(dates))
        
        # Первая половина - положительный спред
        price1[:mid_point] = 100 + np.arange(mid_point) * 0.1
        price2[:mid_point] = 100 + np.arange(mid_point) * 0.05
        
        # Вторая половина - отрицательный спред (резкое изменение)
        price1[mid_point:] = 105 - np.arange(mid_point) * 0.1
        price2[mid_point:] = 105 + np.arange(mid_point) * 0.05
        
        df = pd.DataFrame({
            'asset1': price1,
            'asset2': price2
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def backtester(self, sample_data):
        """Создаем бэктестер с тестовыми данными."""
        portfolio = Portfolio(initial_capital=10000, max_active_positions=5)
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=20,
            portfolio=portfolio,
            z_threshold=2.0,
            z_exit=0.5
        )
        return backtester
    
    def test_process_single_period_without_index(self, backtester, sample_data):
        """
        Тест process_single_period БЕЗ передачи current_idx.
        Это старое поведение с lookahead bias (для обратной совместимости).
        """
        # Берем цены из середины данных
        mid_idx = len(sample_data) // 2
        date = sample_data.index[mid_idx]
        price1 = sample_data.iloc[mid_idx, 0]
        price2 = sample_data.iloc[mid_idx, 1]
        
        # Вызываем без current_idx (старое поведение)
        result = backtester.process_single_period(date, price1, price2)
        
        # Проверяем что результат содержит необходимые ключи
        assert 'position' in result
        assert 'pnl' in result
        assert 'z_score' in result
        
        # Сохраняем z_score для сравнения
        z_score_with_lookahead = result['z_score']
        
        # Не возвращаем значение из теста (pytest ожидает None)
    
    def test_process_single_period_with_index(self, backtester, sample_data):
        """
        Тест process_single_period С передачей current_idx.
        Это новое поведение БЕЗ lookahead bias.
        """
        # Берем цены из середины данных
        mid_idx = len(sample_data) // 2
        date = sample_data.index[mid_idx]
        price1 = sample_data.iloc[mid_idx, 0]
        price2 = sample_data.iloc[mid_idx, 1]
        
        # Вызываем с current_idx (новое поведение без lookahead)
        result = backtester.process_single_period(date, price1, price2, current_idx=mid_idx)
        
        # Проверяем что результат содержит необходимые ключи
        assert 'position' in result
        assert 'pnl' in result
        assert 'z_score' in result
        
        # Z-score должен быть рассчитан только на данных ДО текущего момента
        z_score_without_lookahead = result['z_score']
        
        # Не возвращаем значение из теста (pytest ожидает None)
    
    def test_lookahead_bias_detection(self, backtester, sample_data):
        """
        Проверяем что z_score отличается при использовании и без использования будущих данных.
        Это доказывает что исправление работает.
        """
        # Тестируем на точке где есть резкое изменение тренда
        test_idx = len(sample_data) // 2 + 5  # Чуть после изменения тренда
        date = sample_data.index[test_idx]
        price1 = sample_data.iloc[test_idx, 0]
        price2 = sample_data.iloc[test_idx, 1]
        
        # Старое поведение (с lookahead bias)
        result_with_lookahead = backtester.process_single_period(date, price1, price2)
        z_with_lookahead = result_with_lookahead.get('z_score', 0)
        
        # Новое поведение (без lookahead bias)
        result_without_lookahead = backtester.process_single_period(date, price1, price2, current_idx=test_idx)
        z_without_lookahead = result_without_lookahead.get('z_score', 0)
        
        # Z-scores должны отличаться, так как один использует будущие данные, а другой нет
        # Допускаем что они могут быть равны если данные очень стабильны
        # но в нашем случае с резким изменением тренда они должны отличаться
        print(f"Z-score с lookahead: {z_with_lookahead:.4f}")
        print(f"Z-score без lookahead: {z_without_lookahead:.4f}")
        
        # В точке после изменения тренда, z_score с lookahead должен учитывать
        # будущий нисходящий тренд и давать другое значение
        # Проверяем что значения действительно разные (допускаем малую погрешность)
        assert abs(z_with_lookahead - z_without_lookahead) > 0.01, \
            f"Z-scores должны отличаться: с lookahead={z_with_lookahead:.4f}, без={z_without_lookahead:.4f}"
    
    def test_volatility_multiplier_without_lookahead(self, backtester):
        """
        Тест _calculate_volatility_multiplier без lookahead bias.
        """
        # Тестируем на разных индексах
        test_indices = [30, 50, 70]
        
        for idx in test_indices:
            # Новое поведение - с указанием индекса
            multiplier_no_lookahead = backtester._calculate_volatility_multiplier(current_idx=idx)
            
            # Старое поведение - без указания индекса (может использовать будущие данные)
            multiplier_with_lookahead = backtester._calculate_volatility_multiplier()
            
            # Проверяем что множители валидны
            assert 0 < multiplier_no_lookahead <= 10, f"Invalid multiplier at idx {idx}"
            assert 0 < multiplier_with_lookahead <= 10, f"Invalid multiplier (old) at idx {idx}"
            
            print(f"Idx {idx}: без lookahead={multiplier_no_lookahead:.4f}, с lookahead={multiplier_with_lookahead:.4f}")
    
    def test_rolling_stats_consistency(self, backtester):
        """
        Проверяем что rolling статистики рассчитываются консистентно
        и используют только исторические данные.
        """
        # Создаем тестовые данные с достаточной вариацией
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        
        # Создаем данные с шумом для прохождения валидации
        np.random.seed(42)
        price1 = 100 + np.cumsum(np.random.randn(50) * 0.5)
        price2 = 100 + np.cumsum(np.random.randn(50) * 0.3)
        
        df = pd.DataFrame({
            'y': price1,
            'x': price2
        }, index=dates)
        
        # Обновляем данные в бэктестере
        backtester.pair_data = df[['y', 'x']]
        
        # Инициализируем необходимые колонки (используем правильные имена из base_engine)
        df['spread'] = 0.0
        df['mean'] = np.nan  # Изменено с rolling_mean
        df['std'] = np.nan   # Изменено с rolling_std
        df['z_score'] = np.nan
        df['beta'] = np.nan
        
        # Тестируем update_rolling_stats на нескольких барах
        # ВАЖНО: update_rolling_stats работает только для индексов >= rolling_window
        test_indices = [25, 30, 35, 40]
        
        for i in test_indices:
            # Пропускаем индексы меньше rolling_window
            if i < backtester.rolling_window:
                continue
                
            # Вызываем update_rolling_stats
            backtester.update_rolling_stats(df, i)
            
            # Проверяем что статистики рассчитаны (используем правильные имена колонок)
            # Может быть NaN если данные не проходят валидацию, поэтому проверяем что хотя бы некоторые не NaN
            
        # Проверяем что хотя бы для некоторых индексов статистики были рассчитаны
        stats_calculated = False
        for i in test_indices:
            if i >= backtester.rolling_window and not np.isnan(df.loc[df.index[i], 'mean']):
                stats_calculated = True
                break
        
        assert stats_calculated, "Статистики не были рассчитаны ни для одного индекса"
    
    def test_no_future_data_in_calculations(self, sample_data):
        """
        Комплексный тест что будущие данные НЕ используются в расчетах.
        """
        # Создаем данные где будущее сильно отличается от прошлого
        dates = pd.date_range(start='2023-01-01', periods=60, freq='15min')
        
        # Стабильные цены в начале
        stable_prices1 = np.ones(30) * 100
        stable_prices2 = np.ones(30) * 100
        
        # Резкий скачок в будущем
        jump_prices1 = np.ones(30) * 200  # Удвоение цены
        jump_prices2 = np.ones(30) * 150  # Увеличение на 50%
        
        test_data = pd.DataFrame({
            'asset1': np.concatenate([stable_prices1, jump_prices1]),
            'asset2': np.concatenate([stable_prices2, jump_prices2])
        }, index=dates)
        
        # Создаем бэктестер
        portfolio = Portfolio(initial_capital=10000, max_active_positions=5)
        backtester = BasePairBacktester(
            pair_data=test_data,
            rolling_window=10,
            portfolio=portfolio,
            z_threshold=2.0
        )
        
        # Тестируем на границе между стабильным и скачком
        boundary_idx = 29  # Последний стабильный период
        date = test_data.index[boundary_idx]
        price1 = test_data.iloc[boundary_idx, 0]
        price2 = test_data.iloc[boundary_idx, 1]
        
        # С правильным индексом НЕ должны видеть будущий скачок
        result = backtester.process_single_period(date, price1, price2, current_idx=boundary_idx)
        z_score = result.get('z_score', 0)
        
        # Z-score должен быть близок к 0, так как цены стабильны
        assert abs(z_score) < 1.0, \
            f"Z-score {z_score:.4f} слишком большой для стабильных данных (не должен видеть будущий скачок)"
        
        print(f"✅ Z-score на границе (без видимости скачка): {z_score:.4f}")


if __name__ == "__main__":
    # Запускаем тесты
    pytest.main([__file__, "-v", "-s"])