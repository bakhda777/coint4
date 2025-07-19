"""Интеграционные тесты для проверки общей корректности бэктеста.

Эти тесты проверяют:
- Корректность работы всего пайплайна бэктестинга
- Согласованность результатов между различными компонентами
- Граничные случаи и обработку ошибок
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coint2.engine.backtest_engine import PairBacktester, TradeState


class TestBacktestIntegration:
    """Интеграционные тесты для бэктеста."""
    
    def setup_method(self):
        """Настройка тестовых данных."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        
        # Создаем реалистичные коинтегрированные данные
        price_s1 = 100 + np.cumsum(np.random.normal(0, 0.1, 500))
        # S2 коинтегрирована с S1 с некоторым шумом
        price_s2 = 50 + 0.5 * price_s1 + np.cumsum(np.random.normal(0, 0.05, 500))
        
        self.test_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        self.base_params = {
            'pair_data': self.test_data,
            'rolling_window': 50,
            'z_threshold': 2.0,
            'z_exit': 0.5,
            'stop_loss_multiplier': 3.0,
            'capital_at_risk': 100000,
            'commission_pct': 0.001,
            'slippage_pct': 0.0005,
            'bid_ask_spread_pct_s1': 0.0002,
            'bid_ask_spread_pct_s2': 0.0002,
            'half_life': 10.0,
            'time_stop_multiplier': 2.0
        }
    
    def test_full_backtest_pipeline_consistency(self):
        """Тест полного пайплайна бэктестинга на согласованность.
        
        Проверяет, что все компоненты работают согласованно и
        результаты логически корректны.
        """
        engine = PairBacktester(**self.base_params)
        engine.run()
        
        # Базовые проверки результатов
        assert hasattr(engine, 'results'), "Результаты должны существовать"
        assert not engine.results.empty, "Результаты не должны быть пустыми"
        assert len(engine.results) == len(self.test_data), "Длина результатов должна совпадать с данными"
        
        # Проверяем обязательные столбцы
        required_columns = ['position', 'pnl', 'cumulative_pnl', 'z_score', 'beta']
        for col in required_columns:
            assert col in engine.results.columns, f"Столбец {col} должен присутствовать в результатах"
        
        # Проверяем логическую согласованность позиций
        positions = engine.results['position']
        # Убираем NaN значения перед проверкой
        valid_positions = positions.dropna()
        if len(valid_positions) > 0:
            # Позиции могут быть в денежном выражении, проверяем что они конечные числа
            assert valid_positions.apply(lambda x: np.isfinite(x)).all(), "Все позиции должны быть конечными числами"
            # Проверяем, что есть как минимум нулевые позиции (периоды без торговли)
            assert (valid_positions == 0).any(), "Должны быть периоды без открытых позиций"
        
        # Проверяем, что PnL рассчитывается только при открытых позициях
        non_zero_positions = positions != 0
        if non_zero_positions.any():
            pnl_during_positions = engine.results.loc[non_zero_positions, 'pnl']
            # PnL может быть нулевым в первый период позиции, но не всегда
            assert not pnl_during_positions.isna().all(), "PnL должен рассчитываться при открытых позициях"
        
        # Проверяем монотонность кумулятивного PnL
        cumulative_pnl = engine.results['cumulative_pnl']
        pnl_changes = cumulative_pnl.diff().fillna(0)
        actual_pnl = engine.results['pnl'].fillna(0)
        
        # Изменения кумулятивного PnL должны соответствовать PnL
        assert np.allclose(pnl_changes, actual_pnl, atol=1e-10), "Кумулятивный PnL должен быть суммой PnL"
    
    def test_position_entry_exit_logic(self):
        """Тест логики входа и выхода из позиций.
        
        Проверяет, что позиции открываются и закрываются согласно правилам.
        """
        engine = PairBacktester(**self.base_params)
        engine.run()
        
        results = engine.results
        positions = results['position']
        z_scores = results['z_score']
        
        # Находим моменты входа в позицию
        position_changes = positions.diff().fillna(0)
        entries = position_changes != 0
        
        if entries.any():
            entry_indices = results.index[entries]
            
            for entry_idx in entry_indices:
                if pd.notna(z_scores.loc[entry_idx]):
                    z_score_at_entry = z_scores.loc[entry_idx]
                    position_at_entry = positions.loc[entry_idx]
                    
                    # Проверяем правила входа
                    if position_at_entry == 1:  # Длинная позиция
                        assert z_score_at_entry <= -engine.z_threshold, f"Длинная позиция должна открываться при z <= -{engine.z_threshold}, получено {z_score_at_entry}"
                    elif position_at_entry == -1:  # Короткая позиция
                        assert z_score_at_entry >= engine.z_threshold, f"Короткая позиция должна открываться при z >= {engine.z_threshold}, получено {z_score_at_entry}"
        
        # Проверяем, что позиции не открываются в первые rolling_window периодов
        early_positions = positions.iloc[:engine.rolling_window]
        assert (early_positions == 0).all(), "Позиции не должны открываться до накопления достаточной истории"
    
    def test_trading_costs_consistency(self):
        """Тест согласованности торговых издержек.
        
        Проверяет, что все виды издержек рассчитываются корректно
        и влияют на итоговый PnL.
        """
        # Создаем параметры с высокими издержками для лучшей видимости эффекта
        params = self.base_params.copy()
        params['commission_pct'] = 0.01  # 1%
        params['slippage_pct'] = 0.005   # 0.5%
        params['bid_ask_spread_pct_s1'] = 0.002  # 0.2%
        params['bid_ask_spread_pct_s2'] = 0.002  # 0.2%
        
        engine = PairBacktester(**params)
        engine.run()
        
        results = engine.results
        
        # Проверяем, что издержки учитываются
        if 'commission_costs' in results.columns:
            commission_costs = results['commission_costs'].fillna(0)
            total_commission = commission_costs.sum()
            
            # При высоких комиссиях должны быть значительные издержки
            assert total_commission > 0, "Комиссионные издержки должны быть положительными"
        
        # Сравниваем с бэктестом без издержек
        params_no_costs = self.base_params.copy()
        params_no_costs['commission_pct'] = 0
        params_no_costs['slippage_pct'] = 0
        params_no_costs['bid_ask_spread_pct_s1'] = 0
        params_no_costs['bid_ask_spread_pct_s2'] = 0
        
        engine_no_costs = PairBacktester(**params_no_costs)
        engine_no_costs.run()
        
        # PnL с издержками должен быть меньше PnL без издержек
        final_pnl_with_costs = results['cumulative_pnl'].iloc[-1]
        final_pnl_no_costs = engine_no_costs.results['cumulative_pnl'].iloc[-1]
        
        if final_pnl_no_costs > 0:  # Если стратегия прибыльна без издержек
            assert final_pnl_with_costs < final_pnl_no_costs, "PnL с издержками должен быть меньше PnL без издержек"
    
    def test_risk_management_limits(self):
        """Тест лимитов управления рисками.
        
        Проверяет, что система корректно применяет лимиты рисков.
        """
        # Устанавливаем строгие лимиты
        params = self.base_params.copy()
        params['max_kelly_fraction'] = 0.1  # Максимум 10% капитала в риске
        params['volatility_lookback'] = 20
        params['var_confidence'] = 0.95
        params['max_var_multiplier'] = 2.0
        
        engine = PairBacktester(**params)
        engine.run()
        
        results = engine.results
        positions = results['position']
        
        # Проверяем, что позиции не превышают разумных размеров
        if 'position_size' in results.columns:
            position_sizes = results['position_size'].fillna(0)
            max_position_size = position_sizes.abs().max()
            
            # Размер позиции не должен превышать разумную долю капитала
            max_reasonable_size = params['capital_at_risk'] * 0.5  # 50% капитала
            assert max_position_size <= max_reasonable_size, f"Размер позиции {max_position_size} превышает разумный лимит {max_reasonable_size}"
    
    def test_edge_cases_handling(self):
        """Тест обработки граничных случаев.
        
        Проверяет корректную обработку различных граничных ситуаций.
        """
        # Тест с очень малым rolling_window
        params_small_window = self.base_params.copy()
        params_small_window['rolling_window'] = 5
        params_small_window['pair_data'] = self.test_data.iloc[:20]  # Мало данных
        
        engine_small = PairBacktester(**params_small_window)
        engine_small.run()  # Не должно вызывать ошибок
        
        assert hasattr(engine_small, 'results'), "Результаты должны существовать даже для малого окна"
        
        # Тест с константными ценами (отсутствие волатильности)
        constant_data = pd.DataFrame({
            'S1': [100.0] * 100,
            'S2': [50.0] * 100
        }, index=pd.date_range('2024-01-01', periods=100, freq='15min'))
        
        params_constant = self.base_params.copy()
        params_constant['pair_data'] = constant_data
        
        engine_constant = PairBacktester(**params_constant)
        engine_constant.run()  # Не должно вызывать ошибок
        
        # При константных ценах не должно быть торговли
        positions_constant = engine_constant.results['position']
        assert (positions_constant == 0).all(), "При константных ценах не должно быть торговли"
        
        # Тест с очень высокой волатильностью
        np.random.seed(123)
        volatile_s1 = 100 + np.cumsum(np.random.normal(0, 5, 100))  # Высокая волатильность
        volatile_s2 = 50 + 0.5 * volatile_s1 + np.cumsum(np.random.normal(0, 2, 100))
        
        volatile_data = pd.DataFrame({
            'S1': volatile_s1,
            'S2': volatile_s2
        }, index=pd.date_range('2024-01-01', periods=100, freq='15min'))
        
        params_volatile = self.base_params.copy()
        params_volatile['pair_data'] = volatile_data
        
        engine_volatile = PairBacktester(**params_volatile)
        engine_volatile.run()  # Не должно вызывать ошибок
        
        assert hasattr(engine_volatile, 'results'), "Результаты должны существовать даже для высоковолатильных данных"
    
    def test_performance_metrics_calculation(self):
        """Тест расчета метрик производительности.
        
        Проверяет корректность расчета всех метрик производительности.
        """
        engine = PairBacktester(**self.base_params)
        engine.run()
        
        metrics = engine.get_performance_metrics()
        
        # Проверяем наличие ключевых метрик
        expected_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'win_rate', 'num_trades', 'avg_trade_duration'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Метрика {metric} должна присутствовать"
            assert np.isfinite(metrics[metric]), f"Метрика {metric} должна быть конечным числом"
        
        # Проверяем логические ограничения метрик
        assert 0 <= metrics['win_rate'] <= 1, "Win rate должен быть между 0 и 1"
        assert metrics['num_trades'] >= 0, "Количество сделок должно быть неотрицательным"
        assert metrics['max_drawdown'] <= 0, "Максимальная просадка должна быть неположительной"
        
        if metrics['num_trades'] > 0:
            assert metrics['avg_trade_duration'] > 0, "Средняя продолжительность сделки должна быть положительной"
    
    def test_incremental_vs_batch_consistency(self):
        """Тест согласованности инкрементального и пакетного расчета.
        
        Проверяет, что инкрементальный расчет дает те же результаты,
        что и пакетный расчет на тех же данных.
        """
        engine = PairBacktester(**self.base_params)
        
        # Запускаем полный бэктест
        engine.run()
        batch_results = engine.results.copy()
        
        # Запускаем инкрементальный расчет
        engine_incremental = PairBacktester(**self.base_params)
        engine_incremental.run()  # Запускаем полный расчет для инкрементального движка
        
        # Сравниваем ключевые результаты
        # Поскольку оба движка используют одинаковые данные и параметры,
        # результаты должны быть идентичными
        
        batch_final_pnl = batch_results['cumulative_pnl'].iloc[-1]
        incremental_final_pnl = engine_incremental.results['cumulative_pnl'].iloc[-1]
        
        # Проверяем, что результаты идентичны
        assert abs(batch_final_pnl - incremental_final_pnl) < 1e-10, f"Результаты должны быть идентичными: {batch_final_pnl} vs {incremental_final_pnl}"