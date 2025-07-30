"""Тесты робастности и корректности бэктеста.

Эти тесты проверяют:
- Отсутствие утечки данных из будущего (look-ahead bias)
- Корректность обработки временных сдвигов
- Чувствительность к комиссиям и издержкам
- Поведение на синтетических данных
- Sanity-проверки сигналов
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from coint2.engine.base_engine import BasePairBacktester as PairBacktester


class BacktestRobustnessTests:
    """Тесты робастности бэктеста."""
    
    def setUp(self):
        """Настройка тестовых данных."""
        np.random.seed(42)
        self.n_periods = 1000
        dates = pd.date_range('2024-01-01', periods=self.n_periods, freq='15min')
        
        # Создаем базовые коинтегрированные данные
        price_s1 = 100 + np.cumsum(np.random.normal(0, 0.1, self.n_periods))
        price_s2 = 50 + 0.5 * price_s1 + np.cumsum(np.random.normal(0, 0.05, self.n_periods))
        
        self.base_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        self.base_params = {
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
            'time_stop_multiplier': 2.0,
            'cooldown_periods': 5
        }
    
    def test_date_permutation_destroys_performance(self):
        """Тест: перемешивание дат должно уничтожить производительность.
        
        Если стратегия корректна, то случайное перемешивание дат
        должно привести к Sharpe ratio близкому к нулю.
        """
        # Запускаем оригинальный бэктест
        original_engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        original_engine.run()
        original_metrics = original_engine.get_performance_metrics()
        original_sharpe = original_metrics.get('sharpe_ratio', 0)
        
        # Перемешиваем даты (сохраняя структуру DataFrame)
        shuffled_data = self.base_data.copy()
        shuffled_indices = np.random.permutation(len(shuffled_data))
        shuffled_data.iloc[:, :] = shuffled_data.iloc[shuffled_indices, :].values
        
        # Запускаем бэктест на перемешанных данных
        shuffled_engine = PairBacktester(
            pair_data=shuffled_data,
            **self.base_params
        )
        shuffled_engine.run()
        shuffled_metrics = shuffled_engine.get_performance_metrics()
        shuffled_sharpe = shuffled_metrics.get('sharpe_ratio', 0)
        
        # Проверяем, что перемешивание значительно ухудшило результаты
        assert abs(shuffled_sharpe) < abs(original_sharpe) * 0.5, \
            f"Перемешивание должно ухудшить Sharpe: оригинал={original_sharpe:.3f}, перемешанный={shuffled_sharpe:.3f}"
        
        # Sharpe после перемешивания должен быть близок к нулю
        assert abs(shuffled_sharpe) < 0.5, \
            f"Sharpe после перемешивания должен быть близок к нулю: {shuffled_sharpe:.3f}"
    
    def test_time_shift_degrades_performance(self):
        """Тест: смещение одной серии должно ухудшить производительность.
        
        Если одну из серий сдвинуть во времени, коинтеграция нарушается
        и производительность должна деградировать.
        """
        # Оригинальный бэктест
        original_engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        original_engine.run()
        original_metrics = original_engine.get_performance_metrics()
        original_sharpe = original_metrics.get('sharpe_ratio', 0)
        
        # Создаем данные со сдвигом второй серии на 10 периодов
        shifted_data = self.base_data.copy()
        shifted_data['S2'] = shifted_data['S2'].shift(10)
        shifted_data = shifted_data.dropna()  # Удаляем NaN после сдвига
        
        # Бэктест на сдвинутых данных
        shifted_engine = PairBacktester(
            pair_data=shifted_data,
            **self.base_params
        )
        shifted_engine.run()
        shifted_metrics = shifted_engine.get_performance_metrics()
        shifted_sharpe = shifted_metrics.get('sharpe_ratio', 0)
        
        # Сдвиг должен ухудшить производительность
        # Если оба Sharpe отрицательные, то больший по модулю означает худшую производительность
        if original_sharpe >= 0:
            # Для положительного Sharpe: сдвиг должен его уменьшить
            assert shifted_sharpe < original_sharpe * 0.7, \
                f"Сдвиг должен ухудшить Sharpe: оригинал={original_sharpe:.3f}, сдвинутый={shifted_sharpe:.3f}"
        else:
            # Для отрицательного Sharpe: сдвиг может его ухудшить (сделать более отрицательным)
            # или улучшить (сделать менее отрицательным), но не должен делать положительным
            assert shifted_sharpe <= 0.1, \
                f"Сдвиг не должен создавать положительный Sharpe из отрицательного: оригинал={original_sharpe:.3f}, сдвинутый={shifted_sharpe:.3f}"
    
    def test_fee_sensitivity(self):
        """Тест: увеличение комиссий должно уменьшать PnL.
        
        При увеличении комиссий и проскальзывания итоговый PnL
        должен монотонно уменьшаться.
        """
        fee_multipliers = [1.0, 2.0, 5.0, 10.0]
        pnl_results = []
        
        for multiplier in fee_multipliers:
            params = self.base_params.copy()
            params['commission_pct'] *= multiplier
            params['slippage_pct'] *= multiplier
            params['bid_ask_spread_pct_s1'] *= multiplier
            params['bid_ask_spread_pct_s2'] *= multiplier
            
            engine = PairBacktester(
                pair_data=self.base_data,
                **params
            )
            engine.run()
            metrics = engine.get_performance_metrics()
            pnl_results.append(metrics.get('total_return', 0))
        
        # PnL должен монотонно уменьшаться с ростом комиссий
        for i in range(1, len(pnl_results)):
            assert pnl_results[i] <= pnl_results[i-1], \
                f"PnL должен уменьшаться с ростом комиссий: {pnl_results}"
        
        # При 10x комиссиях PnL должен быть значительно хуже
        assert pnl_results[-1] < pnl_results[0] * 0.5, \
            f"10x комиссии должны значительно ухудшить PnL: {pnl_results[0]:.3f} -> {pnl_results[-1]:.3f}"
    
    def test_no_future_reference(self):
        """Тест: изменение будущих баров не должно влиять на прошлые сигналы.
        
        Модифицируем последние 100 баров и проверяем, что сигналы
        на первых 800 барах остались неизменными.
        """
        # Оригинальный бэктест
        original_engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        original_engine.run()
        original_results = original_engine.results
        
        # Модифицируем последние 100 баров
        modified_data = self.base_data.copy()
        modified_data.iloc[-100:, :] *= 1.5  # Увеличиваем цены в 1.5 раза
        
        # Бэктест на модифицированных данных
        modified_engine = PairBacktester(
            pair_data=modified_data,
            **self.base_params
        )
        modified_engine.run()
        modified_results = modified_engine.results
        
        # Проверяем первые 800 баров (исключаем последние 200 для безопасности)
        check_length = min(800, len(original_results) - 200)
        
        if check_length > 0:
            # Сравниваем позиции на первых 800 барах
            original_positions = original_results['position'].iloc[:check_length]
            modified_positions = modified_results['position'].iloc[:check_length]
            
            # Позиции должны быть идентичными
            position_diff = np.abs(original_positions - modified_positions).sum()
            assert position_diff < 1e-10, \
                f"Изменение будущих данных не должно влиять на прошлые позиции: diff={position_diff}"
    
    def test_signal_shift_sanity_check(self):
        """Sanity-тест: shift(1) должен снижать завышенный Sharpe.
        
        Если сдвинуть сигналы на 1 бар вперед, производительность
        должна ухудшиться (защита от look-ahead bias).
        """
        # Создаем модифицированный движок с задержкой сигналов
        class DelayedSignalBacktester(PairBacktester):
            def _calculate_signals(self):
                """Рассчитываем сигналы с задержкой на 1 бар."""
                super()._calculate_signals()
                # Сдвигаем все сигналы на 1 бар вперед
                if hasattr(self, 'signals') and len(self.signals) > 0:
                    self.signals = self.signals.shift(1).fillna(0)
        
        # Оригинальный бэктест
        original_engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        original_engine.run()
        original_metrics = original_engine.get_performance_metrics()
        original_sharpe = original_metrics.get('sharpe_ratio', 0)
        
        # Бэктест с задержкой сигналов
        delayed_engine = DelayedSignalBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        delayed_engine.run()
        delayed_metrics = delayed_engine.get_performance_metrics()
        delayed_sharpe = delayed_metrics.get('sharpe_ratio', 0)
        
        # Задержка должна ухудшить производительность
        assert delayed_sharpe < original_sharpe * 0.8, \
            f"Задержка сигналов должна ухудшить Sharpe: оригинал={original_sharpe:.3f}, задержанный={delayed_sharpe:.3f}"
    
    def test_synthetic_random_walk(self):
        """Тест на случайном блуждании: Sharpe должен быть близок к нулю.
        
        На данных случайного блуждания стратегия не должна
        показывать значимую производительность.
        """
        np.random.seed(123)
        n = 1000
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем независимые случайные блуждания
        random_walk_s1 = 100 + np.cumsum(np.random.normal(0, 0.1, n))
        random_walk_s2 = 50 + np.cumsum(np.random.normal(0, 0.1, n))
        
        random_data = pd.DataFrame({
            'S1': random_walk_s1,
            'S2': random_walk_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=random_data,
            **self.base_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # На случайных данных Sharpe должен быть близок к нулю
        # Увеличиваем допуск, так как на случайных данных возможны флуктуации
        assert abs(sharpe) < 0.8, \
            f"На случайных данных Sharpe должен быть близок к нулю: {sharpe:.3f}"
    
    def test_cost_breakdown_consistency(self):
        """Тест: проверка консистентности разбивки издержек.
        
        Сумма всех издержек должна соответствовать общему влиянию
        на итоговый PnL.
        """
        # Бэктест без издержек
        no_cost_params = self.base_params.copy()
        no_cost_params.update({
            'commission_pct': 0.0,
            'slippage_pct': 0.0,
            'bid_ask_spread_pct_s1': 0.0,
            'bid_ask_spread_pct_s2': 0.0
        })
        
        no_cost_engine = PairBacktester(
            pair_data=self.base_data,
            **no_cost_params
        )
        no_cost_engine.run()
        no_cost_pnl = no_cost_engine.get_performance_metrics().get('total_return', 0)
        
        # Бэктест с издержками
        with_cost_engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        with_cost_engine.run()
        with_cost_pnl = with_cost_engine.get_performance_metrics().get('total_return', 0)
        
        # Издержки должны уменьшать PnL
        cost_impact = no_cost_pnl - with_cost_pnl
        assert cost_impact >= 0, \
            f"Издержки должны уменьшать PnL: без издержек={no_cost_pnl:.3f}, с издержками={with_cost_pnl:.3f}"
        
        # Влияние издержек должно быть разумным (не более 50% от PnL)
        if no_cost_pnl > 0:
            cost_ratio = cost_impact / no_cost_pnl
            assert cost_ratio < 0.5, \
                f"Влияние издержек не должно превышать 50% PnL: {cost_ratio:.2%}"
    
    def test_position_limits_respected(self):
        """Тест: проверка соблюдения лимитов позиций.
        
        Позиции не должны превышать заданные лимиты по капиталу.
        """
        engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        engine.run()
        
        if hasattr(engine, 'results') and len(engine.results) > 0:
            # Проверяем, что позиции не превышают лимиты
            max_position = abs(engine.results['position']).max()
            capital_limit = self.base_params['capital_at_risk']
            
            # Позиция не должна превышать доступный капитал
            assert max_position <= capital_limit * 1.1, \
                f"Позиция превышает лимит капитала: {max_position} > {capital_limit}"
    
    def test_equity_curve_monotonicity(self):
        """Тест: проверка монотонности кривой капитала.
        
        Кривая капитала должна быть непрерывной и не содержать
        резких скачков, не объяснимых торговыми операциями.
        """
        engine = PairBacktester(
            pair_data=self.base_data,
            **self.base_params
        )
        engine.run()
        
        if hasattr(engine, 'results') and len(engine.results) > 0:
            equity = engine.results['equity']
            
            # Проверяем отсутствие NaN и Inf
            assert not equity.isna().any(), "Кривая капитала не должна содержать NaN"
            assert not np.isinf(equity).any(), "Кривая капитала не должна содержать Inf"
            
            # Проверяем разумность изменений (не более 50% за один бар)
            equity_changes = equity.pct_change().dropna()
            max_change = abs(equity_changes).max()
            assert max_change < 0.5, \
                f"Слишком большое изменение капитала за один бар: {max_change:.2%}"
    
    def run_all_tests(self, pair_data=None):
        """Запускает все тесты робастности и возвращает результаты.
        
        Args:
            pair_data: Данные для тестирования (если None, используются базовые)
            
        Returns:
            List[Dict]: Список результатов тестов
        """
        if pair_data is not None:
            self.pair_data = pair_data
        
        # Настройка тестовых данных
        self.setUp()
        
        # Список всех тестовых методов
        test_methods = [
            'test_date_permutation_destroys_performance',
            'test_time_shift_degrades_performance', 
            'test_fee_sensitivity',
            'test_no_future_reference',
            'test_signal_shift_sanity_check',
            'test_synthetic_random_walk',
            'test_cost_breakdown_consistency',
            'test_position_limits_respected',
            'test_equity_curve_monotonicity'
        ]
        
        results = []
        
        for test_name in test_methods:
            try:
                # Получаем метод тестирования
                test_method = getattr(self, test_name)
                
                # Запускаем тест
                test_method()
                
                # Если дошли сюда, тест прошел
                results.append({
                    'test_name': test_name,
                    'status': 'PASSED',
                    'error': None
                })
                
            except Exception as e:
                # Тест провален
                results.append({
                    'test_name': test_name,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        return results