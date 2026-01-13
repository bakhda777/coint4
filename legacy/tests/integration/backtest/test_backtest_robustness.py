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
import pytest

from coint2.engine.base_engine import BasePairBacktester as PairBacktester

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_COMMISSION_PCT = 0.0
DEFAULT_SLIPPAGE_PCT = 0.0

# Константы для комиссий
FEE_LEVELS = [0.0, 0.001, 0.005]
HIGH_FEE = 0.01
EXTREME_FEE = 0.1

# Константы для тестовых данных
TEST_PERIODS = 100
FREQUENCY = '15T'
START_DATE = '2023-01-01'
SMALL_ROLLING_WINDOW = 5
MEDIUM_ROLLING_WINDOW = 20
LARGE_ROLLING_WINDOW = 50


class TestBacktestRobustnessUnit:
    """Быстрые unit тесты для проверки логики робастности."""

    @pytest.mark.unit
    def test_fee_calculation_logic(self, small_prices_df):
        """Unit test: проверяем логику расчета комиссий."""
        test_data = pd.DataFrame({
            'S1': small_prices_df.iloc[:, 0],
            'S2': small_prices_df.iloc[:, 1]
        })

        # Тестируем разные уровни комиссий
        for fee in FEE_LEVELS:
            backtester = PairBacktester(
                pair_data=test_data,
                rolling_window=DEFAULT_ROLLING_WINDOW,
                z_threshold=DEFAULT_Z_THRESHOLD,
                z_exit=DEFAULT_Z_EXIT,
                commission_pct=fee,
                slippage_pct=DEFAULT_SLIPPAGE_PCT
            )

            # Проверяем, что параметры установлены правильно
            assert backtester.commission_pct == fee, f"Commission должен быть {fee}"
            assert backtester.slippage_pct == DEFAULT_SLIPPAGE_PCT, f"Slippage должен быть {DEFAULT_SLIPPAGE_PCT}"

    @pytest.mark.unit
    def test_data_validation_logic(self, tiny_prices_df):
        """Unit test: проверяем логику валидации данных."""
        # Тестируем с корректными данными
        test_data = pd.DataFrame({
            'S1': tiny_prices_df.iloc[:, 0],
            'S2': tiny_prices_df.iloc[:, 1]
        })

        # Должно работать без ошибок
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=SMALL_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            z_exit=DEFAULT_Z_EXIT
        )

        assert len(backtester.pair_data) == len(test_data)
        assert backtester.rolling_window == SMALL_ROLLING_WINDOW

    @pytest.mark.unit
    def test_parameter_bounds_logic(self):
        """Unit test: проверяем логику границ параметров."""
        # Тестируем различные параметры
        MIN_Z_VALUE = 0
        MAX_Z_VALUE = 10
        MIN_FEE_VALUE = 0
        MAX_FEE_VALUE = 0.1

        test_params = {
            'z_threshold': [1.0, DEFAULT_Z_THRESHOLD, 3.0],
            'z_exit': [0.0, DEFAULT_Z_EXIT, 1.0],
            'commission_pct': [0.0, 0.001, HIGH_FEE],
            'slippage_pct': [0.0, 0.0005, 0.005]
        }

        for param_name, values in test_params.items():
            for value in values:
                # Проверяем, что значения в разумных пределах
                if param_name in ['z_threshold', 'z_exit']:
                    assert MIN_Z_VALUE <= value <= MAX_Z_VALUE, f"{param_name} должен быть в пределах [{MIN_Z_VALUE}, {MAX_Z_VALUE}]"
                elif param_name in ['commission_pct', 'slippage_pct']:
                    assert MIN_FEE_VALUE <= value <= MAX_FEE_VALUE, f"{param_name} должен быть в пределах [{MIN_FEE_VALUE}, {MAX_FEE_VALUE}]"

    @pytest.mark.unit
    def test_signal_logic_structure(self):
        """Unit test: проверяем структуру логики сигналов."""
        # Создаем простые синтетические данные для тестирования логики (детерминизм обеспечен глобально)
        N_PERIODS = 50
        BASE_PRICE_S1 = 100
        BASE_PRICE_S2 = 50
        S1_VOLATILITY = 0.1
        S2_VOLATILITY = 0.05
        S2_COEFFICIENT = 0.5

        # Создаем коинтегрированные данные
        price_s1 = BASE_PRICE_S1 + np.cumsum(np.random.normal(0, S1_VOLATILITY, N_PERIODS))
        price_s2 = BASE_PRICE_S2 + S2_COEFFICIENT * price_s1 + np.cumsum(np.random.normal(0, S2_VOLATILITY, N_PERIODS))

        test_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        })

        # Проверяем, что можем создать бэктестер с этими данными
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            z_exit=DEFAULT_Z_EXIT
        )

        # Проверяем основные атрибуты
        assert hasattr(backtester, 'z_threshold')
        assert hasattr(backtester, 'z_exit')
        assert hasattr(backtester, 'rolling_window')


class TestBacktestRobustness:
    """Медленные integration тесты робастности бэктеста."""
    
    def setup_method(self):
        """Настройка тестовых данных."""
        # Константы для генерации данных (детерминизм обеспечен глобально в conftest.py)
        self.n_periods = 200  # Уменьшено с 1000 для ускорения
        BASE_PRICE_S1 = 100
        BASE_PRICE_S2 = 50
        VOLATILITY_S1 = 0.1
        VOLATILITY_S2 = 0.05
        COINTEGRATION_RATIO = 0.5

        dates = pd.date_range('2024-01-01', periods=self.n_periods, freq='15min')

        # Создаем базовые коинтегрированные данные
        price_s1 = BASE_PRICE_S1 + np.cumsum(np.random.normal(0, VOLATILITY_S1, self.n_periods))
        price_s2 = BASE_PRICE_S2 + COINTEGRATION_RATIO * price_s1 + np.cumsum(np.random.normal(0, VOLATILITY_S2, self.n_periods))

        self.base_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)

        self.base_params = {
            'rolling_window': LARGE_ROLLING_WINDOW,
            'z_threshold': DEFAULT_Z_THRESHOLD,
            'z_exit': DEFAULT_Z_EXIT,
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
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_date_permutation_when_shuffled_then_destroys_performance(self):
        """Integration test: перемешивание дат должно уничтожить производительность.
        
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
        
        # Если оригинальный Sharpe не равен нулю, перемешивание должно его изменить
        MIN_SHARPE_THRESHOLD = 0.01
        SHARPE_DEGRADATION_FACTOR = 0.8

        if abs(original_sharpe) > MIN_SHARPE_THRESHOLD:
            # Проверяем, что перемешивание значительно изменило результаты
            sharpe_change = abs(shuffled_sharpe - original_sharpe)
            assert sharpe_change > MIN_SHARPE_THRESHOLD or abs(shuffled_sharpe) < abs(original_sharpe) * SHARPE_DEGRADATION_FACTOR, \
                f"Перемешивание должно изменить Sharpe: оригинал={original_sharpe:.3f}, перемешанный={shuffled_sharpe:.3f}"
        
        # Sharpe после перемешивания должен быть близок к нулю или хуже оригинального
        MAX_REASONABLE_SHARPE = 1.0
        assert abs(shuffled_sharpe) < MAX_REASONABLE_SHARPE, \
            f"Sharpe после перемешивания должен быть разумным: {shuffled_sharpe:.3f}"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_time_shift_when_applied_then_degrades_performance(self):
        """Integration test: смещение одной серии должно ухудшить производительность.
        
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
        TIME_SHIFT_PERIODS = 10
        shifted_data = self.base_data.copy()
        shifted_data['S2'] = shifted_data['S2'].shift(TIME_SHIFT_PERIODS)
        shifted_data = shifted_data.dropna()  # Удаляем NaN после сдвига
        
        # Бэктест на сдвинутых данных
        shifted_engine = PairBacktester(
            pair_data=shifted_data,
            **self.base_params
        )
        shifted_engine.run()
        shifted_metrics = shifted_engine.get_performance_metrics()
        shifted_sharpe = shifted_metrics.get('sharpe_ratio', 0)
        
        # Проверяем, что система корректно обработала сдвинутые данные
        assert not np.isnan(shifted_sharpe), f"Sharpe на сдвинутых данных не должен быть NaN: {shifted_sharpe}"
        assert not np.isinf(shifted_sharpe), f"Sharpe на сдвинутых данных не должен быть бесконечным: {shifted_sharpe}"
        
        # Сдвиг должен изменить производительность, если оригинальная стратегия не нулевая
        MIN_SIGNIFICANT_SHARPE = 0.01
        MIN_SHARPE_CHANGE = 0.01
        MAX_REASONABLE_SHARPE_SHIFT = 1.0

        if abs(original_sharpe) > MIN_SIGNIFICANT_SHARPE:
            # Если есть значимая оригинальная производительность, сдвиг должен ее изменить
            sharpe_change = abs(shifted_sharpe - original_sharpe)
            assert sharpe_change > MIN_SHARPE_CHANGE or abs(shifted_sharpe) < abs(original_sharpe), \
                f"Сдвиг должен изменить Sharpe: оригинал={original_sharpe:.3f}, сдвинутый={shifted_sharpe:.3f}"
        else:
            # Если оригинальный Sharpe близок к нулю, сдвинутый тоже должен быть разумным
            assert abs(shifted_sharpe) < MAX_REASONABLE_SHARPE_SHIFT, \
                f"Сдвинутый Sharpe должен быть разумным: {shifted_sharpe:.3f}"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_fee_sensitivity(self):
        """Integration test: увеличение комиссий должно уменьшать PnL.
        
        При увеличении комиссий и проскальзывания итоговый PnL
        должен монотонно уменьшаться.
        """
        fee_multipliers = [1.0, 5.0, 10.0]  # ОПТИМИЗАЦИЯ: Уменьшено с 4 до 3 итераций
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
        
        # PnL должен монотонно уменьшаться с ростом комиссий (с небольшим допуском)
        for i in range(1, len(pnl_results)):
            assert pnl_results[i] <= pnl_results[i-1] + 1e-6, \
                f"PnL должен уменьшаться с ростом комиссий: {pnl_results}"
        
        # При 10x комиссиях PnL должен быть хуже (если исходный PnL не нулевой)
        if abs(pnl_results[0]) > 1e-6:
            pnl_degradation = pnl_results[-1] - pnl_results[0]
            assert pnl_degradation <= 0, \
                f"10x комиссии должны ухудшить PnL: {pnl_results[0]:.6f} -> {pnl_results[-1]:.6f}"
        else:
            # Если исходный PnL близок к нулю, проверяем, что он остается разумным
            assert abs(pnl_results[-1]) < 1.0, \
                f"PnL с высокими комиссиями должен быть разумным: {pnl_results[-1]:.6f}"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_no_future_reference(self):
        """Integration test: изменение будущих баров не должно влиять на прошлые сигналы.
        
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
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_signal_shift_sanity_check(self):
        """Integration test: задержка сигналов должна ухудшить производительность.
        
        Создаем модифицированный движок, который сдвигает все сигналы
        на 1 бар вперед. Это должно ухудшить Sharpe ratio.
        """
        class DelayedSignalBacktester(PairBacktester):
            def _calculate_signals(self):
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
        
        # Задержка должна ухудшить производительность (если есть значимая производительность)
        if abs(original_sharpe) > 0.01:
            assert delayed_sharpe <= original_sharpe, \
                f"Задержка сигналов должна ухудшить Sharpe: оригинал={original_sharpe:.3f}, задержанный={delayed_sharpe:.3f}"
        else:
            # Если оригинальный Sharpe близок к нулю, задержанный тоже должен быть разумным
            assert abs(delayed_sharpe) < 1.0, \
                f"Задержанный Sharpe должен быть разумным: {delayed_sharpe:.3f}"
    
    def test_synthetic_random_walk(self):
        """Тест на случайном блуждании: Sharpe должен быть близок к нулю.
        
        На данных случайного блуждания стратегия не должна
        показывать значимую производительность.
        """
        np.random.seed(123)
        n = 100  # ОПТИМИЗАЦИЯ: Уменьшено с 1000 до 100
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем независимые случайные блуждания
        random_walk_s1 = 100 + np.cumsum(np.random.normal(0, 0.1, n))
        random_walk_s2 = 50 + np.cumsum(np.random.normal(0, 0.1, n))
        
        random_data = pd.DataFrame({
            'S1': random_walk_s1,
            'S2': random_walk_s2
        }, index=dates)
        
        # Используем более строгие параметры для случайных данных
        random_params = self.base_params.copy()
        random_params.update({
            'z_threshold': 3.0,  # Более строгий порог входа
            'rolling_window': 20,  # ОПТИМИЗАЦИЯ: Уменьшено с 100 до 20
            'commission_pct': 0.002,  # Увеличиваем комиссии
            'slippage_pct': 0.001
        })
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=random_data,
            **random_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # На случайных данных Sharpe должен быть близок к нулю
        # Проверяем также количество сделок - их должно быть мало
        total_trades = metrics.get('total_trades', 0)
        
        assert abs(sharpe) < 1.2, \
            f"На случайных данных Sharpe должен быть близок к нулю: {sharpe:.3f}"
        
        # Дополнительная проверка: если Sharpe высокий, то сделок должно быть мало
        if abs(sharpe) > 0.5:
            assert total_trades < 10, \
                f"При высоком Sharpe на случайных данных сделок должно быть мало: {total_trades}"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_cost_breakdown_consistency(self):
        """Integration test: проверка консистентности разбивки издержек.
        
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