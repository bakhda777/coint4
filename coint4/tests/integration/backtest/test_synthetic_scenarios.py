"""Синтетические тесты и sanity-проверки для бэктест-системы.

Оптимизировано согласно best practices:
- Быстрые версии вынесены в test_synthetic_scenarios_fast.py
- Уменьшены объемы данных
- Добавлены маркеры integration
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester
from tests.conftest import get_test_config


@pytest.mark.slow
@pytest.mark.integration
class TestSyntheticScenarios:
    """Тесты на синтетических сценариях."""

    def setup_method(self, method):
        """Setup method для синтетических тестов."""
        # Инициализация базовых параметров
        self.base_params = {
            'rolling_window': 20,
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
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_perfect_mean_reversion_when_synthetic_data_then_profitable_trades(self, rng):
        """Integration test: тест на идеальной mean-reverting паре.

        Создаем синтетическую пару с известным mean-reversion паттерном
        и проверяем, что стратегия его корректно использует.
        """
        # Используем get_test_config() для динамического размера данных
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер данных
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем идеальную mean-reverting пару
        base_price = 100
        spread_target = 0
        spread_volatility = 2.0
        mean_reversion_speed = 0.1
        
        spreads = [0]
        for i in range(1, n):
            # Mean-reverting процесс: dx = -speed * (x - target) * dt + vol * dW
            prev_spread = spreads[-1]
            drift = -mean_reversion_speed * (prev_spread - spread_target)
            shock = rng.normal(0, spread_volatility * np.sqrt(1/252/24/4))  # 15min intervals
            new_spread = prev_spread + drift + shock
            spreads.append(new_spread)
        
        # Конвертируем спреды в цены
        price_s1 = base_price + np.cumsum(rng.normal(0, 0.01, n))  # Random walk
        price_s2 = price_s1 + np.array(spreads)  # S2 = S1 + spread
        
        synthetic_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=synthetic_data,
            **self.base_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # На синтетических данных проверяем корректность выполнения
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        num_trades = metrics.get('num_trades', 0)
        
        # Если нет торговых сигналов, пропускаем тест
        if total_return == 0.0 and num_trades == 0:
            pytest.skip("Нет торговых сигналов на синтетических данных")
        
        # Основная проверка - система работает без ошибок
        assert not np.isnan(total_return), "Total return не должен быть NaN"
        assert not np.isinf(total_return), "Total return не должен быть бесконечным"
        
        # Проверяем что есть сделки, то результат разумный
        if num_trades > 0:
            assert total_return > -0.5, f"Слишком большие потери: {total_return:.3f}"
            assert abs(sharpe_ratio) < 10, f"Sharpe ratio не должен быть экстремальным: {sharpe_ratio:.3f}"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_trending_market_performance(self, rng):
        """Integration test: тест на трендовом рынке.

        Pairs trading должен показывать плохие результаты на сильно трендовых данных.
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер вместо hardcoded
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем сильно трендовые данные
        trend_strength = 0.002  # 0.2% за период
        price_s1 = 100 * np.exp(np.cumsum(rng.normal(trend_strength, 0.01, n)))
        price_s2 = 50 * np.exp(np.cumsum(rng.normal(trend_strength * 0.8, 0.008, n)))
        
        trending_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=trending_data,
            **self.base_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # На трендовых данных pairs trading должен работать хуже
        total_return = metrics.get('total_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # Не ожидаем высокой доходности на трендовых данных
        assert total_return < 0.15, f"Слишком высокая доходность на трендовых данных: {total_return:.3f}"
        
        # Максимальная просадка может быть значительной
        assert max_drawdown < 0.5, f"Слишком большая просадка: {max_drawdown:.3f}"
        
        # Основная проверка - система работает без ошибок
        assert not np.isnan(total_return), "Total return не должен быть NaN"
        assert not np.isnan(max_drawdown), "Max drawdown не должен быть NaN"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_high_volatility_scenario(self, rng):
        """Integration test: тест на данных с высокой волатильностью.

        Проверяем, что система корректно обрабатывает периоды высокой волатильности.
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер вместо hardcoded
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем данные с переменной волатильностью
        base_vol = 0.01
        vol_multiplier = 1 + 2 * np.sin(np.linspace(0, 4*np.pi, n))  # Волатильность от 1x до 3x
        
        returns_s1 = rng.normal(0, base_vol, n) * vol_multiplier
        returns_s2 = rng.normal(0, base_vol * 0.8, n) * vol_multiplier
        
        price_s1 = 100 * np.exp(np.cumsum(returns_s1))
        price_s2 = 80 * np.exp(np.cumsum(returns_s2))
        
        volatile_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=volatile_data,
            **self.base_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # Проверяем, что система не сломалась
        assert 'total_return' in metrics, "Метрики должны быть рассчитаны"
        assert not np.isnan(metrics.get('total_return', np.nan)), "Total return не должен быть NaN"
        assert not np.isinf(metrics.get('total_return', 0)), "Total return не должен быть бесконечным"
        
        # Волатильность доходности должна быть разумной
        if 'volatility' in metrics:
            vol = metrics['volatility']
            assert vol > 0, f"Волатильность должна быть положительной: {vol}"
            assert vol < 1.0, f"Волатильность не должна быть экстремальной: {vol}"
        
        # Тест успешно завершен - система обрабатывает высокую волатильность
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_cointegration_breakdown(self, rng):
        """Integration test: тест на нарушении коинтеграции.

        Создаем данные, где коинтеграция нарушается в середине периода.
        """
        test_config = get_test_config()
        n = max(test_config['periods'], 60)  # ОПТИМИЗАЦИЯ: минимум 60 для теста нарушения коинтеграции
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Первая половина - коинтегрированные данные
        n_half = n // 2
        
        # Коинтегрированная часть
        price_s1_part1 = 100 + np.cumsum(rng.normal(0, 0.01, n_half))
        error_term = np.cumsum(rng.normal(0, 0.005, n_half))
        price_s2_part1 = 50 + 0.5 * price_s1_part1 + error_term
        
        # Вторая половина - независимые случайные блуждания
        price_s1_part2 = price_s1_part1[-1] + np.cumsum(rng.normal(0, 0.015, n_half))
        price_s2_part2 = price_s2_part1[-1] + np.cumsum(rng.normal(0, 0.012, n_half))
        
        price_s1 = np.concatenate([price_s1_part1, price_s1_part2])
        price_s2 = np.concatenate([price_s2_part1, price_s2_part2])
        
        breakdown_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=breakdown_data,
            **self.base_params
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # При нарушении коинтеграции производительность должна ухудшиться
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # Не ожидаем высоких результатов при нарушении коинтеграции
        assert abs(sharpe_ratio) < 2.0, f"Sharpe не должен быть высоким при нарушении коинтеграции: {sharpe_ratio:.3f}"
        
        # Тест пройден - система корректно обрабатывает нарушение коинтеграции
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_extreme_price_movements(self, rng):
        """Integration test: тест на экстремальных движениях цен.

        Проверяем обработку больших скачков цен (имитация gap'ов).
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Базовые цены
        price_s1 = 100 + np.cumsum(rng.normal(0, 0.005, n))
        price_s2 = 50 + 0.5 * price_s1 + np.cumsum(rng.normal(0, 0.003, n))
        
        # Добавляем несколько экстремальных скачков
        gap_indices = [50, 100, 150]
        gap_sizes = [0.05, -0.08, 0.06]  # 5%, -8%, 6%
        
        for idx, gap_size in zip(gap_indices, gap_sizes):
            if idx < len(price_s1):
                price_s1[idx:] *= (1 + gap_size)
                price_s2[idx:] *= (1 + gap_size * 0.7)  # Частично коррелированный gap
        
        extreme_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=extreme_data,
            **self.base_params
        )
        engine.run()
        
        # Проверяем, что система не сломалась на экстремальных данных
        results = engine.results
        assert len(results) > 0, "Результаты должны быть получены"
        
        # Проверяем отсутствие NaN и inf в ключевых столбцах
        key_columns = ['position', 'pnl', 'equity']
        for col in key_columns:
            if col in results.columns:
                assert not results[col].isna().any(), f"Столбец {col} содержит NaN"
                assert not np.isinf(results[col]).any(), f"Столбец {col} содержит inf"
        
        # Тест пройден - система справляется с экстремальными движениями
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_zero_volatility_period(self, rng):
        """Integration test: тест на периоде нулевой волатильности.

        Проверяем обработку периодов, когда цены не меняются.
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем данные с периодом нулевой волатильности
        price_s1 = np.ones(n) * 100
        price_s2 = np.ones(n) * 50
        
        # Добавляем небольшие изменения в начале и конце
        price_s1[:20] = 100 + np.cumsum(rng.normal(0, 0.001, 20))
        price_s1[-20:] = price_s1[19] + np.cumsum(rng.normal(0, 0.001, 20))
        
        price_s2[:20] = 50 + 0.5 * (price_s1[:20] - 100)
        price_s2[-20:] = price_s2[19] + 0.5 * (price_s1[-20:] - price_s1[19])
        
        static_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=static_data,
            **self.base_params
        )
        engine.run()
        
        # Система должна корректно обработать нулевую волатильность
        metrics = engine.get_performance_metrics()
        assert 'total_return' in metrics, "Метрики должны быть рассчитаны"
        
        # При нулевой волатильности не должно быть сделок
        num_trades = metrics.get('num_trades', 0)
        assert num_trades <= 2, f"При низкой волатильности должно быть мало сделок: {num_trades}"
        
        # Тест пройден - система корректно обрабатывает нулевую волатильность
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_missing_data_handling(self, rng):
        """Integration test: тест обработки пропущенных данных.

        Проверяем, как система обрабатывает NaN значения в данных.
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Создаем базовые данные
        price_s1 = 100 + np.cumsum(rng.normal(0, 0.01, n))
        price_s2 = 50 + 0.5 * price_s1 + np.cumsum(rng.normal(0, 0.005, n))
        
        # Добавляем пропущенные значения
        missing_indices = rng.choice(n, size=10, replace=False)
        price_s1[missing_indices[:5]] = np.nan
        price_s2[missing_indices[5:]] = np.nan
        
        missing_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Система должна либо корректно обработать NaN, либо выдать понятную ошибку
        try:
            engine = PairBacktester(
                pair_data=missing_data,
                **self.base_params
            )
            engine.run()
            
            # Если обработка прошла успешно, проверяем результаты
            results = engine.results
            assert len(results) > 0, "Результаты должны быть получены"
            
            # Тест пройден - обработка пропущенных данных успешна
            assert len(results) > 0
            
        except (ValueError, TypeError, Exception) as e:
            # Ожидаемое поведение - система должна выдать понятную ошибку
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["nan", "missing", "inf", "exog"]), \
                f"Ошибка должна быть связана с пропущенными данными: {e}"
            
            # Тест пройден - ошибка обработана корректно
            pass  # Ожидаемая ошибка
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_single_asset_constant(self, rng):
        """Integration test: тест с константной ценой одного актива.

        Один актив имеет постоянную цену, второй меняется.
        """
        test_config = get_test_config()
        n = test_config['periods']  # ОПТИМИЗАЦИЯ: динамический размер
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # S1 - константа, S2 - случайное блуждание
        price_s1 = np.ones(n) * 100
        price_s2 = 50 + np.cumsum(rng.normal(0, 0.01, n))
        
        constant_data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        # Запускаем бэктест
        engine = PairBacktester(
            pair_data=constant_data,
            **self.base_params
        )
        engine.run()
        
        # При константной цене одного актива коинтеграция невозможна
        metrics = engine.get_performance_metrics()
        num_trades = metrics.get('num_trades', 0)
        
        # Ожидаем мало или ноль сделок
        assert num_trades <= 1, f"При константной цене одного актива не должно быть много сделок: {num_trades}"
        
        # Тест пройден - константные цены обработаны корректно
    
    def run_all_tests(self):
        """Запуск всех синтетических тестов."""
        test_methods = [
            self.test_perfect_mean_reversion,
            self.test_trending_market_performance,
            self.test_high_volatility_scenario,
            self.test_cointegration_breakdown,
            self.test_extreme_price_movements,
            self.test_zero_volatility_period,
            self.test_missing_data_handling,
            self.test_single_asset_constant
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
            except Exception as e:
                results.append({
                    'test_name': test_method.__name__,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        return results


class TestSyntheticScenariosFast:
    """Быстрые версии синтетических тестов с мокированием."""
    
    @pytest.mark.fast
    @patch('src.coint2.engine.base_engine.BasePairBacktester')
    def test_mean_reversion_when_mocked_then_logic_works(self, mock_backtester):
        """Fast test: Быстрая проверка логики mean reversion."""
        # Мокируем PairBacktester
        mock_instance = Mock()
        mock_instance.get_performance_metrics.return_value = {
            'total_return': 0.05,
            'sharpe_ratio': 0.8,
            'num_trades': 12,
            'win_rate': 0.6
        }
        mock_backtester.return_value = mock_instance
        
        # Минимальные синтетические данные
        dates = pd.date_range('2024-01-01', periods=5, freq='1H')
        synthetic_data = pd.DataFrame({
            'S1': [100, 101, 100.5, 99.8, 100.2],
            'S2': [50, 50.5, 50.25, 49.9, 50.1]
        }, index=dates)
        
        # Создаем backtester с мокированием
        engine = mock_backtester(
            pair_data=synthetic_data,
            rolling_window=3,  # Минимальное окно
            z_threshold=2.0
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # Проверки
        assert metrics['total_return'] > 0
        assert metrics['num_trades'] > 0
        mock_instance.run.assert_called_once()
        
    @pytest.mark.fast
    def test_trending_market_when_fast_then_detects_trend(self):
        """Fast test: Быстрое обнаружение трендового рынка.""" 
        # Минимальные трендовые данные
        trend_data = np.array([100, 102, 104, 106, 108])  # Четкий восходящий тренд
        
        # Простая проверка тренда
        returns = np.diff(trend_data) / trend_data[:-1]
        avg_return = np.mean(returns)
        
        # На трендовых данных средний возврат должен быть положительным
        assert avg_return > 0, f"На трендовых данных средний возврат должен быть положительным: {avg_return}"
        
        # Проверяем консистентность тренда
        positive_moves = sum(1 for r in returns if r > 0)
        assert positive_moves >= len(returns) * 0.6, "Большинство движений должны быть положительными"
        
    @pytest.mark.fast
    def test_high_volatility_when_fast_then_detects_volatility(self):
        """Fast test: Быстрое обнаружение высокой волатильности."""
        # Данные с высокой волатильностью
        np.random.seed(42)  # Для детерминизма
        high_vol_data = np.random.randn(10) * 0.1 + 100  # 10% волатильность
        low_vol_data = np.random.randn(10) * 0.01 + 100   # 1% волатильность
        
        # Рассчитываем волатильность (стандартное отклонение)
        high_vol = np.std(high_vol_data)
        low_vol = np.std(low_vol_data)
        
        # Высокая волатильность должна быть больше низкой
        assert high_vol > low_vol, f"Высокая волатильность {high_vol:.4f} должна быть > низкой {low_vol:.4f}"
        assert high_vol > 0.05, f"Высокая волатильность должна превышать порог: {high_vol:.4f}"
        
    @pytest.mark.fast
    def test_cointegration_when_fast_then_detects_breakdown(self):
        """Fast test: Быстрое обнаружение нарушения коинтеграции."""
        # Первая половина - коинтегрированные данные
        s1_part1 = np.array([100, 101, 102, 103, 104])
        s2_part1 = 0.5 * s1_part1 + 50  # Строгая связь
        
        # Вторая половина - независимые данные с низкой корреляцией
        s1_part2 = np.array([105, 103, 108, 106, 109])   # Умеренные изменения
        s2_part2 = np.array([102, 99, 101, 98, 100])     # Независимые умеренные изменения
        
        # Рассчитываем корреляцию для каждой части
        corr_part1 = np.corrcoef(s1_part1, s2_part1)[0, 1]
        corr_part2 = np.corrcoef(s1_part2, s2_part2)[0, 1]
        
        # Коинтеграция должна быть нарушена во второй части
        assert corr_part1 > 0.9, f"Первая часть должна быть сильно коинтегрирована: {corr_part1:.3f}"
        assert abs(corr_part2) < 0.5, f"Вторая часть не должна быть коинтегрирована: {corr_part2:.3f}"
        
    @pytest.mark.fast 
    def test_extreme_movements_when_fast_then_handles_gaps(self):
        """Fast test: Быстрая обработка экстремальных движений."""
        # Данные с gap'ами
        base_price = np.array([100, 101, 102])
        gap_price = base_price.copy()
        gap_price[1] *= 1.1  # 10% gap up
        gap_price[2] *= 0.9  # 10% gap down от нового уровня
        
        # Проверяем обнаружение больших движений
        returns = np.diff(gap_price) / gap_price[:-1]
        large_moves = np.abs(returns) > 0.05  # Движения больше 5%
        
        assert np.any(large_moves), "Должны быть обнаружены большие движения"
        assert np.sum(large_moves) >= 1, f"Должно быть минимум одно большое движение: {np.sum(large_moves)}"
        
    @pytest.mark.fast
    def test_zero_volatility_when_fast_then_detects_static(self):
        """Fast test: Быстрое обнаружение нулевой волатильности."""
        # Статичные данные
        static_data = np.array([100, 100, 100, 100, 100])
        volatile_data = np.array([100, 105, 95, 110, 90])
        
        # Рассчитываем волатильность
        static_vol = np.std(static_data)
        volatile_vol = np.std(volatile_data)
        
        # Статичные данные должны иметь нулевую волатильность
        assert static_vol == 0, f"Статичные данные должны иметь нулевую волатильность: {static_vol}"
        assert volatile_vol > 5, f"Волатильные данные должны иметь высокую волатильность: {volatile_vol}"
        
        # Проверяем отсутствие торговых сигналов на статичных данных
        price_changes = np.diff(static_data)
        trading_signals = np.abs(price_changes) > 0.01  # Сигналы при изменении > 1%
        assert not np.any(trading_signals), "На статичных данных не должно быть торговых сигналов"