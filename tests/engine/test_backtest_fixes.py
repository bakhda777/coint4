"""Тесты для проверки исправленных ошибок в бэктесте."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from coint2.engine.backtest_engine import PairBacktester


class TestBacktestFixes:
    """Тесты для проверки исправлений критических и логических ошибок."""

    def test_constant_data_handling(self):
        """Тест 1: Проверяет обработку константных данных без IndexError.
        
        Проверяет, что бэктестер корректно обрабатывает случаи с константными
        данными, где OLS регрессия может вернуть менее 2 параметров.
        """
        # Создаем данные с константными значениями
        data = pd.DataFrame({
            'asset1': [100.0] * 20,  # Константные данные
            'asset2': [50.0] * 20    # Константные данные
        })
        
        bt = PairBacktester(
            data,
            rolling_window=5,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        # Должно выполниться без ошибок
        bt.run()
        results = bt.get_results()
        
        # Проверяем, что результаты получены
        assert 'spread' in results
        assert 'z_score' in results
        assert len(results['spread']) == len(data)
        
        # При константных данных все z_score должны быть NaN или 0
        z_scores = pd.Series(results['z_score']).dropna()
        if not z_scores.empty:
            assert all(abs(z) < 1e-6 for z in z_scores), "Z-scores должны быть близки к нулю для константных данных"

    def test_zero_std_protection(self):
        """Тест 2: Проверяет защиту от деления на ноль в стандартном отклонении.
        
        Проверяет, что функция _calculate_ols_with_cache корректно обрабатывает
        случаи с нулевым стандартным отклонением.
        """
        # Создаем данные с очень малой вариацией
        np.random.seed(42)
        data = pd.DataFrame({
            'asset1': [100.0 + 1e-10 * i for i in range(20)],  # Почти константные
            'asset2': [50.0 + 1e-10 * i for i in range(20)]    # Почти константные
        })
        
        bt = PairBacktester(
            data,
            rolling_window=5,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        # Тестируем напрямую функцию кеширования
        y_win = data['asset1'].iloc[:5]
        x_win = data['asset2'].iloc[:5]
        
        beta, mean, std = bt._calculate_ols_with_cache(y_win, x_win)
        
        # Стандартное отклонение должно быть защищено от нуля
        assert std >= 1e-8, f"Стандартное отклонение ({std}) должно быть >= 1e-8"
        assert np.isfinite(beta), "Beta должна быть конечным числом"
        assert np.isfinite(mean), "Mean должно быть конечным числом"

    def test_position_size_zero_division_protection(self):
        """Тест 3: Проверяет защиту от деления на ноль в расчете размера позиции.
        
        Проверяет, что функция _calculate_position_size корректно обрабатывает
        случаи с нулевым риском или нулевой стоимостью сделки.
        """
        data = pd.DataFrame({
            'asset1': np.linspace(100, 120, 20),
            'asset2': np.linspace(50, 60, 20)
        })
        
        bt = PairBacktester(
            data,
            rolling_window=5,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        # Тестируем случай с нулевым риском (spread_curr == stop_loss_price)
        entry_z = 2.0
        spread_curr = 10.0
        mean = 8.0
        std = 1.0
        beta = 1.5
        price_s1 = 100.0
        price_s2 = 50.0
        
        # Случай 1: Нормальный расчет
        size = bt._calculate_position_size(entry_z, spread_curr, mean, std, beta, price_s1, price_s2)
        assert size >= 0, "Размер позиции должен быть неотрицательным"
        assert np.isfinite(size), "Размер позиции должен быть конечным"
        
        # Случай 2: Очень маленький риск (spread_curr близко к stop_loss_price)
        stop_loss_z = np.sign(entry_z) * bt.stop_loss_multiplier
        stop_loss_price = mean + stop_loss_z * std
        size_small_risk = bt._calculate_position_size(entry_z, stop_loss_price, mean, std, beta, price_s1, price_s2)
        
        # С новой защитой от микроскопического риска, позиция не должна быть 0,
        # но должна быть ограничена min_risk_per_unit = max(0.1 * std, EPSILON)
        min_risk_expected = max(0.1 * std, 1e-8)  # = max(0.1, 1e-8) = 0.1
        trade_value = price_s1 + abs(beta) * price_s2  # = 100 + 1.5 * 50 = 175
        expected_size = min(bt.capital_at_risk / min_risk_expected, bt.capital_at_risk / trade_value)
        expected_size = min(expected_size, 1000 / 175)  # учитываем trade_value ограничение
        
        assert abs(size_small_risk - expected_size) < 1e-6, \
            f"При очень маленьком риске размер позиции должен быть ограничен: {size_small_risk} vs {expected_size}"

    def test_take_profit_logic_correction(self):
        """Тест 4: Проверяет исправленную логику take-profit.
        
        Проверяет, что take-profit срабатывает при движении z-score к нулю,
        а не от него, и что валидация параметров работает корректно.
        """
        data = pd.DataFrame({
            'asset1': [100, 102, 104, 103, 101, 100, 99, 98, 100, 102],
            'asset2': [50, 51, 52, 51.5, 50.5, 50, 49.5, 49, 50, 51]
        })
        
        # Тест валидации: take_profit_multiplier >= 1.0 теперь разрешен с предупреждением
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bt_warning = PairBacktester(
                data,
                rolling_window=3,
                z_threshold=1.0,
                take_profit_multiplier=1.5,  # Теперь разрешено с предупреждением
                capital_at_risk=1000.0,
                stop_loss_multiplier=2.0
            )
            # Проверяем, что выдается предупреждение
            assert len(w) > 0, "Должно быть предупреждение для take_profit_multiplier >= 1.0"
            assert "take_profit_multiplier" in str(w[0].message)
            
        # Тест валидации: отрицательный take_profit_multiplier должен вызывать ошибку
        with pytest.raises(ValueError, match="take_profit_multiplier.*must be positive"):
            PairBacktester(
                data,
                rolling_window=3,
                z_threshold=1.0,
                take_profit_multiplier=-0.5,  # Отрицательное значение
                capital_at_risk=1000.0,
                stop_loss_multiplier=2.0
            )
        
        # Тест корректной работы take-profit
        bt = PairBacktester(
            data,
            rolling_window=3,
            z_threshold=1.0,
            take_profit_multiplier=0.5,  # Правильное значение
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        bt.run()
        results = bt.get_results()
        
        # Проверяем, что есть записи о выходах по take-profit
        trades_log = results['trades_log']
        if trades_log:
            exit_reasons = [trade.get('exit_reason', '') for trade in trades_log]
            # Если были сделки, проверяем что take-profit может сработать
            assert all(reason in ['take_profit', 'z_exit', 'end_of_test', 'stop_loss', 'time_stop'] 
                      for reason in exit_reasons), f"Неожиданные причины выхода: {exit_reasons}"

    def test_time_calculation_consistency_15min(self):
        """Тест 5: Проверяет консистентность расчета времени для 15-минутных данных.
        
        Проверяет, что функция _calculate_trade_duration корректно обрабатывает
        различные типы индексов и всегда возвращает время в часах.
        """
        # Создаем данные с datetime индексом (15-минутные интервалы)
        start_time = pd.Timestamp('2023-01-01 09:00:00')
        datetime_index = pd.date_range(start_time, periods=20, freq='15T')
        data_datetime = pd.DataFrame({
            'asset1': np.linspace(100, 120, 20),
            'asset2': np.linspace(50, 60, 20)
        }, index=datetime_index)
        
        bt_datetime = PairBacktester(
            data_datetime,
            rolling_window=5,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        # Тест с datetime индексом
        current_time = datetime_index[10]
        entry_time = datetime_index[5]
        duration = bt_datetime._calculate_trade_duration(current_time, entry_time, 10, 5)
        expected_duration = 5 * 0.25  # 5 периодов по 15 минут = 1.25 часа
        assert abs(duration - expected_duration) < 1e-6, f"Ожидалось {expected_duration} часов, получено {duration}"
        
        # Тест с числовым индексом
        data_numeric = pd.DataFrame({
            'asset1': np.linspace(100, 120, 20),
            'asset2': np.linspace(50, 60, 20)
        })
        
        bt_numeric = PairBacktester(
            data_numeric,
            rolling_window=5,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        duration_numeric = bt_numeric._calculate_trade_duration(10, 5, 10, 5)
        expected_duration_numeric = 5 * 0.25  # 5 периодов по 15 минут = 1.25 часа
        assert abs(duration_numeric - expected_duration_numeric) < 1e-6, \
            f"Ожидалось {expected_duration_numeric} часов, получено {duration_numeric}"

    def test_cache_management_and_security(self):
        """Тест 6: Проверяет улучшенное управление кешем и безопасность хеширования.
        
        Проверяет, что кеш работает корректно, имеет фиксированный размер
        и использует безопасное хеширование.
        """
        data = pd.DataFrame({
            'asset1': np.random.normal(100, 10, 100),
            'asset2': np.random.normal(50, 5, 100)
        })
        
        bt = PairBacktester(
            data,
            rolling_window=10,
            z_threshold=1.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=2.0
        )
        
        # Проверяем, что кеш изначально пуст
        assert len(bt._ols_cache) == 0, "Кеш должен быть изначально пуст"
        
        # Проверяем фиксированный размер кеша
        assert bt._ols_cache_max_size == 1000, "Максимальный размер кеша должен быть 1000"
        
        # Выполняем несколько расчетов
        y_win1 = data['asset1'].iloc[0:10]
        x_win1 = data['asset2'].iloc[0:10]
        result1 = bt._calculate_ols_with_cache(y_win1, x_win1)
        
        assert len(bt._ols_cache) == 1, "После первого расчета в кеше должна быть 1 запись"
        
        # Повторный расчет с теми же данными должен использовать кеш
        result2 = bt._calculate_ols_with_cache(y_win1, x_win1)
        assert result1 == result2, "Результаты из кеша должны совпадать"
        assert len(bt._ols_cache) == 1, "Размер кеша не должен измениться при повторном запросе"
        
        # Проверяем информацию о кеше
        cache_info = bt.get_ols_cache_info()
        assert cache_info["max_size"] == 1000, "Максимальный размер должен быть 1000"
        assert cache_info["current_size"] == 1, "Текущий размер должен быть 1"
        
        # Заполняем кеш несколькими записями
        for i in range(10):
            y_win = data['asset1'].iloc[i:i+10]
            x_win = data['asset2'].iloc[i:i+10]
            bt._calculate_ols_with_cache(y_win, x_win)
        
        # Кеш не должен превышать максимальный размер
        assert len(bt._ols_cache) <= bt._ols_cache_max_size, \
            f"Размер кеша ({len(bt._ols_cache)}) не должен превышать {bt._ols_cache_max_size}"
        
        # Проверяем функцию очистки кеша
        bt.clear_ols_cache()
        assert len(bt._ols_cache) == 0, "Кеш должен быть пуст после очистки"

    def test_parameter_validation_enhancements(self):
        """Тест 7: Проверяет улучшенную валидацию параметров.
        
        Проверяет новые проверки валидации, включая минимальное количество
        наблюдений и разумность временных параметров.
        """
        data = pd.DataFrame({
            'asset1': np.linspace(100, 120, 20),
            'asset2': np.linspace(50, 60, 20)
        })
        
        # Тест предупреждения о малом rolling_window
        with pytest.warns(UserWarning, match="rolling_window.*is very small"):
            bt = PairBacktester(
                data,
                rolling_window=5,  # Малое окно
                z_threshold=1.0,
                capital_at_risk=1000.0,
                stop_loss_multiplier=2.0
            )
        
        # Тест валидации временных параметров
        with pytest.raises(ValueError, match="time_stop_limit.*is too small"):
            PairBacktester(
                data,
                rolling_window=10,
                z_threshold=1.0,
                half_life=0.1,  # Очень малое значение
                time_stop_multiplier=0.5,  # В сумме < 1.0
                capital_at_risk=1000.0,
                stop_loss_multiplier=2.0
            )
        
        # Тест валидации z_exit >= z_threshold
        with pytest.raises(ValueError, match="z_exit.*must be less than z_threshold"):
            PairBacktester(
                data,
                rolling_window=10,
                z_threshold=1.0,
                z_exit=1.5,  # Больше z_threshold
                capital_at_risk=1000.0,
                stop_loss_multiplier=2.0
            )

    def test_sharpe_ratio_calculation_accuracy(self):
        """Тест 8: Проверяет точность расчета Sharpe Ratio.
        
        Проверяет, что конвертация PnL в доходности выполняется корректно
        и Sharpe Ratio рассчитывается правильно.
        """
        # Создаем данные с известными характеристиками
        np.random.seed(42)
        data = pd.DataFrame({
            'asset1': 100 + np.cumsum(np.random.normal(0, 1, 50)),
            'asset2': 50 + np.cumsum(np.random.normal(0, 0.5, 50))
        })
        
        bt = PairBacktester(
            data,
            rolling_window=10,
            z_threshold=1.5,
            capital_at_risk=10000.0,  # Большой капитал для точности
            stop_loss_multiplier=2.0,
            annualizing_factor=365
        )
        
        bt.run()
        metrics = bt.get_performance_metrics()
        
        # Проверяем, что метрики рассчитаны
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_pnl' in metrics
        
        # Sharpe ratio должен быть конечным числом
        sharpe = metrics['sharpe_ratio']
        assert np.isfinite(sharpe), f"Sharpe ratio должен быть конечным, получено: {sharpe}"
        
        # Max drawdown должен быть неположительным
        max_dd = metrics['max_drawdown']
        assert max_dd <= 0, f"Max drawdown должен быть <= 0, получено: {max_dd}"
        
        # Проверяем консистентность с ручным расчетом
        results = bt.get_results()
        pnl_series = pd.Series(results['pnl']).dropna()
        if not pnl_series.empty:
            manual_returns = pnl_series / bt.capital_at_risk
            manual_sharpe = np.sqrt(365) * manual_returns.mean() / manual_returns.std() if manual_returns.std() > 0 else 0.0
            
            # Допускаем небольшую погрешность в расчетах
            assert abs(sharpe - manual_sharpe) < 1e-10, \
                f"Sharpe ratio не совпадает: ожидалось {manual_sharpe}, получено {sharpe}"

    def test_integration_all_fixes(self):
        """Интеграционный тест: Проверяет работу всех исправлений вместе.
        
        Комплексный тест, который проверяет, что все исправления работают
        корректно в совокупности на реалистичных данных.
        """
        # Создаем реалистичные данные с различными сценариями
        np.random.seed(123)
        n_points = 100
        
        # Базовые цены с трендом и волатильностью
        base_trend = np.linspace(0, 10, n_points)
        noise1 = np.random.normal(0, 2, n_points)
        noise2 = np.random.normal(0, 1, n_points)
        
        data = pd.DataFrame({
            'STOCK_A': 100 + base_trend + noise1,
            'STOCK_B': 50 + 0.5 * base_trend + noise2
        })
        
        # Добавляем несколько константных периодов для тестирования
        data.iloc[20:25] = data.iloc[20:25].mean()  # Константный период
        
        bt = PairBacktester(
            data,
            rolling_window=15,
            z_threshold=1.5,
            z_exit=0.2,
            take_profit_multiplier=0.6,
            commission_pct=0.001,
            slippage_pct=0.0005,
            bid_ask_spread_pct_s1=0.001,
            bid_ask_spread_pct_s2=0.001,
            capital_at_risk=100000.0,
            stop_loss_multiplier=2.5,
            cooldown_periods=2,
            half_life=1.0,  # 1 день
            time_stop_multiplier=3.0,  # 3 дня
            annualizing_factor=365
        )
        
        # Должно выполниться без ошибок
        bt.run()
        results = bt.get_results()
        metrics = bt.get_performance_metrics()
        
        # Проверяем основные результаты
        assert len(results['spread']) == len(data)
        assert len(results['position']) == len(data)
        assert len(results['pnl']) == len(data)
        
        # Проверяем, что все метрики рассчитаны
        required_metrics = ['sharpe_ratio', 'max_drawdown', 'total_pnl']
        for metric in required_metrics:
            assert metric in metrics
            assert np.isfinite(metrics[metric]), f"Метрика {metric} должна быть конечной"
        
        # Проверяем детализацию издержек
        cost_columns = ['commission_costs', 'slippage_costs', 'bid_ask_costs', 'impact_costs']
        for col in cost_columns:
            assert col in results
            assert len(results[col]) == len(data)
        
        # Проверяем лог сделок
        trades_log = results['trades_log']
        assert isinstance(trades_log, list)
        
        # Если были сделки, проверяем их структуру
        if trades_log:
            for trade in trades_log:
                required_fields = ['pair', 'entry_datetime', 'exit_datetime', 
                                 'position_type', 'pnl', 'exit_reason', 'trade_duration_hours']
                for field in required_fields:
                    assert field in trade, f"Поле {field} отсутствует в логе сделки"
                
                # Проверяем, что время сделки положительное
                assert trade['trade_duration_hours'] >= 0, "Время сделки должно быть неотрицательным"
        
        print(f"Интеграционный тест пройден успешно:")
        print(f"- Обработано {len(data)} точек данных")
        print(f"- Выполнено {len(trades_log)} сделок")
        print(f"- Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"- Max drawdown: {metrics['max_drawdown']:.4f}")
        print(f"- Total PnL: {metrics['total_pnl']:.2f}")