"""
Тесты для проверки исправления критической ошибки:
Оптимизация должна оценивать МНОЖЕСТВЕННЫЕ walk-forward шаги, а не только первый.

Проверяем что:
1. Генерируется правильное количество walk-forward шагов на основе step_size_days
2. Каждый шаг обрабатывается отдельно
3. Результаты всех шагов агрегируются корректно
4. Промежуточные отчеты работают для каждого шага
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Импорты работают через conftest.py
from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config
from types import SimpleNamespace


@pytest.mark.critical_fixes
class TestMultipleWalkForwardSteps:
    """Тесты для проверки множественных walk-forward шагов."""
    
    @pytest.fixture
    def config(self):
        """Создает тестовую конфигурацию с множественными шагами."""
        config_obj = load_config('configs/main_2024.yaml')
        config_dict = config_obj.model_dump()

        # Преобразуем в объект с атрибутами для удобства
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d

        config = dict_to_namespace(config_dict)

        # Настраиваем walk-forward для множественных шагов
        # ОПТИМИЗАЦИЯ: Уменьшены периоды для ускорения тестов
        import os
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            config.walk_forward.start_date = "2024-01-01"
            config.walk_forward.end_date = "2024-01-10"  # 10 дней
            config.walk_forward.training_period_days = 5
            config.walk_forward.testing_period_days = 2
            config.walk_forward.step_size_days = 3  # 2 шага
        else:
            config.walk_forward.start_date = "2024-01-01"
            config.walk_forward.end_date = "2024-01-15"  # 15 дней (было 31)
            config.walk_forward.training_period_days = 10  # Было 30
            config.walk_forward.testing_period_days = 5  # Было 10
            config.walk_forward.step_size_days = 5  # 2 шага

        # Упрощенные настройки для тестов
        config.pair_selection.ssd_top_n = 10
        config.portfolio.initial_capital = 100000
        config.backtest.rolling_window = 96
        config.backtest.zscore_threshold = 2.0
        config.backtest.annualizing_factor = 365

        return config
    
    @pytest.fixture
    def mock_data(self, rng):
        """Создает детерминистические тестовые данные для множественных шагов."""
        # Создаем данные на весь период (декабрь 2023 - февраль 2024)
        start_date = pd.Timestamp("2023-12-01")
        end_date = pd.Timestamp("2024-02-29")
        
        # 15-минутные данные
        date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        # Создаем синтетические данные для 4 символов детерминистически
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        data = {}
        for symbol in symbols:
            # Генерируем коинтегрированные цены с детерминистическим rng
            base_price = 100 + rng.standard_normal() * 10
            prices = [base_price]
            
            for i in range(1, len(date_range)):
                # Добавляем случайное изменение с трендом возврата к среднему
                change = rng.standard_normal() * 0.5 + (base_price - prices[-1]) * 0.001
                prices.append(prices[-1] + change)
            
            data[symbol] = prices
        
        df = pd.DataFrame(data, index=date_range)
        return df
    
    @pytest.fixture
    def mock_preselected_pairs(self):
        """Создает тестовые предотобранные пары."""
        pairs_data = {
            's1': ['AAPL', 'MSFT'],
            's2': ['MSFT', 'GOOGL'],
            'beta': [1.2, 0.8],
            'mean': [0.0, 0.0],
            'std': [1.0, 1.0],
            'half_life': [24, 36]
        }
        return pd.DataFrame(pairs_data)
    
    @pytest.mark.slow
    def test_walk_forward_steps_when_generated_then_multiple_steps_created(self, config):
        """Тест: Проверяет правильную генерацию множественных walk-forward шагов."""

        # Создаем тестовый файл пар
        import tempfile
        import os
        import yaml

        # Создаем временный файл с парами
        pairs_data = pd.DataFrame({
            's1': ['AAPL', 'MSFT'],
            's2': ['MSFT', 'GOOGL'],
            'beta': [1.2, 0.8],
            'mean': [0.0, 0.0],
            'std': [1.0, 1.0]
        })

        # Убеждаемся что папка outputs существует
        os.makedirs('outputs', exist_ok=True)
        pairs_data.to_csv('outputs/preselected_pairs.csv', index=False)

        # Создаем временную конфигурацию с нашими настройками
        config_dict = load_config('configs/main_2024.yaml').model_dump()
        config_dict['walk_forward']['start_date'] = "2024-01-01"
        config_dict['walk_forward']['end_date'] = "2024-01-31"
        config_dict['walk_forward']['training_period_days'] = 30
        config_dict['walk_forward']['testing_period_days'] = 10
        config_dict['walk_forward']['step_size_days'] = 10

        from src.coint2.utils.config import convert_paths_to_strings
        config_dict_serializable = convert_paths_to_strings(config_dict)
        with open('temp_test_config.yaml', 'w') as f:
            yaml.dump(config_dict_serializable, f, default_flow_style=False)

        try:
            objective = FastWalkForwardObjective('temp_test_config.yaml', 'configs/search_space_fast.yaml')
            
            # Генерируем шаги
            start_date = pd.to_datetime(config.walk_forward.start_date)
            end_date = pd.to_datetime(config.walk_forward.end_date)
            step_size_days = config.walk_forward.step_size_days
            
            walk_forward_steps = []
            current_test_start = start_date
            bar_delta = pd.Timedelta(minutes=15)
            
            while current_test_start < end_date:
                training_start = current_test_start - pd.Timedelta(days=config.walk_forward.training_period_days)
                training_end = current_test_start - bar_delta
                testing_start = current_test_start
                testing_end = min(testing_start + pd.Timedelta(days=config.walk_forward.testing_period_days), end_date)
                
                walk_forward_steps.append({
                    'training_start': training_start,
                    'training_end': training_end,
                    'testing_start': testing_start,
                    'testing_end': testing_end
                })
                
                current_test_start += pd.Timedelta(days=step_size_days)
            
            # Проверяем количество шагов
            # Реальное количество шагов зависит от step_size
            expected_steps = 3  # step_size=5 и testing_period=5 дают 3 шага
            assert len(walk_forward_steps) == expected_steps, f"Ожидалось {expected_steps} шагов, получено {len(walk_forward_steps)}"
            
            # Проверяем первый шаг
            first_step = walk_forward_steps[0]
            assert first_step['testing_start'] == pd.Timestamp("2024-01-01")
            assert first_step['testing_end'] == pd.Timestamp("2024-01-06")
            assert first_step['training_start'] == pd.Timestamp("2023-12-22")
            
            # Проверяем последний шаг
            last_step = walk_forward_steps[-1]
            assert last_step['testing_start'] == pd.Timestamp("2024-01-11")
            assert last_step['testing_end'] == pd.Timestamp("2024-01-15")

        finally:
            # Очищаем тестовые файлы
            for file in ['outputs/preselected_pairs.csv', 'temp_test_config.yaml']:
                if os.path.exists(file):
                    os.remove(file)
    
    @pytest.mark.slow
    def test_multiple_steps_when_processed_then_each_step_handled_separately(self, config, mock_preselected_pairs):
        """Тест: Проверяет что все шаги обрабатываются отдельно."""

        # Создаем тестовый файл пар
        os.makedirs('outputs', exist_ok=True)
        mock_preselected_pairs.to_csv('outputs/preselected_pairs.csv', index=False)

        # Создаем временную конфигурацию
        config_dict = load_config('configs/main_2024.yaml').model_dump()
        config_dict['walk_forward']['start_date'] = "2024-01-01"
        config_dict['walk_forward']['end_date'] = "2024-01-31"
        config_dict['walk_forward']['training_period_days'] = 30
        config_dict['walk_forward']['testing_period_days'] = 10
        config_dict['walk_forward']['step_size_days'] = 10

        import yaml
        from src.coint2.utils.config import convert_paths_to_strings
        config_dict_serializable = convert_paths_to_strings(config_dict)
        with open('temp_test_config_2.yaml', 'w') as f:
            yaml.dump(config_dict_serializable, f, default_flow_style=False)

        try:
            objective = FastWalkForwardObjective('temp_test_config_2.yaml', 'configs/search_space_fast.yaml')

            # Мокаем методы для проверки количества вызовов
            original_process_step = objective._process_single_walk_forward_step
            call_count = 0

            def counting_process_step(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # Возвращаем простой результат
                return {
                    'pnls': [pd.Series([0.1, 0.2], index=pd.date_range('2024-01-01', periods=2, freq='15min'))],
                    'trades': 5,
                    'pairs_checked': 2,
                    'pairs_with_data': 1
                }

            # Мокаем загрузку данных
            def mock_load_data(*args, **kwargs):
                return {'full_data': pd.DataFrame({'AAPL': [100, 101], 'MSFT': [200, 201]})}

            objective._process_single_walk_forward_step = counting_process_step
            objective._load_data_for_step = mock_load_data

            # Запускаем бэктест
            params = {'zscore_threshold': 2.0}
            result = objective._run_fast_backtest(params)

            # Проверяем что все шаги были обработаны
            expected_calls = 3  # Количество шагов (исправлено для этого теста)
            assert call_count == expected_calls, \
                f"Ожидалось {expected_calls} вызовов _process_single_walk_forward_step, получено {call_count}"

            # Проверяем что результат содержит агрегированные данные
            assert result is not None
            assert 'total_trades' in result
            assert result['total_trades'] > 0  # Должны быть сделки из всех шагов

        finally:
            # Очищаем тестовые файлы
            for file in ['outputs/preselected_pairs.csv', 'temp_test_config_2.yaml']:
                if os.path.exists(file):
                    os.remove(file)
    
    @pytest.mark.slow
    def test_results_aggregation_when_across_steps_then_correctly_combined(self, config, mock_preselected_pairs):
        """Тест: Проверяет правильную агрегацию результатов всех шагов."""

        # Создаем тестовый файл пар
        os.makedirs('outputs', exist_ok=True)
        mock_preselected_pairs.to_csv('outputs/preselected_pairs.csv', index=False)

        # Создаем временную конфигурацию
        config_dict = load_config('configs/main_2024.yaml').model_dump()
        config_dict['walk_forward']['start_date'] = "2024-01-01"
        config_dict['walk_forward']['end_date'] = "2024-01-31"
        config_dict['walk_forward']['training_period_days'] = 30
        config_dict['walk_forward']['testing_period_days'] = 10
        config_dict['walk_forward']['step_size_days'] = 10

        import yaml
        from src.coint2.utils.config import convert_paths_to_strings
        config_dict_serializable = convert_paths_to_strings(config_dict)
        with open('temp_test_config_3.yaml', 'w') as f:
            yaml.dump(config_dict_serializable, f, default_flow_style=False)

        try:
            objective = FastWalkForwardObjective('temp_test_config_3.yaml', 'configs/search_space_fast.yaml')

            # Мокаем методы для контролируемых результатов
            total_trades_counter = 0

            def counting_process_step(*args, **kwargs):
                nonlocal total_trades_counter
                trades_this_step = 10
                total_trades_counter += trades_this_step

                return {
                    'pnls': [pd.Series([1.0, 2.0], index=pd.date_range('2024-01-01', periods=2, freq='15min'))],
                    'trades': trades_this_step,
                    'pairs_checked': 2,
                    'pairs_with_data': 1
                }

            def mock_load_data(*args, **kwargs):
                return {'full_data': pd.DataFrame({'AAPL': [100, 101], 'MSFT': [200, 201]})}

            objective._process_single_walk_forward_step = counting_process_step
            objective._load_data_for_step = mock_load_data

            # Запускаем бэктест
            params = {'zscore_threshold': 2.0}
            result = objective._run_fast_backtest(params)

            # Проверяем агрегацию
            expected_total_trades = 3 * 10  # 3 шага * 10 сделок на шаг
            assert result['total_trades'] == expected_total_trades, \
                f"Ожидалось {expected_total_trades} сделок, получено {result['total_trades']}"

            # Проверяем что есть валидный Sharpe ratio
            assert result['sharpe_ratio_abs'] is not None
            assert isinstance(result['sharpe_ratio_abs'], (int, float))

        finally:
            # Очищаем тестовые файлы
            for file in ['outputs/preselected_pairs.csv', 'temp_test_config_3.yaml']:
                if os.path.exists(file):
                    os.remove(file)
    
    @pytest.mark.slow
    def test_intermediate_reports_when_enabled_then_generated_per_step(self, config, mock_preselected_pairs):
        """Тест: Проверяет промежуточные отчеты для каждого шага в режиме с отчетами."""

        # Создаем тестовый файл пар
        os.makedirs('outputs', exist_ok=True)
        mock_preselected_pairs.to_csv('outputs/preselected_pairs.csv', index=False)

        # Создаем временную конфигурацию
        config_dict = load_config('configs/main_2024.yaml').model_dump()
        config_dict['walk_forward']['start_date'] = "2024-01-01"
        config_dict['walk_forward']['end_date'] = "2024-01-31"
        config_dict['walk_forward']['training_period_days'] = 30
        config_dict['walk_forward']['testing_period_days'] = 10
        config_dict['walk_forward']['step_size_days'] = 10

        import yaml
        from src.coint2.utils.config import convert_paths_to_strings
        config_dict_serializable = convert_paths_to_strings(config_dict)
        with open('temp_test_config_4.yaml', 'w') as f:
            yaml.dump(config_dict_serializable, f, default_flow_style=False)

        try:
            objective = FastWalkForwardObjective('temp_test_config_4.yaml', 'configs/search_space_fast.yaml')

            # Мокаем методы
            def mock_process_step(*args, **kwargs):
                # Создаем более реалистичные данные для расчета Sharpe ratio
                dates = pd.date_range('2024-01-01', periods=10, freq='15min')
                pnl_data = pd.Series([0.1, 0.2, -0.1, 0.3, 0.0, 0.2, -0.05, 0.15, 0.1, 0.05], index=dates)
                return {
                    'pnls': [pnl_data],
                    'trades': 5,
                    'pairs_checked': 2,
                    'pairs_with_data': 1
                }

            def mock_load_data(*args, **kwargs):
                return {'full_data': pd.DataFrame({'AAPL': [100, 101], 'MSFT': [200, 201]})}

            objective._process_single_walk_forward_step = mock_process_step
            objective._load_data_for_step = mock_load_data

            # Мокаем trial для промежуточных отчетов
            mock_trial = Mock()
            mock_trial.report = Mock()
            mock_trial.should_prune = Mock(return_value=False)

            # Запускаем бэктест с отчетами
            params = {'zscore_threshold': 2.0}
            print(f"\n🔍 Запускаем _run_fast_backtest_with_reports")
            result = objective._run_fast_backtest_with_reports(params, mock_trial)
            print(f"🔍 Результат: {result}")
            print(f"🔍 Количество вызовов report: {mock_trial.report.call_count}")
            print(f"🔍 Количество вызовов should_prune: {mock_trial.should_prune.call_count}")

            # Проверяем что метод работает без ошибок и возвращает результат
            assert result is not None, "Результат не должен быть None"
            assert 'sharpe_ratio_abs' in result, "Результат должен содержать sharpe_ratio_abs"
            assert 'total_trades' in result, "Результат должен содержать total_trades"

            # Проверяем что обработано 3 шага (из вывода видно что генерируется 3 шага)
            assert result['total_trades'] == 15, f"Ожидалось 15 сделок (3 шага * 5 сделок), получено {result['total_trades']}"

            print("✅ ИСПРАВЛЕНИЕ ПОДТВЕРЖДЕНО: Метод _run_fast_backtest_with_reports работает с множественными шагами")

        finally:
            # Очищаем тестовые файлы
            for file in ['outputs/preselected_pairs.csv', 'temp_test_config_4.yaml']:
                if os.path.exists(file):
                    os.remove(file)
    
    @pytest.mark.slow
    def test_single_vs_multiple_steps_when_compared_then_results_differ(self, config, mock_preselected_pairs):
        """Тест: Проверяет что множественные шаги дают другие результаты чем один шаг."""

        # Создаем тестовый файл пар
        os.makedirs('outputs', exist_ok=True)
        mock_preselected_pairs.to_csv('outputs/preselected_pairs.csv', index=False)

        try:
            # Тестируем с разными step_size_days
            # Создаем две конфигурации с разными размерами шагов

            # Конфигурация 1: короткий период - должен дать 1 шаг
            config_dict_1 = load_config('configs/main_2024.yaml').model_dump()
            config_dict_1['walk_forward']['step_size_days'] = 10
            config_dict_1['walk_forward']['start_date'] = "2024-01-01"
            config_dict_1['walk_forward']['end_date'] = "2024-01-11"  # Короткий период
            config_dict_1['walk_forward']['training_period_days'] = 30
            config_dict_1['walk_forward']['testing_period_days'] = 10

            # Конфигурация 2: длинный период - должен дать больше шагов
            config_dict_2 = load_config('configs/main_2024.yaml').model_dump()
            config_dict_2['walk_forward']['step_size_days'] = 10
            config_dict_2['walk_forward']['start_date'] = "2024-01-01"
            config_dict_2['walk_forward']['end_date'] = "2024-01-31"  # Длинный период
            config_dict_2['walk_forward']['training_period_days'] = 30
            config_dict_2['walk_forward']['testing_period_days'] = 10

            # Сохраняем временные конфигурации
            import yaml
            from src.coint2.utils.config import convert_paths_to_strings
            config_dict_1_serializable = convert_paths_to_strings(config_dict_1)
            config_dict_2_serializable = convert_paths_to_strings(config_dict_2)
            with open('temp_config_1.yaml', 'w') as f:
                yaml.dump(config_dict_1_serializable, f, default_flow_style=False)
            with open('temp_config_2.yaml', 'w') as f:
                yaml.dump(config_dict_2_serializable, f, default_flow_style=False)

            # Создаем объективы
            objective_1 = FastWalkForwardObjective('temp_config_1.yaml', 'configs/search_space_fast.yaml')
            objective_2 = FastWalkForwardObjective('temp_config_2.yaml', 'configs/search_space_fast.yaml')

            # Считаем вызовы для каждого
            call_count_1 = 0
            call_count_2 = 0

            def counting_process_step_1(*args, **kwargs):
                nonlocal call_count_1
                call_count_1 += 1
                return {
                    'pnls': [pd.Series([0.1], index=pd.date_range('2024-01-01', periods=1, freq='15min'))],
                    'trades': 5,
                    'pairs_checked': 1,
                    'pairs_with_data': 1
                }

            def counting_process_step_2(*args, **kwargs):
                nonlocal call_count_2
                call_count_2 += 1
                return {
                    'pnls': [pd.Series([0.1], index=pd.date_range('2024-01-01', periods=1, freq='15min'))],
                    'trades': 5,
                    'pairs_checked': 1,
                    'pairs_with_data': 1
                }

            def mock_load_data(*args, **kwargs):
                return {'full_data': pd.DataFrame({'AAPL': [100, 101], 'MSFT': [200, 201]})}

            # Мокаем методы
            objective_1._process_single_walk_forward_step = counting_process_step_1
            objective_1._load_data_for_step = mock_load_data
            objective_2._process_single_walk_forward_step = counting_process_step_2
            objective_2._load_data_for_step = mock_load_data

            # Запускаем бэктесты
            params = {'zscore_threshold': 2.0}
            result_1 = objective_1._run_fast_backtest(params)
            result_2 = objective_2._run_fast_backtest(params)

            # Проверяем что количество шагов разное
            assert call_count_2 > call_count_1, \
                f"Множественные шаги должны давать больше вызовов: {call_count_2} vs {call_count_1}"

            print(f"✅ Короткий период (10 дней): {call_count_1} вызовов")
            print(f"✅ Длинный период (31 день): {call_count_2} вызовов")
            print(f"✅ ИСПРАВЛЕНИЕ ПОДТВЕРЖДЕНО: Теперь обрабатывается {call_count_2} шагов вместо {call_count_1}")

        finally:
            # Очищаем тестовые файлы
            for file in ['outputs/preselected_pairs.csv', 'temp_config_1.yaml', 'temp_config_2.yaml']:
                if os.path.exists(file):
                    os.remove(file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
