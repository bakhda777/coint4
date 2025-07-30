"""Тест для проверки исправления оптимизации Optuna.

Проверяет что FastWalkForwardObjective корректно считывает параметры
из configs/search_space.yaml и использует их в оптимизации.
"""

import pytest
import optuna
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimiser.fast_objective import FastWalkForwardObjective


class TestOptunaConfigFix:
    """Тесты для проверки исправления конфигурации Optuna."""
    
    def test_search_space_structure_matches_code(self):
        """Проверяет что структура search_space.yaml соответствует ожиданиям кода.
        
        Этот тест проверяет что:
        1. Группа называется 'risk_management', а не 'risk'
        2. Параметры risk_per_position_pct и max_position_size_pct находятся в группе 'portfolio'
        3. Параметр max_active_positions имеет формат {low, high, step}
        """
        search_space_path = Path("configs/search_space.yaml")
        assert search_space_path.exists(), "Файл search_space.yaml не найден"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Проверяем наличие группы risk_management (не risk)
        assert 'risk_management' in search_space, "Группа 'risk_management' не найдена в search_space.yaml"
        assert 'risk' not in search_space, "Старая группа 'risk' все еще присутствует в search_space.yaml"
        
        # Проверяем что параметры риска находятся в правильных группах
        portfolio = search_space.get('portfolio', {})
        assert 'risk_per_position_pct' in portfolio, "risk_per_position_pct должен быть в группе 'portfolio'"
        assert 'max_position_size_pct' in portfolio, "max_position_size_pct должен быть в группе 'portfolio'"
        
        # Проверяем формат max_active_positions
        max_active_pos = portfolio.get('max_active_positions', {})
        assert isinstance(max_active_pos, dict), "max_active_positions должен быть словарем, а не списком"
        assert 'low' in max_active_pos, "max_active_positions должен содержать ключ 'low'"
        assert 'high' in max_active_pos, "max_active_positions должен содержать ключ 'high'"
        assert 'step' in max_active_pos, "max_active_positions должен содержать ключ 'step'"
        
        # Проверяем типы значений
        assert isinstance(max_active_pos['low'], int), "max_active_positions['low'] должен быть целым числом"
        assert isinstance(max_active_pos['high'], int), "max_active_positions['high'] должен быть целым числом"
        assert isinstance(max_active_pos['step'], int), "max_active_positions['step'] должен быть целым числом"
        
        # Проверяем логичность значений
        assert max_active_pos['low'] < max_active_pos['high'], "low должен быть меньше high"
        assert max_active_pos['step'] > 0, "step должен быть положительным"
    
    def test_parameter_suggestion_with_step(self):
        """Проверяет что метод _suggest_parameters корректно использует step для max_active_positions.
        
        Этот тест проверяет что:
        1. Параметр step передается в trial.suggest_int
        2. Все остальные параметры также корректно считываются
        """
        # Создаем тестовую конфигурацию
        test_search_space = {
            'signals': {
                'zscore_threshold': {'low': 1.2, 'high': 1.8},
                'zscore_exit': {'low': -0.2, 'high': 0.2}
            },
            'risk_management': {
                'stop_loss_multiplier': {'low': 2.5, 'high': 4.5},
                'time_stop_multiplier': {'low': 1.5, 'high': 3.0}
            },
            'portfolio': {
                'max_active_positions': {'low': 5, 'high': 25, 'step': 5},
                'risk_per_position_pct': {'low': 0.015, 'high': 0.035},
                'max_position_size_pct': {'low': 0.05, 'high': 0.12}
            }
        }
        
        # Создаем временные файлы
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_search_space, f)
            search_space_path = f.name
        
        # Мокаем зависимости
        mock_config = Mock()
        mock_config.walk_forward.start_date = '2024-01-01'
        mock_config.walk_forward.training_period_days = 90
        mock_config.walk_forward.testing_period_days = 30
        
        with patch('src.optimiser.fast_objective.load_config', return_value=mock_config), \
             patch('src.optimiser.fast_objective.Path') as mock_path, \
             patch('src.optimiser.fast_objective.pd.read_csv') as mock_read_csv:
            
            # Настраиваем моки
            mock_path.return_value.exists.return_value = True
            mock_read_csv.return_value = Mock()
            mock_read_csv.return_value.__len__ = Mock(return_value=10)
            
            # Создаем объект для тестирования
            objective = FastWalkForwardObjective('dummy_config.yaml', search_space_path)
            objective.search_space = test_search_space
            
            # Создаем мок trial
            mock_trial = Mock()
            suggest_calls = {}
            
            def mock_suggest_float(name, low, high):
                suggest_calls[name] = {'type': 'float', 'low': low, 'high': high}
                return (low + high) / 2
            
            def mock_suggest_int(name, low, high, step=1):
                suggest_calls[name] = {'type': 'int', 'low': low, 'high': high, 'step': step}
                return low
            
            mock_trial.suggest_float = mock_suggest_float
            mock_trial.suggest_int = mock_suggest_int
            
            # Вызываем метод
            params = objective._suggest_parameters(mock_trial)
            
            # Проверяем что все ожидаемые параметры присутствуют
            expected_params = ['zscore_threshold', 'zscore_exit', 'stop_loss_multiplier', 'time_stop_multiplier', 
                             'risk_per_position_pct', 'max_position_size_pct', 'max_active_positions']
            
            for param in expected_params:
                assert param in params, f"Параметр {param} не был сгенерирован"
            
            # Проверяем что max_active_positions использует step
            assert 'max_active_positions' in suggest_calls, "max_active_positions не был вызван через suggest_int"
            max_active_call = suggest_calls['max_active_positions']
            assert max_active_call['step'] == 5, f"step должен быть 5, получен {max_active_call['step']}"
            assert max_active_call['low'] == 5, f"low должен быть 5, получен {max_active_call['low']}"
            assert max_active_call['high'] == 25, f"high должен быть 25, получен {max_active_call['high']}"
            
            # Проверяем диапазоны других параметров
            assert suggest_calls['zscore_threshold']['low'] == 1.2
            assert suggest_calls['zscore_threshold']['high'] == 1.8
            assert suggest_calls['zscore_exit']['low'] == -0.2
            assert suggest_calls['zscore_exit']['high'] == 0.2
            assert suggest_calls['stop_loss_multiplier']['low'] == 2.5
            assert suggest_calls['stop_loss_multiplier']['high'] == 4.5
            assert suggest_calls['risk_per_position_pct']['low'] == 0.015
            assert suggest_calls['risk_per_position_pct']['high'] == 0.035
        
        # Очищаем временный файл
        Path(search_space_path).unlink()
    
    def test_all_parameters_are_used_in_optimization(self):
        """Проверяет что все параметры из search_space.yaml используются в оптимизации.
        
        Этот тест проверяет что ни один параметр не игнорируется из-за
        неправильного имени группы или расположения.
        """
        search_space_path = Path("configs/search_space.yaml")
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Собираем все параметры из конфигурации
        expected_params = set()
        
        if 'signals' in search_space:
            for param in search_space['signals']:
                if param == 'zscore_threshold':
                    expected_params.add('zscore_threshold')
                elif param == 'zscore_exit':
                    expected_params.add('zscore_exit')
        
        if 'risk_management' in search_space:
            for param in search_space['risk_management']:
                if param == 'stop_loss_multiplier':
                    expected_params.add('stop_loss_multiplier')
                elif param == 'time_stop_multiplier':
                    expected_params.add('time_stop_multiplier')
        
        if 'portfolio' in search_space:
            for param in search_space['portfolio']:
                if param == 'risk_per_position_pct':
                    expected_params.add('risk_per_position_pct')
                elif param == 'max_position_size_pct':
                    expected_params.add('max_position_size_pct')
                elif param == 'max_active_positions':
                    expected_params.add('max_active_positions')
        
        # Проверяем что у нас есть ожидаемые параметры
        assert len(expected_params) >= 6, f"Ожидается минимум 6 параметров, найдено {len(expected_params)}"
        
        # Мокаем зависимости и тестируем
        mock_config = Mock()
        mock_config.walk_forward.start_date = '2024-01-01'
        
        with patch('src.optimiser.fast_objective.load_config', return_value=mock_config), \
             patch('src.optimiser.fast_objective.Path') as mock_path, \
             patch('src.optimiser.fast_objective.pd.read_csv') as mock_read_csv:
            
            mock_path.return_value.exists.return_value = True
            mock_read_csv.return_value = Mock()
            mock_read_csv.return_value.__len__ = Mock(return_value=10)
            
            objective = FastWalkForwardObjective('dummy_config.yaml', str(search_space_path))
            
            # Создаем мок trial
            mock_trial = Mock()
            generated_params = set()
            
            def track_suggest_float(name, low, high, log=False):
                generated_params.add(name)
                return (low + high) / 2

            def track_suggest_int(name, low, high, step=1, log=False):
                generated_params.add(name)
                return low
            
            mock_trial.suggest_float = track_suggest_float
            mock_trial.suggest_int = track_suggest_int
            
            # Вызываем метод
            params = objective._suggest_parameters(mock_trial)
            
            # Проверяем что все ожидаемые параметры были сгенерированы
            missing_params = expected_params - generated_params
            assert len(missing_params) == 0, f"Параметры не были сгенерированы: {missing_params}"
            
            # Проверяем что все сгенерированные параметры есть в результате
            for param in expected_params:
                assert param in params, f"Параметр {param} отсутствует в результате _suggest_parameters"
    
    def test_parameter_ranges_are_reasonable(self):
        """Проверяет что диапазоны параметров в search_space.yaml разумны для торговли.
        
        Этот тест проверяет что значения параметров находятся в разумных
        диапазонах для торговой стратегии.
        """
        search_space_path = Path("configs/search_space.yaml")
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Проверяем сигналы
        if 'signals' in search_space:
            signals = search_space['signals']
            
            if 'zscore_threshold' in signals:
                z_thresh = signals['zscore_threshold']
                assert z_thresh['low'] > 0, "zscore_threshold low должен быть положительным"
                assert z_thresh['high'] > z_thresh['low'], "zscore_threshold high должен быть больше low"
                assert z_thresh['low'] >= 1.0, "zscore_threshold low должен быть >= 1.0 для разумной торговли"
                assert z_thresh['high'] <= 3.0, "zscore_threshold high должен быть <= 3.0 для достаточной частоты сделок"
            
            if 'zscore_exit' in signals:
                z_exit = signals['zscore_exit']
                assert z_exit['low'] <= 0, "zscore_exit low должен быть <= 0"
                assert z_exit['high'] >= 0, "zscore_exit high должен быть >= 0"
                assert abs(z_exit['low']) <= 1.0, "zscore_exit low не должен быть слишком отрицательным"
                assert z_exit['high'] <= 1.0, "zscore_exit high не должен быть слишком большим"
        
        # Проверяем управление рисками
        if 'risk_management' in search_space:
            risk = search_space['risk_management']
            
            if 'stop_loss_multiplier' in risk:
                sl_mult = risk['stop_loss_multiplier']
                assert sl_mult['low'] > 1.0, "stop_loss_multiplier должен быть > 1.0"
                assert sl_mult['high'] <= 10.0, "stop_loss_multiplier не должен быть слишком большим"
            
            if 'time_stop_multiplier' in risk:
                time_mult = risk['time_stop_multiplier']
                assert time_mult['low'] > 0, "time_stop_multiplier должен быть положительным"
                assert time_mult['high'] <= 10.0, "time_stop_multiplier не должен быть слишком большим"
        
        # Проверяем портфель
        if 'portfolio' in search_space:
            portfolio = search_space['portfolio']
            
            if 'risk_per_position_pct' in portfolio:
                risk_per_pos = portfolio['risk_per_position_pct']
                assert risk_per_pos['low'] > 0, "risk_per_position_pct должен быть положительным"
                assert risk_per_pos['high'] <= 0.1, "risk_per_position_pct не должен превышать 10%"
                assert risk_per_pos['low'] >= 0.005, "risk_per_position_pct должен быть >= 0.5%"
            
            if 'max_position_size_pct' in portfolio:
                max_pos_size = portfolio['max_position_size_pct']
                assert max_pos_size['low'] > 0, "max_position_size_pct должен быть положительным"
                assert max_pos_size['high'] <= 0.5, "max_position_size_pct не должен превышать 50%"
            
            if 'max_active_positions' in portfolio:
                max_active = portfolio['max_active_positions']
                assert max_active['low'] >= 1, "max_active_positions должен быть >= 1"
                assert max_active['high'] <= 100, "max_active_positions не должен быть слишком большим"
                assert max_active['step'] >= 1, "max_active_positions step должен быть >= 1"