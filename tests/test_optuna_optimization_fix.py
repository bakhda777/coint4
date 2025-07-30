"""Тест для проверки исправления оптимизации Optuna.

Проверяет, что параметры из configs/search_space.yaml корректно считываются
и используются в процессе оптимизации.
"""

import pytest
import yaml
import optuna
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimiser.fast_objective import FastWalkForwardObjective


class TestOptunaOptimizationFix:
    """Тесты для проверки исправления оптимизации Optuna."""
    
    def test_search_space_yaml_structure(self):
        """Проверяет корректную структуру файла search_space.yaml.
        
        Тест проверяет:
        - Наличие группы 'risk_management' (не 'risk')
        - Наличие параметров risk_per_position_pct и max_position_size_pct в группе 'portfolio'
        - Корректный формат max_active_positions с ключами low, high, step
        """
        search_space_path = Path("configs/search_space.yaml")
        assert search_space_path.exists(), "Файл search_space.yaml не найден"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Проверяем наличие группы risk_management (не risk)
        assert 'risk_management' in search_space, "Группа 'risk_management' не найдена в search_space.yaml"
        assert 'risk' not in search_space, "Группа 'risk' не должна существовать, используйте 'risk_management'"
        
        # Проверяем параметры в группе risk_management
        risk_mgmt = search_space['risk_management']
        assert 'stop_loss_multiplier' in risk_mgmt, "Параметр 'stop_loss_multiplier' не найден в risk_management"
        assert 'time_stop_multiplier' in risk_mgmt, "Параметр 'time_stop_multiplier' не найден в risk_management"
        
        # Проверяем группу portfolio
        assert 'portfolio' in search_space, "Группа 'portfolio' не найдена в search_space.yaml"
        portfolio = search_space['portfolio']
        
        # Проверяем, что risk параметры перенесены в portfolio
        assert 'risk_per_position_pct' in portfolio, "Параметр 'risk_per_position_pct' должен быть в группе 'portfolio'"
        assert 'max_position_size_pct' in portfolio, "Параметр 'max_position_size_pct' должен быть в группе 'portfolio'"
        
        # Проверяем корректный формат max_active_positions
        assert 'max_active_positions' in portfolio, "Параметр 'max_active_positions' не найден в portfolio"
        max_active_pos = portfolio['max_active_positions']
        assert isinstance(max_active_pos, dict), "max_active_positions должен быть словарем, не списком"
        assert 'low' in max_active_pos, "max_active_positions должен содержать ключ 'low'"
        assert 'high' in max_active_pos, "max_active_positions должен содержать ключ 'high'"
        assert 'step' in max_active_pos, "max_active_positions должен содержать ключ 'step'"
        
        # Проверяем разумные значения
        assert max_active_pos['low'] >= 1, "Минимальное количество позиций должно быть >= 1"
        assert max_active_pos['high'] > max_active_pos['low'], "Максимальное количество позиций должно быть больше минимального"
        assert max_active_pos['step'] >= 1, "Шаг должен быть >= 1"
    
    @patch('src.optimiser.fast_objective.pd.read_csv')
    @patch('src.optimiser.fast_objective.Path.exists')
    @patch('src.optimiser.fast_objective.load_master_dataset')
    def test_parameter_suggestion_from_yaml(self, mock_load_data, mock_exists, mock_read_csv):
        """Проверяет корректное считывание параметров из YAML файла.
        
        Тест проверяет:
        - Все параметры из search_space.yaml корректно считываются
        - Параметр max_active_positions использует step корректно
        - Нет ошибок при обращении к несуществующим группам
        """
        # Мокаем зависимости
        mock_exists.return_value = True
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=10)
        mock_read_csv.return_value = mock_df
        mock_load_data.return_value = Mock()
        
        # Создаем тестовый объект
        with patch('src.optimiser.fast_objective.load_config') as mock_config:
            mock_config.return_value = Mock()
            
            objective = FastWalkForwardObjective(
                base_config_path="configs/main_2024.yaml",
                search_space_path="configs/search_space.yaml"
            )
        
        # Создаем мок trial
        trial = Mock(spec=optuna.Trial)
        trial.suggest_float = Mock(side_effect=lambda name, low, high: (low + high) / 2)
        trial.suggest_int = Mock(side_effect=lambda name, low, high, step=1: low + step)
        
        # Вызываем метод
        params = objective._suggest_parameters(trial)
        
        # Проверяем, что все ожидаемые параметры присутствуют
        expected_params = ['zscore_threshold', 'zscore_exit', 'stop_loss_multiplier', 'time_stop_multiplier', 'risk_per_position_pct', 'max_position_size_pct', 'max_active_positions']
        for param in expected_params:
            assert param in params, f"Параметр '{param}' не был сгенерирован"
        
        # Проверяем, что suggest_int для max_active_positions вызван с step
        trial.suggest_int.assert_called_with(
            "max_active_positions",
            5,  # low из YAML
            25,  # high из YAML
            step=5  # step из YAML
        )
        
        # Проверяем, что suggest_float вызван для всех float параметров
        float_calls = trial.suggest_float.call_args_list
        float_param_names = [call[0][0] for call in float_calls]
        expected_float_params = ['zscore_threshold', 'zscore_exit', 'stop_loss_multiplier', 'time_stop_multiplier', 'risk_per_position_pct', 'max_position_size_pct']
        
        for param in expected_float_params:
            assert param in float_param_names, f"suggest_float не был вызван для параметра '{param}'"
    
    def test_parameter_ranges_are_reasonable(self):
        """Проверяет, что диапазоны параметров в YAML файле разумны.
        
        Тест проверяет:
        - Диапазоны параметров не содержат отрицательных значений где это неуместно
        - Минимальные значения меньше максимальных
        - Значения находятся в разумных пределах для торговой стратегии
        """
        search_space_path = Path("configs/search_space.yaml")
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Проверяем сигналы
        signals = search_space['signals']
        z_threshold = signals['zscore_threshold']
        assert z_threshold['low'] > 0, "zscore_threshold должен быть положительным"
        assert z_threshold['high'] > z_threshold['low'], "Максимальный zscore_threshold должен быть больше минимального"
        assert z_threshold['high'] <= 3.0, "zscore_threshold не должен быть слишком большим (>3.0)"
        
        # Проверяем управление рисками
        risk_mgmt = search_space['risk_management']
        sl_mult = risk_mgmt['stop_loss_multiplier']
        assert sl_mult['low'] > 0, "stop_loss_multiplier должен быть положительным"
        assert sl_mult['high'] > sl_mult['low'], "Максимальный stop_loss_multiplier должен быть больше минимального"
        
        # Проверяем портфель
        portfolio = search_space['portfolio']
        risk_per_pos = portfolio['risk_per_position_pct']
        assert risk_per_pos['low'] > 0, "risk_per_position_pct должен быть положительным"
        assert risk_per_pos['high'] <= 0.1, "risk_per_position_pct не должен превышать 10%"
        
        max_pos_size = portfolio['max_position_size_pct']
        assert max_pos_size['low'] > 0, "max_position_size_pct должен быть положительным"
        assert max_pos_size['high'] <= 0.5, "max_position_size_pct не должен превышать 50%"
    
    def test_no_legacy_risk_group_usage(self):
        """Проверяет, что код не пытается использовать устаревшую группу 'risk'.
        
        Тест проверяет:
        - В коде fast_objective.py нет обращений к группе 'risk'
        - Все параметры считываются из правильных групп
        """
        fast_objective_path = Path("src/optimiser/fast_objective.py")
        assert fast_objective_path.exists(), "Файл fast_objective.py не найден"
        
        with open(fast_objective_path, 'r') as f:
            code_content = f.read()
        
        # Проверяем, что код ищет правильную группу
        assert "'risk_management'" in code_content, "Код должен искать группу 'risk_management'"
        
        # Проверяем, что код не ищет устаревшую группу 'risk' (кроме комментариев)
        lines = code_content.split('\n')
        for i, line in enumerate(lines, 1):
            if "'risk'" in line and not line.strip().startswith('#'):
                # Разрешаем только в контексте 'risk_management' или 'risk_per_position_pct'
                if "'risk_management'" not in line and "'risk_per_position_pct'" not in line and "'risk_per_pos'" not in line:
                    pytest.fail(f"Строка {i} содержит обращение к устаревшей группе 'risk': {line.strip()}")