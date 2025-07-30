#!/usr/bin/env python3
"""
Тест для проверки валидации параметров и выявления конкретных проблем.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.metric_utils import validate_params, normalize_params


class TestParameterValidationFix:
    """Тесты для проверки валидации параметров из search_space."""
    
    def test_search_space_parameter_validation(self):
        """Проверяет что параметры из search_space проходят валидацию."""
        
        # Загружаем реальный search space
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        print(f"Search space: {search_space}")
        
        # Создаем параметры из средних значений диапазонов
        test_params = {}
        
        # Signals
        if 'signals' in search_space:
            signals = search_space['signals']
            if 'zscore_threshold' in signals:
                cfg = signals['zscore_threshold']
                test_params['zscore_threshold'] = (cfg['low'] + cfg['high']) / 2
            if 'zscore_exit' in signals:
                cfg = signals['zscore_exit']
                test_params['zscore_exit'] = (cfg['low'] + cfg['high']) / 2
            if 'rolling_window' in signals:
                cfg = signals['rolling_window']
                test_params['rolling_window'] = int((cfg['low'] + cfg['high']) / 2)
        
        # Risk management
        if 'risk_management' in search_space:
            risk = search_space['risk_management']
            if 'stop_loss_multiplier' in risk:
                cfg = risk['stop_loss_multiplier']
                test_params['stop_loss_multiplier'] = (cfg['low'] + cfg['high']) / 2
            if 'time_stop_multiplier' in risk:
                cfg = risk['time_stop_multiplier']
                test_params['time_stop_multiplier'] = (cfg['low'] + cfg['high']) / 2
            if 'cooldown_hours' in risk:
                cfg = risk['cooldown_hours']
                test_params['cooldown_hours'] = int((cfg['low'] + cfg['high']) / 2)
        
        # Portfolio
        if 'portfolio' in search_space:
            portfolio = search_space['portfolio']
            if 'risk_per_position_pct' in portfolio:
                cfg = portfolio['risk_per_position_pct']
                test_params['risk_per_position_pct'] = (cfg['low'] + cfg['high']) / 2
            if 'max_position_size_pct' in portfolio:
                cfg = portfolio['max_position_size_pct']
                test_params['max_position_size_pct'] = (cfg['low'] + cfg['high']) / 2
            if 'max_active_positions' in portfolio:
                cfg = portfolio['max_active_positions']
                test_params['max_active_positions'] = int((cfg['low'] + cfg['high']) / 2)
        
        # Costs
        if 'costs' in search_space:
            costs = search_space['costs']
            if 'commission_pct' in costs:
                cfg = costs['commission_pct']
                test_params['commission_pct'] = (cfg['low'] + cfg['high']) / 2
            if 'slippage_pct' in costs:
                cfg = costs['slippage_pct']
                test_params['slippage_pct'] = (cfg['low'] + cfg['high']) / 2
        
        # Normalization
        if 'normalization' in search_space:
            norm = search_space['normalization']
            if 'normalization_method' in norm:
                test_params['normalization_method'] = norm['normalization_method'][0]
            if 'min_history_ratio' in norm:
                cfg = norm['min_history_ratio']
                test_params['min_history_ratio'] = (cfg['low'] + cfg['high']) / 2
        
        print(f"Тестовые параметры: {test_params}")
        
        # Проверяем валидацию
        try:
            validated_params = validate_params(test_params)
            print(f"✓ Валидация прошла успешно: {validated_params}")
            
            # Проверяем ключевые ограничения
            assert validated_params['zscore_threshold'] > validated_params['zscore_exit'], \
                f"zscore_threshold ({validated_params['zscore_threshold']}) должен быть больше zscore_exit ({validated_params['zscore_exit']})"
            
            assert 0 < validated_params['risk_per_position_pct'] <= 1.0, \
                f"risk_per_position_pct должен быть в диапазоне (0, 1], получен: {validated_params['risk_per_position_pct']}"
            
            assert validated_params['max_active_positions'] > 0, \
                f"max_active_positions должен быть положительным, получен: {validated_params['max_active_positions']}"
            
        except ValueError as e:
            print(f"✗ ОШИБКА ВАЛИДАЦИИ: {e}")
            
            # Анализируем каждый параметр отдельно
            for key, value in test_params.items():
                try:
                    single_param = {key: value}
                    validate_params(single_param)
                    print(f"  ✓ {key}: {value} - OK")
                except ValueError as param_error:
                    print(f"  ✗ {key}: {value} - ОШИБКА: {param_error}")
            
            pytest.fail(f"Валидация параметров не прошла: {e}")
    
    def test_zscore_threshold_exit_relationship(self):
        """Проверяет корректность соотношения zscore_threshold и zscore_exit в search_space."""
        
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        if 'signals' in search_space:
            signals = search_space['signals']
            
            if 'zscore_threshold' in signals and 'zscore_exit' in signals:
                threshold_cfg = signals['zscore_threshold']
                exit_cfg = signals['zscore_exit']
                
                print(f"zscore_threshold config: {threshold_cfg}")
                print(f"zscore_exit config: {exit_cfg}")
                
                # Проверяем что минимальный threshold больше максимального exit
                min_threshold = threshold_cfg['low']
                max_exit = exit_cfg['high']
                
                print(f"min_threshold: {min_threshold}, max_exit: {max_exit}")
                
                if min_threshold <= max_exit:
                    print(f"⚠️  ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА: min zscore_threshold ({min_threshold}) <= max zscore_exit ({max_exit})")
                    print("   Это может привести к невалидным комбинациям параметров")
                    
                    # Тестируем проблемную комбинацию
                    problem_params = {
                        'zscore_threshold': min_threshold,
                        'zscore_exit': max_exit
                    }
                    
                    with pytest.raises(ValueError, match="z_entry должен быть больше z_exit"):
                        validate_params(problem_params)
                    
                    print("✓ Валидация корректно отклоняет невалидные комбинации")
                else:
                    print("✓ Диапазоны zscore_threshold и zscore_exit не пересекаются")
    
    def test_risk_parameters_ranges(self):
        """Проверяет разумность диапазонов риск-параметров."""
        
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        if 'portfolio' in search_space:
            portfolio = search_space['portfolio']
            
            # Проверяем risk_per_position_pct
            if 'risk_per_position_pct' in portfolio:
                risk_cfg = portfolio['risk_per_position_pct']
                print(f"risk_per_position_pct config: {risk_cfg}")
                
                assert risk_cfg['low'] > 0, f"risk_per_position_pct low должен быть > 0, получен: {risk_cfg['low']}"
                assert risk_cfg['high'] <= 1.0, f"risk_per_position_pct high должен быть <= 1.0, получен: {risk_cfg['high']}"
                assert risk_cfg['low'] < risk_cfg['high'], "risk_per_position_pct low должен быть < high"
                
                # Проверяем разумность диапазона (обычно 0.5% - 10%)
                if risk_cfg['low'] < 0.005:
                    print(f"⚠️  risk_per_position_pct low ({risk_cfg['low']}) очень мал (< 0.5%)")
                if risk_cfg['high'] > 0.15:
                    print(f"⚠️  risk_per_position_pct high ({risk_cfg['high']}) очень велик (> 15%)")
            
            # Проверяем max_position_size_pct
            if 'max_position_size_pct' in portfolio:
                size_cfg = portfolio['max_position_size_pct']
                print(f"max_position_size_pct config: {size_cfg}")
                
                assert size_cfg['low'] > 0, f"max_position_size_pct low должен быть > 0, получен: {size_cfg['low']}"
                assert size_cfg['high'] <= 1.0, f"max_position_size_pct high должен быть <= 1.0, получен: {size_cfg['high']}"
                assert size_cfg['low'] < size_cfg['high'], "max_position_size_pct low должен быть < high"
            
            # Проверяем max_active_positions
            if 'max_active_positions' in portfolio:
                pos_cfg = portfolio['max_active_positions']
                print(f"max_active_positions config: {pos_cfg}")
                
                assert pos_cfg['low'] > 0, f"max_active_positions low должен быть > 0, получен: {pos_cfg['low']}"
                assert pos_cfg['low'] < pos_cfg['high'], "max_active_positions low должен быть < high"
                
                # Проверяем разумность диапазона (обычно 3-50)
                if pos_cfg['low'] < 1:
                    print(f"⚠️  max_active_positions low ({pos_cfg['low']}) очень мал")
                if pos_cfg['high'] > 100:
                    print(f"⚠️  max_active_positions high ({pos_cfg['high']}) очень велик")
    
    def test_edge_case_parameter_combinations(self):
        """Тестирует граничные случаи комбинаций параметров."""
        
        # Тест 1: Минимальные значения
        min_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': -0.5,
            'risk_per_position_pct': 0.01,
            'max_position_size_pct': 0.03,
            'max_active_positions': 3,
            'stop_loss_multiplier': 2.0,
            'time_stop_multiplier': 1.0
        }
        
        try:
            validated_min = validate_params(min_params)
            print(f"✓ Минимальные параметры валидны: {validated_min}")
        except ValueError as e:
            print(f"✗ Минимальные параметры невалидны: {e}")
            pytest.fail(f"Минимальные параметры должны быть валидными: {e}")
        
        # Тест 2: Максимальные значения
        max_params = {
            'zscore_threshold': 2.2,
            'zscore_exit': 0.5,
            'risk_per_position_pct': 0.05,
            'max_position_size_pct': 0.15,
            'max_active_positions': 30,
            'stop_loss_multiplier': 5.0,
            'time_stop_multiplier': 4.0
        }
        
        try:
            validated_max = validate_params(max_params)
            print(f"✓ Максимальные параметры валидны: {validated_max}")
        except ValueError as e:
            print(f"✗ Максимальные параметры невалидны: {e}")
            pytest.fail(f"Максимальные параметры должны быть валидными: {e}")
        
        # Тест 3: Проблемная комбинация - zscore_threshold слишком близко к zscore_exit
        problem_params = {
            'zscore_threshold': 1.01,  # Очень близко к exit
            'zscore_exit': 1.0,
            'risk_per_position_pct': 0.02
        }
        
        try:
            validate_params(problem_params)
            print("⚠️  Проблемная комбинация прошла валидацию - возможно нужно ужесточить проверки")
        except ValueError as e:
            print(f"✓ Проблемная комбинация корректно отклонена: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
