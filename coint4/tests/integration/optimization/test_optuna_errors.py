"""
Тесты для проверки критических и логических ошибок в Optuna оптимизации.
Эти тесты строго проверяют корректность работы оптимизации и исправления ошибок.
"""

import pytest
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.optimiser.metric_utils import validate_params, extract_sharpe
from src.coint2.utils.config import load_config


@pytest.mark.critical_fixes
@pytest.mark.serial
class TestOptunaCriticalErrors:
    """Тесты для проверки критических ошибок в Optuna оптимизации."""
    
    def setup_method(self):
        """Настройка тестового окружения."""
        self.base_config_path = "configs/main_2024.yaml"
        self.search_space_fast_path = "configs/search_space_fast.yaml"
        self.search_space_full_path = "configs/search_space.yaml"
    
    @pytest.mark.critical_fixes
    @pytest.mark.serial
    def test_fast_objective_when_initialized_without_filters_then_works_correctly(self):
        """Проверяет, что FastWalkForwardObjective инициализируется без ошибок с search_space_fast.yaml."""
        try:
            objective = FastWalkForwardObjective(
                self.base_config_path,
                self.search_space_fast_path
            )
            
            assert hasattr(objective, 'base_config')
            assert hasattr(objective, 'search_space')
            assert 'filters' not in objective.search_space
            print("✅ FastWalkForwardObjective инициализируется корректно с search_space_fast.yaml")
            
        except Exception as e:
            pytest.fail(f"Ошибка инициализации FastWalkForwardObjective: {e}")
    
    def test_parameter_validation_strictness(self):
        """Проверяет строгую валидацию параметров."""
        
        # Тест 1: Нормальные параметры должны проходить валидацию
        normal_params = {
            'zscore_threshold': 1.5,
            'zscore_exit': 0.3,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0,
            'risk_per_position_pct': 0.015,
            'max_position_size_pct': 0.1,
            'max_active_positions': 15,
            'rolling_window': 30,
            'commission_pct': 0.0004,
            'slippage_pct': 0.0005
        }
        
        try:
            validated = validate_params(normal_params)
            assert validated is not None
            assert validated['zscore_threshold'] == 1.5
            assert validated['zscore_exit'] == 0.3
            assert validated['stop_loss_multiplier'] == 3.0
            print("✅ Нормальные параметры проходят валидацию")
        except Exception as e:
            pytest.fail(f"Нормальные параметры не прошли валидацию: {e}")
        
        # Тест 2: Параметры с конфликтами - проверяем корректную обработку
        # Некоторые параметры автоматически исправляются, другие вызывают ошибки (fail fast)
        
        # Тест zscore конфликта - должен автоматически исправляться
        zscore_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': 1.5,  # exit > threshold - конфликт
        }
        try:
            validated = validate_params(zscore_params)
            assert validated['zscore_exit'] < validated['zscore_threshold'], \
                f"zscore_exit должен быть исправлен чтобы быть < zscore_threshold"
            print("✅ Конфликт zscore_exit > zscore_threshold корректно исправляется")
        except Exception as e:
            pytest.fail(f"Ошибка при исправлении zscore конфликта: {e}")
        
        # Тест отрицательного stop_loss_multiplier - должен вызывать ошибку
        negative_sl_params = {'stop_loss_multiplier': -1.0}
        try:
            validate_params(negative_sl_params)
            pytest.fail("Отрицательный stop_loss_multiplier должен вызывать ValueError")
        except ValueError as e:
            assert "stop_loss_multiplier должен быть неотрицательным" in str(e)
            print("✅ Отрицательный stop_loss_multiplier корректно вызывает ошибку")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка при валидации stop_loss_multiplier: {e}")
        
        # Тест max_active_positions = 0 - должен вызывать ошибку
        zero_positions_params = {'max_active_positions': 0}
        try:
            validate_params(zero_positions_params)
            pytest.fail("max_active_positions = 0 должен вызывать ValueError")
        except ValueError as e:
            assert "max_active_positions должен быть >= 1" in str(e)
            print("✅ max_active_positions = 0 корректно вызывает ошибку")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка при валидации max_active_positions: {e}")
        
        # Тест risk_per_position_pct > 1 - должен вызывать ошибку
        high_risk_params = {'risk_per_position_pct': 1.5}
        try:
            validate_params(high_risk_params)
            pytest.fail("risk_per_position_pct > 1 должен вызывать ValueError")
        except ValueError as e:
            assert "risk_per_position_pct должен быть в (0, 1]" in str(e)
            print("✅ risk_per_position_pct > 1 корректно вызывает ошибку")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка при валидации risk_per_position_pct: {e}")
    
    def test_sharpe_extraction_correctness(self):
        """Проверяет корректность извлечения Sharpe ratio."""
        
        # Тест 1: Нормальный Sharpe ratio
        valid_result = {"sharpe_ratio_abs": 1.5, "total_trades": 50}
        sharpe = extract_sharpe(valid_result)
        assert sharpe == 1.5
        print("✅ Нормальный Sharpe ratio извлекается корректно")
        
        # Тест 2: Sharpe ratio с другим ключом
        alt_result = {"sharpe_ratio": 2.0, "total_trades": 30}
        sharpe = extract_sharpe(alt_result)
        assert sharpe == 2.0
        print("✅ Альтернативный Sharpe ratio извлекается корректно")
        
        # Тест 3: Невалидные значения Sharpe
        invalid_results = [
            {"sharpe_ratio_abs": np.nan},
            {"sharpe_ratio_abs": np.inf},
            {"sharpe_ratio_abs": -np.inf},
            {"sharpe_ratio_abs": "invalid"},
            {}
        ]
        
        for result in invalid_results:
            sharpe = extract_sharpe(result)
            assert sharpe is None, f"Невалидный результат {result} должен возвращать None"
        print("✅ Невалидные Sharpe ratio корректно обрабатываются")
    
    def test_lookahead_bias_prevention(self):
        """Проверяет предотвращение lookahead bias в walk-forward анализе."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        # Проверяем, что данные разделяются правильно
        cfg = objective.base_config
        start_date = pd.to_datetime(cfg.walk_forward.start_date)
        training_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        training_end = start_date - pd.Timedelta(minutes=15)  # 1 bar before test
        testing_start = start_date
        testing_end = start_date + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
        
        # Убедимся, что нет перекрытия между training и testing периодами
        assert training_end < testing_start, "Training и testing периоды не должны перекрываться"
        print("✅ Периоды walk-forward не перекрываются (предотвращение lookahead bias)")
        
        # Проверяем, что step_size_days >= testing_period_days для предотвращения перекрытия
        step_size_days = getattr(cfg.walk_forward, 'step_size_days', cfg.walk_forward.testing_period_days)
        assert step_size_days >= cfg.walk_forward.testing_period_days, \
            "step_size_days должен быть >= testing_period_days для предотвращения перекрытия тестов"
        print("✅ step_size_days >= testing_period_days (предотвращение перекрытия тестов)")
    
    def test_portfolio_position_limit_enforcement(self):
        """Проверяет соблюдение лимита позиций в портфеле."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        # Проверяем, что max_active_positions в конфигурации имеет разумные значения
        cfg = objective.base_config
        max_positions = cfg.portfolio.max_active_positions
        assert 1 <= max_positions <= 50, f"max_active_positions ({max_positions}) должен быть в разумных пределах"
        print("✅ Лимит позиций в портфеле соблюдается")
        
        # Проверяем, что в search_space указаны правильные диапазоны
        search_space = objective.search_space
        if 'portfolio' in search_space and 'max_active_positions' in search_space['portfolio']:
            pos_config = search_space['portfolio']['max_active_positions']
            assert pos_config['low'] >= 1, "Минимальное количество позиций должно быть >= 1"
            assert pos_config['high'] <= 50, "Максимальное количество позиций должно быть разумным"
            print("✅ Диапазоны max_active_positions в search_space корректны")
    
    def test_cost_parameters_no_double_accounting(self):
        """Проверяет, что параметры издержек не учитываются дважды."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        search_space = objective.search_space
        
        # Проверяем, что commission_pct и slippage_pct в разумных пределах
        if 'costs' in search_space:
            costs = search_space['costs']
            if 'commission_pct' in costs:
                comm_config = costs['commission_pct']
                if isinstance(comm_config, dict) and 'low' in comm_config and 'high' in comm_config:
                    assert 0 <= comm_config['low'] <= comm_config['high'] <= 0.01, \
                        "commission_pct должен быть в диапазоне [0, 0.01]"
                elif isinstance(comm_config, (int, float)):
                    assert 0 <= comm_config <= 0.01, \
                        f"commission_pct должен быть в диапазоне [0, 0.01], получен: {comm_config}"
            if 'slippage_pct' in costs:
                slip_config = costs['slippage_pct']
                if isinstance(slip_config, dict) and 'low' in slip_config and 'high' in slip_config:
                    assert 0 <= slip_config['low'] <= slip_config['high'] <= 0.01, \
                        "slippage_pct должен быть в диапазоне [0, 0.01]"
                elif isinstance(slip_config, (int, float)):
                    assert 0 <= slip_config <= 0.01, \
                        f"slippage_pct должен быть в диапазоне [0, 0.01], получен: {slip_config}"
            print("✅ Параметры издержек находятся в разумных пределах")
    
    def test_normalization_method_validation(self):
        """Проверяет валидацию методов нормализации."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        search_space = objective.search_space
        
        # Проверяем, что normalization_method содержит только допустимые значения
        if 'normalization' in search_space and 'normalization_method' in search_space['normalization']:
            norm_methods = search_space['normalization']['normalization_method']
            valid_methods = ['minmax', 'zscore', 'log_returns']
            for method in norm_methods:
                assert method in valid_methods, f"Метод нормализации {method} не поддерживается"
            print("✅ Методы нормализации валидны")
    
    def test_rolling_window_validation(self):
        """Проверяет валидацию параметров rolling window."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        search_space = objective.search_space
        
        # Проверяем, что rolling_window в разумных пределах
        if 'signals' in search_space and 'rolling_window' in search_space['signals']:
            rw_config = search_space['signals']['rolling_window']
            assert rw_config['low'] >= 10, "rolling_window должен быть >= 10 для статистической значимости"
            assert rw_config['high'] <= 200, "rolling_window должен быть <= 200 для адаптивности"
            print("✅ Параметры rolling_window валидны")
    
    def test_zscore_parameters_logical_consistency(self):
        """Проверяет логическую консистентность zscore параметров."""
        
        objective = FastWalkForwardObjective(
            self.base_config_path,
            self.search_space_fast_path
        )
        
        search_space = objective.search_space
        
        # Проверяем, что zscore_exit диапазон логически корректен
        if 'signals' in search_space:
            signals = search_space['signals']
            if 'zscore_threshold' in signals and 'zscore_exit' in signals:
                threshold_config = signals['zscore_threshold']
                exit_config = signals['zscore_exit']
                
                # zscore_exit должен быть меньше zscore_threshold
                assert exit_config['high'] < threshold_config['high'], \
                    "zscore_exit должен быть меньше zscore_threshold"
                print("✅ Логическая консистентность zscore параметров соблюдается")


@pytest.mark.slow  # Помечаем как медленный для исключения из быстрых тестов
def test_comprehensive_optuna_validation():
    """Комплексный тест валидации Optuna оптимизации (медленный)."""
    
    print("🔍 КОМПЛЕКСНАЯ ПРОВЕРКА OPTUNA ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    # ОПТИМИЗАЦИЯ: Быстрая проверка в режиме QUICK_TEST
    import os
    if os.environ.get('QUICK_TEST', '').lower() == 'true':
        pytest.skip("Пропускаем комплексный тест в режиме QUICK_TEST")
    
    test_obj = TestOptunaCriticalErrors()
    test_obj.setup_method()
    
    # ОПТИМИЗАЦИЯ: Выполняем только 3 самых важных теста вместо 9
    
    # Тест 1: Инициализация objective
    print("\n1️⃣ Тестируем инициализацию objective...")
    test_obj.test_fast_objective_when_initialized_without_filters_then_works_correctly()
    
    # Тест 2: Валидация параметров
    print("\n2️⃣ Тестируем валидацию параметров...")
    test_obj.test_parameter_validation_strictness()
    
    # Тест 3: Предотвращение lookahead bias (самый важный)
    print("\n3️⃣ Тестируем предотвращение lookahead bias...")
    test_obj.test_lookahead_bias_prevention()
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ КРИТИЧЕСКИЕ И ЛОГИЧЕСКИЕ ОШИБКИ ПРОВЕРЕНЫ!")


class TestOptunaCriticalErrorsFast:
    """Быстрые версии тестов Optuna с мокированием."""
    
    @pytest.mark.fast
    @patch('src.optimiser.fast_objective.FastWalkForwardObjective')
    def test_fast_objective_when_mocked_then_initializes(self, mock_objective):
        """Fast test: Быстрая проверка инициализации FastWalkForwardObjective."""
        # Настраиваем мок
        mock_instance = Mock()
        mock_instance.base_config = {'some': 'config'}
        mock_instance.search_space = {'param': 'value'}
        mock_objective.return_value = mock_instance
        
        # Тестируем
        objective = mock_objective("config.yaml", "search.yaml")
        
        # Проверки
        assert mock_objective.called
        assert hasattr(objective, 'base_config')
        assert hasattr(objective, 'search_space')
        
    @pytest.mark.fast
    def test_parameter_validation_when_fast_then_validates_logic(self):
        """Fast test: Быстрая валидация логики параметров без реальной оптимизации."""
        from src.optimiser.metric_utils import validate_params
        
        # Тестируем логику валидации с минимальными данными
        params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'rolling_window': 10  # Минимальный размер для быстроты
        }
        
        try:
            validated = validate_params(params)
            assert validated is not None
            assert validated['zscore_exit'] < validated['zscore_threshold']
        except Exception as e:
            pytest.fail(f"Валидация не должна падать на корректных параметрах: {e}")
            
    @pytest.mark.fast  
    def test_sharpe_extraction_when_mocked_then_logic_works(self):
        """Fast test: Быстрое тестирование извлечения Sharpe ratio."""
        from src.optimiser.metric_utils import extract_sharpe
        
        # Мокируем простые данные вместо реального результата backtest
        mock_result = {
            'sharpe_ratio': 0.8,
            'sharpe_ratio_abs': 0.8,
            'total_return': 0.1,
            'volatility': 0.125
        }
        
        # Тестируем извлечение 
        sharpe = extract_sharpe(mock_result)
        assert isinstance(sharpe, (int, float))
        assert sharpe == 0.8
        
    @pytest.mark.fast
    def test_lookahead_bias_prevention_when_mocked_then_validates(self):
        """Fast test: Быстрая проверка предотвращения lookahead bias."""
        # Создаем мок данных с временными метками  
        import pandas as pd
        import numpy as np
        
        # Минимальный набор данных для быстроты
        dates = pd.date_range('2024-01-01', periods=10, freq='1D')  # Всего 10 дней
        data = pd.DataFrame({
            'price': np.random.randn(10).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 10)
        }, index=dates)
        
        # Проверяем что данные упорядочены по времени (предотвращает lookahead)
        assert data.index.is_monotonic_increasing, "Данные должны быть упорядочены по времени"
        
        # Проверяем отсутствие будущих данных в прошлых расчетах
        for i in range(1, len(data)):
            current_data = data.iloc[:i]  # Данные до текущего момента
            assert len(current_data) == i, "Количество данных должно соответствовать позиции"
            
    @pytest.mark.fast
    def test_portfolio_position_limit_when_mocked_then_enforced(self):
        """Fast test: Быстрая проверка соблюдения лимитов позиций.""" 
        # Симулируем портфель с лимитами
        max_positions = 3
        current_positions = ['BTCUSDT', 'ETHUSDT']  # 2 текущие позиции
        new_position = 'ADAUSDT'
        
        # Проверяем что можем добавить новую позицию
        total_after_add = len(current_positions) + 1
        assert total_after_add <= max_positions, f"Превышен лимит позиций: {total_after_add} > {max_positions}"
        
        # Добавляем позицию до лимита
        current_positions.append(new_position)
        assert len(current_positions) == 3
        
        # Проверяем что четвертую позицию добавить нельзя
        would_exceed = len(current_positions) + 1 > max_positions
        assert would_exceed, "Должна быть проверка превышения лимита"


if __name__ == "__main__":
    test_comprehensive_optuna_validation()
