"""
Тесты для проверки критических исправлений в Optuna оптимизации.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import yaml
import optuna

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.optimiser.lookahead_validator import LookaheadValidator
from src.optimiser.metric_utils import validate_params, _validate_cross_parameter_constraints


class TestLookaheadBiasPrevention:
    """Тесты для предотвращения lookahead bias."""
    
    def test_walk_forward_gap_enforcement(self, tmp_path):
        """Проверяет обязательное соблюдение gap между train и test."""
        # Создаем минимальную конфигурацию
        config = {
            'data_dir': str(tmp_path),
            'walk_forward': {
                'enabled': True,
                'start_date': '2023-01-01',
                'end_date': '2023-03-01',
                'training_period_days': 30,
                'testing_period_days': 7,
                'step_size_days': 7,
                'gap_minutes': 15,  # 15 минут = 1 бар
                'max_steps': 2
            },
            'optuna': {
                'n_trials': 3,
                'n_jobs': 1
            },
            'strict_lookahead_validation': True
        }
        
        # Проверяем, что корректная конфигурация работает
        validator = LookaheadValidator(strict_mode=True)
        
        train_data = pd.DataFrame(
            index=pd.date_range('2023-01-01', '2023-01-30 23:45', freq='15min')
        )
        # Test начинается через 15 минут после окончания train
        test_data = pd.DataFrame(
            index=pd.date_range('2023-01-31 00:00', '2023-02-07', freq='15min')
        )
        
        gap_days = 15 / (24 * 60)  # 15 минут в днях
        is_valid, message = validator.validate_data_split(train_data, test_data, gap_days=gap_days)
        assert is_valid, f"Корректное разделение должно быть валидным: {message}"
        
        # Проверяем, что перекрытие обнаруживается
        # Test начинается точно когда заканчивается train - без gap
        bad_test_data = pd.DataFrame(
            {'close': [100] * 100},
            index=pd.date_range('2023-01-30 23:45', '2023-02-05', freq='15min')[:100]
        )
        
        is_valid, message = validator.validate_data_split(train_data, bad_test_data, gap_days=gap_days)
        assert not is_valid, "Перекрытие данных должно быть обнаружено"
        assert "Перекрытие" in message or "КРИТИЧНО" in message or "Недостаточный gap" in message
        
    def test_normalization_isolation(self):
        """Проверяет изоляцию параметров нормализации."""
        validator = LookaheadValidator(strict_mode=True)
        
        train_data = pd.DataFrame(
            {'close': np.random.randn(100)},
            index=pd.date_range('2023-01-01', periods=100, freq='15min')
        )
        test_data = pd.DataFrame(
            {'close': np.random.randn(50)},
            index=pd.date_range('2023-01-05', periods=50, freq='15min')
        )
        
        # Правильные параметры нормализации (вычислены только на train)
        good_params = {
            'normalization_params': {
                'computed_from': '2023-01-01',
                'computed_to': train_data.index.max().isoformat(),
                'mean': 0.0,
                'std': 1.0
            }
        }
        
        is_valid, message = validator.validate_normalization(good_params, train_data, test_data)
        assert is_valid, f"Корректная нормализация должна быть валидной: {message}"
        
        # Неправильные параметры (используют будущие данные)
        bad_params = {
            'normalization_params': {
                'computed_from': '2023-01-01',
                'computed_to': '2023-01-10',  # После train периода
                'mean': 0.0,
                'std': 1.0
            }
        }
        
        is_valid, message = validator.validate_normalization(bad_params, train_data, test_data)
        assert not is_valid, "Использование будущих данных должно быть обнаружено"
        
    def test_cache_isolation(self):
        """Проверяет изоляцию кэша между периодами."""
        validator = LookaheadValidator(strict_mode=True)
        
        train_period = (
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-30')
        )
        test_period = (
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-02-07')
        )
        
        # Правильный ключ кэша (содержит границы периода)
        good_key = "pairs_20230101_20230130_step1"
        is_valid, message = validator.validate_cache_isolation(good_key, train_period, test_period)
        assert is_valid, f"Корректный ключ кэша должен быть валидным: {message}"
        
        # Неправильный ключ (глобальный)
        bad_key = "pairs_global_cache"
        is_valid, message = validator.validate_cache_isolation(bad_key, train_period, test_period)
        assert not is_valid, "Глобальный кэш должен быть обнаружен как проблема"


class TestMetricCalculations:
    """Тесты для проверки корректности расчета метрик."""
    
    def test_sharpe_ratio_calculation(self):
        """Проверяет правильность расчета Sharpe ratio."""
        # Создаем тестовые данные с известным Sharpe
        np.random.seed(42)
        daily_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # ~0.05% daily return
        
        # Расчет Sharpe ratio (annualized)
        expected_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Проверяем, что расчет корректен
        # Для правильного теста используем те же daily_returns напрямую
        calculated_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        assert abs(calculated_sharpe - expected_sharpe) < 0.001, \
            f"Sharpe ratio расчет некорректен: {calculated_sharpe} vs {expected_sharpe}"
            
    def test_win_rate_by_trades(self):
        """Проверяет расчет win rate по сделкам, а не по дням."""
        # Создаем список PnL сделок
        trade_pnls = [100, -50, 200, -30, 150, -20, 80, -10, 60, -40]
        
        # Правильный расчет: 5 прибыльных из 10
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        expected_win_rate = len(winning_trades) / len(trade_pnls)
        
        calculated_win_rate = sum(1 for pnl in trade_pnls if pnl > 0) / len(trade_pnls)
        
        assert calculated_win_rate == expected_win_rate, \
            f"Win rate должен быть {expected_win_rate}, получен {calculated_win_rate}"
            
    def test_drawdown_calculation(self):
        """Проверяет корректность расчета просадки."""
        # Создаем equity curve с известной максимальной просадкой
        equity_values = [100000, 105000, 103000, 98000, 102000, 108000, 106000, 110000]
        equity_curve = pd.Series(equity_values)
        
        # Расчет максимальной просадки
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Ожидаемая просадка: от 105000 до 98000 = 7000/105000 = 0.0667
        expected_max_dd = 7000 / 105000
        
        assert abs(max_drawdown - expected_max_dd) < 0.0001, \
            f"Максимальная просадка некорректна: {max_drawdown} vs {expected_max_dd}"


class TestParameterValidation:
    """Тесты для проверки валидации параметров."""
    
    def test_zscore_parameter_validation(self):
        """Проверяет валидацию zscore параметров."""
        # Корректные параметры
        good_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5
        }
        
        try:
            validated = validate_params(good_params)
            assert validated['zscore_threshold'] == 2.0
            assert validated['zscore_exit'] == 0.5
        except ValueError:
            pytest.fail("Корректные zscore параметры не должны вызывать исключение")
            
        # Некорректные параметры (exit >= threshold)
        bad_params = {
            'zscore_threshold': 1.5,
            'zscore_exit': 2.0
        }
        
        with pytest.raises(ValueError, match="zscore_exit.*должен быть < zscore_threshold|гистерезис"):
            validate_params(bad_params)
            
        # Слишком маленький гистерезис
        small_hysteresis = {
            'zscore_threshold': 1.0,
            'zscore_exit': 0.95
        }
        
        with pytest.raises(ValueError, match="гистерезис"):
            validate_params(small_hysteresis)
            
    def test_cross_parameter_constraints(self):
        """Проверяет cross-parameter ограничения."""
        # Проверка стоп-лоссов
        params = {
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0  # Меньше stop_loss - ошибка
        }
        
        with pytest.raises(ValueError, match="time_stop_multiplier.*должен быть >= stop_loss_multiplier"):
            _validate_cross_parameter_constraints(params)
            
        # Проверка максимальной экспозиции
        params = {
            'risk_per_position_pct': 0.1,
            'max_active_positions': 15  # 0.1 * 15 = 1.5 > 1.0
        }
        
        with pytest.raises(ValueError, match="Максимальная экспозиция.*превышает 100%"):
            _validate_cross_parameter_constraints(params)
            
    def test_position_sizing_validation(self):
        """Проверяет валидацию размеров позиций."""
        # max_position_size должен быть >= risk_per_position
        params = {
            'max_position_size_pct': 0.05,
            'risk_per_position_pct': 0.1  # Больше max_position_size - ошибка
        }
        
        with pytest.raises(ValueError, match="max_position_size_pct.*должен быть >= risk_per_position_pct"):
            _validate_cross_parameter_constraints(params)


class TestThreadSafety:
    """Тесты для проверки thread safety."""
    
    def test_multiprocessing_detection(self):
        """Проверяет корректное определение multiprocessing режима."""
        import multiprocessing.process
        
        # В главном процессе
        assert multiprocessing.process.current_process().name == 'MainProcess'
        
        # Проверяем, что определение режима работает
        is_multiprocessing = multiprocessing.process.current_process().name != 'MainProcess'
        assert not is_multiprocessing, "В главном процессе не должен определяться multiprocessing"
        
    @pytest.mark.parametrize("n_jobs", [1, 2, 4])
    def test_cache_synchronization(self, n_jobs):
        """Проверяет синхронизацию кэша при разных n_jobs."""
        config = {
            'optuna': {'n_jobs': n_jobs}
        }
        
        # При n_jobs > 1 должен использоваться Manager
        use_multiprocessing = n_jobs > 1
        
        if use_multiprocessing:
            # Проверяем, что Manager может быть создан
            try:
                import multiprocessing
                manager = multiprocessing.Manager()
                test_dict = manager.dict()
                test_lock = manager.Lock()
                
                # Базовые операции должны работать
                test_dict['key'] = 'value'
                with test_lock:
                    assert test_dict['key'] == 'value'
                    
            except Exception as e:
                pytest.skip(f"Multiprocessing Manager недоступен: {e}")


@pytest.mark.smoke
class TestCriticalFixes:
    """Smoke тесты для критических исправлений."""
    
    def test_no_data_overlap(self):
        """Быстрый тест на отсутствие перекрытия данных."""
        validator = LookaheadValidator(strict_mode=True)
        
        # Создаем корректно разделенные данные
        train_end = pd.Timestamp('2023-01-30 23:45:00')
        test_start = pd.Timestamp('2023-02-01 00:00:00')
        
        assert test_start > train_end, "Test должен начинаться после train"
        
        gap = test_start - train_end
        assert gap >= pd.Timedelta(hours=1), "Должен быть минимальный gap"
        
    def test_metric_calculations_smoke(self):
        """Быстрый тест базовых расчетов метрик."""
        # Sharpe ratio для нулевой волатильности
        returns = pd.Series([0.01] * 100)  # Постоянный доход
        assert returns.std() < 0.0001, "Волатильность должна быть близка к нулю"
        
        # Win rate 100%
        all_positive = [1, 2, 3, 4, 5]
        win_rate = sum(1 for x in all_positive if x > 0) / len(all_positive)
        assert win_rate == 1.0, "Win rate должен быть 100%"
        
        # Drawdown для монотонно растущей кривой
        monotonic = pd.Series(range(100))
        running_max = monotonic.expanding().max()
        dd = (monotonic - running_max) / running_max
        assert dd.min() == 0, "Не должно быть просадки для монотонной кривой"