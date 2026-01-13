"""Тесты валидации параметров и обработки ошибок в BasePairBacktester."""

import numpy as np
import pandas as pd
import pytest

from coint2.engine.base_engine import BasePairBacktester


class TestParameterValidation:
    """Тесты валидации параметров."""

    @pytest.mark.unit
    def test_parameter_validation_when_invalid_then_raises_error(self):
        """Test parameter validation in constructor."""
        SAMPLE_DATA = pd.DataFrame({"Y": [1, 2, 3], "X": [1, 2, 3]})
        VALID_ROLLING_WINDOW = 2
        VALID_Z_THRESHOLD = 1.0
        INVALID_Z_THRESHOLD = -1.0
        INVALID_Z_EXIT = 1.5  # Greater than z_threshold

        # Test invalid z_threshold
        try:
            BasePairBacktester(SAMPLE_DATA, rolling_window=VALID_ROLLING_WINDOW, z_threshold=INVALID_Z_THRESHOLD)
            assert False, "Should raise ValueError for negative z_threshold"
        except ValueError:
            pass

        # Test invalid z_exit >= z_threshold
        try:
            BasePairBacktester(SAMPLE_DATA, rolling_window=VALID_ROLLING_WINDOW, z_threshold=VALID_Z_THRESHOLD, z_exit=INVALID_Z_EXIT)
            assert False, "Should raise ValueError for z_exit >= z_threshold"
        except ValueError:
            pass
        
        # Test invalid take_profit vs stop_loss
        try:
            BasePairBacktester(
                SAMPLE_DATA, rolling_window=2, z_threshold=1.0,
                stop_loss_multiplier=2.0, take_profit_multiplier=3.0
            )
            assert False, "Should raise ValueError for take_profit >= stop_loss"
        except ValueError:
            pass

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_cost_validation_when_excessive_then_raises_error(self):
        """Проверяет валидацию общих торговых издержек."""
        data = pd.DataFrame({
            "Y": np.linspace(1, 10, 10),
            "X": np.linspace(1, 10, 10),
        })
        
        # Тест с нормальными издержками - должен пройти
        bt_normal = BasePairBacktester(
            data,
            rolling_window=3,
            z_threshold=1.0,
            commission_pct=0.001,
            slippage_pct=0.001,
            bid_ask_spread_pct_s1=0.001,
            bid_ask_spread_pct_s2=0.001,
            capital_at_risk=100.0,
            stop_loss_multiplier=2.0,
        )
        # Не должно быть исключения
        assert bt_normal is not None
        
        # Тест с чрезмерно высокими издержками - должен вызвать исключение
        try:
            bt_high_costs = BasePairBacktester(
                data,
                rolling_window=3,
                z_threshold=1.0,
                commission_pct=0.02,  # 2%
                slippage_pct=0.02,    # 2%
                bid_ask_spread_pct_s1=0.02,  # 2%
                bid_ask_spread_pct_s2=0.02,  # 2%
                capital_at_risk=100.0,
                stop_loss_multiplier=2.0,
            )
            assert False, "Должно быть исключение при чрезмерно высоких издержках"
        except ValueError as e:
            assert "Total trading costs" in str(e), "Должно быть сообщение о высоких издержках"


class TestDataHandling:
    """Тесты обработки данных."""

    @pytest.mark.unit
    def test_empty_data_when_provided_then_handled_correctly(self):
        """Test handling of empty or insufficient data."""
        # Константы для тестирования
        ROLLING_WINDOW_SMALL = 3
        ROLLING_WINDOW_LARGE = 5
        Z_THRESHOLD = 1.0
        MIN_DATA_SIZE = 6

        # Test empty data validation - now allowed for incremental mode
        empty_data = pd.DataFrame({"Y": [], "X": []})
        # Empty data should be allowed now (for incremental mode)
        bt_empty = BasePairBacktester(empty_data, rolling_window=ROLLING_WINDOW_SMALL, z_threshold=Z_THRESHOLD)
        # Should not raise exception during initialization

        # Test insufficient data
        small_data = pd.DataFrame({"Y": [1, 2], "X": [1, 2]})
        try:
            BasePairBacktester(small_data, rolling_window=ROLLING_WINDOW_LARGE, z_threshold=Z_THRESHOLD)
            assert False, "Should raise ValueError for insufficient data"
        except ValueError as e:
            assert "Data length" in str(e)

        # Test minimum valid data
        min_data = pd.DataFrame({
            "Y": list(range(1, MIN_DATA_SIZE + 1)),
            "X": list(range(1, MIN_DATA_SIZE + 1))
        })
        bt = BasePairBacktester(min_data, rolling_window=ROLLING_WINDOW_SMALL, z_threshold=Z_THRESHOLD)
        bt.run()  # Should not raise exceptions
        results = bt.get_results()
        assert "position" in results

    @pytest.mark.unit
    def test_division_by_zero_when_zero_values_then_protected(self):
        """Test protection against division by zero in position sizing."""
        # Create data that could cause division by zero
        N_SAMPLES = 7
        CONSTANT_Y_VALUE = 1
        ZERO_X_VALUE = 0

        data = pd.DataFrame({
            "Y": [CONSTANT_Y_VALUE] * N_SAMPLES,  # Constant values
            "X": [ZERO_X_VALUE] * N_SAMPLES      # Zero values
        })
        
        bt = BasePairBacktester(
            data,
            rolling_window=3,  # Minimum required rolling window
            z_threshold=0.5,
            z_exit=0.0,
            commission_pct=0.001,
            slippage_pct=0.001,
            capital_at_risk=100.0,
            stop_loss_multiplier=2.0,
            cooldown_periods=0,
        )
        
        # Should not raise any exceptions
        bt.run()
        results = bt.get_results()
        
        # All positions should be zero due to division by zero protection
        assert all(pos == 0.0 for pos in results["position"])

    @pytest.mark.unit
    def test_zero_std_when_spread_constant_then_handled_correctly(self):
        """Проверяет корректность работы при нулевом стандартном отклонении спреда."""
        # Создаем данные с постоянным спредом (нулевое стандартное отклонение)
        N_SAMPLES = 10
        COINTEGRATION_RATIO = 2

        data = pd.DataFrame({
            "Y": COINTEGRATION_RATIO * np.arange(1, N_SAMPLES + 1),
            "X": np.arange(1, N_SAMPLES + 1)
        })

        ROLLING_WINDOW = 3

        bt = BasePairBacktester(
            data,
            rolling_window=ROLLING_WINDOW,
            z_threshold=1.0,
            z_exit=0.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            capital_at_risk=100.0,
            stop_loss_multiplier=2.0,
            cooldown_periods=0,
        )
        bt.run()
        result = bt.get_results()

        # Проверяем, что система работает без ошибок
        assert isinstance(result["trades_log"], list)

        metrics = bt.get_performance_metrics()
        expected_zero_metrics = {
            "sharpe_ratio": 0.0, 
            "max_drawdown": 0.0, 
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "kelly_criterion": 0.0
        }
        # Проверяем только ожидаемые ключи, игнорируя дополнительные
        for key, expected_value in expected_zero_metrics.items():
            assert key in metrics, f"Ключ {key} отсутствует в метриках"
            assert np.isclose(metrics[key], expected_value), f"Метрика {key}: ожидалось {expected_value}, получено {metrics[key]}"
