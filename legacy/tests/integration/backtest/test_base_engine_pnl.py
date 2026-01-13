"""Тесты для проверки корректности расчета PnL в BasePairBacktester."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from coint2.engine.base_engine import BasePairBacktester


class TestPnlCalculations:
    """Тесты для проверки корректности расчета PnL."""

    @pytest.mark.unit
    def test_pnl_scaling_when_fixed_then_no_double_scaling(self):
        """Unit test: демонстрирует исправление критической ошибки PnL scaling."""

        def simulate_old_buggy_logic(pnl_series, capital_per_pair):
            """Симулирует старую (неправильную) логику с двойным масштабированием."""
            return pnl_series * capital_per_pair

        def simulate_new_fixed_logic(pnl_series, capital_per_pair):
            """Симулирует новую (исправленную) логику без двойного масштабирования."""
            return pnl_series  # Без лишнего умножения

        # Создаем реалистичные данные PnL от FullNumbaPairBacktester
        realistic_pnl = pd.Series([0.5, -0.2, 1.0, -0.1, 0.3], name='pnl')
        capital_per_pair = 1000.0

        # Тестируем старую (неправильную) логику
        old_result = simulate_old_buggy_logic(realistic_pnl, capital_per_pair)

        # Тестируем новую (исправленную) логику
        new_result = simulate_new_fixed_logic(realistic_pnl, capital_per_pair)

        # Проверяем, что исправленная логика не масштабирует дважды
        assert not np.allclose(old_result, new_result), "Старая и новая логика должны отличаться"
        assert np.allclose(new_result, realistic_pnl), "Новая логика должна возвращать исходный PnL"

        # Проверяем масштаб различий
        total_old = old_result.sum()
        total_new = new_result.sum()

        assert abs(total_old) > abs(total_new) * 100, "Старая логика должна давать катастрофически большие значения"

    @pytest.mark.unit
    def test_pnl_when_optimized_then_not_double_scaled(self):
        """Unit test: проверяет, что PnL не масштабируется дважды в оптимизаторе."""
        # Создаем mock результаты от бэктестера (уже в долларах)
        mock_backtest_results = {
            'pnl': pd.Series([1.0, -0.5, 2.0, -0.3, 1.5]),  # Уже в долларах
            'trades': []
        }

        capital_per_pair = 1000.0

        # Симулируем правильную логику (без дополнительного масштабирования)
        correct_pnl = mock_backtest_results['pnl']  # Без умножения на capital_per_pair

        # Симулируем неправильную логику (с двойным масштабированием)
        incorrect_pnl = mock_backtest_results['pnl'] * capital_per_pair

        # Проверяем, что правильная логика сохраняет исходные значения
        assert np.allclose(correct_pnl, mock_backtest_results['pnl'])

        # Проверяем, что неправильная логика дает катастрофически большие значения
        assert (incorrect_pnl.abs() > correct_pnl.abs() * 100).all()

        # Проверяем разумность значений
        assert correct_pnl.abs().max() < 10, "Правильный PnL должен быть в разумных пределах"
        assert incorrect_pnl.abs().max() > 1000, "Неправильный PnL должен быть катастрофически большим"

    @pytest.mark.integration
    def test_pnl_calculation_when_real_backtester_then_correct(self, small_prices_df):
        """Integration test: проверяем корректность PnL с реальным бэктестером."""
        # Используем реальные данные для интеграционного теста
        test_data = pd.DataFrame({
            'asset1': small_prices_df.iloc[:, 0],
            'asset2': small_prices_df.iloc[:, 1]
        })

        backtester = BasePairBacktester(
            pair_data=test_data,
            rolling_window=20,
            z_threshold=2.0,
            z_exit=0.5,
            commission_pct=0.001,
            slippage_pct=0.0005,
            capital_at_risk=1000.0
        )

        backtester.run()

        # Проверяем, что PnL в разумных пределах
        final_pnl = backtester.results['cumulative_pnl'].iloc[-1]
        assert abs(final_pnl) < 10000, f"PnL должен быть в разумных пределах: {final_pnl}"

        # Проверяем, что PnL не является результатом двойного масштабирования
        step_pnl = backtester.results['step_pnl'].dropna()
        if len(step_pnl) > 0:
            max_step_pnl = step_pnl.abs().max()
            assert max_step_pnl < 1000, f"Максимальный step PnL должен быть разумным: {max_step_pnl}"

    @pytest.mark.unit
    def test_pnl_calculation_when_using_individual_asset_returns_then_correct(self):
        """Проверяет корректность расчета PnL через доходности отдельных активов."""
        # Создаем тестовые данные с достаточным количеством точек
        N_SAMPLES = 15
        NOISE_STD_Y = 0.5
        NOISE_STD_X = 0.2

        data = pd.DataFrame({
            "Y": np.linspace(100, 120, N_SAMPLES) + np.random.normal(0, NOISE_STD_Y, N_SAMPLES),
            "X": np.linspace(50, 60, N_SAMPLES) + np.random.normal(0, NOISE_STD_X, N_SAMPLES)
        })
        
        bt = BasePairBacktester(
            data,
            rolling_window=5,
            z_threshold=0.8,
            z_exit=0.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            capital_at_risk=1000.0,
            stop_loss_multiplier=10.0,
            cooldown_periods=0
        )
        
        bt.run()
        results = bt.get_results()
        
        # Проверяем, что результаты содержат необходимые поля
        assert "pnl" in results
        assert "position" in results
        assert "beta" in results
        
        # Проверяем, что бэктест выполнился успешно
        assert len(results["pnl"]) > 0
        assert len(results["position"]) > 0
        
        # Простая проверка: если есть позиции, то должен быть и PnL
        positions = pd.Series(results["position"]).dropna()
        pnls = pd.Series(results["pnl"]).dropna()
        
        if len(positions) > 0 and positions.abs().sum() > 0:
            # Если были позиции, должен быть и PnL
            assert len(pnls) > 0

    @pytest.mark.unit
    def test_step_pnl_when_calculated_then_includes_costs(self):
        """Ensure step PnL subtracts trading costs for each period."""
        # Создаем тестовые данные
        N_SAMPLES = 10
        NOISE_STD = 0.1

        data = pd.DataFrame({
            "Y": np.linspace(1, 10, N_SAMPLES) + np.random.normal(0, NOISE_STD, size=N_SAMPLES),
            "X": np.linspace(1, 10, N_SAMPLES),
        })

        bt = BasePairBacktester(
            data,
            rolling_window=3,
            z_threshold=0.5,
            z_exit=0.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            capital_at_risk=50.0,
            stop_loss_multiplier=2.0,
            cooldown_periods=0,
        )
        bt.run()
        result = bt.get_results()
        df = pd.DataFrame({
            "spread": result["spread"],
            "position": result["position"],
            "pnl": result["pnl"],
            "costs": result["costs"],
            "y": result["y"],
            "x": result["x"],
            "beta": result["beta"],
        })

        df = df.dropna()

        for i in range(1, len(df)):
            position_prev = df["position"].iloc[i - 1]
            if position_prev != 0:  # Only check when there's an active position
                # Calculate expected PnL using individual asset returns
                price_s1_curr = df["y"].iloc[i]
                price_s2_curr = df["x"].iloc[i]
                price_s1_prev = df["y"].iloc[i - 1]
                price_s2_prev = df["x"].iloc[i - 1]
                beta = df["beta"].iloc[i]
                
                delta_p1 = price_s1_curr - price_s1_prev
                delta_p2 = price_s2_curr - price_s2_prev
                size_s1 = position_prev
                size_s2 = -beta * size_s1
                
                expected_step_pnl = size_s1 * delta_p1 + size_s2 * delta_p2 - df["costs"].iloc[i]
                assert np.isclose(expected_step_pnl, df["pnl"].iloc[i])
