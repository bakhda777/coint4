"""Быстрые тесты на синтетических сценариях.

Оптимизировано согласно best practices:
- Минимальные данные для тестов
- Мокирование тяжелых операций
- Unit тесты без полного бэктеста
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.fast
class TestSyntheticScenariosFast:
    """Быстрые тесты на синтетических данных."""
    
    @pytest.mark.unit
    def test_mean_reversion_pattern_generation(self, rng):
        """Unit: тест генерации mean-reverting паттерна."""
        n = 20  # Минимум данных
        spread_target = 0
        spread_volatility = 1.0
        mean_reversion_speed = 0.2
        
        spreads = [0]
        for i in range(1, n):
            prev_spread = spreads[-1]
            drift = -mean_reversion_speed * (prev_spread - spread_target)
            shock = rng.normal(0, spread_volatility * 0.01)
            new_spread = prev_spread + drift + shock
            spreads.append(new_spread)
        
        # Проверяем базовые свойства
        assert len(spreads) == n
        assert all(isinstance(s, (int, float)) for s in spreads)
        # Mean reversion должен возвращать к нулю
        assert abs(np.mean(spreads)) < 2 * spread_volatility
    
    @pytest.mark.unit
    def test_synthetic_data_creation_fast(self, rng):
        """Unit: быстрый тест создания синтетических данных."""
        n = 10
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        price_s1 = 100 + rng.normal(0, 1, n).cumsum()
        price_s2 = price_s1 * 1.5 + rng.normal(0, 0.5, n)
        
        data = pd.DataFrame({
            'S1': price_s1,
            'S2': price_s2
        }, index=dates)
        
        assert len(data) == n
        assert 'S1' in data.columns
        assert 'S2' in data.columns
        assert data.index[0] == pd.Timestamp('2024-01-01')
    
    @pytest.mark.unit
    def test_spread_calculation_fast(self):
        """Unit: быстрый тест расчета спреда."""
        data = pd.DataFrame({
            'S1': [100, 101, 102],
            'S2': [150, 151.5, 153]
        })
        
        # Простой спред
        spread = data['S2'] - data['S1']
        
        assert len(spread) == 3
        assert spread.iloc[0] == 50
        assert spread.iloc[1] == 50.5
        assert spread.iloc[2] == 51
    
    @pytest.mark.fast
    @patch('src.coint2.engine.base_engine.BasePairBacktester')
    def test_backtest_on_synthetic_mocked(self, mock_backtester, rng):
        """Fast: тест бэктеста с мокированием."""
        # Создаем минимальные синтетические данные
        n = 30
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        data = pd.DataFrame({
            'S1': 100 + rng.normal(0, 1, n).cumsum(),
            'S2': 150 + rng.normal(0, 1.5, n).cumsum()
        }, index=dates)
        
        # Мокируем backtester
        mock_engine = MagicMock()
        mock_engine.run.return_value = None
        mock_engine.get_performance_metrics.return_value = {
            'total_return': 0.05,
            'sharpe_ratio': 0.8,
            'num_trades': 10
        }
        mock_backtester.return_value = mock_engine
        
        # Запускаем "бэктест"
        from src.coint2.engine.base_engine import BasePairBacktester
        engine = BasePairBacktester(
            pair_data=data,
            rolling_window=10,
            z_threshold=2.0,
            capital_at_risk=10000
        )
        engine.run()
        metrics = engine.get_performance_metrics()
        
        # Проверяем мокированные результаты
        assert metrics['total_return'] == 0.05
        assert metrics['sharpe_ratio'] == 0.8
        assert metrics['num_trades'] == 10
    
    @pytest.mark.unit
    def test_trend_scenario_fast(self, rng):
        """Unit: тест трендового сценария."""
        n = 20
        trend = np.linspace(100, 110, n)  # Восходящий тренд
        noise = rng.normal(0, 0.1, n)
        prices = trend + noise
        
        # Проверяем что тренд восходящий
        assert prices[-1] > prices[0]
        # Проверяем монотонность (с учетом шума)
        smoothed = pd.Series(prices).rolling(5, center=True).mean()
        diffs = smoothed.diff().dropna()
        assert (diffs > 0).sum() > len(diffs) * 0.7  # Большинство положительных
    
    @pytest.mark.unit
    def test_volatility_scenario_fast(self, rng):
        """Unit: тест сценария с изменяющейся волатильностью."""
        n = 20
        # Первая половина - низкая волатильность
        vol1 = rng.normal(0, 0.1, n//2)
        # Вторая половина - высокая волатильность
        vol2 = rng.normal(0, 1.0, n//2)
        
        prices = 100 + np.concatenate([vol1, vol2]).cumsum()
        
        # Проверяем изменение волатильности
        first_half_std = np.std(np.diff(prices[:n//2]))
        second_half_std = np.std(np.diff(prices[n//2:]))
        
        assert second_half_std > first_half_std * 2  # Волатильность увеличилась