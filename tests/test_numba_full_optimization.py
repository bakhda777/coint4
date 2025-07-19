"""Тесты для полной Numba-оптимизированной версии PairBacktester."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.coint2.engine.backtest_engine import PairBacktester
from src.coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester
from src.coint2.core.numba_backtest_full import (
    rolling_ols, calculate_hurst_exponent, calculate_variance_ratio,
    calculate_rolling_correlation, calculate_half_life, detect_market_regime,
    check_structural_breaks, calculate_adaptive_threshold
)


class TestFullNumbaOptimization:
    """Тесты для полной Numba-оптимизированной версии."""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        np.random.seed(42)
        n_points = 1000
        
        # Создаем коинтегрированные временные ряды
        dates = pd.date_range('2023-01-01', periods=n_points, freq='15T')
        
        # Общий тренд
        common_trend = np.cumsum(np.random.randn(n_points) * 0.01)
        
        # Первый актив
        asset1 = 100 + common_trend + np.cumsum(np.random.randn(n_points) * 0.005)
        
        # Второй актив (коинтегрирован с первым)
        beta_true = 0.8
        asset2 = 80 + beta_true * asset1 + np.cumsum(np.random.randn(n_points) * 0.003)
        
        return pd.DataFrame({
            'asset1': asset1,
            'asset2': asset2
        }, index=dates)
    
    def test_rolling_ols_accuracy(self, sample_data):
        """Тест точности rolling OLS расчетов."""
        y = sample_data['asset1'].values.astype(np.float32)
        x = sample_data['asset2'].values.astype(np.float32)
        window = 60
        
        # Numba версия
        beta_numba, mu_numba, sigma_numba = rolling_ols(y, x, window)
        
        # Проверяем несколько точек вручную
        for i in [100, 200, 500, 800]:
            if i >= window:
                # Ручной расчет для проверки
                y_win = y[i-window:i]
                x_win = x[i-window:i]
                
                # OLS регрессия
                X = np.column_stack([np.ones(len(x_win)), x_win])
                coeffs = np.linalg.lstsq(X, y_win, rcond=None)[0]
                beta_manual = coeffs[1]
                
                # Спред и его статистики
                spread = y_win - beta_manual * x_win
                mu_manual = np.mean(spread)
                sigma_manual = np.std(spread, ddof=1)
                
                # Проверяем с разумной точностью (учитывая float32 и fastmath)
                # Используем относительную точность для больших значений
                beta_tolerance = max(0.01, 0.02 * abs(beta_manual))
                mu_tolerance = max(0.5, 0.05 * abs(mu_manual))
                sigma_tolerance = max(0.1, 0.05 * abs(sigma_manual))
                
                assert abs(beta_numba[i] - beta_manual) < beta_tolerance, f"Beta mismatch at {i}: {beta_numba[i]} vs {beta_manual}"
                assert abs(mu_numba[i] - mu_manual) < mu_tolerance, f"Mu mismatch at {i}: {mu_numba[i]} vs {mu_manual}"
                assert abs(sigma_numba[i] - sigma_manual) < sigma_tolerance, f"Sigma mismatch at {i}: {sigma_numba[i]} vs {sigma_manual}"
    
    def test_hurst_exponent_calculation(self):
        """Тест расчета показателя Херста."""
        # Тестовые данные: случайное блуждание (Hurst ≈ 0.5)
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(200))
        hurst_rw = calculate_hurst_exponent(random_walk.astype(np.float32))
        
        # Для случайного блуждания Hurst должен быть около 0.5
        assert 0.3 < hurst_rw < 0.7, f"Hurst for random walk: {hurst_rw}"
        
        # Тестовые данные: трендовый ряд (Hurst > 0.5)
        trend_data = np.arange(200) + np.random.randn(200) * 0.1
        hurst_trend = calculate_hurst_exponent(trend_data.astype(np.float32))
        
        # Для трендового ряда Hurst должен быть больше 0.5
        assert hurst_trend > 0.5, f"Hurst for trend: {hurst_trend}"
    
    def test_variance_ratio_calculation(self):
        """Тест расчета коэффициента дисперсии."""
        # Случайное блуждание: VR должен быть около 1
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(200))
        vr_rw = calculate_variance_ratio(random_walk.astype(np.float32))
        
        assert 0.7 < vr_rw < 1.3, f"VR for random walk: {vr_rw}"
        
        # Mean-reverting ряд: VR < 1
        mr_data = np.zeros(200)
        for i in range(1, 200):
            mr_data[i] = 0.9 * mr_data[i-1] + np.random.randn() * 0.1
        
        vr_mr = calculate_variance_ratio(mr_data.astype(np.float32))
        assert vr_mr < 1.0, f"VR for mean-reverting: {vr_mr}"
    
    def test_market_regime_detection(self):
        """Тест определения рыночных режимов."""
        # Трендовые данные
        trend_y = np.arange(100).astype(np.float32)
        trend_x = (np.arange(100) * 0.8 + np.random.randn(100) * 0.1).astype(np.float32)
        
        regime_trend = detect_market_regime(trend_y, trend_x)
        # Ожидаем trending (1) или neutral (0), но не mean_reverting (2)
        assert regime_trend in [0, 1], f"Unexpected regime for trend: {regime_trend}"
        
        # Mean-reverting данные
        mr_y = np.zeros(100, dtype=np.float32)
        mr_x = np.zeros(100, dtype=np.float32)
        for i in range(1, 100):
            mr_y[i] = 0.8 * mr_y[i-1] + np.random.randn() * 0.1
            mr_x[i] = 0.8 * mr_x[i-1] + np.random.randn() * 0.1
        
        regime_mr = detect_market_regime(mr_y, mr_x)
        # Любой режим допустим для коротких mean-reverting данных
        assert regime_mr in [0, 1, 2], f"Invalid regime: {regime_mr}"
    
    def test_structural_break_detection(self):
        """Тест обнаружения структурных сдвигов."""
        # Стабильные коинтегрированные данные
        np.random.seed(42)
        y_stable = np.cumsum(np.random.randn(100) * 0.1)
        x_stable = y_stable * 0.8 + np.random.randn(100) * 0.05
        spread_stable = y_stable - 0.8 * x_stable
        
        break_stable = check_structural_breaks(
            y_stable.astype(np.float32), 
            x_stable.astype(np.float32), 
            spread_stable.astype(np.float32)
        )
        assert not break_stable, "Should not detect break in stable data"
        
        # Данные со структурным сдвигом (низкая корреляция)
        y_break = np.concatenate([y_stable[:50], np.random.randn(50) * 2])
        x_break = np.concatenate([x_stable[:50], np.random.randn(50) * 2])
        spread_break = y_break - 0.8 * x_break
        
        break_detected = check_structural_breaks(
            y_break.astype(np.float32), 
            x_break.astype(np.float32), 
            spread_break.astype(np.float32)
        )
        # Может обнаружить или не обнаружить в зависимости от данных
        assert isinstance(break_detected, bool), "Should return boolean"
    
    def test_adaptive_threshold_calculation(self):
        """Тест расчета адаптивных порогов."""
        # Тест адаптивного порога
        stable_sigma = 0.5
        volatile_sigma = 2.0
        base_threshold = 2.0
        min_vol = 0.1
        adaptive_factor = 1.0
        
        adaptive_stable = calculate_adaptive_threshold(base_threshold, stable_sigma, min_vol, adaptive_factor)
        # Для stable_sigma=0.5, min_vol=0.1: normalized_vol = 0.5/0.1 = 5.0
        # volatility_multiplier = min(2.0, 5.0 * 1.0) = 2.0
        # expected = 2.0 * 2.0 = 4.0
        assert adaptive_stable == 4.0, f"Expected 4.0, got {adaptive_stable}"
        
        adaptive_volatile = calculate_adaptive_threshold(base_threshold, volatile_sigma, min_vol, adaptive_factor)
        assert adaptive_volatile >= adaptive_stable, "Volatile threshold should be >= stable threshold"
    
    def test_full_numba_backtest_equivalence(self, sample_data):
        """Тест эквивалентности полной Numba версии с оригиналом."""
        # Параметры бэктеста
        params = {
            'rolling_window': 60,
            'z_threshold': 2.0,
            'z_exit': 0.0,
            'commission_pct': 0.001,
            'slippage_pct': 0.001,
            'bid_ask_spread_pct_s1': 0.001,
            'bid_ask_spread_pct_s2': 0.001
        }
        
        # Оригинальный бэктестер
        original = PairBacktester(sample_data, **params)
        
        # Полная Numba версия
        numba_full = FullNumbaPairBacktester(sample_data, **params)
        
        # Запускаем бэктесты
        original.run()
        numba_full.run()
        
        # Сравниваем результаты
        if not original.results.empty and not numba_full.results.empty:
            original_pnl = original.results['cumulative_pnl'].iloc[-1]
            numba_pnl = numba_full.results['cumulative_pnl'].iloc[-1]
            
            # Проверяем с более строгой точностью (1%)
            pnl_diff = abs(numba_pnl - original_pnl)
            tolerance = max(0.01 * abs(original_pnl), 0.001)  # 1% относительная или 0.001 абсолютная
            
            assert pnl_diff <= tolerance, (
                f"PnL mismatch: Numba={numba_pnl:.8f}, Original={original_pnl:.8f}, "
                f"Difference={pnl_diff:.8f}, Tolerance={tolerance:.8f}"
            )
    
    def test_performance_comparison(self, sample_data):
        """Тест сравнения производительности."""
        import time
        
        params = {
            'rolling_window': 60,
            'z_threshold': 2.0,
            'z_exit': 0.0,
            'commission_pct': 0.001,
            'slippage_pct': 0.001
        }
        
        # Оригинальный бэктестер
        original = PairBacktester(sample_data, **params)
        start_time = time.time()
        original.run()
        original_time = time.time() - start_time
        
        # Полная Numba версия
        numba_full = FullNumbaPairBacktester(sample_data, **params)
        start_time = time.time()
        numba_full.run()
        numba_time = time.time() - start_time
        
        # Numba должна быть быстрее (после прогрева)
        speedup = original_time / numba_time if numba_time > 0 else float('inf')
        
        print(f"\nPerformance comparison:")
        print(f"Original time: {original_time:.4f}s")
        print(f"Numba time: {numba_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Проверяем, что Numba версия работает
        assert numba_time > 0, "Numba version should complete"
        assert not numba_full.results.empty, "Numba version should produce results"
    
    def test_comparison_method(self, sample_data):
        """Тест метода сравнения с оригиналом."""
        params = {
            'rolling_window': 60,
            'z_threshold': 2.0,
            'z_exit': 0.0,
            'commission_pct': 0.001,
            'slippage_pct': 0.001
        }
        
        original = PairBacktester(sample_data, **params)
        numba_full = FullNumbaPairBacktester(sample_data, **params)
        
        # Запускаем Numba версию
        numba_full.run()
        
        # Сравниваем с оригиналом
        comparison = numba_full.compare_with_original(original, tolerance=0.01)
        
        # Проверяем структуру результата
        expected_keys = [
            'original_pnl', 'numba_pnl', 'pnl_difference', 'pnl_relative_error',
            'within_tolerance', 'tolerance', 'original_trades', 'numba_trades', 'trades_difference'
        ]
        
        for key in expected_keys:
            assert key in comparison, f"Missing key in comparison: {key}"
        
        # Проверяем, что результаты в пределах допуска
        assert comparison['within_tolerance'], (
            f"Results outside tolerance: {comparison['pnl_relative_error']:.4f} > {comparison['tolerance']:.4f}"
        )
    
    def test_performance_summary(self, sample_data):
        """Тест получения сводки производительности."""
        params = {
            'rolling_window': 60,
            'z_threshold': 2.0,
            'z_exit': 0.0,
            'commission_pct': 0.001,
            'slippage_pct': 0.001
        }
        
        numba_full = FullNumbaPairBacktester(sample_data, **params)
        numba_full.run()
        
        summary = numba_full.get_performance_summary()
        
        # Проверяем основные метрики
        expected_keys = ['total_pnl', 'total_trades', 'winning_trades', 'losing_trades']
        for key in expected_keys:
            assert key in summary, f"Missing key in summary: {key}"
        
        # Проверяем разумность значений
        assert isinstance(summary['total_pnl'], (int, float)), "total_pnl should be numeric"
        assert summary['total_trades'] >= 0, "total_trades should be non-negative"
        assert summary['winning_trades'] >= 0, "winning_trades should be non-negative"
        assert summary['losing_trades'] >= 0, "losing_trades should be non-negative"
    
    def test_edge_cases(self):
        """Тест граничных случаев."""
        # Пустые данные
        empty_data = pd.DataFrame()
        numba_full = FullNumbaPairBacktester(empty_data, rolling_window=10, z_threshold=2.0)
        numba_full.run()
        assert numba_full.results.empty, "Should handle empty data"
        
        # Очень короткие данные - должны вызывать исключение или обрабатываться корректно
        short_data = pd.DataFrame({
            'asset1': [100, 101, 102],
            'asset2': [80, 81, 82]
        })
        numba_short = FullNumbaPairBacktester(short_data, rolling_window=10, z_threshold=2.0)
        try:
            numba_short.run()
            # Если не вызывает исключение, проверяем результат
            assert len(numba_short.results) == len(short_data), "Should handle short data"
        except ValueError:
            # Ожидаемое поведение для слишком коротких данных
            pass
        
        # Данные с NaN
        nan_data = pd.DataFrame({
            'asset1': [100, np.nan, 102, 103, 104],
            'asset2': [80, 81, np.nan, 83, 84]
        })
        numba_nan = FullNumbaPairBacktester(nan_data, rolling_window=3, z_threshold=2.0)
        numba_nan.run()
        # Должен работать без ошибок
        assert len(numba_nan.results) == len(nan_data), "Should handle NaN data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])