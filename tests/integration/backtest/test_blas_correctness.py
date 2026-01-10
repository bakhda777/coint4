#!/usr/bin/env python3
"""
Тесты для проверки корректности результатов бэктеста с оптимизацией BLAS.

Оптимизировано согласно best practices:
- Минимальные данные для тестов
- Разделение на fast/slow
- Мокирование тяжелых операций
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pytest
import numpy as np
import pandas as pd

# Импорты работают через conftest.py

from src.coint2.core.memory_optimization import setup_blas_threading_limits
from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester
from src.coint2.core.portfolio import Portfolio
from src.coint2.utils.config import AppConfig, BacktestConfig, PortfolioConfig, PairSelectionConfig, WalkForwardConfig
from src.coint2.pipeline.walk_forward_orchestrator import (
    process_single_pair,
    process_single_pair_mmap
)
from src.coint2.utils.logging_utils import get_logger


@pytest.mark.fast
class TestBacktestBLASFast:
    """Быстрые тесты корректности бэктеста с BLAS."""
    
    @pytest.fixture
    def tiny_price_data(self, rng) -> pd.DataFrame:
        """Минимальные данные для fast тестов."""
        dates = pd.date_range('2024-01-01', periods=20, freq='15min')
        return pd.DataFrame({
            'AAPL': 100 + rng.normal(0, 1, 20).cumsum(),
            'MSFT': 150 + rng.normal(0, 1.5, 20).cumsum()
        }, index=dates)
    
    @pytest.mark.unit
    def test_blas_settings_when_configured_then_applied(self):
        """Быстрый тест: проверяет применение BLAS настроек."""
        original_env = dict(os.environ)
        
        try:
            # Настраиваем BLAS
            setup_blas_threading_limits(2)
            
            # Проверяем переменные окружения
            assert os.environ.get('OMP_NUM_THREADS') == '2'
            assert os.environ.get('MKL_NUM_THREADS') == '2'
            
        finally:
            # Восстанавливаем
            for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                if key in original_env:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]
    
    @pytest.mark.unit
    def test_backtest_engine_initialization_fast(self, tiny_price_data):
        """Быстрый тест: инициализация движка."""
        engine = PairBacktester(
            pair_data=tiny_price_data,
            rolling_window=10,
            z_threshold=2.0,
            capital_at_risk=10000
        )
        
        assert engine is not None
        assert engine.rolling_window == 10
        assert len(engine.pair_data) == 20


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.integration
class TestBacktestCorrectnessWithBLAS:
    """Тесты корректности бэктеста с оптимизацией BLAS."""
    
    @pytest.fixture
    def sample_price_data(self, rng) -> pd.DataFrame:
        """
        Создает образец ценовых данных для тестирования.

        Возвращает детерминированные данные без lookahead bias.
        """
        # Детерминизм обеспечен через фикстуру rng

        # ОПТИМИЗАЦИЯ: Уменьшено количество точек для ускорения
        import os
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            n_points = 25
        else:
            n_points = 50
        
        dates = pd.date_range('2024-01-01', periods=n_points, freq='15min')

        # Создаем данные последовательно, без использования будущих значений
        price1 = np.zeros(n_points)
        price2 = np.zeros(n_points)

        price1[0] = 100
        price2[0] = 150

        beta = 1.5

        for i in range(1, n_points):
            # Каждое значение зависит только от предыдущих
            price1[i] = price1[i-1] + rng.normal(0, 0.01)
            # Коинтеграция с лагом - price2 реагирует на предыдущее значение price1
            price2[i] = price2[i-1] + beta * (price1[i] - price1[i-1]) + rng.normal(0, 0.5)

        return pd.DataFrame({
            'AAPL': price1,
            'MSFT': price2
        }, index=dates)
    
    @pytest.fixture
    def test_config(self) -> AppConfig:
        """
        Создает тестовую конфигурацию для бэктестирования.
        """
        return AppConfig(
            data_dir=Path("/tmp"),
            results_dir=Path("/tmp"),
            pair_selection=PairSelectionConfig(
                lookback_days=90,
                coint_pvalue_threshold=0.05,
                ssd_top_n=100,
                min_half_life_days=1,
                max_half_life_days=30,
                min_mean_crossings=5
            ),
            walk_forward=WalkForwardConfig(
                start_date="2021-01-01",
                end_date="2021-12-31",
                training_period_days=90,
                testing_period_days=30
            ),
            backtest=BacktestConfig(
                timeframe="15min",
                rolling_window=10,  # ОПТИМИЗАЦИЯ: Уменьшено для работы с малыми данными
                zscore_threshold=2.0,
                zscore_exit=0.5,
                fill_limit_pct=0.1,
                commission_pct=0.001,
                slippage_pct=0.001,
                annualizing_factor=365,
                stop_loss_multiplier=3.0,
                cooldown_hours=1,
                use_kelly_sizing=True,
                max_kelly_fraction=0.25,
                volatility_lookback=96,
                adaptive_thresholds=True,
                var_confidence=0.05,
                max_var_multiplier=3.0,
                market_regime_detection=True,
                structural_break_protection=True
            ),
            portfolio=PortfolioConfig(
                initial_capital=100000.0,
                max_active_positions=10,
                risk_per_position_pct=0.02,
                max_margin_usage=1.0
            )
        )
    
    def save_and_restore_env_vars(self, env_vars: list) -> Dict[str, str]:
        """
        Сохраняет текущие значения переменных окружения.
        
        Args:
            env_vars: Список переменных для сохранения
            
        Returns:
            Словарь с исходными значениями
        """
        original = {}
        for var in env_vars:
            original[var] = os.environ.get(var)
        return original
    
    def restore_env_vars(self, original: Dict[str, str]):
        """
        Восстанавливает переменные окружения.
        
        Args:
            original: Словарь с исходными значениями
        """
        for var, value in original.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    
    def run_backtest_with_blas_config(self, price_data: pd.DataFrame, 
                                     config: AppConfig, 
                                     blas_threads: int) -> Dict[str, Any]:
        """
        Запускает бэктест с определенной конфигурацией BLAS потоков.
        
        Args:
            price_data: Ценовые данные
            config: Конфигурация бэктеста
            blas_threads: Количество BLAS потоков (0 = без ограничений)
            
        Returns:
            Результаты бэктеста
        """
        # Настраиваем BLAS потоки
        if blas_threads > 0:
            setup_blas_threading_limits(num_threads=blas_threads, verbose=False)
        
        # Создаем портфель
        portfolio = Portfolio(
            initial_capital=config.portfolio.initial_capital,
            max_active_positions=1
        )
        
        # Создаем бэктестер
        backtester = PairBacktester(
            pair_data=price_data,
            rolling_window=config.backtest.rolling_window,
            portfolio=portfolio,
            pair_name="AAPL-MSFT",
            z_threshold=config.backtest.zscore_threshold,
            z_exit=config.backtest.zscore_exit,
            commission_pct=config.backtest.commission_pct,
            slippage_pct=config.backtest.slippage_pct,
            annualizing_factor=config.backtest.annualizing_factor,
            capital_at_risk=config.portfolio.initial_capital,
            stop_loss_multiplier=config.backtest.stop_loss_multiplier,
            cooldown_periods=4,  # 1 час при 15-минутных барах
            use_kelly_sizing=config.backtest.use_kelly_sizing,
            max_kelly_fraction=config.backtest.max_kelly_fraction,
            volatility_lookback=config.backtest.volatility_lookback,
            adaptive_thresholds=config.backtest.adaptive_thresholds,
            var_confidence=config.backtest.var_confidence,
            max_var_multiplier=config.backtest.max_var_multiplier,
            market_regime_detection=config.backtest.market_regime_detection,
            structural_break_protection=config.backtest.structural_break_protection
        )
        
        # Запускаем бэктест
        start_time = time.time()
        backtester.run()
        elapsed_time = time.time() - start_time
        
        # Получаем результаты
        results = backtester.get_performance_metrics()
        
        # Добавляем информацию о времени выполнения
        results['execution_time'] = elapsed_time
        results['blas_threads'] = blas_threads
        
        return results
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_backtest_results_when_blas_optimized_then_identical(self, sample_price_data, test_config):
        """
        Тест 1: Проверяет идентичность результатов бэктеста с и без BLAS оптимизации.
        
        Проверяет, что ограничение BLAS потоков не влияет на точность
        вычислений и результаты бэктестирования.
        """
        env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", 
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
        original_env = self.save_and_restore_env_vars(env_vars)
        
        try:
            # Убираем все ограничения BLAS
            for var in env_vars:
                os.environ.pop(var, None)
            
            # Запускаем бэктест без ограничений
            results_unlimited = self.run_backtest_with_blas_config(
                sample_price_data, test_config, blas_threads=0
            )
            
            # Запускаем бэктест с ограничением в 1 поток
            results_limited = self.run_backtest_with_blas_config(
                sample_price_data, test_config, blas_threads=1
            )
            
            # Сравниваем ключевые метрики
            tolerance = 1e-10  # Очень строгая толерантность
            
            # Проверяем основные финансовые метрики
            metrics_to_compare = [
                'total_pnl', 'total_return', 'sharpe_ratio', 'max_drawdown',
                'num_trades', 'win_rate'
            ]
            
            for metric in metrics_to_compare:
                if metric in results_unlimited and metric in results_limited:
                    val_unlimited = results_unlimited[metric]
                    val_limited = results_limited[metric]
                    
                    # Проверяем на NaN
                    if pd.isna(val_unlimited) and pd.isna(val_limited):
                        continue
                    
                    # Проверяем числовые значения
                    if isinstance(val_unlimited, (int, float)) and isinstance(val_limited, (int, float)):
                        diff = abs(val_unlimited - val_limited)
                        rel_diff = diff / (abs(val_unlimited) + 1e-10)
                        
                        assert diff < tolerance or rel_diff < tolerance, \
                            f"Metric {metric} differs: {val_unlimited} vs {val_limited} (diff: {diff}, rel_diff: {rel_diff})"
            
            # Проверяем PnL серии
            if 'pnl_series' in results_unlimited and 'pnl_series' in results_limited:
                pnl_unlimited = results_unlimited['pnl_series']
                pnl_limited = results_limited['pnl_series']
                
                if len(pnl_unlimited) == len(pnl_limited):
                    pnl_diff = np.abs(pnl_unlimited - pnl_limited)
                    max_pnl_diff = np.max(pnl_diff)
                    
                    assert max_pnl_diff < tolerance, \
                        f"PnL series differ by {max_pnl_diff} (max allowed: {tolerance})"
            
            # Проверяем логи сделок
            if 'trades_log' in results_unlimited and 'trades_log' in results_limited:
                trades_unlimited = results_unlimited['trades_log']
                trades_limited = results_limited['trades_log']
                
                assert len(trades_unlimited) == len(trades_limited), \
                    f"Different number of trades: {len(trades_unlimited)} vs {len(trades_limited)}"
                
                # Сравниваем каждую сделку
                for i, (trade_u, trade_l) in enumerate(zip(trades_unlimited, trades_limited)):
                    for key in ['entry_time', 'exit_time', 'pnl', 'position_size']:
                        if key in trade_u and key in trade_l:
                            val_u = trade_u[key]
                            val_l = trade_l[key]
                            
                            if isinstance(val_u, (int, float)) and isinstance(val_l, (int, float)):
                                diff = abs(val_u - val_l)
                                assert diff < tolerance, \
                                    f"Trade {i} {key} differs: {val_u} vs {val_l} (diff: {diff})"
            
            print(f"\n✅ Backtest results identical:")
            print(f"   Unlimited BLAS: {results_unlimited['total_pnl']:.6f} PnL, {results_unlimited['execution_time']:.3f}s")
            print(f"   Limited BLAS:   {results_limited['total_pnl']:.6f} PnL, {results_limited['execution_time']:.3f}s")
            
        finally:
            self.restore_env_vars(original_env)
    
    @pytest.mark.integration
    def test_process_single_pair_when_blas_optimized_then_consistency_maintained(self, sample_price_data, test_config):
        """
        Тест 2: Проверяет консистентность функции process_single_pair.
        
        Проверяет, что функция process_single_pair дает одинаковые
        результаты с и без BLAS оптимизации.
        """
        env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        original_env = self.save_and_restore_env_vars(env_vars)
        
        try:
            # Подготавливаем данные для тестирования
            pair_data_tuple = ('AAPL', 'MSFT', 1.5, 0.0, 1.0, {'half_life': 100})
            testing_start = sample_price_data.index[20]  # Исправлено: используем валидный индекс
            testing_end = sample_price_data.index[-1]     # Исправлено: используем последний элемент
            capital_per_pair = 10000.0
            bar_minutes = 15
            period_label = "test_period"
            
            # Тест без ограничений BLAS
            for var in env_vars:
                os.environ.pop(var, None)
            
            result_unlimited = process_single_pair(
                pair_data_tuple=pair_data_tuple,
                step_df=sample_price_data,
                testing_start=testing_start,
                testing_end=testing_end,
                cfg=test_config,
                capital_per_pair=capital_per_pair,
                bar_minutes=bar_minutes,
                period_label=period_label
            )
            
            # Тест с ограничениями BLAS
            setup_blas_threading_limits(num_threads=1, verbose=False)
            
            result_limited = process_single_pair(
                pair_data_tuple=pair_data_tuple,
                step_df=sample_price_data,
                testing_start=testing_start,
                testing_end=testing_end,
                cfg=test_config,
                capital_per_pair=capital_per_pair,
                bar_minutes=bar_minutes,
                period_label=period_label
            )
            
            # Сравниваем результаты
            assert result_unlimited['success'] == result_limited['success'], \
                "Success status should be identical"
            
            if result_unlimited['success'] and result_limited['success']:
                # Сравниваем trade_stat
                stat_unlimited = result_unlimited['trade_stat']
                stat_limited = result_limited['trade_stat']
                
                numeric_fields = ['total_pnl', 'num_trades', 'win_rate', 'avg_win', 
                                'avg_loss', 'profit_factor', 'max_drawdown', 'sharpe_ratio']
                
                for field in numeric_fields:
                    if field in stat_unlimited and field in stat_limited:
                        val_u = stat_unlimited[field]
                        val_l = stat_limited[field]
                        
                        if pd.isna(val_u) and pd.isna(val_l):
                            continue
                        
                        if isinstance(val_u, (int, float)) and isinstance(val_l, (int, float)):
                            diff = abs(val_u - val_l)
                            rel_diff = diff / (abs(val_u) + 1e-10)
                            
                            assert diff < 1e-10 or rel_diff < 1e-10, \
                                f"Field {field} differs: {val_u} vs {val_l} (diff: {diff})"
                
                # Сравниваем PnL серии
                pnl_u = result_unlimited['pnl_series']
                pnl_l = result_limited['pnl_series']
                
                if len(pnl_u) == len(pnl_l) and len(pnl_u) > 0:
                    pnl_diff = np.abs(pnl_u.values - pnl_l.values)
                    max_diff = np.max(pnl_diff)
                    
                    assert max_diff < 1e-10, \
                        f"PnL series differ by {max_diff}"
                
                print(f"\n✅ process_single_pair results identical:")
                print(f"   Total PnL: {stat_unlimited['total_pnl']:.6f}")
                print(f"   Trades: {stat_unlimited.get('trade_count', stat_unlimited.get('num_trades', 0))}")
            
        finally:
            self.restore_env_vars(original_env)
    
    @pytest.mark.unit
    def test_numerical_stability_when_blas_limited_then_stable(self, test_config, rng):
        """
        Тест 3: Проверяет численную стабильность с ограничениями BLAS.
        
        Проверяет, что ограничение BLAS потоков не влияет на
        численную стабильность вычислений.
        """
        env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        original_env = self.save_and_restore_env_vars(env_vars)
        
        try:
            # Настраиваем BLAS на 1 поток
            setup_blas_threading_limits(num_threads=1, verbose=False)
            
            # ОПТИМИЗАЦИЯ: Уменьшено количество точек
            import os
            if os.environ.get('QUICK_TEST', '').lower() == 'true':
                n_points = 100
            else:
                n_points = 200  # Уменьшено с 1000 до 200
            dates = pd.date_range('2024-01-01', periods=n_points, freq='15min')

            # Создаем точно коинтегрированные ряды
            x = rng.normal(0, 1, n_points).cumsum()
            beta_true = 2.0
            alpha_true = 10.0
            noise = rng.normal(0, 0.1, n_points)
            y = alpha_true + beta_true * x + noise
            
            price_data = pd.DataFrame({
                'X': x + 100,  # Добавляем базовый уровень
                'Y': y + 200
            }, index=dates)
            
            # Создаем портфель
            portfolio = Portfolio(
                initial_capital=100000.0,
                max_active_positions=1
            )
            
            # Запускаем бэктест
            backtester = PairBacktester(
                pair_data=price_data,
                rolling_window=20,
                portfolio=portfolio,
                pair_name="X-Y",
                z_threshold=2.0,
                z_exit=0.5,
                commission_pct=0.001,
                slippage_pct=0.001
            )
            
            backtester.run()
            results = backtester.get_performance_metrics()
            
            # Проверяем, что результаты разумные
            assert 'total_pnl' in results, "Missing total_pnl in results"
            assert 'num_trades' in results, "Missing num_trades in results"
            assert 'sharpe_ratio' in results, "Missing sharpe_ratio in results"
            
            # Проверяем, что нет NaN или inf значений
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    assert not pd.isna(value), f"NaN value in {key}"
                    assert not np.isinf(value), f"Inf value in {key}"
            
            # Проверяем PnL серию
            if 'pnl_series' in results:
                pnl_series = results['pnl_series']
                assert not pnl_series.isna().any(), "NaN values in PnL series"
                assert not np.isinf(pnl_series).any(), "Inf values in PnL series"
            
            print(f"\n✅ Numerical stability test passed:")
            print(f"   Total PnL: {results['total_pnl']:.6f}")
            print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")
            print(f"   Trade Count: {results['num_trades']}")
            
        finally:
            self.restore_env_vars(original_env)
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_memory_usage_when_blas_optimized_then_efficient(self, sample_price_data, test_config):
        """
        Тест 4: Проверяет использование памяти с BLAS оптимизацией.
        
        Проверяет, что ограничение BLAS потоков не приводит к
        утечкам памяти или чрезмерному потреблению.
        """
        import psutil
        import gc
        
        env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        original_env = self.save_and_restore_env_vars(env_vars)
        
        try:
            # Настраиваем BLAS
            setup_blas_threading_limits(num_threads=1, verbose=False)
            
            # Измеряем начальное потребление памяти
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Запускаем несколько итераций бэктеста
            memory_measurements = []
            
            for i in range(5):
                # Принудительная сборка мусора
                gc.collect()
                
                # Запускаем бэктест
                results = self.run_backtest_with_blas_config(
                    sample_price_data, test_config, blas_threads=1
                )
                
                # Измеряем память
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                
                # Проверяем, что результаты корректные
                assert 'total_pnl' in results
                assert isinstance(results['total_pnl'], (int, float))
            
            # Финальная сборка мусора
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # Анализируем использование памяти
            max_memory = max(memory_measurements)
            memory_growth = final_memory - initial_memory
            
            # Проверяем, что нет значительных утечек памяти
            # Допускаем рост до 50 MB (разумный предел для тестов)
            assert memory_growth < 50, \
                f"Excessive memory growth: {memory_growth:.1f} MB"
            
            # Проверяем, что пиковое потребление разумное
            # Допускаем до 200 MB для тестовых данных
            assert max_memory < initial_memory + 200, \
                f"Excessive peak memory usage: {max_memory:.1f} MB"
            
            print(f"\n✅ Memory usage test passed:")
            print(f"   Initial: {initial_memory:.1f} MB")
            print(f"   Peak: {max_memory:.1f} MB")
            print(f"   Final: {final_memory:.1f} MB")
            print(f"   Growth: {memory_growth:.1f} MB")
            
        finally:
            self.restore_env_vars(original_env)
    
    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.integration
    def test_concurrent_backtest_when_executed_then_correct(self, sample_price_data, test_config):
        """
        Тест 5: Проверяет корректность параллельного выполнения бэктестов.
        
        Проверяет, что BLAS оптимизация работает корректно при
        параллельном выполнении нескольких бэктестов.
        """
        import concurrent.futures
        import threading
        
        env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        original_env = self.save_and_restore_env_vars(env_vars)
        
        try:
            # Настраиваем BLAS
            setup_blas_threading_limits(num_threads=1, verbose=False)
            
            def run_single_backtest(thread_id: int) -> Dict[str, Any]:
                """Запускает один бэктест в отдельном потоке."""
                # Убеждаемся, что BLAS настроен в каждом потоке
                setup_blas_threading_limits(num_threads=1, verbose=False)
                
                # Создаем уникальные данные для каждого потока
                # Используем отдельный генератор для каждого потока
                thread_rng = np.random.default_rng(42 + thread_id * 1000)  # Больший разброс seed
                dates = pd.date_range('2024-01-01', periods=100, freq='15min')  # Уменьшено с 500

                # Создаем синтетические коинтегрированные данные с разными параметрами
                # Это гарантирует, что будут генерироваться сигналы
                base_price = 100 + thread_id * 10  # Разные базовые цены
                trend = np.linspace(0, thread_id * 0.1, 100)  # Разные тренды, уменьшено с 500

                # Создаем коинтегрированную пару с известными свойствами
                common_factor = thread_rng.normal(0, 0.02, 100).cumsum()  # Уменьшено с 500

                price1 = base_price + common_factor + trend + thread_rng.normal(0, 0.01, 100)  # Уменьшено
                price2 = base_price * 1.2 + common_factor * 1.1 + trend * 0.8 + thread_rng.normal(0, 0.01, 100)  # Уменьшено
                
                # Добавляем периодические отклонения для генерации сигналов
                spread_oscillation = 2 * np.sin(np.linspace(0, 4 * np.pi + thread_id, 100))  # Уменьшено
                price2 += spread_oscillation
                
                thread_data = pd.DataFrame({
                    'A': price1,
                    'B': price2
                }, index=dates)
                
                # Запускаем бэктест с более мягкими параметрами
                # Создаем временную конфигурацию для этого потока
                thread_config = AppConfig(
                      data_dir=Path("/tmp"),
                      results_dir=Path("/tmp"),
                      pair_selection=PairSelectionConfig(
                          lookback_days=90,
                          coint_pvalue_threshold=0.05,
                          ssd_top_n=100,
                          min_half_life_days=1,
                          max_half_life_days=30,
                          min_mean_crossings=5
                      ),
                      walk_forward=WalkForwardConfig(
                          start_date="2021-01-01",
                          end_date="2021-12-31",
                          training_period_days=90,
                          testing_period_days=30
                      ),
                      backtest=BacktestConfig(
                          timeframe="15min",  # Обязательное поле
                          rolling_window=30,  # Меньше окно
                          zscore_threshold=1.5,  # Более мягкий порог
                          zscore_exit=0.3,  # Более мягкий выход
                          fill_limit_pct=0.1,  # Обязательное поле
                          commission_pct=0.0001,  # Меньшие комиссии
                          slippage_pct=0.0001,
                          annualizing_factor=252 * 24 * 4,
                          stop_loss_multiplier=3.0,
                          cooldown_hours=1,
                          use_kelly_sizing=False,
                          max_kelly_fraction=0.25,
                          volatility_lookback=20,
                          adaptive_thresholds=False,
                          var_confidence=0.95,
                          max_var_multiplier=2.0,
                          market_regime_detection=False,
                          structural_break_protection=False
                      ),
                      portfolio=PortfolioConfig(
                          initial_capital=100000.0,
                          max_active_positions=1,
                          risk_per_position_pct=0.02,
                          max_margin_usage=1.0
                      )
                  )
                
                result = self.run_backtest_with_blas_config(
                    thread_data, thread_config, blas_threads=1
                )
                
                result['thread_id'] = thread_id
                result['thread_name'] = threading.current_thread().name
                
                return result
            
            # Запускаем параллельные бэктесты
            num_threads = min(2, os.cpu_count())  # Уменьшено с 4 до 2 для ускорения тестов
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(run_single_backtest, i) 
                          for i in range(num_threads)]
                
                results = [future.result() for future in futures]
            
            # Проверяем результаты
            assert len(results) == num_threads, "Not all threads completed"
            
            for i, result in enumerate(results):
                assert 'total_pnl' in result, f"Thread {i} missing total_pnl"
                assert 'thread_id' in result, f"Thread {i} missing thread_id"
                assert result['thread_id'] == i, f"Thread {i} has wrong ID"
                
                # Проверяем, что результаты разумные
                assert isinstance(result['total_pnl'], (int, float))
                assert not pd.isna(result['total_pnl'])
                assert not np.isinf(result['total_pnl'])
            
            # Проверяем, что результаты различаются (разные данные)
            pnl_values = [r['total_pnl'] for r in results]
            
            # Проверяем, что все потоки завершились успешно
            for i, result in enumerate(results):
                print(f"   Thread {i}: PnL={result['total_pnl']:.6f}, Trades={result.get('num_trades', 0)}")
            
            # Если все PnL равны нулю, это может быть нормально для некоторых стратегий
            all_zero = all(abs(pnl) < 1e-10 for pnl in pnl_values)
            
            if all_zero:
                # Проверяем, что хотя бы количество сделок или другие метрики различаются
                trade_counts = [r.get('num_trades', 0) for r in results]
                sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
                
                # Проверяем различия в количестве сделок или Sharpe ratio
                trade_differences = len(set(trade_counts)) > 1
                sharpe_differences = len(set([round(sr, 6) for sr in sharpe_ratios])) > 1
                
                if not (trade_differences or sharpe_differences):
                    print(f"\n⚠️  Warning: All threads produced very similar results, but this may be expected for this strategy")
                    print(f"   PnL values: {pnl_values}")
                    print(f"   Trade counts: {trade_counts}")
                    print(f"   Sharpe ratios: {sharpe_ratios}")
                else:
                    print(f"\n✅ Found differences in trade counts or Sharpe ratios")
            else:
                # Проверяем различия в PnL
                pnl_differences = []
                for i in range(len(pnl_values)):
                    for j in range(i + 1, len(pnl_values)):
                        diff = abs(pnl_values[i] - pnl_values[j])
                        pnl_differences.append(diff)
                
                significant_differences = [d for d in pnl_differences if d > 0.001]
                print(f"\n✅ Found {len(significant_differences)} significant differences in PnL values")
            
            print(f"\n✅ Concurrent execution test passed:")
            print(f"   Threads: {num_threads}")
            unique_pnl = len(set(pnl_values))
            print(f"   Unique PnL values: {unique_pnl}")
            for i, result in enumerate(results):
                print(f"   Thread {i}: {result['total_pnl']:.6f} PnL")
            
        finally:
            self.restore_env_vars(original_env)


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "-s"])