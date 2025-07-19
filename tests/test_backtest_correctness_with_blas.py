#!/usr/bin/env python3
"""
Тесты для проверки корректности результатов бэктеста с оптимизацией BLAS.

Эти тесты обеспечивают, что оптимизация BLAS потоков не влияет на
точность и корректность результатов бэктестирования.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pytest
import numpy as np
import pandas as pd

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.core.memory_optimization import setup_blas_threading_limits
from coint2.engine.backtest_engine import PairBacktester
from coint2.core.portfolio import Portfolio
from coint2.utils.config import AppConfig, BacktestConfig, PortfolioConfig, PairSelectionConfig, WalkForwardConfig
from coint2.pipeline.walk_forward_orchestrator import (
    process_single_pair,
    process_single_pair_mmap
)
from coint2.utils.logging_utils import get_logger


class TestBacktestCorrectnessWithBLAS:
    """Тесты корректности бэктеста с оптимизацией BLAS."""
    
    @pytest.fixture
    def sample_price_data(self) -> pd.DataFrame:
        """
        Создает образец ценовых данных для тестирования.
        
        Возвращает детерминированные данные для воспроизводимых тестов.
        """
        np.random.seed(42)  # Фиксированный seed для воспроизводимости
        
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        
        # Создаем коинтегрированные временные ряды
        # Первый ряд - случайное блуждание
        price1 = np.random.normal(0, 0.01, 1000).cumsum() + 100
        
        # Второй ряд коинтегрирован с первым + шум
        beta = 1.5
        price2 = beta * price1 + np.random.normal(0, 0.5, 1000) + 50
        
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
                rolling_window=50,
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
    
    def test_backtest_results_identical_with_blas_optimization(self, sample_price_data, test_config):
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
    
    def test_process_single_pair_consistency(self, sample_price_data, test_config):
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
            testing_start = sample_price_data.index[100]
            testing_end = sample_price_data.index[800]
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
    
    def test_numerical_stability_with_blas_limits(self, test_config):
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
            
            # Создаем данные с известными свойствами
            np.random.seed(123)
            n_points = 1000
            dates = pd.date_range('2024-01-01', periods=n_points, freq='15min')
            
            # Создаем точно коинтегрированные ряды
            x = np.random.normal(0, 1, n_points).cumsum()
            beta_true = 2.0
            alpha_true = 10.0
            noise = np.random.normal(0, 0.1, n_points)
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
                rolling_window=50,
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
    
    def test_memory_usage_with_blas_optimization(self, sample_price_data, test_config):
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
    
    def test_concurrent_backtest_execution(self, sample_price_data, test_config):
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
                np.random.seed(42 + thread_id)
                dates = pd.date_range('2024-01-01', periods=500, freq='15min')
                
                price1 = np.random.normal(0, 0.01, 500).cumsum() + 100
                price2 = 1.5 * price1 + np.random.normal(0, 0.5, 500) + 50
                
                thread_data = pd.DataFrame({
                    'A': price1,
                    'B': price2
                }, index=dates)
                
                # Запускаем бэктест
                result = self.run_backtest_with_blas_config(
                    thread_data, test_config, blas_threads=1
                )
                
                result['thread_id'] = thread_id
                result['thread_name'] = threading.current_thread().name
                
                return result
            
            # Запускаем параллельные бэктесты
            num_threads = min(4, os.cpu_count())
            
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
            unique_pnl = len(set(pnl_values))
            
            # Должно быть хотя бы 2 разных значения PnL
            assert unique_pnl >= 2, "All threads produced identical results (unexpected)"
            
            print(f"\n✅ Concurrent execution test passed:")
            print(f"   Threads: {num_threads}")
            print(f"   Unique PnL values: {unique_pnl}")
            for i, result in enumerate(results):
                print(f"   Thread {i}: {result['total_pnl']:.6f} PnL")
            
        finally:
            self.restore_env_vars(original_env)


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "-s"])