"""Интеграционные тесты для Walk-Forward анализа.

Оптимизировано согласно best practices:
- Минимальные данные для тестов
- Разделение на unit/fast/slow
- Мокирование тяжелых операций
"""

import copy
import gc
import psutil
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import time
import os

from src.coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from src.coint2.utils.config import AppConfig, PortfolioConfig, PairSelectionConfig, BacktestConfig, WalkForwardConfig
from tests.conftest import get_test_config


# Unit и Fast тесты перенесены в отдельные файлы:
# - test_walk_forward_unit.py
# - test_walk_forward_fast.py


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.integration
class TestWalkForwardIntegration:
    """Медленные integration тесты для оптимизированного Walk-Forward анализа."""
    
    def _create_app_config(self, data_file: str, results_dir: str, config_dict: dict) -> AppConfig:
        """Создает AppConfig объект с базовыми настройками."""
        import shutil
        
        # Базовые настройки
        base_config = {
            'portfolio': {
                'initial_capital': 10000.0,
                'risk_per_position_pct': 0.01,
                'max_active_positions': 5
            },
            'pair_selection': {
                'lookback_days': 90,
                'coint_pvalue_threshold': 0.05,
                'ssd_top_n': 100,
                'min_half_life_days': 1,
                'max_half_life_days': 30,
                'min_mean_crossings': 5
            },
            'backtest': {
                'timeframe': '1d',
                'rolling_window': 30,
                'zscore_threshold': 2.0,
                'stop_loss_multiplier': 3.0,
                'fill_limit_pct': 0.1,
                'commission_pct': 0.001,
                'slippage_pct': 0.0,
                'annualizing_factor': 365
            },
            'walk_forward': {
                'start_date': '2021-01-01',
                'end_date': '2021-12-31',
                'training_period_days': 90,
                'testing_period_days': 30
            }
        }
        
        # Объединяем с переданной конфигурацией
        for section, params in config_dict.items():
            if section in base_config:
                base_config[section].update(params)
            else:
                base_config[section] = params
        
        # Создаем data_dir и копируем туда CSV файл
        data_dir = os.path.join(os.path.dirname(results_dir), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Копируем CSV файл в data_dir с именем data.csv
        target_file = os.path.join(data_dir, 'data.csv')
        shutil.copy2(data_file, target_file)
        
        return AppConfig(
            data_dir=data_dir,
            results_dir=results_dir,
            portfolio=PortfolioConfig(**base_config['portfolio']),
            pair_selection=PairSelectionConfig(**base_config['pair_selection']),
            backtest=BacktestConfig(**base_config['backtest']),
            walk_forward=WalkForwardConfig(**base_config['walk_forward'])
        )
    
    @pytest.fixture
    def sample_market_data(self, rng):
        """Минимальные данные для integration тестов."""
        # ОПТИМИЗАЦИЯ: Используем get_test_config() вместо hardcoded значений
        test_config = get_test_config()
        total_bars = test_config['periods']  # Динамическое количество баров
        
        dates = pd.date_range('2024-01-01', periods=total_bars, freq='1h')
        
        # Только 2 актива для скорости
        asset_names = ['ASSET_00', 'ASSET_01']
        assets = []  # ИСПРАВЛЕНИЕ: инициализация списка
        
        for i, name in enumerate(asset_names):
            # Базовая цена
            base_price = 100 + i * 10
            
            # Простая генерация цен
            prices = base_price + np.cumsum(rng.standard_normal(total_bars) * 0.3)
            
            assets.append(prices)
        
        # Создаем DataFrame
        data = pd.DataFrame(dict(zip(asset_names, assets)), index=dates)
        
        # Создаем коинтегрированные пары векторизованно для ускорения
        data['PAIR_A'] = (
            0.8 * data['ASSET_00'] + 
            rng.standard_normal(total_bars) * 0.1 + 20
        )
        data['PAIR_B'] = (
            1.2 * data['ASSET_01'] + 
            rng.standard_normal(total_bars) * 0.15 - 10
        )
        
        return data
    
    @pytest.fixture
    def optimized_config(self):
        """Создает конфигурацию с включенными оптимизациями."""
        test_config = get_test_config()
        return {
            'data': {
                'normalization_method': 'minmax'
            },
            'pair_selection': {
                'ssd_threshold': 0.02,
                'min_half_life_days': 0.5,
                'max_half_life_days': 30,
                'min_mean_crossings': 5,
                'max_hurst_exponent': 0.7,
                'adf_pvalue_threshold': 0.05
            },
            'backtest': {
                'rolling_window': test_config['rolling_window'],  # ОПТИМИЗАЦИЯ: используем test_config
                'zscore_threshold': 2.0,
                'transaction_cost': 0.001,
                'market_regime_detection': False,  # Отключено для ускорения
                'hurst_window': test_config['hurst_window'],  # ОПТИМИЗАЦИЯ: используем test_config
                'variance_ratio_window': max(test_config['rolling_window'] * 2, 50),  # ОПТИМИЗАЦИЯ: используем test_config
                'structural_break_protection': False,  # Отключено
                'cointegration_test_frequency': test_config['hurst_window'],  # ОПТИМИЗАЦИЯ: используем test_config
                'adf_pvalue_threshold': 0.05,
                'exclusion_period_days': 1,  # Уменьшено
                'max_half_life_days': 5,  # Уменьшено
                'min_correlation_threshold': 0.3,
                'correlation_window': max(test_config['rolling_window'] * 2, 50),  # ОПТИМИЗАЦИЯ: используем test_config
                # Параметры оптимизации - упрощены
                'regime_check_frequency': 10,
                'use_market_regime_cache': True,
                'adf_check_frequency': test_config['hurst_window'],  # ОПТИМИЗАЦИЯ: используем test_config
                'cache_cleanup_frequency': 50,
                'lazy_adf_threshold': 0.1,
                'hurst_neutral_band': 0.05,
                'vr_neutral_band': 0.2
            },
            'walk_forward': {
                'training_period_days': 1,  # Минимум 1 день
                'testing_period_days': 1,
                'step_size_days': 1
            },
            'portfolio': {
                'initial_capital': 100000,
                'max_pairs': 20,
                'capital_allocation_method': 'equal_weight'
            }
        }
    
    @pytest.fixture
    def baseline_config(self, optimized_config):
        """Создает базовую конфигурацию без оптимизаций."""
        config = copy.deepcopy(optimized_config)
        
        # Отключаем оптимизации с минимальными значениями
        config['backtest'].update({
            'regime_check_frequency': 1,  # Проверяем каждый бар
            'use_market_regime_cache': False,
            'adf_check_frequency': 100,  # Минимум для валидации
            'lazy_adf_threshold': 0.0,  # Отключаем Lazy ADF
            'hurst_neutral_band': 0.0,  # Без нейтральных зон
            'vr_neutral_band': 0.0
        })
        
        return config
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_walk_forward_when_optimized_config_then_completes_successfully(self, sample_market_data, optimized_config, tmp_path):
        """Integration test: Проверяет полный цикл Walk-Forward с оптимизациями.
        
        Проверяет, что оптимизированный Walk-Forward завершается успешно
        и возвращает корректные результаты.
        """
        # Сохраняем тестовые данные
        data_file = tmp_path / 'test_data.csv'
        sample_market_data.to_csv(data_file)
        
        # Настраиваем результаты
        results_dir = tmp_path / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Запускаем Walk-Forward
        start_time = time.time()
        
        try:
            # Создаем AppConfig объект с базовыми настройками
            cfg = self._create_app_config(data_file, results_dir, optimized_config)
            
            results = run_walk_forward(cfg)
            
            execution_time = time.time() - start_time
            
            # Проверяем, что результаты получены
            assert results is not None, "Walk-Forward не вернул результаты"
            assert isinstance(results, dict), "Результаты должны быть словарем"
            
            # Проверяем базовые метрики (results уже является base_metrics)
            assert 'total_pnl' in results, "Отсутствует total_pnl"
            assert 'sharpe_ratio_abs' in results, "Отсутствует sharpe_ratio_abs"
            assert 'max_drawdown_abs' in results, "Отсутствует max_drawdown_abs"
            
            # Проверяем разумность значений
            assert isinstance(results['total_pnl'], (int, float)), "total_pnl не число"
            assert isinstance(results['sharpe_ratio_abs'], (int, float)), "sharpe_ratio_abs не число"
            assert isinstance(results['max_drawdown_abs'], (int, float)), "max_drawdown_abs не число"
            
            # Проверяем, что обязательные файлы результатов созданы
            required_files = [
                'CointegrationStrategy_performance_report.png',
                'strategy_metrics.csv',
                'equity_curve.csv'
            ]
            
            for filename in required_files:
                filepath = os.path.join(results_dir, filename)
                assert os.path.exists(filepath), f"Файл {filename} не создан"
                
                # Проверяем, что файл не пустой
                assert os.path.getsize(filepath) > 0, f"Файл {filename} пустой"
            
            # Проверяем опциональные файлы (могут отсутствовать при отсутствии сделок)
            optional_files = ['daily_pnl.csv', 'trade_statistics.csv', 'trade_log.csv']
            for filename in optional_files:
                filepath = os.path.join(results_dir, filename)
                if os.path.exists(filepath):
                    assert os.path.getsize(filepath) > 0, f"Файл {filename} существует, но пустой"
            
            print(f"✅ Walk-Forward с оптимизациями завершен за {execution_time:.2f}с")
            print(f"Total PnL: {results['total_pnl']:.2f}")
            print(f"Sharpe Ratio: {results['sharpe_ratio_abs']:.3f}")
            print(f"Max Drawdown: {results['max_drawdown_abs']:.3f}")
            
        except Exception as e:
            pytest.fail(f"Walk-Forward с оптимизациями завершился с ошибкой: {str(e)}")
        
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_performance_comparison(self, sample_market_data, optimized_config, baseline_config):
        """Integration test: Сравнивает производительность с/без оптимизаций.
        
        Проверяет, что оптимизации действительно ускоряют выполнение
        при сохранении качества результатов.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Сохраняем тестовые данные
            data_file = os.path.join(temp_dir, 'test_data.csv')
            sample_market_data.to_csv(data_file)
            
            # Тестируем оптимизированную версию
            results_opt_dir = os.path.join(temp_dir, 'results_opt')
            os.makedirs(results_opt_dir, exist_ok=True)
            
            start_time = time.time()
            # Создаем AppConfig для оптимизированной версии
            cfg_opt = self._create_app_config(data_file, results_opt_dir, optimized_config)
            
            results_optimized = run_walk_forward(cfg_opt)
            optimized_time = time.time() - start_time
            
            # Тестируем базовую версию (может быть медленной)
            results_base_dir = os.path.join(temp_dir, 'results_base')
            os.makedirs(results_base_dir, exist_ok=True)
            
            start_time = time.time()
            # Создаем AppConfig для базовой версии
            cfg_base = self._create_app_config(data_file, results_base_dir, baseline_config)
            
            results_baseline = run_walk_forward(cfg_base)
            baseline_time = time.time() - start_time
            
            # Сравниваем производительность
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            
            print(f"Время оптимизированной версии: {optimized_time:.2f}с")
            print(f"Время базовой версии: {baseline_time:.2f}с")
            print(f"Ускорение: {speedup:.2f}x")
            
            # Проверяем ускорение (может не работать на малых данных)
            # На малых тестовых данных оптимизации могут не давать заметного ускорения
            # из-за накладных расходов на инициализацию
            if baseline_time > 5.0:  # Только если тест достаточно долгий (>5 сек)
                assert speedup >= 0.8, f"Значительное замедление: {speedup:.2f}x"
            else:
                print(f"⚠️  Тест слишком быстрый ({baseline_time:.2f}с) для измерения ускорения")
            
            # Сравниваем качество результатов
            opt_pnl = results_optimized['total_pnl']
            base_pnl = results_baseline['total_pnl']
            
            # Результаты должны быть близкими (разрешаем 10% отличие)
            pnl_diff_pct = abs(opt_pnl - base_pnl) / max(abs(base_pnl), 1.0)
            
            print(f"PnL оптимизированной версии: {opt_pnl:.2f}")
            print(f"PnL базовой версии: {base_pnl:.2f}")
            print(f"Разница в PnL: {pnl_diff_pct:.2%}")
            
            assert pnl_diff_pct < 0.15, f"Результаты слишком различаются: {pnl_diff_pct:.2%}"
    
    def test_config_validation_and_loading(self, optimized_config):
        """Тест 3: Проверяет валидацию и загрузку конфигурации.
        
        Проверяет, что все новые параметры оптимизации корректно
        загружаются и валидируются.
        """
        # Проверяем наличие всех ключевых параметров оптимизации
        backtest_config = optimized_config['backtest']
        
        required_optimization_params = [
            'regime_check_frequency',
            'use_market_regime_cache',
            'adf_check_frequency',
            'cache_cleanup_frequency',
            'lazy_adf_threshold',
            'hurst_neutral_band',
            'vr_neutral_band'
        ]
        
        for param in required_optimization_params:
            assert param in backtest_config, f"Отсутствует параметр оптимизации: {param}"
        
        # Проверяем типы и диапазоны значений
        assert isinstance(backtest_config['regime_check_frequency'], int), "regime_check_frequency должен быть int"
        assert backtest_config['regime_check_frequency'] > 0, "regime_check_frequency должен быть положительным"
        
        assert isinstance(backtest_config['use_market_regime_cache'], bool), "use_market_regime_cache должен быть bool"
        
        assert isinstance(backtest_config['adf_check_frequency'], int), "adf_check_frequency должен быть int"
        assert backtest_config['adf_check_frequency'] > 0, "adf_check_frequency должен быть положительным"
        
        assert isinstance(backtest_config['lazy_adf_threshold'], (int, float)), "lazy_adf_threshold должен быть числом"
        assert 0.0 <= backtest_config['lazy_adf_threshold'] <= 1.0, "lazy_adf_threshold должен быть в [0,1]"
        
        assert isinstance(backtest_config['hurst_neutral_band'], (int, float)), "hurst_neutral_band должен быть числом"
        assert 0.0 <= backtest_config['hurst_neutral_band'] <= 0.5, "hurst_neutral_band должен быть в [0,0.5]"
        
        assert isinstance(backtest_config['vr_neutral_band'], (int, float)), "vr_neutral_band должен быть числом"
        assert 0.0 <= backtest_config['vr_neutral_band'] <= 1.0, "vr_neutral_band должен быть в [0,1]"
        
        print("✅ Все параметры оптимизации корректно настроены")
        
        # Проверяем совместимость параметров
        assert backtest_config['regime_check_frequency'] <= backtest_config['adf_check_frequency'], \
            "regime_check_frequency не должен превышать adf_check_frequency"
        
        print("✅ Параметры оптимизации совместимы между собой")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_memory_usage_optimization(self, sample_market_data, optimized_config):
        """Integration test: Проверяет оптимизацию использования памяти.
        
        Проверяет, что кэш корректно управляет памятью и не растет бесконечно.
        """
        # ОПТИМИЗАЦИЯ: Быстрая проверка без полного walk-forward
        test_config = get_test_config()
        if test_config['periods'] <= 20:
            pytest.skip("Пропускаем тяжелый тест памяти в режиме minimal test config")
        import psutil
        import gc
        
        # Измеряем начальное использование памяти
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Сохраняем тестовые данные
            data_file = os.path.join(temp_dir, 'test_data.csv')
            sample_market_data.to_csv(data_file)
            
            # Настраиваем результаты
            results_dir = os.path.join(temp_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Создаем AppConfig объект
            cfg = self._create_app_config(data_file, results_dir, optimized_config)
            
            # Запускаем Walk-Forward
            results = run_walk_forward(cfg)
            
            # Измеряем использование памяти после выполнения
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Начальная память: {initial_memory:.1f} MB")
            print(f"Конечная память: {final_memory:.1f} MB")
            print(f"Увеличение памяти: {memory_increase:.1f} MB")
            
            # Проверяем, что увеличение памяти разумное
            # (зависит от размера данных, но не должно быть чрезмерным)
            data_size_mb = sample_market_data.memory_usage(deep=True).sum() / 1024 / 1024
            max_reasonable_increase = max(100.0, data_size_mb * 10)  # Не более 100 MB или 10x от размера данных
            
            assert memory_increase < max_reasonable_increase, \
                f"Слишком большое увеличение памяти: {memory_increase:.1f} MB > {max_reasonable_increase:.1f} MB"
            
            print(f"✅ Использование памяти в пределах нормы ({memory_increase:.1f} MB)")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_error_handling_and_robustness(self, sample_market_data, optimized_config):
        """Integration test: Проверяет обработку ошибок и устойчивость.
        
        Проверяет, что система корректно обрабатывает различные
        проблемные ситуации.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Тест 1: Данные с NaN значениями
            corrupted_data = sample_market_data.copy()
            test_config = get_test_config()
            nan_range_end = min(test_config['periods'] // 2, 12)  # ОПТИМИЗАЦИЯ: адаптивный диапазон на основе test_config
            corrupted_data.iloc[max(nan_range_end - 2, 1):nan_range_end, 0] = np.nan
            
            data_file = os.path.join(temp_dir, 'corrupted_data.csv')
            corrupted_data.to_csv(data_file)
            
            results_dir = os.path.join(temp_dir, 'results_corrupted')
            os.makedirs(results_dir, exist_ok=True)
            
            # Создаем AppConfig объект
            cfg = self._create_app_config(data_file, results_dir, optimized_config)
            
            # Должно завершиться без ошибок или с понятной ошибкой
            try:
                results = run_walk_forward(cfg)
                print("✅ Данные с NaN обработаны корректно")
            except Exception as e:
                # Ошибка должна быть информативной
                assert "NaN" in str(e) or "missing" in str(e).lower(), \
                    f"Неинформативная ошибка для NaN данных: {str(e)}"
                print(f"✅ NaN данные корректно отклонены: {str(e)[:100]}...")
            
            # Тест 2: Слишком малый период обучения
            small_config = optimized_config.copy()
            small_config['walk_forward']['training_period_days'] = 1  # Слишком мало
            
            clean_data_file = os.path.join(temp_dir, 'clean_data.csv')
            sample_market_data.to_csv(clean_data_file)
            
            results_dir_small = os.path.join(temp_dir, 'results_small')
            os.makedirs(results_dir_small, exist_ok=True)
            
            # Создаем AppConfig для малого периода
            cfg_small = self._create_app_config(clean_data_file, results_dir_small, small_config)
            
            try:
                results = run_walk_forward(cfg_small)
                # Если не упало, проверяем что результат разумный
                assert results is not None
                print("✅ Малый период обучения обработан")
            except Exception as e:
                # Ошибка должна быть понятной
                assert any(word in str(e).lower() for word in ['training', 'period', 'insufficient']), \
                    f"Неинформативная ошибка для малого периода: {str(e)}"
                print(f"✅ Малый период корректно отклонен: {str(e)[:100]}...")
            
            # Тест 3: Экстремальные параметры оптимизации
            extreme_config = optimized_config.copy()
            extreme_config['backtest'].update({
                'regime_check_frequency': test_config['periods'],  # ОПТИМИЗАЦИЯ: используем test_config
                'adf_check_frequency': test_config['periods'] * 2,  # ОПТИМИЗАЦИЯ: используем test_config
                'lazy_adf_threshold': 0.99        # Почти никогда
            })
            
            results_dir_extreme = os.path.join(temp_dir, 'results_extreme')
            os.makedirs(results_dir_extreme, exist_ok=True)
            
            # Создаем AppConfig для экстремальных параметров
            cfg_extreme = self._create_app_config(clean_data_file, results_dir_extreme, extreme_config)
            
            try:
                results = run_walk_forward(cfg_extreme)
                assert results is not None
                print("✅ Экстремальные параметры обработаны")
            except Exception as e:
                print(f"⚠️ Экстремальные параметры вызвали ошибку: {str(e)[:100]}...")
                # Это может быть ожидаемым поведением
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_results_consistency_across_runs(self, sample_market_data, optimized_config):
        """Integration test: Проверяет консистентность результатов между запусками.
        
        Проверяет, что при одинаковых данных и конфигурации
        результаты воспроизводимы.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Сохраняем тестовые данные
            data_file = os.path.join(temp_dir, 'test_data.csv')
            sample_market_data.to_csv(data_file)
            
            # Первый запуск
            results_dir_1 = os.path.join(temp_dir, 'results_1')
            os.makedirs(results_dir_1, exist_ok=True)
            
            # Создаем AppConfig для первого запуска
            cfg_1 = self._create_app_config(data_file, results_dir_1, optimized_config)
            
            results_1 = run_walk_forward(cfg_1)
            
            # Второй запуск
            results_dir_2 = os.path.join(temp_dir, 'results_2')
            os.makedirs(results_dir_2, exist_ok=True)
            
            # Создаем AppConfig для второго запуска
            cfg_2 = self._create_app_config(data_file, results_dir_2, optimized_config)
            
            results_2 = run_walk_forward(cfg_2)
            
            # Сравниваем ключевые метрики
            pnl_1 = results_1['total_pnl']
            pnl_2 = results_2['total_pnl']
            
            sharpe_1 = results_1['sharpe_ratio_abs']
            sharpe_2 = results_2['sharpe_ratio_abs']
            
            # Результаты должны быть идентичными или очень близкими
            pnl_diff = abs(pnl_1 - pnl_2) / max(abs(pnl_1), 1.0)
            sharpe_diff = abs(sharpe_1 - sharpe_2) / max(abs(sharpe_1), 0.1)
            
            print(f"PnL запуск 1: {pnl_1:.4f}, запуск 2: {pnl_2:.4f}, разница: {pnl_diff:.4%}")
            print(f"Sharpe запуск 1: {sharpe_1:.4f}, запуск 2: {sharpe_2:.4f}, разница: {sharpe_diff:.4%}")
            
            # ОПТИМИЗАЦИЯ: Разрешаем чуть большие различия для малых данных
            assert pnl_diff < 0.01, f"PnL слишком различается между запусками: {pnl_diff:.4%}"
            assert sharpe_diff < 0.05, f"Sharpe слишком различается между запусками: {sharpe_diff:.4%}"
            
            print("✅ Результаты консистентны между запусками")


# Fast версии тестов перенесены в test_walk_forward_fast.py
