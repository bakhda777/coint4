#!/usr/bin/env python3
"""
Тесты для проверки исправлений в системе оптимизации.
Проверяют корректность генерации сделок и работы бэктестера.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.coint2.utils.config import load_config
from src.coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester
from src.coint2.core.portfolio import Portfolio
from src.optimiser.fast_objective import FastWalkForwardObjective


class TestOptimizationFixes:
    """Тесты для проверки исправлений в оптимизации."""
    
    def test_simple_backtest_generates_trades(self):
        """Тест: простой бэктест должен генерировать сделки."""
        
        # Создаем синтетические данные с трендом
        np.random.seed(42)
        n_periods = 500
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
        
        # Создаем коинтегрированные данные
        price1 = 100 + np.cumsum(np.random.normal(0, 0.5, n_periods))
        noise = np.random.normal(0, 0.2, n_periods)
        price2 = 0.8 * price1 + 10 + noise  # Коинтегрированная серия
        
        test_data = pd.DataFrame({
            'price1': price1,
            'price2': price2
        }, index=dates)
        
        # Создаем портфель
        portfolio = Portfolio(initial_capital=10000, max_active_positions=1)
        
        # Создаем бэктестер с мягкими параметрами
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=30,
            z_threshold=1.0,  # Низкий порог для генерации сделок
            z_exit=0.0,
            commission_pct=0.0001,
            slippage_pct=0.0001,
            stop_loss_multiplier=5.0,
            time_stop_multiplier=10.0,
            portfolio=portfolio,
            pair_name="TEST/PAIR",
            capital_at_risk=1000.0
        )
        
        # Запускаем бэктест
        backtester.run()
        results = backtester.get_results()

        # Проверяем результаты
        assert results is not None, "Результаты бэктеста не должны быть None"

        if isinstance(results, dict):
            # Если результаты - словарь, проверяем наличие ключей
            assert 'pnl' in results, "В результатах должен быть ключ 'pnl'"
            pnl_data = results['pnl']
            if isinstance(pnl_data, pd.Series):
                trades_count = len(pnl_data[pnl_data != 0])
            else:
                trades_count = 1 if pnl_data != 0 else 0
        else:
            # Если результаты - DataFrame
            assert not results.empty, "Результаты бэктеста не должны быть пустыми"
            assert 'pnl' in results.columns, "В результатах должна быть колонка 'pnl'"
            trades = results[results['position'] != 0]
            trades_count = len(trades)

        assert trades_count > 0, f"Должны быть сгенерированы сделки, получено: {trades_count}"

        print(f"✅ Тест пройден: сгенерировано {trades_count} сделок")
        
    def test_fast_objective_with_simple_params(self):
        """Тест: FastWalkForwardObjective должен работать с простыми параметрами."""
        
        # Проверяем наличие необходимых файлов
        required_files = [
            "configs/main_2024.yaml",
            "configs/search_space.yaml",
            "outputs/preselected_pairs.csv"
        ]
        
        for file_path in required_files:
            assert Path(file_path).exists(), f"Файл {file_path} не найден"
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space.yaml"
        )
        
        # Простые параметры для тестирования
        simple_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': 0.0,
            'stop_loss_multiplier': 5.0,
            'time_stop_multiplier': 10.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.1,
            'max_active_positions': 1,
            'commission_pct': 0.0001,
            'slippage_pct': 0.0001,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.5,
            'trial_number': 999
        }
        
        # Запускаем тест
        result = objective(simple_params)
        
        # Проверяем результат
        assert result is not None, "Результат не должен быть None"
        assert isinstance(result, (int, float)), f"Результат должен быть числом, получен: {type(result)}"
        assert result > -999, f"Результат не должен быть штрафным значением: {result}"
        
        print(f"✅ Тест пройден: FastWalkForwardObjective вернул результат: {result}")
        
    def test_zscore_threshold_usage(self):
        """Тест: проверяем правильное использование zscore_entry_threshold."""
        
        # Создаем простые данные
        test_data = pd.DataFrame({
            'price1': [100, 101, 102, 103, 104, 105],
            'price2': [80, 81, 82, 83, 84, 85]
        })
        
        portfolio = Portfolio(initial_capital=10000, max_active_positions=1)
        
        # Создаем бэктестер
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=3,
            z_threshold=1.5,  # Это должно стать zscore_entry_threshold
            z_exit=0.0,
            portfolio=portfolio,
            pair_name="TEST/PAIR"
        )
        
        # Проверяем, что zscore_entry_threshold установлен правильно
        assert hasattr(backtester, 'zscore_entry_threshold'), "Должен быть атрибут zscore_entry_threshold"
        assert backtester.zscore_entry_threshold == 1.5, f"zscore_entry_threshold должен быть 1.5, получен: {backtester.zscore_entry_threshold}"
        
        print("✅ Тест пройден: zscore_entry_threshold установлен правильно")
        
    def test_data_loading_efficiency(self):
        """Тест: проверяем, что данные загружаются эффективно."""
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space.yaml"
        )
        
        # Проверяем, что предотобранные пары загружены
        assert hasattr(objective, 'preselected_pairs'), "Должны быть загружены предотобранные пары"
        assert len(objective.preselected_pairs) > 0, "Должны быть предотобранные пары"
        
        print(f"✅ Тест пройден: загружено {len(objective.preselected_pairs)} предотобранных пар")


def test_config_parameters():
    """Тест: проверяем корректность параметров в конфигурации."""
    
    cfg = load_config("configs/main_2024.yaml")
    
    # Проверяем основные параметры
    assert hasattr(cfg.backtest, 'zscore_threshold'), "В конфигурации должен быть zscore_threshold"
    assert cfg.backtest.zscore_threshold > 0, "zscore_threshold должен быть положительным"
    assert cfg.backtest.zscore_threshold < 3.0, "zscore_threshold не должен быть слишком высоким"
    
    # Проверяем walk-forward параметры
    assert hasattr(cfg, 'walk_forward'), "В конфигурации должна быть секция walk_forward"
    assert cfg.walk_forward.training_period_days > 0, "training_period_days должен быть положительным"
    assert cfg.walk_forward.testing_period_days > 0, "testing_period_days должен быть положительным"
    
    print("✅ Тест пройден: параметры конфигурации корректны")


if __name__ == "__main__":
    # Запускаем тесты
    test_instance = TestOptimizationFixes()
    
    print("🧪 ЗАПУСК ТЕСТОВ ИСПРАВЛЕНИЙ ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    try:
        test_instance.test_simple_backtest_generates_trades()
        test_instance.test_zscore_threshold_usage()
        test_instance.test_data_loading_efficiency()
        test_config_parameters()
        test_instance.test_fast_objective_with_simple_params()
        
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        
    except Exception as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
