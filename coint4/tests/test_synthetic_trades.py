"""
Тесты для проверки генерации сделок на синтетических данных.
Используется для изоляции проблемы "0 trades".
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from coint2.engine.reference_engine import ReferenceEngine, make_synthetic_pair
from coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester


class TestSyntheticTrades:
    """Тесты на синтетических данных для проверки генерации сделок."""
    
    def test_reference_engine_generates_trades(self):
        """Проверка что reference engine генерирует сделки на синтетике."""
        # Создаем синтетическую пару
        data = make_synthetic_pair(n=1000, seed=42)
        
        # Создаем движок
        engine = ReferenceEngine(
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            verbose=False
        )
        
        # Запускаем бэктест
        results = engine.backtest(data)
        
        # Проверки
        assert results['num_trades'] > 0, "Reference engine должен генерировать сделки"
        assert results['num_trades'] >= 2, "Должно быть минимум 2 смены позиции (вход и выход)"
        assert np.any(results['positions'] != 0), "Должны быть ненулевые позиции"
        
        print(f"✅ Reference engine: {results['num_trades']} сделок")
    
    def test_numba_engine_generates_trades(self):
        """Проверка что numba engine генерирует сделки на синтетике."""
        # Создаем синтетическую пару
        data = make_synthetic_pair(n=1000, seed=42)
        
        # Подготовка данных для numba engine
        pair_data = pd.DataFrame({
            'symbol1': data['symbol1'].values,
            'symbol2': data['symbol2'].values
        })
        
        # Создаем движок
        engine = FullNumbaPairBacktester(
            pair_data=pair_data,
            rolling_window=60,
            z_threshold=2.0,
            z_exit=0.5,
            commission_pct=0.0004,
            slippage_pct=0.0005
        )
        
        # Запускаем бэктест
        results = engine.run_numba_full()
        
        # Подсчет сделок
        num_trades = np.sum(results.trades_series != 0)
        
        # Проверки
        assert num_trades > 0, "Numba engine должен генерировать сделки"
        assert np.any(results.positions != 0), "Должны быть ненулевые позиции"
        
        print(f"✅ Numba engine: {num_trades} сделок")
    
    def test_engines_consistency(self):
        """Сравнение reference и numba engines на одинаковых данных."""
        # Создаем синтетическую пару
        data = make_synthetic_pair(n=1000, seed=42)
        
        # Reference engine
        ref_engine = ReferenceEngine(
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            verbose=False
        )
        ref_results = ref_engine.backtest(data)
        
        # Numba engine
        pair_data = pd.DataFrame({
            'symbol1': data['symbol1'].values,
            'symbol2': data['symbol2'].values
        })
        
        numba_engine = FullNumbaPairBacktester(
            pair_data=pair_data,
            rolling_window=60,
            z_threshold=2.0,
            z_exit=0.5,
            commission_pct=0.0004,
            slippage_pct=0.0005
        )
        numba_results = numba_engine.run_numba_full()
        numba_trades = np.sum(numba_results.trades_series != 0)
        
        print(f"\nСравнение движков:")
        print(f"Reference: {ref_results['num_trades']} сделок")
        print(f"Numba: {numba_trades} сделок")
        
        # Не требуем точного совпадения, но оба должны торговать
        assert ref_results['num_trades'] > 0, "Reference должен торговать"
        assert numba_trades > 0, "Numba должен торговать"
        
        # Проверяем что расхождение не слишком большое (в пределах 50%)
        if ref_results['num_trades'] > 0 and numba_trades > 0:
            ratio = max(ref_results['num_trades'], numba_trades) / min(ref_results['num_trades'], numba_trades)
            assert ratio < 2.0, f"Слишком большое расхождение между движками: {ratio:.1f}x"
    
    def test_different_parameters(self):
        """Тест с разными параметрами для проверки чувствительности."""
        data = make_synthetic_pair(n=1000, seed=42)
        
        test_params = [
            {'z_enter': 1.5, 'z_exit': 0.3, 'window': 40},
            {'z_enter': 2.5, 'z_exit': 0.5, 'window': 80},
            {'z_enter': 1.0, 'z_exit': 0.0, 'window': 30},
        ]
        
        for params in test_params:
            engine = ReferenceEngine(
                rolling_window=params['window'],
                z_enter=params['z_enter'],
                z_exit=params['z_exit'],
                verbose=False
            )
            
            results = engine.backtest(data)
            
            print(f"Params {params}: {results['num_trades']} trades, Sharpe={results['sharpe_ratio']:.2f}")
            
            # Хотя бы с одним набором параметров должны быть сделки
            if results['num_trades'] > 0:
                print(f"✅ Найдены рабочие параметры: {params}")
                break
        else:
            pytest.fail("Ни один набор параметров не сгенерировал сделки")
    
    def test_edge_cases(self):
        """Тест граничных случаев."""
        # Слишком короткие данные
        short_data = make_synthetic_pair(n=100, seed=42)
        engine = ReferenceEngine(rolling_window=30, z_enter=2.0, z_exit=0.5, verbose=False)
        results = engine.backtest(short_data)
        assert 'num_trades' in results  # Должен работать даже с короткими данными
        
        # Экстремальные пороги
        data = make_synthetic_pair(n=500, seed=42)
        
        # Очень низкий порог входа
        low_threshold_engine = ReferenceEngine(
            rolling_window=30,
            z_enter=0.5,
            z_exit=0.0,
            verbose=False
        )
        low_results = low_threshold_engine.backtest(data)
        assert low_results['num_trades'] > 0, "С низким порогом должны быть частые сделки"
        
        # Очень высокий порог входа
        high_threshold_engine = ReferenceEngine(
            rolling_window=30,
            z_enter=5.0,
            z_exit=2.0,
            verbose=False
        )
        high_results = high_threshold_engine.backtest(data)
        # С очень высоким порогом может не быть сделок - это нормально
        print(f"High threshold trades: {high_results['num_trades']}")


if __name__ == "__main__":
    # Запуск тестов
    test = TestSyntheticTrades()
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ СДЕЛОК")
    print("=" * 60)
    
    test.test_reference_engine_generates_trades()
    test.test_numba_engine_generates_trades()
    test.test_engines_consistency()
    test.test_different_parameters()
    test.test_edge_cases()
    
    print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")