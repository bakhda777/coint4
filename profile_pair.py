"""Профилирование производительности PairBacktester для выявления hotspots."""

import cProfile
import pstats
import io
import time
import numpy as np
import pandas as pd
from src.coint2.engine.backtest_engine import PairBacktester
from src.coint2.engine.numba_backtest_engine import NumbaPairBacktester


def load_example_prices():
    """Создает тестовые данные для профилирования."""
    np.random.seed(42)
    n_periods = 4000  # Достаточно данных для профилирования
    dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
    
    # Создаем коинтегрированные ряды
    x = np.cumsum(np.random.randn(n_periods)) + 100
    y = 0.8 * x + np.cumsum(np.random.randn(n_periods) * 0.1) + 20
    
    return pd.DataFrame({
        'TICK1': y,
        'TICK2': x
    }, index=dates)


def profile_original():
    """Профилирование оригинальной версии."""
    print("=== ПРОФИЛИРОВАНИЕ ОРИГИНАЛЬНОЙ ВЕРСИИ ===")
    df = load_example_prices()[['TICK1', 'TICK2']]
    
    params = {
        'rolling_window': 100,
        'z_threshold': 2.0,
        'z_exit': 0.5,
        'commission_pct': 0.0002,
        'slippage_pct': 0.0001,
        'market_regime_detection': False,
        'structural_break_protection': False
    }
    
    bt = PairBacktester(df, **params)
    
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    
    bt.run()
    
    execution_time = time.time() - start_time
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumtime')
    ps.print_stats(20)
    
    print(f"Время выполнения: {execution_time:.4f}s")
    print("\nТоп функций по времени:")
    print(s.getvalue())
    
    if bt.results is not None:
        print(f"\nСтатистика: {len(bt.results)} периодов, "
              f"{bt.results['trades'].sum():.1f} сделок, "
              f"PnL: {bt.results['pnl'].sum():.4f}")
    
    return execution_time, bt.results['pnl'].sum() if bt.results is not None else 0


def profile_numba():
    """Профилирование Numba-версии."""
    print("\n=== ПРОФИЛИРОВАНИЕ NUMBA-ВЕРСИИ ===")
    df = load_example_prices()[['TICK1', 'TICK2']]
    
    params = {
        'rolling_window': 100,
        'z_threshold': 2.0,
        'z_exit': 0.5,
        'commission_pct': 0.0002,
        'slippage_pct': 0.0001,
        'market_regime_detection': False,
        'structural_break_protection': False
    }
    
    bt = NumbaPairBacktester(df, **params)
    
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    
    result = bt.run_numba()
    
    execution_time = time.time() - start_time
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumtime')
    ps.print_stats(20)
    
    print(f"Время выполнения: {execution_time:.4f}s")
    print("\nТоп функций по времени:")
    print(s.getvalue())
    
    print(f"\nСтатистика: {len(result.positions)} периодов, "
          f"{np.sum(result.trades_series > 0):.1f} сделок, "
          f"PnL: {result.total_pnl:.4f}")
    
    return execution_time, result.total_pnl


def main():
    """Основная функция профилирования."""
    print("Загрузка тестовых данных...")
    
    # Профилируем оригинальную версию
    original_time, original_pnl = profile_original()
    
    # Профилируем Numba-версию
    numba_time, numba_pnl = profile_numba()
    
    # Сравнение результатов
    print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
    speedup = original_time / numba_time if numba_time > 0 else float('inf')
    pnl_diff = abs(numba_pnl - original_pnl)
    
    print(f"Ускорение: {speedup:.2f}x")
    print(f"Оригинальное время: {original_time:.4f}s")
    print(f"Numba время: {numba_time:.4f}s")
    print(f"Разница в PnL: {pnl_diff:.8f}")
    print(f"Относительная ошибка: {pnl_diff / max(abs(original_pnl), 1e-8):.2e}")
    
    if speedup >= 10:
        print("✅ Отличное ускорение (≥10x)")
    elif speedup >= 5:
        print("✅ Хорошее ускорение (≥5x)")
    elif speedup >= 2:
        print("⚠️ Умеренное ускорение (≥2x)")
    else:
        print("❌ Недостаточное ускорение (<2x)")
    
    if pnl_diff < 1e-6:
        print("✅ Результаты эквивалентны")
    else:
        print("❌ Результаты различаются")


if __name__ == '__main__':
    main()