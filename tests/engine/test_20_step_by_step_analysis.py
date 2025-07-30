#!/usr/bin/env python3
"""
Тест для проверки корректности вычислений на конкретном шаге цикла бэктеста.
Этот тест должен был поймать lookahead bias в оригинальном коде.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from coint2.engine.base_engine import BasePairBacktester

def test_backtest_step_by_step():
    """
    Тест проверяет корректность вычислений на шаге 7 при rolling_window = 5.
    
    Ключевая проблема: в методе update_rolling_stats() есть ошибка.
    Когда мы на шаге i=7, метод должен:
    1. Взять данные с [2:7] (индексы 2,3,4,5,6) для расчета статистик
    2. Применить эти статистики к шагу i-1=6 для расчета z_score
    3. Использовать z_score[6] для принятия решения на шаге 7
    
    НО в текущем коде есть ошибка в строке:
    prev_z_score = (prev_spread - mean) / std
    df.loc[df.index[i-1], "z_score"] = prev_z_score
    
    Это означает, что z_score для шага 6 рассчитывается с использованием
    статистик, полученных из данных [2:7], что является lookahead bias!
    """
    
    # Создаем простые тестовые данные
    dates = pd.date_range('2023-01-01', periods=15, freq='15T')
    
    # Данные с реальной вариацией для выявления lookahead bias
    y_data = [100.0, 101.5, 102.2, 103.8, 104.1, 105.7, 106.3, 107.9, 108.2, 109.6, 110.4, 111.8, 112.1, 113.5, 114.7]
    x_data = [50.0, 50.3, 50.1, 50.7, 50.2, 50.8, 50.4, 50.9, 50.3, 50.6, 50.5, 50.9, 50.2, 50.7, 50.6]
    
    pair_data = pd.DataFrame({
        'y': y_data,
        'x': x_data
    }, index=dates)
    
    # Создаем бэктестер
    bt = BasePairBacktester(
        pair_data=pair_data,
        rolling_window=5,
        z_threshold=2.0,
        z_exit=0.5,
        capital_at_risk=1000.0
    )
    
    # Запускаем бэктест
    bt.run()
    results = bt.results
    
    print("\n=== АНАЛИЗ ШАГА 7 (индекс 7) ===")
    print(f"rolling_window = {bt.rolling_window}")
    
    # ИСПРАВЛЕНО: На шаге i=7 метод update_rolling_stats(7) использует данные [2:7) = [2,3,4,5,6]
    # для расчета статистик и применяет их к текущему бару 7 (НЕ к предыдущему)
    
    # Рассчитаем статистики "правильно" - используя данные [2:7) как в update_rolling_stats
    target_bar = 7  # Тестируем шаг 7
    start_idx = target_bar - bt.rolling_window  # 7 - 5 = 2
    end_idx = target_bar  # 7
    
    y_window_correct = pair_data['y'].iloc[start_idx:end_idx]  # [2:7) = индексы 2,3,4,5,6
    x_window_correct = pair_data['x'].iloc[start_idx:end_idx]
    
    print(f"\nДанные для расчета статистик (правильно, [2:7) без текущего бара):")
    print(f"y_window: {y_window_correct.values}")
    print(f"x_window: {x_window_correct.values}")
    
    # Рассчитаем beta, mean, std вручную
    X = np.column_stack([np.ones(len(x_window_correct)), x_window_correct.values])
    y = y_window_correct.values
    
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_correct = coeffs[1]
        
        # Рассчитаем spread и его статистики
        spreads = y - beta_correct * x_window_correct.values
        mean_correct = np.mean(spreads)
        std_correct = np.std(spreads, ddof=1)
        
        print(f"\nСтатистики (правильные):")
        print(f"beta = {beta_correct:.10f}")
        print(f"mean = {mean_correct:.10f}")
        print(f"std = {std_correct:.10f}")
        
        # Теперь рассчитаем z_score для шага 7, используя цены шага 7
        y_step7 = pair_data['y'].iloc[target_bar]  # Цена на шаге 7
        x_step7 = pair_data['x'].iloc[target_bar]  # Цена на шаге 7
        spread_step7 = y_step7 - beta_correct * x_step7
        z_score_step7_correct = (spread_step7 - mean_correct) / std_correct
        
        print(f"\nРасчет z_score для шага {target_bar} (правильно):")
        print(f"y[{target_bar}] = {y_step7}")
        print(f"x[{target_bar}] = {x_step7}")
        print(f"spread[{target_bar}] = {spread_step7:.10f}")
        print(f"z_score[{target_bar}] = {z_score_step7_correct:.10f}")
        
    except np.linalg.LinAlgError:
        print("Ошибка при расчете OLS")
        return
    
    # Теперь посмотрим, что получилось в бэктесте
    print(f"\n=== РЕЗУЛЬТАТЫ БЭКТЕСТА ===")
    print(f"beta[{target_bar}] = {results['beta'].iloc[target_bar]:.10f}")
    print(f"mean[{target_bar}] = {results['mean'].iloc[target_bar]:.10f}")
    print(f"std[{target_bar}] = {results['std'].iloc[target_bar]:.10f}")
    print(f"spread[{target_bar}] = {results['spread'].iloc[target_bar]:.10f}")
    print(f"z_score[{target_bar}] = {results['z_score'].iloc[target_bar]:.10f}")
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: z_score должен быть рассчитан БЕЗ использования данных шага 7
    # Если в коде есть lookahead bias, то z_score будет отличаться от правильного
    
    print(f"\n=== ПРОВЕРКА НА LOOKAHEAD BIAS ===")
    print(f"Правильный z_score[{target_bar}]: {z_score_step7_correct:.10f}")
    print(f"Фактический z_score[{target_bar}]: {results['z_score'].iloc[target_bar]:.10f}")
    print(f"Разница: {abs(results['z_score'].iloc[target_bar] - z_score_step7_correct):.10f}")
    
    # Проверяем, что z_score рассчитан корректно (без lookahead bias)
    tolerance = 1e-8
    if abs(results['z_score'].iloc[target_bar] - z_score_step7_correct) <= tolerance:
        print(f"\n✅ Lookahead bias ИСПРАВЛЕН")
        print(f"z_score рассчитан корректно без использования будущих данных")
    else:
        print(f"\n❌ ПРОБЛЕМА: z_score все еще рассчитывается неправильно")
        print(f"Ожидаемый z_score[{target_bar}]: {z_score_step7_correct:.10f}")
        print(f"Фактический z_score[{target_bar}]: {results['z_score'].iloc[target_bar]:.10f}")
        print(f"Разница: {abs(results['z_score'].iloc[target_bar] - z_score_step7_correct):.10f}")
        
        # Проверим, не используются ли все еще неправильные данные
        assert False, f"z_score рассчитан неправильно! Ожидалось {z_score_step7_correct:.10f}, получено {results['z_score'].iloc[target_bar]:.10f}"

if __name__ == "__main__":
    test_backtest_step_by_step()