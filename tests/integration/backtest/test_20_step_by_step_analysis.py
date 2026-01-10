#!/usr/bin/env python3
"""
Тест для проверки корректности вычислений на конкретном шаге цикла бэктеста.
Этот тест должен был поймать lookahead bias в оригинальном коде.
"""

import numpy as np
import pandas as pd
import pytest
from coint2.engine.base_engine import BasePairBacktester
# Константы перенесены в файл теста
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_Z_EXIT = 0.5  
TOLERANCE_PRECISION = 1e-6

# Константы для тестирования
TEST_PERIODS = 15
FREQUENCY = '15T'
START_DATE = '2023-01-01'
ROLLING_WINDOW = 5
Z_THRESHOLD = 2.0
Z_EXIT = 0.5
TARGET_STEP = 7
EXPECTED_WINDOW_START = 2
EXPECTED_WINDOW_END = 7
PREVIOUS_STEP = 6
TOLERANCE = 1e-8

# Тестовые данные с реальной вариацией для выявления lookahead bias
Y_TEST_DATA = [100.0, 101.5, 102.2, 103.8, 104.1, 105.7, 106.3, 107.9, 108.2, 109.6, 110.4, 111.8, 112.1, 113.5, 114.7]
X_TEST_DATA = [50.0, 50.3, 50.1, 50.7, 50.2, 50.8, 50.4, 50.9, 50.3, 50.6, 50.5, 50.9, 50.2, 50.7, 50.6]

@pytest.mark.unit
@pytest.mark.critical_fixes
def test_backtest_rolling_stats_when_step_by_step_analysis_then_no_lookahead_bias():
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
    dates = pd.date_range(START_DATE, periods=TEST_PERIODS, freq=FREQUENCY)

    # Используем предопределенные тестовые данные
    pair_data = pd.DataFrame({
        'y': Y_TEST_DATA,
        'x': X_TEST_DATA
    }, index=dates)
    
    # Создаем бэктестер
    bt = BasePairBacktester(
        pair_data=pair_data,
        rolling_window=ROLLING_WINDOW,
        z_threshold=Z_THRESHOLD,
        z_exit=Z_EXIT,
        capital_at_risk=1000.0
    )
    
    # Запускаем бэктест
    bt.run()
    results = bt.results

    # ИСПРАВЛЕНО: На шаге i=7 метод update_rolling_stats(7) использует данные [2:7) = [2,3,4,5,6]
    # для расчета статистик и применяет их к текущему бару 7 (НЕ к предыдущему)

    # Рассчитаем статистики "правильно" - используя данные [2:7) как в update_rolling_stats
    start_idx = TARGET_STEP - bt.rolling_window  # 7 - 5 = 2
    end_idx = TARGET_STEP  # 7

    y_window_correct = pair_data['y'].iloc[start_idx:end_idx]  # [2:7) = индексы 2,3,4,5,6
    x_window_correct = pair_data['x'].iloc[start_idx:end_idx]
    
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
        
        # Теперь рассчитаем z_score для шага 7, используя цены шага 7
        y_step7 = pair_data['y'].iloc[TARGET_STEP]  # Цена на шаге 7
        x_step7 = pair_data['x'].iloc[TARGET_STEP]  # Цена на шаге 7
        spread_step7 = y_step7 - beta_correct * x_step7
        z_score_step7_correct = (spread_step7 - mean_correct) / std_correct
        
    except np.linalg.LinAlgError:
        pytest.fail("Ошибка при расчете OLS")
    
    # Теперь посмотрим, что получилось в бэктесте
    print(f"\n=== РЕЗУЛЬТАТЫ БЭКТЕСТА ===")
    # Проверяем результаты бэктеста (убираем print statements для соответствия стандартам)
    beta_result = results['beta'].iloc[TARGET_STEP]
    mean_result = results['mean'].iloc[TARGET_STEP]
    std_result = results['std'].iloc[TARGET_STEP]
    spread_result = results['spread'].iloc[TARGET_STEP]
    z_score_result = results['z_score'].iloc[TARGET_STEP]
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: z_score должен быть рассчитан БЕЗ использования данных шага 7
    # Если в коде есть lookahead bias, то z_score будет отличаться от правильного
    
    # Проверяем, что z_score рассчитан корректно (без lookahead bias)
    z_score_difference = abs(z_score_result - z_score_step7_correct)

    assert z_score_difference == pytest.approx(0, abs=TOLERANCE), \
        f"z_score рассчитан неправильно! Ожидалось {z_score_step7_correct:.10f}, получено {z_score_result:.10f}, разница: {z_score_difference:.10f}"

if __name__ == "__main__":
    test_backtest_rolling_stats_when_step_by_step_analysis_then_no_lookahead_bias()