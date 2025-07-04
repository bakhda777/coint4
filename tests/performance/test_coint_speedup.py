"""Демонстрация ускорения cointegration test."""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from coint2.core.fast_coint import fast_coint


def generate_test_pairs(n_pairs=100, n_observations=500):
    """Генерирует тестовые пары для бенчмарка."""
    np.random.seed(42)
    pairs = []
    
    for i in range(n_pairs):
        # Генерируем случайные блуждания
        x = np.random.normal(0, 1, n_observations).cumsum()
        y = np.random.normal(0, 1, n_observations).cumsum()
        
        # Иногда делаем пары коинтегрированными
        if i % 3 == 0:
            # Создаем коинтегрированную пару
            beta = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0, 0.1, n_observations)
            y = beta * x + noise
        
        pairs.append((
            pd.Series(x, name=f'X_{i}'),
            pd.Series(y, name=f'Y_{i}')
        ))
    
    return pairs


def benchmark_statsmodels(pairs):
    """Бенчмарк statsmodels.coint."""
    start_time = time.time()
    results = []
    
    for x, y in pairs:
        try:
            tau, pvalue, _ = coint(x, y, trend='n')
            results.append((tau, pvalue))
        except Exception:
            results.append((np.nan, np.nan))
    
    end_time = time.time()
    return results, end_time - start_time


def benchmark_fast_coint(pairs):
    """Бенчмарк fast_coint."""
    # Предварительная компиляция
    _ = fast_coint(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    
    start_time = time.time()
    results = []
    
    for x, y in pairs:
        try:
            tau, pvalue, _ = fast_coint(x, y, trend='n')
            results.append((tau, pvalue))
        except Exception:
            results.append((np.nan, np.nan))
    
    end_time = time.time()
    return results, end_time - start_time


def compare_accuracy(results_stats, results_fast):
    """Сравнивает точность результатов."""
    tau_diffs = []
    pval_diffs = []
    
    for (tau_s, pval_s), (tau_f, pval_f) in zip(results_stats, results_fast):
        if not (np.isnan(tau_s) or np.isnan(tau_f)):
            tau_diffs.append(abs(tau_s - tau_f))
        if not (np.isnan(pval_s) or np.isnan(pval_f)):
            pval_diffs.append(abs(pval_s - pval_f))
    
    return tau_diffs, pval_diffs


def test_coint_speedup_demo():
    """Демонстрация ускорения cointegration test."""
    print("🚀 ДЕМОНСТРАЦИЯ УСКОРЕНИЯ COINTEGRATION TEST")
    print("=" * 60)
    
    # Генерируем тестовые данные
    n_pairs = 50  # Меньше пар для быстрого теста
    n_obs = 300
    
    print(f"📊 Генерируем {n_pairs} пар с {n_obs} наблюдениями каждая...")
    pairs = generate_test_pairs(n_pairs, n_obs)
    
    # Тестируем statsmodels
    print("⏱️  Тестируем statsmodels.coint...")
    results_stats, time_stats = benchmark_statsmodels(pairs)
    
    # Тестируем fast_coint
    print("⏱️  Тестируем fast_coint...")
    results_fast, time_fast = benchmark_fast_coint(pairs)
    
    # Анализируем точность
    tau_diffs, pval_diffs = compare_accuracy(results_stats, results_fast)
    
    # Выводим результаты
    print("\n📈 РЕЗУЛЬТАТЫ:")
    print("-" * 40)
    print(f"statsmodels время:  {time_stats:.3f} сек")
    print(f"fast_coint время:   {time_fast:.3f} сек")
    print(f"Ускорение:          {time_stats/time_fast:.1f}x")
    print()
    print(f"Обработано пар:     {len(pairs)}")
    print(f"Средняя разность tau:    {np.mean(tau_diffs):.6f}")
    print(f"Средняя разность p-val:  {np.mean(pval_diffs):.6f}")
    print(f"Макс разность tau:       {np.max(tau_diffs):.6f}")
    print(f"Макс разность p-val:     {np.max(pval_diffs):.6f}")
    
    # Проверяем что ускорение достигнуто
    speedup = time_stats / time_fast
    assert speedup > 2, f"Ускорение {speedup:.1f}x меньше ожидаемого (>2x)"
    
    # Проверяем точность (для практических целей торговли)
    assert np.mean(pval_diffs) < 0.1, f"Средняя разность p-value ({np.mean(pval_diffs):.6f}) слишком большая"
    
    print("\n✅ Тест прошел успешно!")
    print(f"✅ Достигнуто ускорение в {speedup:.1f}x раз")
    print("✅ Точность результатов приемлема для практического использования")


if __name__ == "__main__":
    test_coint_speedup_demo() 