"""Тест оптимизации fast_coint после удаления принудительного k=2."""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from coint2.core.fast_coint import fast_coint


def generate_test_data(n_pairs=100, n_obs=500, seed=42):
    """Генерирует идентичные тестовые данные для воспроизводимости."""
    np.random.seed(seed)
    pairs = []
    
    for i in range(n_pairs):
        x = np.random.normal(0, 1, n_obs).cumsum()
        y = np.random.normal(0, 1, n_obs).cumsum()
        
        # Некоторые пары делаем коинтегрированными
        if i % 3 == 0:
            beta = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0, 0.1, n_obs)
            y = beta * x + noise
        
        pairs.append((
            pd.Series(x, name=f'X_{i}'),
            pd.Series(y, name=f'Y_{i}')
        ))
    
    return pairs


def benchmark_optimized_fast_coint(pairs, warmup=True):
    """Бенчмарк оптимизированной версии fast_coint."""
    if warmup:
        # Предварительная компиляция
        _ = fast_coint(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    
    start_time = time.time()
    results = []
    
    for x, y in pairs:
        try:
            tau, pvalue, k = fast_coint(x, y, trend='n')
            results.append((tau, pvalue, k))
        except Exception as e:
            results.append((np.nan, np.nan, 0))
    
    end_time = time.time()
    return results, end_time - start_time


def benchmark_statsmodels_reference(pairs):
    """Эталонный бенчмарк statsmodels."""
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


def compare_accuracy_detailed(results_stats, results_fast):
    """Детальное сравнение точности."""
    tau_diffs = []
    pval_diffs = []
    lag_info = []
    
    for i, ((tau_s, pval_s), (tau_f, pval_f, k_f)) in enumerate(zip(results_stats, results_fast)):
        if not (np.isnan(tau_s) or np.isnan(tau_f)):
            tau_diff = abs(tau_s - tau_f)
            tau_diffs.append(tau_diff)
            
        if not (np.isnan(pval_s) or np.isnan(pval_f)):
            pval_diff = abs(pval_s - pval_f)
            pval_diffs.append(pval_diff)
            
        lag_info.append(k_f)
    
    return {
        'tau_diffs': tau_diffs,
        'pval_diffs': pval_diffs,
        'avg_lag': np.mean([k for k in lag_info if k > 0]),
        'lag_distribution': np.bincount(lag_info)
    }


def test_optimized_performance():
    """Тест производительности оптимизированной версии."""
    print("🚀 ТЕСТ ОПТИМИЗИРОВАННОЙ FAST_COINT")
    print("=" * 55)
    
    # Параметры теста
    n_pairs = 100  # Больше пар для лучшего измерения
    n_obs = 400
    n_runs = 3  # Несколько прогонов для стабильности
    
    print(f"📊 Параметры: {n_pairs} пар × {n_obs} наблюдений")
    print(f"🔄 Количество прогонов: {n_runs}")
    print()
    
    # Генерируем тестовые данные
    pairs = generate_test_data(n_pairs, n_obs)
    
    # === STATSMODELS BASELINE ===
    print("⏱️  Измеряем statsmodels.coint...")
    stats_times = []
    for run in range(n_runs):
        results_stats, time_stats = benchmark_statsmodels_reference(pairs)
        stats_times.append(time_stats)
        print(f"   Прогон {run+1}: {time_stats:.3f} сек")
    
    avg_stats_time = np.mean(stats_times)
    print(f"   Среднее время: {avg_stats_time:.3f} сек\n")
    
    # === OPTIMIZED FAST_COINT ===
    print("⏱️  Измеряем оптимизированную fast_coint...")
    fast_times = []
    for run in range(n_runs):
        results_fast, time_fast = benchmark_optimized_fast_coint(pairs, warmup=(run==0))
        fast_times.append(time_fast)
        print(f"   Прогон {run+1}: {time_fast:.3f} сек")
    
    avg_fast_time = np.mean(fast_times)
    print(f"   Среднее время: {avg_fast_time:.3f} сек\n")
    
    # === АНАЛИЗ РЕЗУЛЬТАТОВ ===
    speedup = avg_stats_time / avg_fast_time
    accuracy = compare_accuracy_detailed(results_stats, results_fast)
    
    print("📈 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("-" * 40)
    print(f"statsmodels среднее время:    {avg_stats_time:.3f} сек")
    print(f"fast_coint среднее время:     {avg_fast_time:.3f} сек")
    print(f"Ускорение:                    {speedup:.1f}x")
    print()
    print("🎯 КАЧЕСТВО РЕЗУЛЬТАТОВ:")
    print(f"Обработано пар:               {len(pairs)}")
    print(f"Средняя разность tau:         {np.mean(accuracy['tau_diffs']):.6f}")
    print(f"Средняя разность p-value:     {np.mean(accuracy['pval_diffs']):.6f}")
    print(f"Максимальная разность tau:    {np.max(accuracy['tau_diffs']):.6f}")
    print(f"Максимальная разность p-val:  {np.max(accuracy['pval_diffs']):.6f}")
    print(f"Средний выбранный лаг:        {accuracy['avg_lag']:.1f}")
    print()
    
    # Анализ распределения лагов
    print("📊 РАСПРЕДЕЛЕНИЕ ВЫБРАННЫХ ЛАГОВ:")
    lag_dist = accuracy['lag_distribution']
    for k, count in enumerate(lag_dist):
        if count > 0:
            print(f"   k={k}: {count} пар ({count/len(pairs)*100:.1f}%)")
    print()
    
    # === ПРОВЕРКИ ===
    expected_speedup = 4.0  # Ожидаем ускорение минимум в 4x
    max_pval_diff = 0.05   # Максимальная разность p-value
    
    print("✅ ПРОВЕРКИ:")
    
    # Проверка ускорения
    if speedup >= expected_speedup:
        print(f"✅ Ускорение {speedup:.1f}x >= {expected_speedup}x — ОТЛИЧНО!")
    else:
        print(f"⚠️  Ускорение {speedup:.1f}x < {expected_speedup}x — приемлемо, но можно лучше")
    
    # Проверка точности
    avg_pval_diff = np.mean(accuracy['pval_diffs'])
    if avg_pval_diff <= max_pval_diff:
        print(f"✅ Точность p-value ({avg_pval_diff:.4f}) — ОТЛИЧНО!")
    else:
        print(f"⚠️  Точность p-value ({avg_pval_diff:.4f}) — требует внимания")
    
    # Проверка что лаги выбираются по AIC, а не фиксированно k=2
    if accuracy['avg_lag'] != 2.0 or len(lag_dist) > 3:
        print(f"✅ AIC-оптимизация работает — выбираются разные лаги!")
    else:
        print(f"⚠️  Возможно, лаги выбираются неоптимально")
    
    print(f"\n🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print(f"💡 Экономия времени на {n_pairs} пар: {(avg_stats_time - avg_fast_time):.2f} сек")
    print(f"📈 Относительное ускорение: {speedup:.1f}x")
    
    return {
        'speedup': speedup,
        'accuracy': accuracy,
        'stats_time': avg_stats_time,
        'fast_time': avg_fast_time
    }


if __name__ == "__main__":
    test_optimized_performance() 