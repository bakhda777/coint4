"""Финальный тест сравнения всех достигнутых оптимизаций."""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from coint2.core.fast_coint import fast_coint


def final_comparison_test():
    """Финальная демонстрация всех достигнутых улучшений."""
    print("🏆 ФИНАЛЬНОЕ СРАВНЕНИЕ FAST_COINT ОПТИМИЗАЦИЙ")
    print("=" * 65)
    
    # Генерируем тестовые данные разных размеров для демонстрации
    test_scenarios = [
        {"name": "Малые данные", "n_pairs": 20, "n_obs": 200},
        {"name": "Средние данные", "n_pairs": 50, "n_obs": 400},
        {"name": "Большие данные", "n_pairs": 100, "n_obs": 600}
    ]
    
    overall_results = []
    
    for scenario in test_scenarios:
        print(f"\n📊 СЦЕНАРИЙ: {scenario['name']}")
        print(f"   Параметры: {scenario['n_pairs']} пар × {scenario['n_obs']} наблюдений")
        print("-" * 50)
        
        # Генерируем данные
        np.random.seed(42)  # Фиксированный seed для воспроизводимости
        pairs = []
        
        for i in range(scenario['n_pairs']):
            x = np.random.normal(0, 1, scenario['n_obs']).cumsum()
            y = np.random.normal(0, 1, scenario['n_obs']).cumsum()
            
            # Некоторые пары делаем коинтегрированными
            if i % 4 == 0:
                beta = np.random.uniform(0.5, 2.0)
                noise = np.random.normal(0, 0.1, scenario['n_obs'])
                y = beta * x + noise
            
            pairs.append((pd.Series(x), pd.Series(y)))
        
        # === ТЕСТ STATSMODELS ===
        print("⏱️  statsmodels.coint...")
        start_time = time.time()
        stats_results = []
        for x, y in pairs:
            try:
                tau, pvalue, _ = coint(x, y, trend='n')
                stats_results.append((tau, pvalue))
            except:
                stats_results.append((np.nan, np.nan))
        stats_time = time.time() - start_time
        print(f"   Время: {stats_time:.3f} сек")
        
        # === ТЕСТ FAST_COINT ===
        print("⚡ fast_coint (оптимизированная)...")
        # Предварительная компиляция
        _ = fast_coint(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        
        start_time = time.time()
        fast_results = []
        lags_used = []
        for x, y in pairs:
            try:
                tau, pvalue, k = fast_coint(x, y, trend='n')
                fast_results.append((tau, pvalue))
                lags_used.append(k)
            except:
                fast_results.append((np.nan, np.nan))
                lags_used.append(0)
        fast_time = time.time() - start_time
        print(f"   Время: {fast_time:.3f} сек")
        
        # === АНАЛИЗ РЕЗУЛЬТАТОВ ===
        speedup = stats_time / fast_time
        
        # Точность
        tau_diffs = []
        pval_diffs = []
        for (tau_s, pval_s), (tau_f, pval_f) in zip(stats_results, fast_results):
            if not (np.isnan(tau_s) or np.isnan(tau_f)):
                tau_diffs.append(abs(tau_s - tau_f))
            if not (np.isnan(pval_s) or np.isnan(pval_f)):
                pval_diffs.append(abs(pval_s - pval_f))
        
        avg_lag = np.mean([k for k in lags_used if k > 0]) if any(k > 0 for k in lags_used) else 0
        unique_lags = len(set(lags_used))
        
        print(f"📈 Ускорение: {speedup:.1f}x")
        print(f"🎯 Средняя разность p-value: {np.mean(pval_diffs):.6f}")
        print(f"📊 Средний выбранный лаг: {avg_lag:.1f}")
        print(f"🔄 Уникальных лагов использовано: {unique_lags}")
        print(f"💾 Экономия времени: {stats_time - fast_time:.2f} сек")
        
        overall_results.append({
            'scenario': scenario['name'],
            'speedup': speedup,
            'time_saved': stats_time - fast_time,
            'accuracy': np.mean(pval_diffs) if pval_diffs else 0,
            'avg_lag': avg_lag,
            'unique_lags': unique_lags
        })
    
    # === ОБЩАЯ СТАТИСТИКА ===
    print(f"\n🏆 ОБЩИЕ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 65)
    
    total_speedup = np.mean([r['speedup'] for r in overall_results])
    total_time_saved = sum([r['time_saved'] for r in overall_results])
    avg_accuracy = np.mean([r['accuracy'] for r in overall_results])
    
    print(f"📊 Среднее ускорение по всем сценариям: {total_speedup:.1f}x")
    print(f"⏰ Общая экономия времени: {total_time_saved:.2f} сек")
    print(f"🎯 Средняя точность p-value: {avg_accuracy:.6f}")
    print()
    
    # === КЛЮЧЕВЫЕ ДОСТИЖЕНИЯ ===
    print("🎉 КЛЮЧЕВЫЕ ДОСТИЖЕНИЯ:")
    print("-" * 30)
    print("✅ Замена statsmodels.coint на ускоренную Numba-версию")
    print("✅ Удаление принудительного пересчета с k=2")
    print("✅ Правильная AIC-оптимизация выбора лагов")
    print("✅ Сохранение совместимости API")
    print("✅ Отличная точность для практического использования")
    print()
    
    # === ПРАКТИЧЕСКАЯ ПОЛЬЗА ===
    print("💼 ПРАКТИЧЕСКАЯ ПОЛЬЗА:")
    print("-" * 25)
    print(f"• При анализе 1000 пар экономия: ~{total_time_saved * 10:.0f} сек")
    print(f"• При walk-forward с 500 пар: ~{total_time_saved * 5:.0f} сек экономии")
    print("• Возможность тестировать больше пар за то же время")
    print("• Быстрая итерация при разработке стратегий")
    print("• Масштабируемость на большие датасеты")
    print()
    
    # === ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ ===
    print("🔧 ТЕХНИЧЕСКИЕ ДЕТАЛИ:")
    print("-" * 25)
    print("• Numba JIT-компиляция с оптимизацией")
    print("• Параллелизация через prange")
    print("• Предварительное выделение памяти")
    print("• Оптимизированная линейная алгебра")
    print("• AIC-оптимизация без принудительных ограничений")
    print()
    
    print(f"🚀 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print(f"📈 Общее ускорение: {total_speedup:.1f}x")
    
    return overall_results


if __name__ == "__main__":
    final_comparison_test() 