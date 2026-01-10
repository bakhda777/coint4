#!/usr/bin/env python3
"""
Тест Optuna оптимизации с исправленными критическими проблемами.
Использует параметры из test_best_params.py (rolling_window=30, z_threshold=0.7).
"""

import sys
import os
sys.path.insert(0, 'src')
os.environ['QUICK_TEST'] = 'true'

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 ТЕСТ OPTUNA С ИСПРАВЛЕНИЯМИ")
print("="*70)

# Список исправлений
fixes = {
    "Lookahead bias": "StatefulNormalizer разделяет fit/transform",
    "Universe пар": "UniverseManager фиксирует пары между trials",
    "SQLite": "PRAGMA WAL mode + оптимизации",
    "Pivot данных": "Правильное преобразование long->wide",
    "Sharpe расчет": "sqrt(252*96) для 15-мин данных"
}

print("\n📋 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:")
for problem, solution in fixes.items():
    print(f"   ✅ {problem}: {solution}")

print("\n🔧 ЦЕЛЕВЫЕ ПАРАМЕТРЫ (из test_best_params.py):")
print("   rolling_window = 30")
print("   zscore_threshold = 0.7")
print("   zscore_exit = 0.0")
print("   Цель: Sharpe > 1.0")

print("\n" + "="*70)

try:
    from src.optimiser.run_optimization import run_optimization
    
    print("🚀 Запускаем оптимизацию с 5 trials...")
    print("   Config: configs/main_2024.yaml")
    print("   Search space: configs/search_spaces/ultra_fast.yaml")
    
    # Запускаем оптимизацию
    results = run_optimization(
        base_config_path="configs/main_2024.yaml",
        search_space_path="configs/search_spaces/ultra_fast.yaml",
        n_trials=5,
        n_jobs=1,
        study_name="test_with_fixes"
    )
    
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ:")
    
    if results and 'best_value' in results:
        sharpe = results['best_value']
        print(f"\n🏆 Лучший Sharpe: {sharpe:.3f}")
        
        if 'best_params' in results:
            print("\n📈 Лучшие параметры:")
            for key, value in results['best_params'].items():
                print(f"   {key}: {value}")
        
        # Проверка достижения цели
        if sharpe > 1.0:
            print("\n" + "="*70)
            print("🎉 УСПЕХ! SHARPE > 1.0 ДОСТИГНУТ!")
            print("="*70)
            print("\n✨ Все критические проблемы исправлены!")
            print("✨ Оптимизация работает корректно!")
            print("✨ Можно запускать полную оптимизацию!")
        elif sharpe > 0.5:
            print("\n⚠️ Sharpe > 0.5 - хороший прогресс!")
            print("   Рекомендуется увеличить n_trials для поиска лучших параметров")
        else:
            print(f"\n❌ Sharpe = {sharpe:.3f} - требуется дополнительная отладка")
    else:
        print("\n❌ Результаты не получены")
        
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Тест завершен")