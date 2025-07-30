#!/usr/bin/env python3
"""
Скрипт для запуска оптимизации с автоматическим обновлением веб-анализа.
Комбинирует оптимизацию и обновление веб-интерфейса в одной команде.
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_optimization(args):
    """Запускает оптимизацию Optuna."""
    
    print("🚀 ЗАПУСК ОПТИМИЗАЦИИ")
    print("=" * 50)
    
    # Формируем команду для оптимизации
    cmd = [
        sys.executable, 
        "src/optimiser/run_optimization.py",
        "--n-trials", str(args.n_trials),
        "--study-name", args.study_name,
        "--storage-path", args.storage_path,
        "--search-space", args.search_space,
        "--base-config", args.base_config,
        "--n-jobs", str(args.n_jobs),
        "--seed", str(args.seed)
    ]
    
    print(f"📊 Команда: {' '.join(cmd)}")
    
    # Запускаем оптимизацию
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Оптимизация завершилась с ошибкой (код: {result.returncode})")
        return False
    
    print(f"✅ Оптимизация завершена успешно!")
    return True

def update_web_analysis(args):
    """Обновляет веб-анализ."""
    
    print("\n🔄 ОБНОВЛЕНИЕ ВЕБ-АНАЛИЗА")
    print("=" * 50)
    
    # Формируем команду для обновления веб-анализа
    cmd = [
        sys.executable,
        "scripts/update_web_analysis.py",
        args.study_name,
        args.storage_path,
        "results"
    ]
    
    print(f"🌐 Команда: {' '.join(cmd)}")
    
    # Запускаем обновление
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Обновление веб-анализа завершилось с ошибкой (код: {result.returncode})")
        return False
    
    print(f"✅ Веб-анализ обновлен успешно!")
    return True

def main():
    """Главная функция."""
    
    parser = argparse.ArgumentParser(description="Запуск оптимизации с автоматическим обновлением веб-анализа")
    
    # Параметры оптимизации
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Количество trials для оптимизации (по умолчанию: 50)")
    parser.add_argument("--study-name", type=str, default="quick_optimization",
                       help="Имя study (по умолчанию: quick_optimization)")
    parser.add_argument("--storage-path", type=str, default="outputs/studies/quick_optimization.db",
                       help="Путь к БД Optuna")
    parser.add_argument("--search-space", type=str, default="configs/search_space_relaxed.yaml",
                       help="Файл пространства поиска")
    parser.add_argument("--base-config", type=str, default="configs/main_2024.yaml",
                       help="Базовая конфигурация")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Количество параллельных процессов")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed для воспроизводимости")
    
    # Дополнительные опции
    parser.add_argument("--skip-optimization", action="store_true",
                       help="Пропустить оптимизацию, только обновить веб-анализ")
    parser.add_argument("--skip-web-update", action="store_true",
                       help="Пропустить обновление веб-анализа")
    
    args = parser.parse_args()
    
    print("🎯 ПОЛНЫЙ ЦИКЛ ОПТИМИЗАЦИИ И АНАЛИЗА")
    print("=" * 60)
    print(f"📊 Study: {args.study_name}")
    print(f"🔢 Trials: {args.n_trials}")
    print(f"📁 Storage: {args.storage_path}")
    print(f"🔍 Search space: {args.search_space}")
    
    success = True
    
    # Шаг 1: Оптимизация
    if not args.skip_optimization:
        success = run_optimization(args)
        if not success:
            print(f"\n❌ ОШИБКА: Оптимизация не удалась")
            return False
    else:
        print(f"\n⏭️  Оптимизация пропущена")
    
    # Шаг 2: Обновление веб-анализа
    if not args.skip_web_update:
        success = update_web_analysis(args)
        if not success:
            print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Веб-анализ не обновлен, но оптимизация прошла успешно")
    else:
        print(f"\n⏭️  Обновление веб-анализа пропущено")
    
    # Финальный отчет
    print(f"\n🎉 ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН!")
    print("=" * 60)
    
    if not args.skip_optimization:
        print(f"✅ Оптимизация: завершена ({args.n_trials} trials)")
        print(f"📊 Результаты: {args.storage_path}")
        print(f"⚙️  Лучшая конфигурация: configs/best_config.yaml")
    
    if not args.skip_web_update:
        web_path = Path("src/web_analysis/index.html").resolve()
        print(f"🌐 Веб-анализ: обновлен")
        print(f"🔗 Откройте: file://{web_path}")
        print(f"🔄 Обновите страницу в браузере (Ctrl+F5)")
    
    print(f"\n💡 Следующие шаги:")
    print(f"   1. Изучите результаты в веб-интерфейсе")
    print(f"   2. Запустите валидацию с лучшими параметрами")
    print(f"   3. При необходимости повторите оптимизацию с большим количеством trials")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
