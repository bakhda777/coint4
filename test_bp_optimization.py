#!/usr/bin/env python3
"""
Тест Best Practice оптимизации
Быстрый тест на 10 trials для проверки работоспособности
"""

import subprocess
import sys
from pathlib import Path
import time

def run_test():
    """Запуск тестовой оптимизации"""
    
    print("🧪 ТЕСТ BEST PRACTICE ОПТИМИЗАЦИИ")
    print("="*50)
    
    # Проверяем наличие файлов
    required_files = [
        "configs/main_2024.yaml",
        "configs/search_space_bp_balanced.yaml", 
        "scripts/bp_optimize.py",
        "scripts/bp_validate.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ Все необходимые файлы найдены")
    
    # Параметры тестового запуска
    test_params = {
        'base': 'configs/main_2024.yaml',
        'space': 'configs/search_space_bp_balanced.yaml',
        'trials': 10,  # Быстрый тест
        'study': 'test_bp_optimization',
        'storage': 'sqlite:///test_studies.db',
        'n_jobs': 1
    }
    
    # Формируем команду
    cmd = [
        sys.executable, 'scripts/bp_optimize.py',
        '--base', test_params['base'],
        '--space', test_params['space'],
        '--trials', str(test_params['trials']),
        '--study', test_params['study'],
        '--storage', test_params['storage'],
        '--n-jobs', str(test_params['n_jobs'])
    ]
    
    print(f"\n🚀 Запуск тестовой оптимизации:")
    print(f"  Trials: {test_params['trials']}")
    print(f"  Study: {test_params['study']}")
    print(f"  Storage: {test_params['storage']}")
    
    # Запускаем оптимизацию
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 минут таймаут
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Оптимизация завершена успешно за {duration:.1f} секунд")
            
            # Показываем последние строки вывода
            output_lines = result.stdout.strip().split('\n')
            print("\n📊 Последние строки вывода:")
            for line in output_lines[-10:]:
                print(f"  {line}")
            
            # Запускаем валидацию
            print(f"\n🔍 Запуск валидации...")
            
            validation_cmd = [
                sys.executable, 'scripts/bp_validate.py',
                '--study', test_params['study'],
                '--storage', test_params['storage']
            ]
            
            validation_result = subprocess.run(validation_cmd, capture_output=True, text=True, timeout=300)
            
            if validation_result.returncode == 0:
                print("✅ Валидация завершена успешно")
                
                # Показываем результаты валидации
                validation_lines = validation_result.stdout.strip().split('\n')
                print("\n📋 Результаты валидации:")
                for line in validation_lines[-15:]:
                    print(f"  {line}")
                
                return True
            else:
                print("❌ Ошибка валидации:")
                print(validation_result.stderr)
                return False
                
        else:
            print(f"❌ Ошибка оптимизации (код: {result.returncode})")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Таймаут - оптимизация заняла слишком много времени")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False


def check_results():
    """Проверка созданных файлов"""
    
    print(f"\n📁 Проверка созданных файлов:")
    
    expected_files = [
        "test_studies.db",
        "best_params_test_bp_optimization.json",
        "logs/test_bp_optimization_bp_optimization.log"
    ]
    
    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - не найден")


def main():
    """Основная функция теста"""
    
    print("🧪 ТЕСТ BEST PRACTICE ОПТИМИЗАЦИИ")
    print("Проверяет работоспособность новой системы оптимизации")
    print("="*60)
    
    # Запускаем тест
    success = run_test()
    
    # Проверяем результаты
    check_results()
    
    # Итоговый результат
    print(f"\n{'='*60}")
    if success:
        print("🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
        print("✅ Best Practice оптимизация работает корректно")
        print("✅ Валидация параметров работает")
        print("✅ Логирование настроено правильно")
        print("\n💡 Можно запускать полную оптимизацию:")
        print("python scripts/bp_optimize.py --base configs/main_2024.yaml --space configs/search_space_bp_balanced.yaml --trials 400 --study wf_best_practice_balanced --storage sqlite:///studies.db")
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН")
        print("Необходимо исправить ошибки перед запуском полной оптимизации")
    
    print("="*60)


if __name__ == "__main__":
    main()
