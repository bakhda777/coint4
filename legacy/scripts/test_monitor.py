#!/usr/bin/env python3
"""
Система мониторинга производительности тестов

Запускает различные наборы тестов и собирает метрики производительности.
Создает отчеты для анализа и оптимизации.
"""

import subprocess
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def run_test_suite(name: str, command: str) -> Dict[str, Any]:
    """Запускает набор тестов и собирает метрики."""
    print(f"🧪 Запуск {name}...")
    
    start_time = time.time()
    result = subprocess.run(
        command, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=Path(__file__).parent.parent
    )
    duration = time.time() - start_time
    
    # Парсим вывод pytest для получения статистики
    output = result.stdout + result.stderr
    
    # Подсчитываем тесты
    passed_count = output.count(' PASSED')
    failed_count = output.count(' FAILED')
    skipped_count = output.count(' SKIPPED')
    error_count = output.count(' ERROR')
    
    # Ищем информацию о времени выполнения
    slowest_tests = []
    if "slowest" in output:
        lines = output.split('\n')
        in_slowest_section = False
        for line in lines:
            if "slowest" in line.lower() and "durations" in line.lower():
                in_slowest_section = True
                continue
            if in_slowest_section and line.strip():
                if line.startswith('=') or line.startswith('-'):
                    break
                if 's call' in line or 's setup' in line:
                    slowest_tests.append(line.strip())
    
    return {
        'name': name,
        'command': command,
        'duration': round(duration, 2),
        'success': result.returncode == 0,
        'return_code': result.returncode,
        'tests': {
            'passed': passed_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'errors': error_count,
            'total': passed_count + failed_count + skipped_count + error_count
        },
        'slowest_tests': slowest_tests[:5],  # Топ 5 самых медленных
        'timestamp': datetime.now().isoformat(),
        'output_sample': output[:500] if result.returncode != 0 else ""  # Образец вывода при ошибках
    }


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Генерирует текстовый отчет."""
    
    report = []
    report.append("📊 ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ ТЕСТОВ")
    report.append("=" * 50)
    report.append(f"Время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Сводная таблица
    report.append("📈 СВОДНАЯ ТАБЛИЦА")
    report.append("-" * 30)
    report.append(f"{'Набор':<20} {'Время':<8} {'Тесты':<8} {'Статус':<10}")
    report.append("-" * 30)
    
    total_duration = 0
    total_tests = 0
    successful_suites = 0
    
    for result in results:
        status = "✅ OK" if result['success'] else "❌ FAIL"
        tests_info = f"{result['tests']['total']}"
        duration_str = f"{result['duration']}s"
        
        report.append(f"{result['name']:<20} {duration_str:<8} {tests_info:<8} {status:<10}")
        
        total_duration += result['duration']
        total_tests += result['tests']['total']
        if result['success']:
            successful_suites += 1
    
    report.append("-" * 30)
    report.append(f"{'ИТОГО':<20} {total_duration:.1f}s{'':<3} {total_tests:<8} {successful_suites}/{len(results)}")
    report.append("")
    
    # Детальная информация по каждому набору
    for result in results:
        report.append(f"🔍 {result['name'].upper()}")
        report.append("-" * 20)
        report.append(f"Команда: {result['command']}")
        report.append(f"Время выполнения: {result['duration']}s")
        report.append(f"Статус: {'✅ Успешно' if result['success'] else '❌ Неудачно'}")
        
        tests = result['tests']
        if tests['total'] > 0:
            report.append(f"Тесты: {tests['passed']} прошли, {tests['failed']} провалились, {tests['skipped']} пропущены")
            
            if tests['failed'] > 0:
                report.append("⚠️ Есть провалившиеся тесты!")
        
        if result['slowest_tests']:
            report.append("🐌 Самые медленные тесты:")
            for test in result['slowest_tests']:
                report.append(f"  • {test}")
        
        if result['output_sample']:
            report.append("📝 Образец вывода ошибки:")
            report.append(result['output_sample'])
        
        report.append("")
    
    return "\n".join(report)


def main():
    """Основная функция мониторинга."""
    
    print("🚀 СИСТЕМА МОНИТОРИНГА ПРОИЗВОДИТЕЛЬНОСТИ ТЕСТОВ")
    print("=" * 60)
    print()
    
    # Определяем наборы тестов для мониторинга
    test_suites = [
        ("smoke", "./scripts/test_smoke.sh"),
        ("fast", "pytest -n auto -m 'fast and not serial' --maxfail=5 -q --tb=short"),
        ("critical_fixes", "pytest -m critical_fixes --maxfail=3 -q --tb=short"),
        ("serial", "pytest -m 'serial and not slow' --maxfail=3 -q --tb=short"),
    ]
    
    results = []
    
    # Запускаем каждый набор тестов
    for name, command in test_suites:
        try:
            result = run_test_suite(name, command)
            results.append(result)
            
            # Показываем промежуточный результат
            status = "✅" if result['success'] else "❌"
            print(f"{status} {name}: {result['duration']}s, {result['tests']['total']} тестов")
            
        except Exception as e:
            print(f"❌ Ошибка при запуске {name}: {e}")
            results.append({
                'name': name,
                'command': command,
                'duration': 0,
                'success': False,
                'return_code': -1,
                'tests': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1, 'total': 1},
                'slowest_tests': [],
                'timestamp': datetime.now().isoformat(),
                'output_sample': str(e)
            })
    
    print()
    print("📊 Генерация отчетов...")
    
    # Сохраняем JSON отчет
    json_path = Path("test_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Генерируем и сохраняем текстовый отчет
    report = generate_report(results)
    report_path = Path("test_performance_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Выводим отчет в консоль
    print(report)
    
    print(f"📁 Отчеты сохранены:")
    print(f"  • JSON: {json_path}")
    print(f"  • Текст: {report_path}")
    
    # Возвращаем код выхода на основе результатов
    failed_suites = [r for r in results if not r['success']]
    if failed_suites:
        print(f"\n⚠️ {len(failed_suites)} наборов тестов завершились неудачно!")
        return 1
    else:
        print(f"\n🎉 Все {len(results)} наборов тестов прошли успешно!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
