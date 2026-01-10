#!/usr/bin/env python3
"""
Упрощенная система мониторинга производительности тестов
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


def run_test_command(name: str, command: str) -> dict:
    """Запускает команду тестирования и собирает метрики."""
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
    
    # Простой парсинг результатов
    output = result.stdout + result.stderr
    passed = output.count(' passed')
    failed = output.count(' failed')
    skipped = output.count(' skipped')
    
    return {
        'name': name,
        'duration': round(duration, 2),
        'success': result.returncode == 0,
        'tests': {'passed': passed, 'failed': failed, 'skipped': skipped},
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Основная функция мониторинга."""
    
    print("📊 УПРОЩЕННЫЙ МОНИТОРИНГ ТЕСТОВ")
    print("=" * 40)
    
    # Определяем тесты для мониторинга
    test_commands = [
        ("smoke", "./scripts/test_smoke.sh"),
        ("super_fast", "./scripts/test_super_fast.sh"),
        ("critical_fixes", "pytest tests/test_critical_fixes_consolidated.py -q"),
    ]
    
    results = []
    total_time = 0
    
    for name, command in test_commands:
        result = run_test_command(name, command)
        results.append(result)
        total_time += result['duration']
        
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name}: {result['duration']}s")
    
    print(f"\n📊 ИТОГИ:")
    print(f"Общее время: {total_time:.1f}s")
    print(f"Успешных: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    # Сохраняем результаты
    with open('test_metrics_simple.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Результаты сохранены в test_metrics_simple.json")
    
    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    exit(main())
