#!/usr/bin/env python3
"""
Скрипт валидации результатов оптимизации.

Проверяет оптимизированные параметры на отложенных данных и 
генерирует отчет о производительности.

Использование:
    python scripts/validate.py --study outputs/studies/bp_optimization.db
    python scripts/validate.py --config configs/best_config.yaml
"""

import sys
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

from optimiser.unified_optimization import validate_best_params


def main():
    """Основная функция валидации."""
    parser = argparse.ArgumentParser(
        description="Валидация результатов оптимизации",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Валидация лучших параметров из Optuna study
  python scripts/validate.py --study outputs/studies/bp_optimization.db
  
  # Валидация конкретного конфига
  python scripts/validate.py --config configs/best_config.yaml
  
  # С кастомными датами валидации
  python scripts/validate.py --study outputs/studies/bp_optimization.db \\
    --start-date 2024-01-01 --end-date 2024-06-30
        """
    )
    
    # Источник параметров
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--study', 
                      help='Путь к Optuna study БД')
    group.add_argument('--config',
                      help='Путь к YAML конфигу с параметрами')
    
    # Базовая конфигурация
    parser.add_argument('--base-config', '-b',
                       default='configs/main_2024.yaml',
                       help='Путь к базовой конфигурации')
    
    # Период валидации
    parser.add_argument('--start-date',
                       help='Начальная дата валидации (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                       help='Конечная дата валидации (YYYY-MM-DD)')
    
    # Выходные файлы
    parser.add_argument('--output-dir', '-o',
                       default='results/validation',
                       help='Директория для сохранения результатов')
    parser.add_argument('--save-trades', action='store_true',
                       help='Сохранить детальный лог сделок')
    
    # Дополнительные опции
    parser.add_argument('--pairs-csv',
                       help='CSV файл с парами для валидации')
    parser.add_argument('--n-jobs', '-j', type=int, default=1,
                       help='Количество параллельных процессов')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    # Запуск валидации (пока заглушка, будет реализовано после рефакторинга)
    print(f"Валидация параметров из {'study' if args.study else 'config'}: {args.study or args.config}")
    print(f"Базовый конфиг: {args.base_config}")
    print(f"Результаты будут сохранены в: {args.output_dir}")
    
    # TODO: вызов функции валидации после рефакторинга модулей


if __name__ == "__main__":
    main()