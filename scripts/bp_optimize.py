#!/usr/bin/env python3
"""
Скрипт оптимизации для BP (Best Practices) режима.
Использует унифицированный оптимизатор в bp режиме.
"""

import sys
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

from optimiser.unified_optimization import main as unified_main


def main():
    """Основная функция BP оптимизации."""
    parser = argparse.ArgumentParser(description="BP оптимизация параметров")
    parser.add_argument('--base-config', '-b', 
                       default='configs/main_2024.yaml',
                       help='Путь к базовой конфигурации')
    parser.add_argument('--search-space', '-s',
                       default='configs/search_space.yaml', 
                       help='Путь к пространству поиска')
    parser.add_argument('--n-trials', '-n', type=int, default=50,
                       help='Количество испытаний')
    parser.add_argument('--output', '-o',
                       default='results/bp_optimization_results.db',
                       help='Путь к файлу результатов')
    
    args = parser.parse_args()
    
    # Формируем аргументы для унифицированного оптимизатора
    unified_args = [
        'bp',  # режим
        '--base-config', args.base_config,
        '--search-space', args.search_space,
        '--n-trials', str(args.n_trials),
        '--output', args.output,
        '--interval-steps', '5',  # interval_steps=5 для стандартизации
        '--n-warmup-steps', '30'  # n_warmup_steps=30 для стандартизации
    ]
    
    # Подменяем sys.argv для унифицированного оптимизатора
    original_argv = sys.argv
    try:
        sys.argv = ['unified_optimization.py'] + unified_args
        unified_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
