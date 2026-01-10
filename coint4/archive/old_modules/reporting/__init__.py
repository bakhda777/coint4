"""Модуль отчетности для системы парного трейдинга.

Включает:
- Генератор детализированных отчетов
- HTML-форматирование
- Анализ чувствительности
- Визуализация результатов
"""

from .detailed_report import DetailedReportGenerator

__all__ = [
    'DetailedReportGenerator'
]