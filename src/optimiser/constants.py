#!/usr/bin/env python3
"""
Константы для Optuna оптимизации.
"""

# Умеренный штраф для исключений и нечисловых метрик (исправлено для TPE)
PENALTY = -5.0

# Минимальное количество сделок для валидного результата
MIN_TRADES_THRESHOLD = 10

# Пороги для штрафов за просадку
MAX_DRAWDOWN_SOFT_THRESHOLD = 0.25  # 25% - начало штрафа
MAX_DRAWDOWN_HARD_THRESHOLD = 0.50  # 50% - усиленный штраф

# Пороги для бонусов/штрафов за win rate
WIN_RATE_BONUS_THRESHOLD = 0.55  # 55% - начало бонуса
WIN_RATE_PENALTY_THRESHOLD = 0.40  # 40% - начало штрафа

# Коэффициенты для расчета композитного скора
DD_PENALTY_SOFT_MULTIPLIER = 3.0
DD_PENALTY_HARD_MULTIPLIER = 5.0
WIN_RATE_BONUS_MULTIPLIER = 0.5
WIN_RATE_PENALTY_MULTIPLIER = 1.0

# Настройки для промежуточных отчетов
INTERMEDIATE_REPORT_INTERVAL = 20  # Отчет каждые N пар
