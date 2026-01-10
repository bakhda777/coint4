#!/usr/bin/env python3
"""
Константы для Optuna оптимизации.
"""

# Оптимизированная система штрафов с более плавными переходами
PENALTY_SOFT = -2.0   # Уменьшен для лучшего баланса - плохие, но валидные результаты
PENALTY_MEDIUM = -10.0  # Добавлен средний уровень для критичных, но не фатальных ошибок
PENALTY_HARD = -50.0  # Сохранен для системных ошибок и невалидных результатов
PENALTY = PENALTY_SOFT  # Обратная совместимость

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

INTERMEDIATE_REPORT_INTERVAL = 10   # Увеличено до 10 пар для снижения overhead при больших объемах
