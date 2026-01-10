"""
Исправленная версия numba_kernels.py с минималистичной торговой логикой.
Убраны все дополнительные фильтры для отладки проблемы "0 сделок".
"""

import numpy as np
import numba as nb
from typing import Tuple

# Простейшие настройки для стабильности
import os
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


@nb.njit(fastmath=True, cache=True)
def rolling_ols_simple(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Упрощенная версия rolling OLS.
    """
    n = y.size
    beta = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    
    for i in range(window, n):
        y_win = y[i-window:i]
        x_win = x[i-window:i]
        
        # Проверка на NaN
        if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
            continue
            
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        # Вычисляем beta
        numerator = 0.0
        denominator = 0.0
        for j in range(window):
            numerator += (x_win[j] - x_mean) * (y_win[j] - y_mean)
            denominator += (x_win[j] - x_mean) ** 2
            
        if abs(denominator) > 1e-10:
            beta_val = numerator / denominator
            
            # Вычисляем spread и его статистики
            spread_win = y_win - beta_val * x_win
            mu_val = np.mean(spread_win)
            sigma_val = np.std(spread_win)
            
            if sigma_val > 1e-10:
                beta[i] = beta_val
                mu[i] = mu_val
                sigma[i] = sigma_val
    
    return beta, mu, sigma


@nb.njit(fastmath=True, cache=True)
def calculate_positions_simple(y: np.ndarray, x: np.ndarray,
                              rolling_window: int,
                              z_enter: float,
                              z_exit: float,
                              commission: float,
                              slippage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Максимально упрощенная торговая логика - как в reference engine.
    
    Returns:
        positions, trades, pnl_series, z_scores, spreads
    """
    n = y.size
    
    # Выходные массивы
    positions = np.zeros(n, dtype=np.float32)
    trades = np.zeros(n, dtype=np.float32)
    pnl_series = np.zeros(n, dtype=np.float32)
    z_scores = np.full(n, np.nan, dtype=np.float32)
    spreads = np.full(n, np.nan, dtype=np.float32)
    
    # Вычисляем rolling статистики
    beta, mu, sigma = rolling_ols_simple(y, x, rolling_window)
    
    # Начальные значения
    position = 0.0
    entry_price = 0.0
    num_trades = 0
    
    # КРИТИЧНО: начинаем с rolling_window + 1 чтобы иметь предыдущий бар
    for i in range(rolling_window + 1, n):
        
        # Проверяем валидность статистик
        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = position
            continue
            
        if sigma[i] < 1e-10:
            positions[i] = position
            continue
        
        # ИСПОЛЬЗУЕМ ПРЕДЫДУЩИЙ БАР для расчета сигнала (избегаем lookahead)
        current_spread = y[i-1] - beta[i] * x[i-1]
        current_z = (current_spread - mu[i]) / sigma[i]
        
        # Сохраняем для диагностики
        z_scores[i] = current_z
        spreads[i] = current_spread
        
        # Простая торговая логика
        new_position = position
        
        if position == 0.0:
            # Входим в позицию
            if current_z > z_enter:
                new_position = -1.0  # Short spread
                entry_price = current_spread
                num_trades += 1
            elif current_z < -z_enter:
                new_position = 1.0   # Long spread
                entry_price = current_spread
                num_trades += 1
                
        else:
            # Выходим из позиции
            if position > 0 and current_z <= z_exit:
                new_position = 0.0
                num_trades += 1
            elif position < 0 and current_z >= -z_exit:
                new_position = 0.0
                num_trades += 1
        
        # Обновляем позицию и считаем PnL
        if new_position != position:
            trades[i] = abs(new_position - position)
            
            # Комиссии
            cost = trades[i] * (commission + slippage)
            
            # PnL от закрытия позиции
            if position != 0.0 and new_position == 0.0:
                # Используем текущий спред для закрытия
                exit_spread = y[i] - beta[i] * x[i]
                price_pnl = position * (exit_spread - entry_price)
                pnl_series[i] = price_pnl - cost
            else:
                # Только комиссии при открытии
                pnl_series[i] = -cost
        
        position = new_position
        positions[i] = position
    
    return positions, trades, pnl_series, z_scores, spreads


@nb.njit(fastmath=True, cache=True)
def calculate_positions_debug(y: np.ndarray, x: np.ndarray,
                             rolling_window: int,
                             z_enter: float,
                             z_exit: float) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Версия для отладки - возвращает диагностическую информацию.
    
    Returns:
        z_scores, positions, num_trades, max_abs_z, pct_above_threshold
    """
    n = y.size
    
    z_scores = np.full(n, np.nan, dtype=np.float32)
    positions = np.zeros(n, dtype=np.float32)
    
    # Вычисляем rolling статистики
    beta, mu, sigma = rolling_ols_simple(y, x, rolling_window)
    
    position = 0.0
    num_trades = 0
    points_above_threshold = 0
    
    for i in range(rolling_window + 1, n):
        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = position
            continue
            
        if sigma[i] < 1e-10:
            positions[i] = position
            continue
        
        # Используем предыдущий бар
        current_spread = y[i-1] - beta[i] * x[i-1]
        current_z = (current_spread - mu[i]) / sigma[i]
        z_scores[i] = current_z
        
        # Считаем статистику
        if abs(current_z) > z_enter:
            points_above_threshold += 1
        
        # Торговая логика
        new_position = position
        
        if position == 0.0:
            if current_z > z_enter:
                new_position = -1.0
                num_trades += 1
            elif current_z < -z_enter:
                new_position = 1.0
                num_trades += 1
        else:
            if (position > 0 and current_z <= z_exit) or (position < 0 and current_z >= -z_exit):
                new_position = 0.0
                num_trades += 1
        
        position = new_position
        positions[i] = position
    
    # Статистика
    valid_z = z_scores[~np.isnan(z_scores)]
    max_abs_z = np.max(np.abs(valid_z)) if len(valid_z) > 0 else 0.0
    pct_above = points_above_threshold / max(1, n - rolling_window - 1) * 100
    
    return z_scores, positions, num_trades, max_abs_z, pct_above


# Функция прогрева
@nb.njit(fastmath=True, cache=True)
def warmup():
    """Прогрев JIT компиляции."""
    n = 100
    test_y = np.arange(n, dtype=np.float32) + 100.0
    test_x = np.arange(n, dtype=np.float32) + 100.1
    
    # Прогреваем функции
    rolling_ols_simple(test_y, test_x, 20)
    calculate_positions_simple(test_y, test_x, 20, 2.0, 0.5, 0.001, 0.0005)
    calculate_positions_debug(test_y, test_x, 20, 2.0, 0.5)
    
    return True


# Автоматический прогрев при импорте
warmup()