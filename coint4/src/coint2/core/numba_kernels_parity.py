"""
Исправленная версия numba_kernels для достижения паритета с reference engine.
"""

import numpy as np
import numba as nb
from typing import Tuple

# Простейшие настройки для стабильности
import os
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


@nb.njit(fastmath=True, cache=False)
def rolling_ols_reference(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling OLS как в reference engine.
    Возвращает: beta, alpha, mu, sigma
    """
    n = y.size
    beta = np.full(n, np.nan, dtype=np.float64)
    alpha = np.full(n, np.nan, dtype=np.float64)
    mu = np.full(n, np.nan, dtype=np.float64)
    sigma = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(window, n):
        y_win = y[i-window:i]
        x_win = x[i-window:i]
        
        # Проверка на NaN
        if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
            continue
            
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        # Вычисляем beta = Cov(x,y) / Var(x)
        cov_xy = 0.0
        var_x = 0.0
        for j in range(window):
            cov_xy += (x_win[j] - x_mean) * (y_win[j] - y_mean)
            var_x += (x_win[j] - x_mean) ** 2
        
        # Деление с защитой
        if window > 1:
            cov_xy = cov_xy / float(window - 1)
            var_x = var_x / float(window - 1)
        else:
            # Для window=1 пропускаем
            continue
        
        if abs(var_x) > 1e-10:
            beta_val = cov_xy / var_x
            alpha_val = y_mean - beta_val * x_mean
            
            # Вычисляем spread = y - (alpha + beta * x)
            spread_win = np.zeros(window)
            for j in range(window):
                spread_win[j] = y_win[j] - (alpha_val + beta_val * x_win[j])
            
            mu_val = np.mean(spread_win)
            sigma_val = np.std(spread_win)
            
            if sigma_val > 1e-8:
                beta[i] = beta_val
                alpha[i] = alpha_val
                mu[i] = mu_val
                sigma[i] = sigma_val
    
    return beta, alpha, mu, sigma


@nb.njit(fastmath=True, cache=False)
def calculate_positions_parity(y: np.ndarray, x: np.ndarray,
                              rolling_window: int,
                              z_enter: float,
                              z_exit: float,
                              max_holding_period: int,
                              commission: float,
                              slippage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Торговая логика максимально близкая к reference engine.
    
    Returns:
        positions, trades, pnl_series, z_scores, spreads
    """
    n = y.size
    
    # Выходные массивы
    positions = np.zeros(n, dtype=np.int32)  # -1, 0, +1 как в reference
    trades = np.zeros(n, dtype=np.float64)
    pnl_series = np.zeros(n, dtype=np.float64)
    z_scores = np.full(n, np.nan, dtype=np.float64)
    spreads = np.full(n, np.nan, dtype=np.float64)
    
    # Вычисляем rolling статистики
    beta, alpha, mu, sigma = rolling_ols_reference(y, x, rolling_window)
    
    # Сначала вычисляем все spreads и z-scores
    for i in range(n):
        if np.isfinite(alpha[i]) and np.isfinite(beta[i]):
            spreads[i] = y[i] - (alpha[i] + beta[i] * x[i])
            
            if np.isfinite(mu[i]) and np.isfinite(sigma[i]) and sigma[i] > 1e-8:
                z_scores[i] = (spreads[i] - mu[i]) / sigma[i]
    
    # Торговая логика (точно как в reference)
    holding_period = 0
    
    for i in range(rolling_window, n):
        # Пропускаем если z-score невалидный
        if not np.isfinite(z_scores[i]):
            positions[i] = positions[i-1] if i > 0 else 0
            continue
        
        current_z = z_scores[i]
        prev_pos = positions[i-1] if i > 0 else 0
        
        # ВЫХОД (приоритет над входом)
        if prev_pos != 0:
            should_exit = False
            
            # Выход по z-score
            if abs(current_z) <= z_exit:
                should_exit = True
            
            # Выход по времени
            elif holding_period >= max_holding_period:
                should_exit = True
            
            if should_exit:
                positions[i] = 0
                holding_period = 0
            else:
                positions[i] = prev_pos
                holding_period += 1
        
        # ВХОД (только если не в позиции)
        elif prev_pos == 0:
            if current_z > z_enter:
                positions[i] = -1  # Short spread
                holding_period = 1
                
            elif current_z < -z_enter:
                positions[i] = 1  # Long spread
                holding_period = 1
                
            else:
                positions[i] = 0
        
        # УДЕРЖАНИЕ
        else:
            positions[i] = prev_pos
            if prev_pos != 0:
                holding_period += 1
    
    # Расчет PnL (как в reference)
    spread_returns = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(spreads[i]) and np.isfinite(spreads[i-1]):
            spread_returns[i] = spreads[i] - spreads[i-1]
    
    # PnL от позиций
    for i in range(n):
        pnl_series[i] = positions[i] * spread_returns[i]
    
    # Учет издержек
    position_changes = np.zeros(n)
    for i in range(1, n):
        position_changes[i] = abs(positions[i] - positions[i-1])
    
    trade_costs = position_changes * (commission + slippage) * 100  # Примерная стоимость
    
    # Net PnL
    for i in range(n):
        pnl_series[i] = pnl_series[i] - trade_costs[i]
        trades[i] = position_changes[i]
    
    return positions.astype(np.float64), trades, pnl_series, z_scores, spreads
