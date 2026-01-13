"""
Версия 2 numba движка для достижения паритета с reference engine.
Полностью новые имена функций чтобы избежать проблем с кэшем.
"""

import numpy as np
import numba as nb
from typing import Tuple


@nb.njit(fastmath=False, cache=False)
def compute_rolling_stats_v2(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling OLS как в reference engine.
    Возвращает: beta, alpha, mu, sigma
    """
    n = len(y)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    
    if window < 2:
        return beta, alpha, mu, sigma
    
    for i in range(window - 1, n):
        start_idx = i - window + 1

        # Проверка на NaN и расчет средних
        has_nan = False
        x_sum = 0.0
        y_sum = 0.0
        for j in range(start_idx, i + 1):
            if np.isnan(y[j]) or np.isnan(x[j]):
                has_nan = True
                break
            x_sum += x[j]
            y_sum += y[j]

        if has_nan:
            continue

        x_mean = x_sum / window
        y_mean = y_sum / window

        # Вычисляем beta = Cov(x,y) / Var(x)
        cov_xy = 0.0
        var_x = 0.0
        for j in range(start_idx, i + 1):
            dx = x[j] - x_mean
            dy = y[j] - y_mean
            cov_xy += dx * dy
            var_x += dx * dx

        cov_xy = cov_xy / (window - 1.0)
        var_x = var_x / (window - 1.0)

        if abs(var_x) < 1e-10:
            continue

        beta_val = cov_xy / var_x
        alpha_val = y_mean - beta_val * x_mean

        beta[i] = beta_val
        alpha[i] = alpha_val

    # Считаем spread с текущими alpha/beta как в reference engine
    spread = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(alpha[i]) and not np.isnan(beta[i]):
            spread[i] = y[i] - (alpha[i] + beta[i] * x[i])

    # Rolling mean/std по spread (как в pandas rolling)
    for i in range(window - 1, n):
        start_idx = i - window + 1
        has_nan = False
        spread_sum = 0.0

        for j in range(start_idx, i + 1):
            val = spread[j]
            if np.isnan(val):
                has_nan = True
                break
            spread_sum += val

        if has_nan:
            continue

        mu_val = spread_sum / window
        spread_sq_sum = 0.0
        for j in range(start_idx, i + 1):
            diff = spread[j] - mu_val
            spread_sq_sum += diff * diff

        sigma_val = np.sqrt(spread_sq_sum / (window - 1.0))
        if sigma_val > 1e-8:
            mu[i] = mu_val
            sigma[i] = sigma_val
    
    return beta, alpha, mu, sigma


@nb.njit(fastmath=False, cache=False)
def compute_positions_v2(y: np.ndarray, x: np.ndarray,
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
    n = len(y)
    
    # Выходные массивы
    positions = np.zeros(n)  # -1, 0, +1 как в reference
    trades = np.zeros(n)
    pnl_series = np.zeros(n)
    z_scores = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    
    # Вычисляем rolling статистики
    beta, alpha, mu, sigma = compute_rolling_stats_v2(y, x, rolling_window)
    
    # Сначала вычисляем все spreads и z-scores
    for i in range(n):
        if not np.isnan(alpha[i]) and not np.isnan(beta[i]):
            spreads[i] = y[i] - (alpha[i] + beta[i] * x[i])
            
            if not np.isnan(mu[i]) and not np.isnan(sigma[i]) and sigma[i] > 1e-8:
                z_scores[i] = (spreads[i] - mu[i]) / sigma[i]
    
    # Торговая логика (точно как в reference)
    holding_period = 0
    
    for i in range(rolling_window, n):
        # Пропускаем если z-score невалидный
        if np.isnan(z_scores[i]):
            if i > 0:
                positions[i] = positions[i-1]
            continue
        
        current_z = z_scores[i]
        prev_pos = positions[i-1] if i > 0 else 0.0
        
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
        
        # Это else никогда не должно выполняться
        else:
            positions[i] = prev_pos
            if prev_pos != 0:
                holding_period += 1
    
    # Расчет PnL (как в reference)
    for i in range(1, n):
        if not np.isnan(spreads[i]) and not np.isnan(spreads[i-1]):
            spread_return = spreads[i] - spreads[i-1]
            pnl_series[i] = positions[i] * spread_return
    
    # Учет издержек
    for i in range(1, n):
        position_change = abs(positions[i] - positions[i-1])
        trades[i] = position_change
        
        if position_change > 0:
            trade_cost = (commission + slippage) * 100.0  # Примерная стоимость
            pnl_series[i] -= trade_cost
    
    return positions, trades, pnl_series, z_scores, spreads
