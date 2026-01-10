"""
Версия 3 numba движка для полного паритета с reference engine.
Учитывает двойной warmup период для mu/sigma.
"""

import numpy as np
import numba as nb
from typing import Tuple


@nb.njit(fastmath=False, cache=False)
def compute_rolling_ols_parity(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling OLS точно как в reference engine.
    Возвращает: beta, alpha
    """
    n = len(y)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    
    if window < 2:
        return beta, alpha
    
    # Начинаем с индекса window-1 (как pandas rolling)
    for i in range(window - 1, n):
        y_win = y[i-window+1:i+1]
        x_win = x[i-window+1:i+1]
        
        # Проверка на NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y_win[j]) or np.isnan(x_win[j]):
                has_nan = True
                break
        
        if has_nan:
            continue
            
        # Средние
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        # Вычисляем Cov(x,y) и Var(x)
        cov_xy = 0.0
        var_x = 0.0
        for j in range(window):
            dx = x_win[j] - x_mean
            dy = y_win[j] - y_mean
            cov_xy += dx * dy
            var_x += dx * dx
        
        # Pandas использует ddof=1 по умолчанию
        if window > 1:
            cov_xy = cov_xy / (window - 1.0)
            var_x = var_x / (window - 1.0)
        
        if abs(var_x) < 1e-10:
            continue
            
        beta[i] = cov_xy / var_x
        alpha[i] = y_mean - beta[i] * x_mean
    
    return beta, alpha


@nb.njit(fastmath=False, cache=False)
def compute_spread_stats_parity(spread: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling mean и std для spread как в reference (pandas rolling).
    """
    n = len(spread)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    
    if window < 2:
        return mu, sigma
    
    # Начинаем с индекса window-1
    for i in range(window - 1, n):
        spread_win = spread[i-window+1:i+1]
        
        # Проверка на NaN
        has_nan = False
        valid_count = 0
        for j in range(window):
            if not np.isnan(spread_win[j]):
                valid_count += 1
            else:
                has_nan = True
        
        # Pandas требует минимум window валидных значений
        if valid_count < window:
            continue
        
        # Среднее
        spread_sum = 0.0
        for j in range(window):
            if not np.isnan(spread_win[j]):
                spread_sum += spread_win[j]
        mu_val = spread_sum / valid_count
        
        # Стандартное отклонение (pandas использует ddof=1)
        spread_sq_sum = 0.0
        for j in range(window):
            if not np.isnan(spread_win[j]):
                diff = spread_win[j] - mu_val
                spread_sq_sum += diff * diff
        
        if valid_count > 1:
            sigma_val = np.sqrt(spread_sq_sum / (valid_count - 1))
        else:
            sigma_val = 0.0
        
        if sigma_val > 1e-8:
            mu[i] = mu_val
            sigma[i] = sigma_val
    
    return mu, sigma


@nb.njit(fastmath=False, cache=False)
def compute_positions_parity(y: np.ndarray, x: np.ndarray,
                            rolling_window: int,
                            z_enter: float,
                            z_exit: float,
                            max_holding_period: int,
                            commission: float,
                            slippage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Торговая логика с полным паритетом к reference engine.
    
    Returns:
        positions, trades, pnl_series, z_scores, spreads
    """
    n = len(y)
    
    # Выходные массивы
    positions = np.zeros(n)
    trades = np.zeros(n)
    pnl_series = np.zeros(n)
    z_scores = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    
    # 1. Вычисляем beta и alpha (rolling OLS)
    beta, alpha = compute_rolling_ols_parity(y, x, rolling_window)
    
    # 2. Вычисляем spread
    for i in range(n):
        if not np.isnan(alpha[i]) and not np.isnan(beta[i]):
            spreads[i] = y[i] - (alpha[i] + beta[i] * x[i])
    
    # 3. Вычисляем mu и sigma для spread (еще один rolling)
    mu, sigma = compute_spread_stats_parity(spreads, rolling_window)
    
    # 4. Вычисляем z-scores
    for i in range(n):
        if not np.isnan(mu[i]) and not np.isnan(sigma[i]) and sigma[i] > 1e-8:
            if not np.isnan(spreads[i]):
                z_scores[i] = (spreads[i] - mu[i]) / sigma[i]
    
    # 5. Торговая логика (точно как в reference)
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
        
        # УДЕРЖАНИЕ (этот блок не должен выполняться)
        else:
            positions[i] = prev_pos
            if prev_pos != 0:
                holding_period += 1
    
    # 6. Расчет PnL (как в reference)
    # spread_returns = np.diff(spread, prepend=spread[0])
    for i in range(1, n):
        if not np.isnan(spreads[i]) and not np.isnan(spreads[i-1]):
            spread_return = spreads[i] - spreads[i-1]
            pnl_series[i] = positions[i] * spread_return
    
    # 7. Учет издержек
    for i in range(1, n):
        position_change = abs(positions[i] - positions[i-1])
        trades[i] = position_change
        
        if position_change > 0:
            # Примерная стоимость как в reference
            trade_cost = (commission + slippage) * 100.0
            pnl_series[i] -= trade_cost
    
    return positions, trades, pnl_series, z_scores, spreads


@nb.njit(fastmath=False, cache=False)
def compute_positions_parity_debug(y: np.ndarray, x: np.ndarray,
                                  rolling_window: int,
                                  z_enter: float,
                                  z_exit: float,
                                  max_holding_period: int,
                                  commission: float,
                                  slippage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Debug версия торговой логики с расширенным выводом для трассировки.
    
    Returns:
        positions, trades, pnl_series, z_scores, spreads,
        entries_idx, exits_idx, mu, sigma, beta, alpha
    """
    n = len(y)
    
    # Выходные массивы (основные)
    positions = np.zeros(n)
    trades = np.zeros(n)
    pnl_series = np.zeros(n)
    z_scores = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    
    # Debug массивы 
    entries_idx = np.zeros(n, dtype=np.bool_)
    exits_idx = np.zeros(n, dtype=np.bool_)
    
    # 1. Вычисляем beta и alpha (rolling OLS)
    beta, alpha = compute_rolling_ols_parity(y, x, rolling_window)
    
    # 2. Вычисляем spread
    for i in range(n):
        if not np.isnan(alpha[i]) and not np.isnan(beta[i]):
            spreads[i] = y[i] - (alpha[i] + beta[i] * x[i])
    
    # 3. Вычисляем mu и sigma для spread (еще один rolling)
    mu, sigma = compute_spread_stats_parity(spreads, rolling_window)
    
    # 4. Вычисляем z-scores
    for i in range(n):
        if not np.isnan(mu[i]) and not np.isnan(sigma[i]) and sigma[i] > 1e-8:
            if not np.isnan(spreads[i]):
                z_scores[i] = (spreads[i] - mu[i]) / sigma[i]
    
    # 5. Торговая логика (точно как в reference)
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
                exits_idx[i] = True  # Debug: маркируем выход
                holding_period = 0
            else:
                positions[i] = prev_pos
                holding_period += 1
        
        # ВХОД (только если не в позиции)
        elif prev_pos == 0:
            if current_z > z_enter:
                positions[i] = -1  # Short spread
                entries_idx[i] = True  # Debug: маркируем вход
                holding_period = 1
                
            elif current_z < -z_enter:
                positions[i] = 1  # Long spread
                entries_idx[i] = True  # Debug: маркируем вход
                holding_period = 1
                
            else:
                positions[i] = 0
        
        # УДЕРЖАНИЕ (этот блок не должен выполняться)
        else:
            positions[i] = prev_pos
            if prev_pos != 0:
                holding_period += 1
    
    # 6. Расчет PnL (как в reference)
    for i in range(1, n):
        if not np.isnan(spreads[i]) and not np.isnan(spreads[i-1]):
            spread_return = spreads[i] - spreads[i-1]
            pnl_series[i] = positions[i] * spread_return
    
    # 7. Учет издержек
    for i in range(1, n):
        position_change = abs(positions[i] - positions[i-1])
        trades[i] = position_change
        
        if position_change > 0:
            # Примерная стоимость как в reference
            trade_cost = (commission + slippage) * 100.0
            pnl_series[i] -= trade_cost
    
    return positions, trades, pnl_series, z_scores, spreads, entries_idx, exits_idx, mu, sigma, beta, alpha