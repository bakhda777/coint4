"""Numba-оптимизированные функции для бэктестинга."""

import numpy as np
import numba as nb
from typing import Tuple


@nb.njit(fastmath=True, cache=True)
def rolling_ols(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Быстрый rolling OLS с использованием кумулятивных сумм.
    
    Parameters
    ----------
    y : np.ndarray
        Зависимая переменная (float32)
    x : np.ndarray  
        Независимая переменная (float32)
    window : int
        Размер окна для rolling расчета
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        beta, mu, sigma массивы
    """
    n = y.size
    beta = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    
    # Кумулятивные суммы для O(1) обновления окна
    Sx = x.cumsum()
    Sy = y.cumsum()
    Sxx = (x * x).cumsum()
    Sxy = (x * y).cumsum()
    Syy = (y * y).cumsum()
    
    for i in range(window, n):
        j = i - window
        
        # Суммы для текущего окна
        sum_x = Sx[i] - Sx[j]
        sum_y = Sy[i] - Sy[j]
        sum_xx = Sxx[i] - Sxx[j]
        sum_xy = Sxy[i] - Sxy[j]
        sum_yy = Syy[i] - Syy[j]
        
        # Расчет beta (коэффициент регрессии)
        denom = (window * sum_xx - sum_x * sum_x)
        if abs(denom) < 1e-12:  # Избегаем деления на ноль
            continue
            
        b = (window * sum_xy - sum_x * sum_y) / denom
        beta[i] = b
        
        # Расчет среднего значения spread
        mu[i] = (sum_y - b * sum_x) / window
        
        # Расчет стандартного отклонения spread
        var = (sum_yy - 2*b*sum_xy + b*b*sum_xx) / window - mu[i]**2
        sigma[i] = np.sqrt(max(var, 1e-12))  # Избегаем отрицательной дисперсии
        
        # Дополнительная проверка на минимальную сигму
        if sigma[i] < 1e-12:
            sigma[i] = 1e-12
        
    return beta, mu, sigma


@nb.njit(cache=True, fastmath=True)
def simulate_trades(spread: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   z_enter: float, z_exit: float,
                   fee_perc: float, slippage: float) -> float:
    """JIT-компилированная торговая логика.
    
    Parameters
    ----------
    spread : np.ndarray
        Массив значений spread
    mu : np.ndarray
        Массив средних значений
    sigma : np.ndarray
        Массив стандартных отклонений
    z_enter : float
        Порог входа по z-score
    z_exit : float
        Порог выхода по z-score
    fee_perc : float
        Комиссия в процентах
    slippage : float
        Проскальзывание
        
    Returns
    -------
    float
        Общий PnL
    """
    n = spread.size
    pnl = 0.0
    pos = 0  # -1 short spread, +1 long spread, 0 flat
    
    for i in range(1, n):
        if np.isnan(spread[i]) or np.isnan(mu[i]) or sigma[i] <= 1e-12:
            continue
            
        z = (spread[i] - mu[i]) / sigma[i]
        
        if pos == 0:  # Нет позиции - ищем вход
            if z > z_enter:  # Short spread (продаем переоцененный)
                pos = -1
            elif z < -z_enter:  # Long spread (покупаем недооцененный)
                pos = 1
        elif pos == 1:  # Long позиция - ищем выход
            if z > -z_exit:  # Закрываем long
                pnl += (spread[i] - spread[i-1]) - slippage*2 - fee_perc
                pos = 0
        elif pos == -1:  # Short позиция - ищем выход
            if z < z_exit:  # Закрываем short
                pnl += (spread[i-1] - spread[i]) - slippage*2 - fee_perc
                pos = 0
                
    return pnl


@nb.njit(cache=True, fastmath=True)
def calculate_z_scores(spread: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Быстрый расчет z-scores.
    
    Parameters
    ----------
    spread : np.ndarray
        Массив значений spread
    mu : np.ndarray
        Массив средних значений
    sigma : np.ndarray
        Массив стандартных отклонений
        
    Returns
    -------
    np.ndarray
        Массив z-scores
    """
    n = spread.size
    z_scores = np.full(n, 0.0, dtype=np.float32)  # Инициализируем нулями вместо NaN
    
    for i in range(n):
        if not np.isnan(spread[i]) and not np.isnan(mu[i]) and not np.isnan(sigma[i]):
            if sigma[i] > 1e-12:
                z_scores[i] = (spread[i] - mu[i]) / sigma[i]
            else:
                z_scores[i] = 0.0
    
    return z_scores


@nb.njit(cache=True, fastmath=True)
def calculate_positions_and_pnl(y: np.ndarray, x: np.ndarray, 
                               beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                               z_enter: float, z_exit: float,
                               commission_pct: float, slippage_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Полная торговая логика с расчетом позиций и PnL.
    
    Parameters
    ----------
    y, x : np.ndarray
        Ценовые ряды
    beta, mu, sigma : np.ndarray
        Параметры регрессии
    z_enter, z_exit : float
        Пороги входа и выхода
    commission_pct, slippage_pct : float
        Торговые издержки
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        positions, trades, pnl_series, total_pnl массивы
    """
    n = y.size
    positions = np.zeros(n, dtype=np.float32)
    trades = np.zeros(n, dtype=np.float32)
    pnl_series = np.zeros(n, dtype=np.float32)
    
    position = 0.0
    total_pnl = 0.0
    
    for i in range(1, n):
        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = position
            continue
            
        # Вычисление z-score с защитой от деления на ноль
        if not np.isnan(sigma[i]) and sigma[i] > 1e-12:
            spread_curr = y[i] - beta[i] * x[i]
            z_curr = (spread_curr - mu[i]) / sigma[i]
        else:
            z_curr = 0.0
        
        # Расчет PnL от изменения цен (как в оригинале)
        if position != 0.0:
            delta_y = y[i] - y[i-1]
            delta_x = x[i] - x[i-1]
            # position представляет size_s1, size_s2 = -beta * size_s1
            size_s1 = position
            size_s2 = -beta[i] * size_s1
            pnl_change = size_s1 * delta_y + size_s2 * delta_x
            total_pnl += pnl_change
            pnl_series[i] = pnl_change
        
        # Торговая логика (упрощенная версия оригинала)
        new_position = position
        
        # Выход из позиции при z-score возврате к уровню выхода
        if position != 0.0 and abs(z_curr) <= z_exit:
            new_position = 0.0
        
        # Вход в позицию только если нет текущей позиции
        elif position == 0.0:
            if z_curr > z_enter:
                new_position = -1.0  # Short spread
            elif z_curr < -z_enter:
                new_position = 1.0   # Long spread
        
        # Расчет торговых издержек при изменении позиции
        if new_position != position:
            trade_size = abs(new_position - position)
            # Применяем комиссии и проскальзывание
            total_cost_pct = commission_pct + slippage_pct
            cost = trade_size * total_cost_pct
            total_pnl -= cost
            trades[i] = trade_size
        
        position = new_position
        positions[i] = position
    
    return positions, trades, pnl_series, total_pnl


# Прогрев функций для компиляции
def _warmup_numba_functions():
    """Прогревает Numba функции для избежания задержек компиляции."""
    dummy = np.random.rand(1000).astype(np.float32)
    dummy_int = 100
    
    # Прогрев rolling_ols
    rolling_ols(dummy, dummy, dummy_int)
    
    # Прогрев simulate_trades
    simulate_trades(dummy, dummy, dummy, 2.0, 1.0, 0.0002, 0.0001)
    
    # Прогрев calculate_z_scores
    calculate_z_scores(dummy, dummy, dummy)
    
    # Прогрев calculate_positions_and_pnl
    calculate_positions_and_pnl(dummy, dummy, dummy, dummy, dummy, 
                               2.0, 1.0, 0.0002, 0.0001)


# Автоматический прогрев при импорте
_warmup_numba_functions()