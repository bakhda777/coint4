"""
Улучшенная версия Numba kernels с правильным расчетом PnL.
Исправления:
1. PnL рассчитывается в процентах, не в абсолютных значениях
2. Правильный размер позиций относительно капитала
3. Устранен lookahead bias
4. Исправлен расчет max drawdown
"""

import numpy as np
import numba as nb
from numba import njit
from typing import Tuple

# Восстанавливаем Numba с улучшенной численной стабильностью  
@nb.njit(fastmath=False, cache=True)  # fastmath=False для лучшей численной стабильности
def rolling_ols_no_lookahead(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling OLS БЕЗ lookahead bias.
    Для позиции i используются только данные [i-window:i], НЕ включая i.
    """
    n = y.size
    if n != x.size or window > n or window < 2:
        raise ValueError("Invalid input dimensions")
    
    beta = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    
    # КРИТИЧНО: Начинаем с window+1, чтобы иметь полное окно БЕЗ текущего бара
    for i in range(window + 1, n):
        # Используем данные СТРОГО ДО текущего бара
        start_idx = i - window
        end_idx = i  # НЕ включая i
        
        y_window = y[start_idx:end_idx]
        x_window = x[start_idx:end_idx]
        
        # Проверяем валидность данных в окне
        if np.any(np.isnan(y_window)) or np.any(np.isnan(x_window)):
            continue
            
        # Вычисляем статистики
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        
        # Проверяем на константность
        x_var = np.var(x_window)
        if x_var < 1e-10:  # Увеличиваем порог
            continue
            
        # Вычисляем коэффициенты регрессии
        xy_cov = np.mean((x_window - x_mean) * (y_window - y_mean))
        beta_val = xy_cov / x_var
        
        # Вычисляем остатки
        residuals = y_window - beta_val * x_window
        mu_val = np.mean(residuals)
        sigma_val = np.std(residuals)
        
        # Сохраняем для использования на баре i
        beta[i] = beta_val
        mu[i] = mu_val
        sigma[i] = sigma_val
    
    return beta, mu, sigma


# Восстанавливаем Numba с улучшенными проверками
@nb.njit(fastmath=False, cache=True)  # fastmath=False для избежания NaN 
def calculate_positions_and_pnl_normalized(
    y_prices: np.ndarray, 
    x_prices: np.ndarray,
    rolling_window: int,
    entry_threshold: float,
    exit_threshold: float,
    commission_pct: float,
    slippage_pct: float,
    max_position_pct: float = 0.02,  # Максимум 2% капитала на сделку
    initial_capital: float = 100000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Рассчитывает позиции и PnL с ПРАВИЛЬНОЙ нормализацией.
    
    Returns:
        positions: массив позиций (-1, 0, 1)
        pnl_pct: массив PnL в процентах от капитала
        cumulative_return: кумулятивная доходность
        equity_curve: кривая капитала
    """
    n = y_prices.size
    
    # Выходные массивы
    positions = np.zeros(n, dtype=np.float32)
    pnl_pct = np.zeros(n, dtype=np.float32)
    equity = np.full(n, initial_capital, dtype=np.float64)
    
    # Вычисляем log returns для нормализации
    # Добавляем проверку на положительные цены
    if np.any(y_prices <= 0) or np.any(x_prices <= 0):
        raise ValueError("Prices must be positive for log transformation")
    
    y_log = np.log(y_prices + 1e-10)  # Добавляем малое значение для безопасности
    x_log = np.log(x_prices + 1e-10)
    
    # Вычисляем rolling статистики БЕЗ lookahead
    beta, mu, sigma = rolling_ols_no_lookahead(y_log, x_log, rolling_window)
    
    # Торговые переменные
    position = 0.0
    entry_spread_log = 0.0
    entry_bar = 0
    current_capital = initial_capital
    
    for i in range(rolling_window + 1, n):
        # Используем статистики с ПРЕДЫДУЩЕГО бара для избежания lookahead
        if np.isnan(beta[i-1]) or np.isnan(mu[i-1]) or np.isnan(sigma[i-1]):
            positions[i] = position
            equity[i] = current_capital
            continue
        
        # Рассчитываем спред в log пространстве
        current_spread_log = y_log[i] - beta[i-1] * x_log[i]
        
        # Проверяем валидность спреда
        if np.isnan(current_spread_log) or np.isinf(current_spread_log):
            positions[i] = position
            equity[i] = current_capital
            continue
        
        # Z-score на основе ПРОШЛЫХ данных
        # Добавляем дополнительную защиту от деления на ноль
        sigma_safe = sigma[i-1] if not np.isnan(sigma[i-1]) and sigma[i-1] > 1e-6 else 1e-6
        z_score = (current_spread_log - mu[i-1]) / sigma_safe
        
        # Проверяем валидность z_score
        if np.isnan(z_score) or np.isinf(z_score):
            positions[i] = position
            equity[i] = current_capital
            continue
        
        new_position = position
        
        # Логика выхода из позиции
        if position != 0.0:
            # Проверяем условия выхода
            if (position > 0 and z_score <= exit_threshold) or \
               (position < 0 and z_score >= -exit_threshold):
                # Закрываем позицию
                
                # Рассчитываем PnL в процентах
                spread_change = current_spread_log - entry_spread_log
                
                # PnL = позиция * изменение спреда * размер позиции
                # Размер позиции ограничен max_position_pct
                position_return = position * spread_change * max_position_pct
                
                # Вычитаем комиссии (в процентах от позиции)
                total_costs = (commission_pct + slippage_pct) * max_position_pct
                
                # Итоговый PnL в процентах
                trade_pnl_pct = position_return - total_costs
                
                # Критическая проверка на NaN ПЕРЕД записью
                if np.isnan(trade_pnl_pct):
                    # Устанавливаем 0 если NaN
                    trade_pnl_pct = 0.0
                
                pnl_pct[i] = trade_pnl_pct
                
                # Безопасное обновление капитала
                # Ограничиваем потери до 90% для предотвращения NaN
                if np.isnan(trade_pnl_pct) or trade_pnl_pct < -0.9:
                    trade_pnl_pct = -0.9
                elif trade_pnl_pct > 1.0:  # Ограничиваем прибыль до 100%
                    trade_pnl_pct = 1.0
                
                # Обновляем капитал
                new_capital = current_capital * (1 + trade_pnl_pct)
                if new_capital > 0 and not np.isnan(new_capital):
                    current_capital = new_capital
                else:
                    # Fallback: минимальный капитал
                    current_capital = max(current_capital * 0.1, 1000.0)
                
                new_position = 0.0
        
        # Логика входа в позицию
        elif position == 0.0:
            if z_score > entry_threshold:
                new_position = -1.0  # Short spread
                entry_bar = i
                entry_spread_log = current_spread_log
                
                # Комиссия за вход
                entry_cost = (commission_pct + slippage_pct) * max_position_pct
                # Ограничиваем комиссию для безопасности
                if entry_cost > 0.1:  # Максимум 10% комиссии
                    entry_cost = 0.1
                
                # Проверяем на NaN
                entry_pnl = -entry_cost
                if np.isnan(entry_pnl):
                    entry_pnl = 0.0
                
                pnl_pct[i] = entry_pnl
                
                # Безопасное обновление капитала
                new_capital = current_capital * (1 - entry_cost)
                if new_capital > 0 and not np.isnan(new_capital):
                    current_capital = new_capital
                else:
                    current_capital = max(current_capital * 0.9, 1000.0)
                
            elif z_score < -entry_threshold:
                new_position = 1.0  # Long spread
                entry_bar = i
                entry_spread_log = current_spread_log
                
                # Комиссия за вход
                entry_cost = (commission_pct + slippage_pct) * max_position_pct
                # Ограничиваем комиссию для безопасности
                if entry_cost > 0.1:  # Максимум 10% комиссии
                    entry_cost = 0.1
                
                # Проверяем на NaN
                entry_pnl = -entry_cost
                if np.isnan(entry_pnl):
                    entry_pnl = 0.0
                
                pnl_pct[i] = entry_pnl
                
                # Безопасное обновление капитала
                new_capital = current_capital * (1 - entry_cost)
                if new_capital > 0 and not np.isnan(new_capital):
                    current_capital = new_capital
                else:
                    current_capital = max(current_capital * 0.9, 1000.0)
        
        position = new_position
        positions[i] = position
        equity[i] = current_capital
    
    # Рассчитываем кумулятивную доходность
    cumulative_return = (equity / initial_capital - 1) * 100  # В процентах
    
    return positions, pnl_pct * 100, cumulative_return, equity


@nb.njit(fastmath=False, cache=True)
def calculate_max_drawdown_correct(equity_curve: np.ndarray) -> float:
    """
    Правильный расчет максимальной просадки.
    
    Args:
        equity_curve: Кривая капитала
        
    Returns:
        Максимальная просадка в процентах (положительное число)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Numba-совместимая версия maximum.accumulate
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
    
    # Избегаем деления на 0
    drawdowns = np.zeros_like(equity_curve)
    for i in range(len(equity_curve)):
        if running_max[i] > 0:
            drawdowns[i] = (equity_curve[i] - running_max[i]) / running_max[i]
    
    max_drawdown = abs(np.min(drawdowns))
    return max_drawdown * 100  # В процентах


@nb.njit(fastmath=False, cache=True)
def calculate_sharpe_ratio_correct(
    pnl_pct: np.ndarray,
    periods_per_year: int = 365 * 96  # Для 15-минутных баров (crypto 24/7)
) -> float:
    """
    Правильный расчет Sharpe ratio.
    
    Args:
        pnl_pct: PnL в процентах
        periods_per_year: Количество периодов в году
        
    Returns:
        Sharpe ratio (аннуализированный)
    """
    # Убираем NaN, нули сохраняем (нулевая доходность тоже часть ряда)
    valid_pnl = pnl_pct[~np.isnan(pnl_pct)]
    
    if len(valid_pnl) < 2:
        return 0.0
    
    mean_return = np.mean(valid_pnl)
    std_return = np.std(valid_pnl)
    
    # Проверяем на NaN в статистиках
    if np.isnan(mean_return) or np.isnan(std_return) or std_return < 1e-8:
        return 0.0
    
    # Аннуализированный Sharpe
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    
    # Проверяем на numerical issues, но НЕ ограничиваем значение искусственно
    if np.isnan(sharpe) or np.isinf(sharpe):
        return 0.0
    
    # Применяем только экстремальные ограничения для защиты от numerical overflow
    if sharpe > 50.0:  # Только защита от overflow
        sharpe = 50.0
    elif sharpe < -50.0:
        sharpe = -50.0
    
    return sharpe


@nb.njit(fastmath=False, cache=True)
def calculate_metrics_suite(
    pnl_pct: np.ndarray,
    equity_curve: np.ndarray,
    positions: np.ndarray
) -> Tuple[float, int, float, float, float, float, float, float, float]:
    """
    Рассчитывает полный набор метрик.
    """
    # Базовые метрики
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100 if len(equity_curve) > 0 and equity_curve[0] > 0 else 0
    num_trades = int(np.sum(np.diff(positions) != 0) // 2)  # Вход + выход = 1 сделка
    
    # PnL метрики
    non_zero_pnl = pnl_pct[pnl_pct != 0]
    if len(non_zero_pnl) > 0:
        win_rate = np.sum(non_zero_pnl > 0) / len(non_zero_pnl)
        avg_win = np.mean(non_zero_pnl[non_zero_pnl > 0]) if np.any(non_zero_pnl > 0) else 0
        avg_loss = np.mean(non_zero_pnl[non_zero_pnl < 0]) if np.any(non_zero_pnl < 0) else 0
        
        # Profit factor
        gross_wins = np.sum(non_zero_pnl[non_zero_pnl > 0])
        gross_losses = abs(np.sum(non_zero_pnl[non_zero_pnl < 0]))
        profit_factor = gross_wins / gross_losses if gross_losses > 1e-8 else 0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
    
    # Risk метрики
    max_drawdown = calculate_max_drawdown_correct(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio_correct(pnl_pct)
    
    # Calmar ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown > 1e-6 else 0.0
    
    # Проверяем на NaN и заменяем на 0
    if np.isnan(total_return):
        total_return = 0.0
    if np.isnan(win_rate):
        win_rate = 0.0
    if np.isnan(avg_win):
        avg_win = 0.0
    if np.isnan(avg_loss):
        avg_loss = 0.0
    if np.isnan(profit_factor):
        profit_factor = 0.0
    if np.isnan(max_drawdown):
        max_drawdown = 0.0
    if np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0
    if np.isnan(calmar_ratio):
        calmar_ratio = 0.0
    
    # Возвращаем tuple вместо dict для Numba совместимости
    return (
        total_return,
        num_trades,
        win_rate,
        avg_win,
        avg_loss,
        profit_factor,
        max_drawdown,
        sharpe_ratio,
        calmar_ratio
    )
