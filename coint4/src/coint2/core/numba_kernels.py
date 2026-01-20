"""
Единый модуль для всех Numba-оптимизированных функций проекта.

Консолидирует функции из:
- numba_backtest.py
- numba_backtest_full.py
- fast_coint.py
- других файлов с Numba-оптимизацией

Обеспечивает:
- Централизованное управление Numba-кодом
- Единую точку прогрева функций
- Оптимизированные настройки компиляции
- Консистентные типы данных
"""

import os
import sys
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, config
from typing import Tuple
import multiprocessing

# Настройка многопоточности для оптимальной производительности
os.environ['OPENBLAS_NUM_THREADS'] = '1'  
os.environ['OMP_NUM_THREADS'] = '1'        
os.environ['MKL_NUM_THREADS'] = '1'

# Устанавливаем количество потоков исходя из доступности системы
# Устанавливаем количество потоков только если еще не установлено
if 'NUMBA_NUM_THREADS' not in os.environ:
    # Для тестового окружения используем 1 поток для стабильности
    if 'PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules:
        os.environ['NUMBA_NUM_THREADS'] = '1'
    else:
        max_threads = min(multiprocessing.cpu_count(), 12)  # Ограничиваем 12 потоками
        os.environ['NUMBA_NUM_THREADS'] = str(max_threads)

# Безопасная установка количества потоков с проверкой доступности
# Отключаем установку потоков в тестовом окружении для избежания конфликтов
if not ('PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules):
    try:
        from numba.np.ufunc.parallel import get_num_threads
        available_threads = get_num_threads()
        max_threads = min(multiprocessing.cpu_count(), 12)
        safe_threads = min(max_threads, available_threads)
        if safe_threads >= 1:
            set_num_threads(safe_threads)
    except (ImportError, ValueError):
        # Если не удается установить потоки, используем значение по умолчанию
        pass
config.THREADING_LAYER = 'threadsafe'


# =============================================================================
# БАЗОВЫЕ СТАТИСТИЧЕСКИЕ ФУНКЦИИ
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def rolling_ols(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Быстрый rolling OLS с использованием кумулятивных сумм.
    
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
    if n != x.size or window > n or window < 2:
        raise ValueError("Invalid input dimensions")
    
    beta = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    
    # Проверяем на константные данные
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return beta, mu, sigma
    
    for i in range(window, n):
        start_idx = i - window
        end_idx = i
        
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
        if x_var < 1e-12:
            continue
            
        # Вычисляем коэффициенты регрессии
        xy_cov = np.mean((x_window - x_mean) * (y_window - y_mean))
        beta_val = xy_cov / x_var
        
        # Вычисляем остатки
        residuals = y_window - beta_val * x_window
        mu_val = np.mean(residuals)
        sigma_val = np.std(residuals)
        
        beta[i] = beta_val
        mu[i] = mu_val
        sigma[i] = sigma_val
    
    return beta, mu, sigma


@nb.njit(fastmath=True, cache=True)
def calculate_z_scores(spread: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Вычисляет z-scores для spread.
    
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
    z_scores = np.full(n, np.nan, dtype=np.float32)
    
    for i in range(n):
        if not np.isnan(sigma[i]) and sigma[i] > 1e-12:
            z_scores[i] = (spread[i] - mu[i]) / sigma[i]
    
    return z_scores


@nb.njit(fastmath=True, cache=True)
def calculate_rolling_correlation(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Вычисляет rolling корреляцию между двумя рядами."""
    n = y.size
    correlations = np.full(n, np.nan, dtype=np.float32)
    
    for i in range(window, n):
        start_idx = i - window
        end_idx = i
        
        y_window = y[start_idx:end_idx]
        x_window = x[start_idx:end_idx]
        
        if np.any(np.isnan(y_window)) or np.any(np.isnan(x_window)):
            continue
            
        # Вычисляем корреляцию
        y_mean = np.mean(y_window)
        x_mean = np.mean(x_window)
        
        y_std = np.std(y_window)
        x_std = np.std(x_window)
        
        if y_std > 1e-12 and x_std > 1e-12:
            covariance = np.mean((y_window - y_mean) * (x_window - x_mean))
            correlations[i] = covariance / (y_std * x_std)
    
    return correlations


@nb.njit(fastmath=True, cache=True)
def calculate_half_life(residuals: np.ndarray) -> float:
    """Вычисляет полупериод mean reversion для остатков."""
    n = residuals.size
    if n < 10:
        return np.nan
    
    # Простая аппроксимация полупериода через AR(1)
    y = residuals[1:]
    x = residuals[:-1]
    
    # Убираем NaN значения
    valid_mask = ~(np.isnan(y) | np.isnan(x))
    if np.sum(valid_mask) < 5:
        return np.nan
    
    y_clean = y[valid_mask]
    x_clean = x[valid_mask]
    
    # Простая регрессия y_t = a * y_{t-1} + e_t
    x_mean = np.mean(x_clean)
    y_mean = np.mean(y_clean)
    
    x_var = np.var(x_clean)
    if x_var < 1e-12:
        return np.nan
    
    xy_cov = np.mean((x_clean - x_mean) * (y_clean - y_mean))
    a = xy_cov / x_var
    
    # Полупериод = -log(2) / log(a)
    if a <= 0 or a >= 1:
        return np.nan
    
    half_life = -0.693147 / np.log(a)  # -log(2) ≈ -0.693147
    
    # Ограничиваем разумными пределами
    if half_life < 1 or half_life > 1000:
        return np.nan
    
    return half_life


# =============================================================================
# ТОРГОВЫЕ ФУНКЦИИ
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def simulate_trades(spread: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   z_enter: float, z_exit: float,
                   fee_perc: float, slippage: float) -> float:
    """
    JIT-компилированная торговая логика.
    
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
    position = 0.0
    total_pnl = 0.0
    entry_price = 0.0
    
    for i in range(1, n):
        if np.isnan(mu[i]) or np.isnan(sigma[i]) or sigma[i] <= 1e-12:
            continue
            
        z_score = (spread[i] - mu[i]) / sigma[i]
        
        # Логика выхода из позиции
        if position != 0.0:
            exit_signal = False
            
            if position > 0 and z_score <= z_exit:
                exit_signal = True
            elif position < 0 and z_score >= -z_exit:
                exit_signal = True
                
            if exit_signal:
                pnl = position * (spread[i] - entry_price)
                cost = abs(position) * (fee_perc + slippage)
                total_pnl += pnl - cost
                position = 0.0
                
        # Логика входа в позицию
        elif position == 0.0:
            if z_score > z_enter:
                position = -1.0  # Short spread
                entry_price = spread[i]
                cost = abs(position) * (fee_perc + slippage)
                total_pnl -= cost
            elif z_score < -z_enter:
                position = 1.0   # Long spread
                entry_price = spread[i]
                cost = abs(position) * (fee_perc + slippage)
                total_pnl -= cost
    
    # Закрываем позицию в конце если она открыта
    if position != 0.0:
        pnl = position * (spread[-1] - entry_price)
        cost = abs(position) * (fee_perc + slippage)
        total_pnl += pnl - cost
    
    return total_pnl


@nb.njit(cache=True, fastmath=True)
def calculate_positions_and_pnl(y: np.ndarray, x: np.ndarray, 
                               beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                               z_enter: float, z_exit: float,
                               commission_pct: float, slippage_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Полная торговая логика с расчетом позиций и PnL.
    
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
            
        # CRITICAL FIX: Calculate z-score using PREVIOUS bar data to avoid lookahead bias
        if i > 1 and not np.isnan(sigma[i]) and sigma[i] > 1e-12:
            # Use previous bar prices with current rolling stats
            prev_spread = y[i-1] - beta[i] * x[i-1]
            z_curr = (prev_spread - mu[i]) / sigma[i]
        else:
            z_curr = 0.0
        
        new_position = position
        
        # Логика выхода из позиции
        if position != 0.0:
            if (position > 0 and z_curr <= z_exit) or (position < 0 and z_curr >= -z_exit):
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


# =============================================================================
# РАСШИРЕННЫЕ ФУНКЦИИ АНАЛИЗА
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def calculate_hurst_exponent(prices: np.ndarray) -> float:
    """Вычисляет показатель Херста для определения трендовости/mean reversion."""
    n = prices.size
    if n < 20:
        return 0.5  # Нейтральное значение

    # Убираем NaN значения
    valid_mask = ~np.isnan(prices)
    if np.sum(valid_mask) < 20:
        return 0.5

    clean_prices = prices[valid_mask]
    n_clean = clean_prices.size

    # Вычисляем логарифмические доходности
    log_returns = np.diff(np.log(clean_prices))

    # Диапазоны для анализа
    max_lag = min(n_clean // 4, 50)
    if max_lag < 5:
        return 0.5

    lags = np.arange(2, max_lag + 1)
    rs_values = np.zeros(len(lags), dtype=np.float32)

    for i, lag in enumerate(lags):
        if lag >= len(log_returns):
            rs_values[i] = np.nan
            continue

        # Разбиваем на блоки
        n_blocks = len(log_returns) // lag
        if n_blocks < 2:
            rs_values[i] = np.nan
            continue

        rs_block_values = np.zeros(n_blocks, dtype=np.float32)

        for j in range(n_blocks):
            start_idx = j * lag
            end_idx = start_idx + lag
            block = log_returns[start_idx:end_idx]

            # Вычисляем среднее и кумулятивные отклонения
            mean_block = np.mean(block)
            cumulative_deviations = np.cumsum(block - mean_block)

            # R/S статистика
            range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            std_val = np.std(block)

            if std_val > 1e-12:
                rs_block_values[j] = range_val / std_val
            else:
                rs_block_values[j] = 1.0

        rs_values[i] = np.mean(rs_block_values)

    # Убираем NaN значения
    valid_rs = ~np.isnan(rs_values)
    if np.sum(valid_rs) < 3:
        return 0.5

    clean_rs = rs_values[valid_rs]
    clean_lags = lags[valid_rs].astype(np.float32)

    # Простая линейная регрессия log(R/S) vs log(lag)
    log_rs = np.log(clean_rs)
    log_lags = np.log(clean_lags)

    # Проверяем валидность логарифмов
    valid_logs = ~(np.isnan(log_rs) | np.isnan(log_lags) | np.isinf(log_rs) | np.isinf(log_lags))
    if np.sum(valid_logs) < 3:
        return 0.5

    log_rs_clean = log_rs[valid_logs]
    log_lags_clean = log_lags[valid_logs]

    # Регрессия
    x_mean = np.mean(log_lags_clean)
    y_mean = np.mean(log_rs_clean)

    x_var = np.var(log_lags_clean)
    if x_var < 1e-12:
        return 0.5

    xy_cov = np.mean((log_lags_clean - x_mean) * (log_rs_clean - y_mean))
    hurst = xy_cov / x_var

    # Ограничиваем разумными пределами
    if hurst < 0.1:
        hurst = 0.1
    elif hurst > 0.9:
        hurst = 0.9

    return hurst


@nb.njit(fastmath=True, cache=True)
def calculate_variance_ratio(prices: np.ndarray) -> float:
    """Вычисляет Variance Ratio для определения случайности блуждания."""
    n = prices.size
    if n < 20:
        return 1.0  # Нейтральное значение

    # Убираем NaN значения
    valid_mask = ~np.isnan(prices)
    if np.sum(valid_mask) < 20:
        return 1.0

    clean_prices = prices[valid_mask]

    # Вычисляем логарифмические доходности
    log_returns = np.diff(np.log(clean_prices))
    n_returns = log_returns.size

    if n_returns < 10:
        return 1.0

    # Variance ratio для периода k=2
    k = 2
    if n_returns < k * 3:
        return 1.0

    # Дисперсия 1-периодных доходностей
    var_1 = np.var(log_returns)
    if var_1 < 1e-12:
        return 1.0

    # k-периодные доходности
    k_returns = np.zeros(n_returns - k + 1, dtype=np.float32)
    for i in range(len(k_returns)):
        k_returns[i] = np.sum(log_returns[i:i+k])

    # Дисперсия k-периодных доходностей
    var_k = np.var(k_returns)

    # Variance ratio
    vr = var_k / (k * var_1)

    # Ограничиваем разумными пределами
    if vr < 0.1:
        vr = 0.1
    elif vr > 10.0:
        vr = 10.0

    return vr


@nb.njit(fastmath=True, cache=True)
def detect_market_regime(y: np.ndarray, x: np.ndarray) -> float:
    """Определяет рыночный режим на основе Hurst и Variance Ratio."""
    if y.size < 50 or x.size < 50:
        return 1.0  # Нейтральный режим

    # Используем spread для анализа
    # Простое приближение beta через корреляцию
    y_mean = np.mean(y)
    x_mean = np.mean(x)

    y_std = np.std(y)
    x_std = np.std(x)

    if y_std < 1e-12 or x_std < 1e-12:
        return 1.0

    # Простая корреляция
    covariance = np.mean((y - y_mean) * (x - x_mean))
    correlation = covariance / (y_std * x_std)

    # Простая оценка beta
    beta_approx = correlation * (y_std / x_std)

    # Создаем spread
    spread = y - beta_approx * x

    # Анализируем режим
    hurst = calculate_hurst_exponent(spread)
    vr = calculate_variance_ratio(spread)

    # Комбинируем показатели
    # Hurst < 0.5 и VR < 1 указывают на mean reversion
    # Hurst > 0.5 и VR > 1 указывают на трендовость

    regime_score = 0.5 * (2.0 - hurst) + 0.5 * (2.0 / vr)

    # Нормализуем в диапазон [0.5, 1.5]
    if regime_score < 0.5:
        regime_score = 0.5
    elif regime_score > 1.5:
        regime_score = 1.5

    return regime_score


@nb.njit(fastmath=True, cache=True)
def check_structural_breaks(y: np.ndarray, x: np.ndarray, min_correlation: float) -> bool:
    """Проверяет наличие структурных сдвигов в коинтеграционной связи."""
    n = y.size
    if n < 100:  # Минимум данных для надежной проверки
        return False

    # Разбиваем на две половины
    mid_point = n // 2

    # Первая половина
    y1 = y[:mid_point]
    x1 = x[:mid_point]

    # Вторая половина
    y2 = y[mid_point:]
    x2 = x[mid_point:]

    # Вычисляем корреляции для каждой половины
    def calc_correlation(y_arr, x_arr):
        if len(y_arr) < 10:
            return 0.0

        y_mean = np.mean(y_arr)
        x_mean = np.mean(x_arr)

        y_std = np.std(y_arr)
        x_std = np.std(x_arr)

        if y_std < 1e-12 or x_std < 1e-12:
            return 0.0

        covariance = np.mean((y_arr - y_mean) * (x_arr - x_mean))
        return covariance / (y_std * x_std)

    corr1 = calc_correlation(y1, x1)
    corr2 = calc_correlation(y2, x2)

    # Проверяем, упала ли корреляция ниже порога
    min_corr = min(abs(corr1), abs(corr2))

    return min_corr < min_correlation


@nb.njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(base_threshold: float, volatility: float,
                               min_vol: float, adaptive_factor: float) -> float:
    """Вычисляет адаптивный порог на основе волатильности."""
    if volatility < min_vol:
        volatility = min_vol

    # Нормализуем волатильность
    vol_ratio = volatility / min_vol

    # Адаптивный множитель
    adaptive_multiplier = 1.0 + adaptive_factor * (vol_ratio - 1.0)

    # Ограничиваем разумными пределами
    if adaptive_multiplier < 0.5:
        adaptive_multiplier = 0.5
    elif adaptive_multiplier > 3.0:
        adaptive_multiplier = 3.0

    return base_threshold * adaptive_multiplier


# =============================================================================
# ПОЛНАЯ ТОРГОВАЯ ФУНКЦИЯ
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def calculate_positions_and_pnl_full(y: np.ndarray, x: np.ndarray,
                                   rolling_window: int,
                                   entry_threshold: float,
                                   exit_threshold: float,
                                   commission: float,
                                   slippage: float,
                                   max_holding_period: int,
                                   enable_regime_detection: bool,
                                   enable_structural_breaks: bool,
                                   min_volatility: float,
                                   adaptive_threshold_factor: float,
                                   cooldown_periods: int = 0,
                                   min_hold_periods: int = 0,
                                   stop_loss_zscore: float = 0.0,
                                   min_spread_move_sigma: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Полная торговая функция с всеми возможностями оригинального алгоритма.

    Parameters
    ----------
    y, x : np.ndarray
        Ценовые ряды
    rolling_window : int
        Размер окна для rolling статистик
    entry_threshold : float
        Базовый порог входа
    exit_threshold : float
        Порог выхода
    commission : float
        Комиссия
    slippage : float
        Проскальзывание
    max_holding_period : int
        Максимальный период удержания позиции
    enable_regime_detection : bool
        Включить определение рыночного режима
    enable_structural_breaks : bool
        Включить защиту от структурных сдвигов
    min_volatility : float
        Минимальная волатильность
    adaptive_threshold_factor : float
        Фактор адаптивности порогов
    cooldown_periods : int
        Период охлаждения после выхода (в барах)
    min_hold_periods : int
        Минимальная длительность удержания позиции (в барах)
    stop_loss_zscore : float
        Stop-loss по z-score
    min_spread_move_sigma : float
        Минимальный сдвиг спреда от последней плоскости (в сигмах)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        positions, pnl_series, cumulative_pnl, costs_series
    """
    n = y.size
    positions = np.zeros(n, dtype=np.float32)
    pnl_series = np.zeros(n, dtype=np.float32)
    costs_series = np.zeros(n, dtype=np.float32)

    # Guard params (avoid negative values / disable features cleanly)
    if max_holding_period <= 0:
        max_holding_period = 2147483647  # effectively disabled
    if cooldown_periods < 0:
        cooldown_periods = 0
    if min_hold_periods < 0:
        min_hold_periods = 0
    if stop_loss_zscore < 0.0:
        stop_loss_zscore = 0.0
    if min_spread_move_sigma < 0.0:
        min_spread_move_sigma = 0.0

    # Вычисляем rolling статистики
    beta, mu, sigma = rolling_ols(y, x, rolling_window)

    position = 0.0
    entry_bar = 0
    entry_spread = 0.0  # Добавляем переменную для хранения спреда при входе
    entry_beta = 1.0    # Бета на момент входа (для консистентного PnL)
    total_pnl = 0.0
    cooldown_until = 0  # bar index; entries allowed when i >= cooldown_until
    last_flat_spread = np.nan

    # Определяем рыночный режим (если включено)
    regime_factor = 1.0
    if enable_regime_detection and n > 100:
        regime_factor = detect_market_regime(y, x)

    # Проверяем структурные сдвиги (если включено)
    has_structural_break = False
    if enable_structural_breaks and n > 200:
        has_structural_break = check_structural_breaks(y, x, 0.3)

    for i in range(rolling_window, n):
        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = position
            continue

        # CRITICAL FIX: Используем данные предыдущего бара и предыдущие статистики
        beta_signal = beta[i]
        if i > rolling_window:
            beta_signal = beta[i - 1]
            mu_prev = mu[i - 1]
            sigma_prev = sigma[i - 1]
            if np.isnan(beta_signal) or np.isnan(mu_prev) or np.isnan(sigma_prev):
                positions[i] = position
                continue

            prev_spread = y[i - 1] - beta_signal * x[i - 1]

            # Проверяем минимальную волатильность
            current_vol = max(sigma_prev, min_volatility)

            # Адаптивные пороги
            adaptive_entry = calculate_adaptive_threshold(
                entry_threshold, current_vol, min_volatility, adaptive_threshold_factor
            )
            adaptive_exit = calculate_adaptive_threshold(
                exit_threshold, current_vol, min_volatility, adaptive_threshold_factor
            )

            # Применяем режимный фактор
            adaptive_entry *= regime_factor
            adaptive_exit *= regime_factor

            # Если есть структурные сдвиги, увеличиваем пороги
            if has_structural_break:
                adaptive_entry *= 1.5
                adaptive_exit *= 1.2

            z_curr = (prev_spread - mu_prev) / current_vol
        else:
            z_curr = 0.0
            adaptive_entry = entry_threshold
            adaptive_exit = exit_threshold
            prev_spread = 0.0
            current_vol = min_volatility

        new_position = position

        # Проверяем максимальный период удержания
        if position != 0.0 and (i - entry_bar) >= max_holding_period:
            new_position = 0.0  # Принудительное закрытие

        # Логика выхода из позиции
        elif position != 0.0:
            # Stop-loss по z-score (override min_hold)
            if stop_loss_zscore > 0.0:
                if (position > 0.0 and z_curr <= -stop_loss_zscore) or (position < 0.0 and z_curr >= stop_loss_zscore):
                    new_position = 0.0

            # Exit band around 0 (aligned with BasePairBacktester: abs(z) <= z_exit)
            if new_position == position:
                if (i - entry_bar) >= min_hold_periods and abs(z_curr) <= abs(adaptive_exit):
                    new_position = 0.0

        # Вход в позицию только если нет текущей позиции
        elif position == 0.0:
            can_enter = i >= cooldown_until
            if can_enter and min_spread_move_sigma > 0.0 and not np.isnan(last_flat_spread):
                if abs(prev_spread - last_flat_spread) < (min_spread_move_sigma * current_vol):
                    can_enter = False

            if can_enter:
                if z_curr > adaptive_entry:
                    new_position = -1.0  # Short spread
                    entry_bar = i
                    entry_beta = beta_signal
                    entry_spread = y[i] - entry_beta * x[i]  # Сохраняем спред при входе
                elif z_curr < -adaptive_entry:
                    new_position = 1.0   # Long spread
                    entry_bar = i
                    entry_beta = beta_signal
                    entry_spread = y[i] - entry_beta * x[i]  # Сохраняем спред при входе

        # Расчет PnL при изменении позиции
        if new_position != position:
            # Торговые издержки
            trade_size = abs(new_position - position)
            total_cost_pct = commission + slippage
            cost_notional = abs(y[i]) + abs(entry_beta * x[i])
            cost = trade_size * cost_notional * total_cost_pct
            costs_series[i] = cost
            
            # Если закрываем позицию, рассчитываем PnL от изменения цены
            if position != 0.0 and new_position == 0.0:
                # Рассчитываем текущий спред
                current_spread = y[i] - entry_beta * x[i]
                # PnL от изменения спреда (position уже содержит знак)
                price_pnl = position * (current_spread - entry_spread)
                # Общий PnL = прибыль от цены минус комиссии
                pnl_series[i] = price_pnl - cost
                total_pnl += price_pnl - cost
                cooldown_until = i + cooldown_periods + 1
                last_flat_spread = y[i] - beta_signal * x[i]
            else:
                # При входе в позицию или изменении размера - только комиссия
                pnl_series[i] = -cost
                total_pnl -= cost

        position = new_position
        positions[i] = position

    # Вычисляем кумулятивный PnL
    cumulative_pnl = np.cumsum(pnl_series)

    return positions, pnl_series, cumulative_pnl, costs_series


# =============================================================================
# ФУНКЦИИ ПРОГРЕВА
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def _warmup_basic_functions():
    """Прогрев базовых Numba функций."""
    n_points = 50
    test_y = np.arange(n_points, dtype=np.float32) + 100.0
    test_x = np.arange(n_points, dtype=np.float32) + 100.1
    test_window = 10

    # Прогрев основных функций
    rolling_ols(test_y, test_x, test_window)
    calculate_z_scores(test_y - test_x, test_y, test_x)
    calculate_rolling_correlation(test_y, test_x, test_window)
    calculate_half_life(test_y - test_x)

    # Прогрев торговых функций
    beta, mu, sigma = rolling_ols(test_y, test_x, test_window)
    simulate_trades(test_y - test_x, mu, sigma, 2.0, 1.0, 0.001, 0.0005)
    calculate_positions_and_pnl(test_y, test_x, beta, mu, sigma, 2.0, 1.0, 0.001, 0.0005)


@nb.njit(fastmath=True, cache=True)
def _warmup_advanced_functions():
    """Прогрев расширенных Numba функций."""
    n_points = 100
    test_y = np.arange(n_points, dtype=np.float32) + 100.0
    test_x = np.arange(n_points, dtype=np.float32) + 100.1

    # Прогрев аналитических функций
    calculate_hurst_exponent(test_y)
    calculate_variance_ratio(test_y)
    detect_market_regime(test_y, test_x)
    check_structural_breaks(test_y, test_x, 0.5)
    calculate_adaptive_threshold(2.0, 1.0, 0.1, 1.0)

    # Прогрев полной торговой функции
    calculate_positions_and_pnl_full(
        test_y, test_x,
        rolling_window=10,
        entry_threshold=2.0,
        exit_threshold=0.5,
        commission=0.001,
        slippage=0.0005,
        max_holding_period=100,
        enable_regime_detection=False,  # Отключаем для прогрева
        enable_structural_breaks=False,  # Отключаем для прогрева
        min_volatility=0.001,
        adaptive_threshold_factor=1.0
    )


def warmup_all_numba_functions():
    """Публичная функция для прогрева всех Numba функций."""
    _warmup_basic_functions()
    _warmup_advanced_functions()


# Автоматический прогрев при импорте модуля
warmup_all_numba_functions()
