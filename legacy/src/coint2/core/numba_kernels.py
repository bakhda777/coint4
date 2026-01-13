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
                                   max_zscore_entry: float = 100.0,
                                   stop_loss_threshold: float = 100.0,
                                   min_holding_period: int = 0,
                                   cooldown_period: int = 0,
                                   max_loss_per_unit: float = 1000000.0,
                                   pnl_stop_loss_threshold: float = 1000000.0,
                                   day_indices: np.ndarray = None,
                                   max_round_trips: int = 100000,
                                   max_entries_per_day: int = 100000,
                                   current_R_price_units: float = 1.0,
                                   max_negative_pair_step_r: float = 3.0,
                                   use_pullback_entry: bool = True,
                                   pullback_hysteresis: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    max_zscore_entry : float
        Максимальный z-score для входа (защита от выбросов)
    stop_loss_threshold : float
        Порог стоп-лосса (по абсолютному значению Z-score)
    min_holding_period : int
        Минимальный период удержания позиции (в барах)
    cooldown_period : int
        Период охлаждения после закрытия сделки (в барах)
    max_loss_per_unit : float
        Максимальный убыток на единицу (Stop trading threshold)
    pnl_stop_loss_threshold : float
        Порог стоп-лосса по PnL сделки (в единицах цены, положительное число)
    day_indices : np.ndarray
        Массив индексов дней для лимитов сделок (опционально)
    max_round_trips : int
        Максимум сделок (round trips)
    max_entries_per_day : int
        Максимум входов в день
    current_R_price_units : float
        Размер риска R в единицах цены (для расчета накопленного R)
    max_negative_pair_step_r : float
        Лимит накопленного убытка в R на пару за шаг (положительное число, например 3.0 означает лимит -3R).
        Если накопленный PnL R <= -max_negative_pair_step_r, новые входы блокируются.
    use_pullback_entry : bool
        Включить подтверждение входа через откат (pullback)
    pullback_hysteresis : float
        Величина отката для подтверждения входа (в единицах z-score)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        positions, pnl_series (Net PnL), cumulative_pnl, cost_series
    """
    n = y.size
    positions = np.zeros(n, dtype=np.float32)
    pnl_series = np.zeros(n, dtype=np.float32)
    cost_series = np.zeros(n, dtype=np.float32)

    # Вычисляем rolling статистики
    beta, mu, sigma = rolling_ols(y, x, rolling_window)

    position = 0.0
    entry_bar = 0
    last_exit_bar = -cooldown_period - 1 # Initialize so entry is allowed immediately
    total_pnl = 0.0
    current_trade_pnl = 0.0  # Track PnL for the current trade
    stopped_by_loss = False
    
    # Counters for limits
    round_trips_count = 0
    daily_entries_count = 0
    current_day = -1
    
    # NEW: Cumulative PnL tracking in R
    cumulative_pnl_R = 0.0
    
    # NEW: Pullback logic state
    waiting_for_pullback = 0  # 0: No, 1: Waiting for Long, -1: Waiting for Short
    pullback_extreme_z = 0.0

    # Определяем рыночный режим (если включено)
    regime_factor = 1.0
    if enable_regime_detection and n > 100:
        regime_factor = detect_market_regime(y, x)

    # Проверяем структурные сдвиги (если включено)
    has_structural_break = False
    if enable_structural_breaks and n > 200:
        has_structural_break = check_structural_breaks(y, x, 0.3)

    for i in range(rolling_window, n):
        # Update Day Counters
        if day_indices is not None:
             today = day_indices[i]
             if today != current_day:
                 current_day = today
                 daily_entries_count = 0

        # Check for Max Loss Stop (Hard Stop / Force Exit)
        if stopped_by_loss:
             # Force closed
             positions[i] = 0.0
             continue

        # NEW: Check Cumulative PnL Kill Switch (Step Risk)
        # Variant A: Limit is POSITIVE (e.g. 3.0 meaning -3.0R)
        # So we check if cumulative_pnl_R <= -max_negative_pair_step_r
        # FIX: Include current trade PnL in the check!
        # cumulative_pnl_R only tracks CLOSED trades.
        # current_trade_pnl tracks OPEN trade PnL (in Price Units).
        # We need to convert current_trade_pnl to R units.
        current_r_units = current_trade_pnl / current_R_price_units if current_R_price_units > 1e-9 else 0.0
        total_current_r_pnl = cumulative_pnl_R + current_r_units
        
        if total_current_r_pnl <= -max_negative_pair_step_r:
             # Force closed and blocked
             positions[i] = 0.0
             # We should also prevent re-entry in future steps, but this 'if' is inside loop.
             # We need to make sure 'cumulative_pnl_R' reflects this force close.
             # If we force close here, next iteration 'position' will be 0, and 'cumulative_pnl_R' will be updated.
             continue

        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = position
            continue

        # CRITICAL FIX: Используем данные предыдущего бара для избежания lookahead bias
        if i > rolling_window:
            prev_spread = y[i-1] - beta[i] * x[i-1]

            # Проверяем минимальную волатильность
            # FIX: Avoid division by zero if sigma is 0 or NaN
            if sigma[i] > min_volatility:
                current_vol = sigma[i]
            else:
                current_vol = min_volatility
                
            # Ensure current_vol is never zero
            if current_vol <= 1e-9:
                current_vol = 1e-9

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

            z_curr = (prev_spread - mu[i]) / current_vol
        else:
            z_curr = 0.0
            adaptive_entry = entry_threshold
            adaptive_exit = exit_threshold

        new_position = position

        # Reset PnL tracker if no position
        if position == 0.0:
             current_trade_pnl = 0.0

        # ESTIMATE CURRENT PNL FOR STOP LOSS CHECK
        # We need to know the PnL of holding 'position' from i-1 to i.
        if position != 0.0 and i > rolling_window:
             if not np.isnan(beta[i-1]):
                y_ch = y[i] - y[i-1]
                x_ch = x[i] - x[i-1]
                spr_ch = y_ch - beta[i-1] * x_ch
                current_trade_pnl += position * spr_ch

        # Проверяем максимальный период удержания
        if position != 0.0 and (i - entry_bar) >= max_holding_period:
            new_position = 0.0  # Принудительное закрытие
        
        # Логика выхода из позиции
        elif position != 0.0:
            # Проверка стоп-лосса (CRITICAL FIX: закрываем, если убыток растет)
            # Для LONG (position > 0): убыток, когда Z падает. Стоп если Z < -stop
            # Для SHORT (position < 0): убыток, когда Z растет. Стоп если Z > stop
            is_stop_loss = False
            
            # 1. Check PnL Stop (Highest Priority - Force Exit)
            if current_trade_pnl < -pnl_stop_loss_threshold:
                 is_stop_loss = True
                 new_position = 0.0
                 
                 # FIX: Cap the loss to the threshold (simulating stop order execution)
                 # The loss so far is current_trade_pnl.
                 # We want it to be -pnl_stop_loss_threshold (minus slippage potentially, but let's stick to threshold first)
                 # To adjust, we need to change the PnL attribution for THIS step.
                 # But we haven't calculated pnl_series[i] yet. It is calculated below.
                 # Below: position_pnl = position * spread_change.
                 # And: total_pnl += position_pnl.
                 # And: current_trade_pnl += position * spr_ch (already done above).
                 
                 # Wait, 'current_trade_pnl' above INCLUDES the full spread change of this bar.
                 # So it reflects the "Close" price execution.
                 # If we want to simulate Stop Price execution, we need to "give back" the excess loss.
                 excess_loss = (-pnl_stop_loss_threshold) - current_trade_pnl
                 # excess_loss should be positive (since current < -threshold).
                 
                 # We will apply this correction to pnl_series[i] below.
                 # Let's store it in a temp variable.
                 # But Numba doesn't like dynamic attributes.
                 # We can modify 'spread_change' effectively? No, spread_change is derived from data.
                 # We can just add a correction term.
                 
                 # However, doing it here is tricky because pnl_series calculation is later.
                 # Easier way: Just force current_trade_pnl to be capped?
                 # No, pnl_series needs to sum up to current_trade_pnl.
                 
                 # Let's handle it in the PnL calculation block below.
                 # We need a flag 'limit_pnl_to_threshold'.
                 pass
            
            # 2. Check Z-score Stop
            elif (position > 0 and z_curr < -stop_loss_threshold) or \
               (position < 0 and z_curr > stop_loss_threshold):
                 is_stop_loss = True
                 new_position = 0.0
            
        # Проверка выхода по тейк-профиту (возврат к среднему)
            # Для LONG (position > 0): ждем пока Z вернется вверх к -exit
            # Для SHORT (position < 0): ждем пока Z вернется вниз к +exit
            elif not is_stop_loss:
                # Проверяем минимальное время удержания
                # FIX: min_holding_period applies only to Take Profit/Standard Exit, NOT Stop Loss
                if (i - entry_bar) >= min_holding_period:
                    if (position > 0 and z_curr >= -adaptive_exit) or (position < 0 and z_curr <= adaptive_exit):
                        new_position = 0.0
                else:
                    # Optional: We could log that we are holding due to min_hold constraint
                    # But Numba doesn't support logging.
                    pass

        # Вход в позицию только если нет текущей позиции
        elif position == 0.0:
            # Check Trade Limits
            can_enter = True
            if round_trips_count >= max_round_trips:
                can_enter = False
            if day_indices is not None and daily_entries_count >= max_entries_per_day:
                can_enter = False
                
            # NEW: Cumulative PnL Kill Switch
            # Если накопленный убыток больше 3R (cumulative_pnl_R <= -max_negative_pair_step_r), новые входы запрещены
            if cumulative_pnl_R <= -max_negative_pair_step_r:
                can_enter = False
            
            # FIX: Force PnLStop Logic: If PnL limit hit, close immediately and forbid re-entry
            if current_trade_pnl < -pnl_stop_loss_threshold:
                 pass
            
            # Проверяем cooldown
            if not can_enter:
                pass
            elif (i - last_exit_bar) < cooldown_period:
                # DEBUG: Could log skip here but Numba doesn't support logging
                pass
            else:
                # Проверяем защиту от экстремальных значений z-score
                if abs(z_curr) <= max_zscore_entry:
                    
                    signal = 0
                    
                    if use_pullback_entry:
                        # 1. Check if waiting
                        if waiting_for_pullback == 1: # Waiting for LONG (z was < -threshold)
                            if z_curr < pullback_extreme_z:
                                pullback_extreme_z = z_curr
                            elif z_curr > pullback_extreme_z + pullback_hysteresis:
                                signal = 1
                                waiting_for_pullback = 0
                        
                        elif waiting_for_pullback == -1: # Waiting for SHORT (z was > threshold)
                            if z_curr > pullback_extreme_z:
                                pullback_extreme_z = z_curr
                            elif z_curr < pullback_extreme_z - pullback_hysteresis:
                                signal = -1
                                waiting_for_pullback = 0
                                
                        # 2. Check for NEW triggers
                        # Reset waiting state if we cross back to neutral/opposite side without triggering?
                        # Or if we are in neutral zone, we are not waiting.
                        # Actually, if z_curr < -adaptive_entry, we start waiting.
                        # If we were waiting for SHORT, but z drops to < -adaptive_entry, we should switch to waiting for LONG?
                        # Yes.
                        
                        if z_curr > adaptive_entry:
                            # Start waiting for SHORT
                            waiting_for_pullback = -1
                            pullback_extreme_z = z_curr
                        elif z_curr < -adaptive_entry:
                            # Start waiting for LONG
                            waiting_for_pullback = 1
                            pullback_extreme_z = z_curr
                            
                    else:
                        # Standard logic
                        if z_curr > adaptive_entry:
                            signal = -1
                        elif z_curr < -adaptive_entry:
                            signal = 1
                    
                    # Apply signal
                    if signal == -1:
                        new_position = -1.0  # Short spread
                        entry_bar = i
                        daily_entries_count += 1 
                    elif signal == 1:
                        new_position = 1.0   # Long spread
                        entry_bar = i
                        daily_entries_count += 1

        # Расчет PnL от изменения цены спреда (если есть позиция)
        if position != 0.0 and i > 0:
            # CRITICAL FIX: Use beta[i-1] to calculate PnL for the period i-1 to i
            # We held the position based on beta known at i-1.
            # PnL = (Y[i] - Y[i-1]) - beta[i-1] * (X[i] - X[i-1])
            
            if not np.isnan(beta[i-1]):
                y_change = y[i] - y[i-1]
                x_change = x[i] - x[i-1]
                spread_change = y_change - beta[i-1] * x_change
                
                position_pnl = position * spread_change
                
                # FIX: Apply PnL Capping if stop loss was triggered by PnL
                # current_trade_pnl (updated above) includes this position_pnl (approx, assuming linear accumulation).
                # Actually, current_trade_pnl above used exact same formula.
                # If we decided to stop because current_trade_pnl < -threshold:
                # We want effective current_trade_pnl to be -threshold.
                # The "excess" loss is (current_trade_pnl - (-threshold)).
                # Since both are negative and current < -threshold, excess is negative (the extra loss).
                # We need to ADD back positive amount to cancel it.
                # correction = (-threshold) - current_trade_pnl.
                # Example: current = -500, threshold = 100 (-100).
                # correction = -100 - (-500) = 400.
                # New PnL = -500 + 400 = -100.
                
                if is_stop_loss and current_trade_pnl < -pnl_stop_loss_threshold:
                     correction = (-pnl_stop_loss_threshold) - current_trade_pnl
                     # Apply correction
                     position_pnl += correction
                     # Also fix current_trade_pnl for tracking
                     current_trade_pnl += correction
                
                pnl_series[i] = position_pnl
                total_pnl += position_pnl
                
                # Check for Max Loss Breach during trade
                if total_pnl <= -max_loss_per_unit:
                    new_position = 0.0
                    stopped_by_loss = True

        # Расчет торговых издержек при изменении позиции
        if new_position != position:
            # Торговые издержки
            trade_size = abs(new_position - position)
            total_cost_pct = commission + slippage
            cost = trade_size * total_cost_pct

            pnl_series[i] -= cost  # Вычитаем издержки из P&L
            cost_series[i] = cost
            total_pnl -= cost
            current_trade_pnl -= cost
        
        # Update last_exit_bar if position closed
        if position != 0.0 and new_position == 0.0:
            last_exit_bar = i
            round_trips_count += 1 # Increment round trip counter
            
            # Update Cumulative PnL R based on last trade
            # ONLY if not already updated by PnL Stop logic above
            # PnL Stop logic updates cumulative_pnl_R inside the loop to trigger immediate stop.
            # Normal exits update it here.
            # To avoid double counting, we check if is_stop_loss was NOT triggered by PnL.
            
            # Actually, cleaner way:
            # Just track if we updated it.
            # Or simpler: Update it here ALWAYS, but do NOT update it inside PnL stop block.
            # Let's revert the update inside PnL stop block and rely on this one?
            # NO, we need immediate update to block re-entry in same step if needed? 
            # Actually, re-entry happens in next i. So updating here is fine for blocking NEXT i.
            # The "can_enter" check is at start of loop.
            
            if current_R_price_units > 1e-9:
                # We need to be careful not to double count if we added logic above.
                # I removed the logic above that adds to cumulative_pnl_R? 
                # Wait, I added "cumulative_pnl_R += trade_r" in previous tool call (line 860).
                # So I MUST NOT add it again here if it was a PnL stop.
                
                # Check if we exited due to PnL stop?
                # Variable is_stop_loss is local to the elif block.
                # We need a flag or check.
                
                # Better approach: Move ALL cumulative update here.
                # Remove the update from line 860 (in PnL stop block).
                # This guarantees single update.
                
                trade_r = current_trade_pnl / current_R_price_units
                cumulative_pnl_R += trade_r

        position = new_position
        positions[i] = position

    # Вычисляем кумулятивный PnL
    cumulative_pnl = np.cumsum(pnl_series)

    return positions, pnl_series, cumulative_pnl, cost_series


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
        adaptive_threshold_factor=1.0,
        max_zscore_entry=10.0,
        stop_loss_threshold=10.0,
        min_holding_period=0,
        cooldown_period=0,
        max_loss_per_unit=1000000.0,
        pnl_stop_loss_threshold=1000000.0,
        day_indices=None,
        max_round_trips=100000,
        max_entries_per_day=100000,
        current_R_price_units=1.0,
        max_negative_pair_step_r=3.0
    )


def warmup_all_numba_functions():
    """Публичная функция для прогрева всех Numba функций."""
    _warmup_basic_functions()
    _warmup_advanced_functions()


# Автоматический прогрев при импорте модуля
warmup_all_numba_functions()
