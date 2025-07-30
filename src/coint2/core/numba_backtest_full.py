"""Полная Numba-оптимизированная версия PairBacktester с всеми функциями оригинала."""

import numpy as np
import numba as nb
from typing import Tuple


@nb.njit(fastmath=True, cache=True)
def rolling_ols(y: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Быстрый rolling OLS, точно повторяющий логику теста."""
    n = y.size
    if n != x.size or window > n or window < 2:
        raise ValueError("Invalid input dimensions")
    
    beta = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    
    # Проверяем на константные данные
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return beta, mu, sigma
    
    for i in range(window - 1, n):
        # Извлекаем окно данных точно как в тесте: y[i-window:i]
        y_win = y[i-window+1:i+1].astype(np.float64)
        x_win = x[i-window+1:i+1].astype(np.float64)
        
        if len(y_win) != window or len(x_win) != window:
            continue
            
        # OLS регрессия с intercept: y = alpha + beta*x
        # Создаем матрицу X = [ones, x_win]
        n_win = len(x_win)
        sum_x = np.sum(x_win)
        sum_y = np.sum(y_win)
        sum_xx = np.sum(x_win * x_win)
        sum_xy = np.sum(x_win * y_win)
        
        # Расчет коэффициентов OLS
        denom = n_win * sum_xx - sum_x * sum_x
        if abs(denom) > 1e-10:
            beta_val = (n_win * sum_xy - sum_x * sum_y) / denom
            alpha_val = (sum_y - beta_val * sum_x) / n_win
            
            beta[i] = np.float32(beta_val)
            
            # Расчет spread как в тесте: y - beta*x
            spread = y_win - beta_val * x_win
            mu[i] = np.float32(np.mean(spread))
            
            # Расчет sigma с ddof=1 как в тесте
            if n_win > 1:
                variance = np.sum((spread - np.mean(spread)) ** 2) / (n_win - 1)
                sigma[i] = np.float32(np.sqrt(max(variance, 1e-12)))
            else:
                sigma[i] = 1e-6
        else:
            # Если знаменатель слишком мал
            beta[i] = 0.0
            mu[i] = np.float32(np.mean(y_win))
            sigma[i] = 1e-6
    
    return beta, mu, sigma


@nb.njit(fastmath=True, cache=True)
def calculate_hurst_exponent(prices: np.ndarray) -> float:
    """Расчет показателя Херста для определения трендовости."""
    if len(prices) < 10:
        return 0.5
    
    # Проверяем на константные цены
    if np.std(prices) < 1e-10:
        return 0.5
    
    # Логарифмические доходности
    log_prices = np.log(np.maximum(prices, 1e-10))  # Избегаем log(0)
    returns = np.diff(log_prices)
    
    if len(returns) < 5:
        return 0.5
    
    # Различные лаги для анализа
    max_lag = min(10, len(returns) // 2)
    if max_lag < 2:
        return 0.5
    
    lags = np.arange(2, max_lag + 1, dtype=np.int32)
    tau = np.zeros(len(lags), dtype=np.float32)
    
    for i, lag in enumerate(lags):
        # Агрегированные доходности
        n_agg = len(returns) // lag
        if n_agg < 2:
            tau[i] = np.nan
            continue
            
        agg_returns = np.zeros(n_agg, dtype=np.float32)
        for j in range(n_agg):
            agg_returns[j] = np.sum(returns[j*lag:(j+1)*lag])
        
        var_agg = np.var(agg_returns)
        if var_agg > 1e-12:
            tau[i] = var_agg
        else:
            tau[i] = np.nan
    
    # Убираем NaN значения
    valid_indices = []
    for i in range(len(tau)):
        if not np.isnan(tau[i]) and tau[i] > 1e-12:
            valid_indices.append(i)
    
    if len(valid_indices) < 3:
        return 0.5
    
    valid_lags = np.zeros(len(valid_indices), dtype=np.float32)
    valid_tau = np.zeros(len(valid_indices), dtype=np.float32)
    
    for i, idx in enumerate(valid_indices):
        valid_lags[i] = float(lags[idx])
        valid_tau[i] = tau[idx]
    
    # Линейная регрессия log(tau) ~ log(lag)
    log_lags = np.log(valid_lags)
    log_tau = np.log(valid_tau)
    
    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xx = np.sum(log_lags * log_lags)
    sum_xy = np.sum(log_lags * log_tau)
    
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.5
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    hurst = slope / 2.0
    
    # Ограничиваем значение в разумных пределах
    return max(0.1, min(0.9, hurst))


@nb.njit(fastmath=True, cache=True)
def calculate_variance_ratio(prices: np.ndarray, lag: int = 2) -> float:
    """Расчет коэффициента вариации для определения mean reversion."""
    if len(prices) < lag + 5:
        return 1.0
    
    # Проверяем на константные цены
    if np.std(prices) < 1e-10:
        return 1.0
    
    # Логарифмические доходности
    log_prices = np.log(np.maximum(prices, 1e-10))  # Избегаем log(0)
    returns = np.diff(log_prices)
    
    if len(returns) < lag + 2:
        return 1.0
    
    # Дисперсия одношаговых доходностей
    var_1 = np.var(returns)
    if var_1 < 1e-12:
        return 1.0
    
    # Дисперсия k-шаговых доходностей
    n_k_returns = len(returns) - lag + 1
    if n_k_returns < 2:
        return 1.0
        
    k_returns = np.zeros(n_k_returns, dtype=np.float32)
    for i in range(n_k_returns):
        k_returns[i] = np.sum(returns[i:i+lag])
    
    var_k = np.var(k_returns)
    if var_k < 1e-12:
        return 1.0
    
    # Коэффициент вариации
    vr = var_k / (lag * var_1)
    
    # Для mean-reverting процессов VR < 1, для trending VR > 1
    return max(0.1, min(3.0, vr))


@nb.njit(fastmath=True, cache=True)
def calculate_rolling_correlation(y: np.ndarray, x: np.ndarray, window: int) -> float:
    """Расчет скользящей корреляции между двумя рядами."""
    if len(y) < window or len(x) < window:
        return 0.0
    
    # Берем последние window значений
    y_win = y[-window:]
    x_win = x[-window:]
    
    # Средние значения
    mean_y = np.mean(y_win)
    mean_x = np.mean(x_win)
    
    # Ковариация и дисперсии
    cov = np.mean((y_win - mean_y) * (x_win - mean_x))
    var_y = np.mean((y_win - mean_y) ** 2)
    var_x = np.mean((x_win - mean_x) ** 2)
    
    # Корреляция
    denom = np.sqrt(var_y * var_x)
    if denom < 1e-12:
        return 0.0
    
    return cov / denom


@nb.njit(fastmath=True, cache=True)
def calculate_half_life(spread: np.ndarray) -> float:
    """Расчет полупериода возврата к среднему для спреда."""
    if len(spread) < 10:
        return np.inf
    
    # Убираем NaN значения
    valid_spread = spread[~np.isnan(spread)]
    if len(valid_spread) < 10:
        return np.inf
    
    # Лагированные значения
    y = valid_spread[1:]
    x = valid_spread[:-1]
    
    # Простая линейная регрессия y = a + b*x
    n = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return np.inf
    
    b = (n * sum_xy - sum_x * sum_y) / denom
    
    # Полупериод = -log(2) / log(b)
    if b <= 0 or b >= 1:
        return np.inf
    
    half_life = -np.log(2.0) / np.log(b)
    return max(1.0, half_life)


@nb.njit(fastmath=True, cache=True)
def detect_market_regime(y_prices: np.ndarray, x_prices: np.ndarray, 
                        hurst_threshold: float = 0.5, 
                        vr_trending_min: float = 1.2,
                        vr_mean_reverting_max: float = 0.8) -> int:
    """Определение рыночного режима.
    
    Returns:
        0: neutral, 1: trending, 2: mean_reverting
    """
    if len(y_prices) < 20 or len(x_prices) < 20:
        return 0  # neutral
    
    # Расчет показателей Херста
    hurst_y = calculate_hurst_exponent(y_prices)
    hurst_x = calculate_hurst_exponent(x_prices)
    avg_hurst = (hurst_y + hurst_x) / 2.0
    
    # Расчет коэффициентов дисперсии
    vr_y = calculate_variance_ratio(y_prices)
    vr_x = calculate_variance_ratio(x_prices)
    avg_vr = (vr_y + vr_x) / 2.0
    
    # Определение режима
    if avg_hurst > hurst_threshold and avg_vr > vr_trending_min:
        return 1  # trending
    elif avg_hurst < hurst_threshold and avg_vr < vr_mean_reverting_max:
        return 2  # mean_reverting
    else:
        return 0  # neutral


@nb.njit(fastmath=True, cache=True)
def check_structural_breaks(y_prices: np.ndarray, x_prices: np.ndarray, spread: float,
                          correlation_window: int = 60,
                          min_correlation: float = 0.7,
                          max_half_life_days: float = 30.0) -> bool:
    """Проверка структурных сдвигов.
    
    Returns:
        True если обнаружен структурный сдвиг
    """
    if len(y_prices) < correlation_window:
        return False
    
    # Проверка корреляции
    correlation = calculate_rolling_correlation(y_prices, x_prices, correlation_window)
    if correlation < min_correlation:
        return True
    
    # Проверка полупериода - создаем массив spread из цен
    spread_array = np.zeros(len(y_prices), dtype=np.float32)
    for i in range(len(y_prices)):
        spread_array[i] = y_prices[i] - x_prices[i]
    
    half_life = calculate_half_life(spread_array)
    # Конвертируем в дни (предполагаем 15-минутные данные)
    half_life_days = half_life * 15.0 / (60.0 * 24.0)
    if half_life_days > max_half_life_days:
        return True
    
    return False


@nb.njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(base_threshold: float, current_sigma: float, 
                               min_volatility: float, adaptive_factor: float) -> float:
    """Расчет адаптивного порога на основе текущей волатильности."""
    # Нормализуем волатильность
    normalized_vol = current_sigma / max(min_volatility, 1e-6)
    
    # Адаптация порога: увеличиваем в периоды высокой волатильности
    volatility_multiplier = max(0.5, min(2.0, normalized_vol * adaptive_factor))
    
    return base_threshold * volatility_multiplier


@nb.njit(fastmath=True, cache=True)
def calculate_positions_and_pnl_full(
    y: np.ndarray, x: np.ndarray, 
    rolling_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    commission: float = 0.001,
    slippage: float = 0.0005,
    max_holding_period: int = 100,
    enable_regime_detection: bool = True,
    enable_structural_breaks: bool = True,
    min_volatility: float = 0.001,
    adaptive_threshold_factor: float = 1.0
) -> tuple:
    """Полная версия расчета позиций и PnL с всеми функциями оригинального алгоритма."""
    
    n = len(y)
    if n != len(x) or n < rolling_window + 10:
        raise ValueError("Insufficient data length")
    
    # Инициализация выходных массивов
    positions = np.zeros(n, dtype=np.float32)
    pnl = np.zeros(n, dtype=np.float32)
    cumulative_pnl = np.zeros(n, dtype=np.float32)
    
    # Расчет скользящих параметров
    beta, mu, sigma = rolling_ols(y, x, rolling_window)
    
    # Инициализация состояния
    current_position = 0.0
    entry_price_y = 0.0
    entry_price_x = 0.0
    entry_beta = 0.0  # CRITICAL FIX: Store entry beta for consistent PnL calculation
    entry_time = -1
    total_pnl = 0.0
    
    for i in range(rolling_window, n):
        if np.isnan(beta[i]) or np.isnan(mu[i]) or np.isnan(sigma[i]):
            positions[i] = current_position
            pnl[i] = 0.0
            cumulative_pnl[i] = total_pnl
            continue
            
        # CRITICAL FIX: Decision at bar 'i' must use information available up to 'i-1'.
        if i > rolling_window:
            beta_prev = beta[i-1]
            mu_prev = mu[i-1]
            sigma_prev = sigma[i-1]
            
            if np.isnan(beta_prev) or np.isnan(mu_prev) or np.isnan(sigma_prev):
                # CRITICAL FIX: Use NaN for z_score when data is invalid to avoid false signals
                z_score = np.nan
                sigma_val = min_volatility
            else:
                spread_for_decision = y[i-1] - beta_prev * x[i-1]
                sigma_val = max(sigma_prev, min_volatility)
                z_score = (spread_for_decision - mu_prev) / sigma_val
        else:
            z_score = 0.0
            sigma_val = min_volatility
        
        # Проверка рыночного режима
        regime_ok = True
        if enable_regime_detection and i >= rolling_window + 20:
            start_idx = max(0, i - 50)
            end_idx = min(i + 1, n)
            if end_idx - start_idx > 20:
                regime = detect_market_regime(y[start_idx:end_idx], x[start_idx:end_idx])
                if regime == 1:  # trending
                    regime_ok = False
        
        # Проверка структурных сдвигов
        structural_ok = True
        if enable_structural_breaks and i >= rolling_window + 20:
            start_idx = max(0, i - 30)
            end_idx = min(i + 1, n)
            if end_idx - start_idx > 10:
                # Use previous spread for structural break detection
                current_spread = y[i-1] - beta[i] * x[i-1] if i > rolling_window else 0.0
                has_break = check_structural_breaks(y[start_idx:end_idx], x[start_idx:end_idx], current_spread)
                if has_break:
                    structural_ok = False
        
        # Адаптивный порог
        current_threshold = calculate_adaptive_threshold(
            entry_threshold, sigma_val, min_volatility, adaptive_threshold_factor
        )
        
        # Логика торговли
        new_position = current_position
        trade_cost = 0.0

        # Проверка на принудительное закрытие
        force_close = False
        if current_position != 0.0:
            # Тайм-стоп
            if entry_time >= 0 and (i - entry_time) >= max_holding_period:
                force_close = True
            # Режим или структурные сдвиги
            if not regime_ok or not structural_ok:
                force_close = True

        if force_close:
            # Принудительное закрытие позиции
            new_position = 0.0
            if current_position != 0.0:
                trade_cost = commission + slippage
                # Reset entry parameters when closing position
                entry_beta = 0.0
        elif current_position == 0.0:
            # Вход в позицию
            if regime_ok and structural_ok and not np.isnan(z_score):
                if z_score > current_threshold:
                    new_position = -1.0  # Short spread
                    # CRITICAL FIX: Use current bar prices for entry execution
                    # Decision made on i-1 data, execution at current bar i prices
                    entry_price_y = y[i]
                    entry_price_x = x[i]
                    # CRITICAL FIX: Store beta used for entry decision for consistent PnL calculation
                    entry_beta = beta_prev if i > rolling_window else beta[i]
                    entry_time = i
                    trade_cost = commission + slippage
                elif z_score < -current_threshold:
                    new_position = 1.0   # Long spread
                    # CRITICAL FIX: Use current bar prices for entry execution
                    # Decision made on i-1 data, execution at current bar i prices
                    entry_price_y = y[i]
                    entry_price_x = x[i]
                    # CRITICAL FIX: Store beta used for entry decision for consistent PnL calculation
                    entry_beta = beta_prev if i > rolling_window else beta[i]
                    entry_time = i
                    trade_cost = commission + slippage
        else:
            # Выход из позиции
            exit_condition = False
            # CRITICAL FIX: Handle NaN z_score - force exit if data is invalid
            if np.isnan(z_score):
                exit_condition = True  # Force exit on invalid data
            elif current_position > 0 and z_score > -exit_threshold:
                exit_condition = True
            elif current_position < 0 and z_score < exit_threshold:
                exit_condition = True

            if exit_condition:
                new_position = 0.0
                trade_cost = commission + slippage
                # Reset entry parameters when closing position
                entry_beta = 0.0
        
        # Расчет PnL
        period_pnl = 0.0
        if current_position != 0.0 and i > 0:
            # PnL от изменения цен
            delta_y = y[i] - y[i-1]
            delta_x = x[i] - x[i-1]
            # CRITICAL FIX: Use entry_beta for consistent PnL calculation
            # This ensures PnL is calculated using the same beta that was used for position sizing
            beta_for_pnl = entry_beta if entry_beta != 0.0 else (beta[i-1] if i > rolling_window else 0.0)
            spread_pnl = current_position * (delta_y - beta_for_pnl * delta_x)
            period_pnl = spread_pnl
        
        # Вычитаем торговые издержки
        period_pnl -= trade_cost
        
        # Обновление состояния
        current_position = new_position
        total_pnl += period_pnl
        
        positions[i] = current_position
        pnl[i] = period_pnl
        cumulative_pnl[i] = total_pnl
    
    return positions, pnl, cumulative_pnl


@nb.njit(fastmath=True, cache=True)
def _warmup_numba_functions_full():
    """Прогрев всех Numba функций для устранения задержки компиляции."""
    # Тестовые данные достаточной длины
    n_points = 50
    test_y = np.arange(n_points, dtype=np.float32) + 100.0
    test_x = np.arange(n_points, dtype=np.float32) + 100.1
    test_window = 10
    
    # Прогрев основных функций
    rolling_ols(test_y, test_x, test_window)
    calculate_hurst_exponent(test_y)
    calculate_variance_ratio(test_y)
    calculate_rolling_correlation(test_y, test_x, test_window)
    calculate_half_life(test_y - test_x)
    detect_market_regime(test_y, test_x)
    check_structural_breaks(test_y, test_x, 1.0)
    calculate_adaptive_threshold(2.0, 1.0, 0.1, 1.0)
    
    # Прогрев полной функции с отключенными сложными функциями
    calculate_positions_and_pnl_full(
        test_y, test_x,
        rolling_window=test_window,
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


# Прогрев функций при импорте
_warmup_numba_functions_full()