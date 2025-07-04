"""Fast cointegration test implementation using Numba."""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit, prange, set_num_threads, config

# Настройка многопоточности для оптимальной производительности
os.environ['OPENBLAS_NUM_THREADS'] = '1'  
os.environ['OMP_NUM_THREADS'] = '1'        
os.environ['MKL_NUM_THREADS'] = '1'

# Устанавливаем количество потоков исходя из доступности системы
import multiprocessing
max_threads = min(multiprocessing.cpu_count(), 12)  # Ограничиваем 12 потоками
os.environ['NUMBA_NUM_THREADS'] = str(max_threads)

set_num_threads(max_threads)
config.THREADING_LAYER = 'threadsafe'


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def precompute_differences(y, maxlag):
    """Предварительное вычисление разностей для ускорения ADF-теста"""
    n = y.shape[0]
    dy = np.zeros(n-1, dtype=np.float64)
    for i in range(n-1):
        dy[i] = y[i+1] - y[i]
    return dy


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def compute_aic_optimized(resid, nobs, k):
    """Оптимизированное вычисление AIC"""
    ssr = resid @ resid
    sigma2 = ssr / nobs
    return np.log(sigma2) + 2 * (k + 1) / nobs


@njit(cache=True, parallel=True, fastmath=True, nogil=True, error_model='numpy')
def _adf_autolag_numba_final(res, du_precomputed, k_max):
    n = res.size
    # Создаем массивы фиксированного размера для хранения результатов
    aic_values = np.full(k_max + 1, 1e300, dtype=np.float64)
    tau_values = np.zeros(k_max + 1, dtype=np.float64)
    valid_k = np.zeros(k_max + 1, dtype=np.bool_)
    
    # Значения по умолчанию
    best_aic = 1e300
    best_tau = 0.0
    best_k   = 0
    
    # Выделяем память для матрицы X максимального размера один раз
    max_n_eff = n - 1
    X_max = np.empty((max_n_eff, k_max + 1), np.float64)
    X_max = np.ascontiguousarray(X_max)
    
    # Создаем единичную матрицу максимального размера один раз
    eye_max = np.eye(k_max + 1, dtype=np.float64)
    
    # Точно соответствуем реализации statsmodels
    for k in range(k_max + 1):
        n_eff = n - k - 1
        if n_eff < 10:
            continue
        
        m = k + 1
        y = du_precomputed[k:]
        # Используем часть предварительно выделенной матрицы
        X = X_max[:n_eff, :m]
        X = np.ascontiguousarray(X)
        X[:, 0] = res[k:-1]
        
        # Формируем матрицу регрессоров точно как в statsmodels
        if k > 0:
            for j in prange(k):
                X[:, j + 1] = du_precomputed[k - j - 1 : n - j - 2]
            
        # Решаем нормальные уравнения через разложение Холецкого
        xtx = X.T @ X
        xty = X.T @ y
        
        # Разложение Холецкого
        L = np.linalg.cholesky(xtx)
        
        # Решаем две треугольные системы для beta
        z = np.linalg.solve(L, xty)
        beta = np.linalg.solve(L.T, z)
        
        # Вычисляем сумму квадратов остатков
        y_hat = X @ beta
        resid_k = y - y_hat
        ssr = resid_k @ resid_k
        nobs = n_eff
        
        # Вычисляем AIC
        aic = compute_aic_optimized(resid_k, nobs, k)
        aic_values[k] = aic
        valid_k[k] = True
        
        # Стандартная ошибка
        df_resid = nobs - m
        if df_resid <= 0:
            continue
        
        sigma2_for_se = ssr / df_resid
        eye_m = eye_max[:m, :m]
        xtx_inv_cols = np.linalg.solve(L.T, np.linalg.solve(L, eye_m))
        
        # Вычисляем стандартную ошибку и t-статистику
        se_b0 = np.sqrt(sigma2_for_se * xtx_inv_cols[0, 0])
        tau = beta[0] / se_b0
        
        tau_values[k] = tau
        
        if aic < best_aic:
            best_aic, best_tau, best_k = aic, tau, k
    
    # Возвращаем лучший результат по AIC (оптимизация: убран принудительный k=2)
    return best_tau, best_k


@njit(cache=True, parallel=True, fastmath=True, nogil=True, error_model='numpy')
def fast_coint_numba_final(x, y, k_max=12):
    """Ускоренная версия cointegration test с использованием Numba.
    
    Parameters
    ----------
    x : np.ndarray
        Первая временная серия
    y : np.ndarray
        Вторая временная серия
    k_max : int
        Максимальное количество лагов для ADF теста
        
    Returns
    -------
    tau : float
        Статистика tau для cointegration test
    pvalue : None
        (заполняется вызывающей функцией через mackinnonp)
    k : int
        Оптимальное количество лагов
    """
    # Преобразуем входные данные в float64 для стабильности
    x64 = x if str(x.dtype) == 'float64' else x.astype(np.float64)
    y64 = y if str(y.dtype) == 'float64' else y.astype(np.float64)
    
    # Быстрое вычисление бета и остатков
    denom = (y64 * y64).sum()
    beta = (x64 * y64).sum() / denom
    
    # Векторизированное вычисление остатков
    resid = x64 - beta * y64
    
    # Предварительно выделяем память для массива разностей
    du_precomputed = precompute_differences(resid, k_max)
    
    # Вызываем оптимизированную функцию для расчета tau и лага
    tau, k = _adf_autolag_numba_final(resid, du_precomputed, k_max)
    
    return tau, None, k


def fast_coint(x, y, trend='n', k_max=12):
    """Быстрая версия cointegration test, совместимая с statsmodels.coint.
    
    Parameters
    ----------
    x : pd.Series or np.ndarray
        Первая временная серия
    y : pd.Series or np.ndarray
        Вторая временная серия
    trend : str
        Тренд для теста ('n' для no trend)
    k_max : int
        Максимальное количество лагов
        
    Returns
    -------
    tau : float
        Статистика tau
    pvalue : float
        P-value теста
    k : int
        Оптимальное количество лагов
    """
    # Конвертируем pandas Series в numpy arrays
    if hasattr(x, 'values'):
        x_vals = x.values
    else:
        x_vals = x
    
    if hasattr(y, 'values'):
        y_vals = y.values
    else:
        y_vals = y
    
    # Убираем NaN значения
    if hasattr(x, 'index') and hasattr(y, 'index'):
        # Для pandas Series - используем их индексы
        combined = pd.DataFrame({'x': x, 'y': y}).dropna()
        x_vals = combined['x'].values
        y_vals = combined['y'].values
    else:
        # Для numpy arrays - убираем NaN построчно
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
    
    # Вызываем быструю Numba-версию
    tau, _, k = fast_coint_numba_final(x_vals, y_vals, k_max)
    
    # Получаем p-value через statsmodels mackinnonp
    # Используем те же параметры, что и в statsmodels.coint
    if trend == 'n':
        pvalue = float(sm.tsa.stattools.mackinnonp(tau, regression="n", N=2))
    else:
        # Для других трендов используем соответствующие параметры
        pvalue = float(sm.tsa.stattools.mackinnonp(tau, regression=trend, N=2))
    
    return tau, pvalue, k 