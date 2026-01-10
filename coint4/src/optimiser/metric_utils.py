"""Утилиты для работы с метриками оптимизации."""

import numpy as np
from typing import Dict, Any, Optional

def extract_sharpe(result: Dict[str, Any]) -> Optional[float]:
    """Извлекает Sharpe ratio из результата, унифицируя доступ к метрике.
    
    Сначала пытается взять result["sharpe_ratio_abs"], потом result["sharpe_ratio"].
    Возвращает None, если ключей нет.
    
    Args:
        result: Словарь с результатами бэктеста
        
    Returns:
        Sharpe ratio или None, если метрика не найдена
    """
    if not isinstance(result, dict):
        return None
        
    # Сначала пытаемся взять sharpe_ratio_abs (приоритетная метрика)
    if "sharpe_ratio_abs" in result:
        sharpe = result["sharpe_ratio_abs"]
        if isinstance(sharpe, (int, float)) and not (sharpe != sharpe) and not np.isinf(sharpe):  # Проверка на NaN и inf
            return float(sharpe)

    # Затем пытаемся взять sharpe_ratio
    if "sharpe_ratio" in result:
        sharpe = result["sharpe_ratio"]
        if isinstance(sharpe, (int, float)) and not (sharpe != sharpe) and not np.isinf(sharpe):  # Проверка на NaN и inf
            return float(sharpe)
    
    # Если ни одного ключа нет
    return None

def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Нормализует параметры, маппя короткие имена на канонические.
    
    Args:
        params: Словарь с параметрами (может содержать короткие имена)
        
    Returns:
        Словарь с каноническими именами параметров
    """
    normalized = params.copy()
    
    # Маппинг коротких имен на канонические
    mapping = {
        'z_entry': 'zscore_threshold',
        'z_exit': 'zscore_exit', 
        'sl_mult': 'stop_loss_multiplier',
        'time_stop_mult': 'time_stop_multiplier',
        'risk_per_pos': 'risk_per_position_pct',
        'max_pos_size': 'max_position_size_pct',
        'max_active_pos': 'max_active_positions'
    }
    
    # Применяем маппинг
    for short_name, canonical_name in mapping.items():
        if short_name in normalized:
            normalized[canonical_name] = normalized.pop(short_name)
    
    return normalized

def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Валидирует и исправляет параметры согласно требованиям парного трейдинга.
    Проверяет совместимость с production торговлей.
    
    Args:
        params: Словарь с параметрами
        
    Returns:
        Словарь с валидными параметрами
        
    Raises:
        ValueError: Если параметры невозможно исправить
    """
    validated = params.copy()
    eps = 1e-6
    
    # НОВОЕ: Валидация production-совместимости нормализации
    if 'normalization_method' in validated:
        norm_method = validated['normalization_method']
        production_methods = ['rolling_zscore', 'percent', 'log_returns']
        
        if norm_method not in production_methods:
            print(f"⚠️ ВНИМАНИЕ: Метод нормализации '{norm_method}' НЕ совместим с production!")
            print(f"   Production поддерживает только: {production_methods}")
            print(f"   Заменяем на 'rolling_zscore' для совместимости")
            validated['normalization_method'] = 'rolling_zscore'
    
    # Валидация z_entry и z_exit
    z_entry = validated.get('zscore_threshold', validated.get('z_entry', 2.0))
    z_exit = validated.get('zscore_exit', validated.get('z_exit', 0.0))
    
    # Handle None values
    if z_entry is None:
        z_entry = 2.0
    if z_exit is None:
        z_exit = 0.0
    
    # z_entry должен быть положительным
    if z_entry <= 0:
        raise ValueError(f"z_entry должен быть положительным, получен: {z_entry}")

    # Проверяем только что |z_exit| < |z_entry| для корректного hysteresis
    if abs(z_exit) >= abs(z_entry):
        # Корректируем z_exit чтобы оставить минимальный hysteresis
        sign = 1 if z_exit >= 0 else -1
        z_exit = sign * max(0, abs(z_entry) - eps)
    
    validated['zscore_threshold'] = z_entry
    validated['zscore_exit'] = z_exit

    sl_mult = validated.get('stop_loss_multiplier', validated.get('sl_mult'))
    time_stop_mult = validated.get('time_stop_multiplier', validated.get('time_stop_mult'))

    if sl_mult is None:
        sl_mult = 2.0
    if time_stop_mult is None:
        time_stop_mult = 5.0  # Добавлено значение по умолчанию
    
    if sl_mult < 0:
        raise ValueError(f"stop_loss_multiplier должен быть неотрицательным, получен: {sl_mult}")
    if time_stop_mult < 0:
        raise ValueError(f"time_stop_multiplier должен быть неотрицательным, получен: {time_stop_mult}")
    
    validated['stop_loss_multiplier'] = sl_mult
    validated['time_stop_multiplier'] = time_stop_mult

    max_active_pos = validated.get('max_active_positions', validated.get('max_active_pos'))
    max_pos_size_pct = validated.get('max_position_size_pct', validated.get('max_pos_size'))
    risk_per_pos_pct = validated.get('risk_per_position_pct', validated.get('risk_per_pos'))

    if max_active_pos is None:
        max_active_pos = 10
    if max_pos_size_pct is None:
        max_pos_size_pct = 1.0
    if risk_per_pos_pct is None:
        risk_per_pos_pct = 0.01
    
    if max_active_pos < 1:
        raise ValueError(f"max_active_positions должен быть >= 1, получен: {max_active_pos}")
    if not (0 < max_pos_size_pct <= 1):
        raise ValueError(f"max_position_size_pct должен быть в (0, 1], получен: {max_pos_size_pct}")
    if not (0 < risk_per_pos_pct <= 1):
        raise ValueError(f"risk_per_position_pct должен быть в (0, 1], получен: {risk_per_pos_pct}")
    
    validated['max_active_positions'] = int(max_active_pos)
    validated['max_position_size_pct'] = max_pos_size_pct
    validated['risk_per_position_pct'] = risk_per_pos_pct
    
    # При конфликте риск-лимита и лимита по размеру позиции берем минимум
    # Это будет обработано в Portfolio.calculate_position_risk_capital

    _validate_parameter_relationships(validated)
    
    # Дополнительная валидация cross-parameter ограничений
    _validate_cross_parameter_constraints(validated)

    _validate_cost_parameters(validated)

    return validated

def _validate_parameter_relationships(params: Dict[str, Any]) -> None:
    """Валидирует базовые ограничения между связанными параметрами.

    Args:
        params: Словарь с параметрами (изменяется на месте)

    Raises:
        ValueError: Если обнаружены несовместимые параметры
    """
    # Валидация half_life параметров
    min_half_life = params.get('min_half_life_days')
    max_half_life = params.get('max_half_life_days')

    if min_half_life is not None and max_half_life is not None:
        if min_half_life > max_half_life:
            raise ValueError(f"min_half_life_days ({min_half_life}) должен быть <= max_half_life_days ({max_half_life})")

        # Проверяем разумные диапазоны
        if min_half_life <= 0:
            raise ValueError(f"min_half_life_days должен быть > 0, получен: {min_half_life}")
        if max_half_life > 365:  # Больше года не имеет смысла
            raise ValueError(f"max_half_life_days не должен превышать 365 дней, получен: {max_half_life}")

    zscore_threshold = params.get('zscore_threshold')
    zscore_exit = params.get('zscore_exit')

    if zscore_threshold is not None and zscore_exit is not None:
        # Проверяем разумные диапазоны сначала
        if zscore_threshold <= 0:
            raise ValueError(f"zscore_threshold должен быть > 0, получен: {zscore_threshold}")
        if zscore_threshold > 5.0:  # Слишком высокий порог
            raise ValueError(f"zscore_threshold не должен превышать 5.0, получен: {zscore_threshold}")

        # Проверяем что |zscore_exit| < |zscore_threshold| для корректного hysteresis
        if abs(zscore_exit) >= abs(zscore_threshold):
            raise ValueError(f"|zscore_exit| ({abs(zscore_exit)}) должен быть < |zscore_threshold| ({abs(zscore_threshold)})")
        
        # ДОПОЛНИТЕЛЬНО: Проверяем разумный гистерезис
        # Для симметричных стратегий гистерезис = |zscore_threshold| - |zscore_exit|
        hysteresis = abs(zscore_threshold) - abs(zscore_exit)
        if hysteresis < 0.05:
            raise ValueError(f"Слишком маленький гистерезис {hysteresis:.3f} между |zscore_threshold| и |zscore_exit|")
        if hysteresis > 2.5:
            raise ValueError(f"Слишком большой гистерезис {hysteresis:.3f} между |zscore_threshold| и |zscore_exit|")

    # Валидация ssd_top_n
    ssd_top_n = params.get('ssd_top_n')
    if ssd_top_n is not None:
        if ssd_top_n < 1000:  # Слишком мало пар
            raise ValueError(f"ssd_top_n должен быть >= 1000 для статистической значимости, получен: {ssd_top_n}")
        if ssd_top_n > 500000:  # Слишком много пар
            raise ValueError(f"ssd_top_n не должен превышать 500000 для производительности, получен: {ssd_top_n}")

    # Валидация rolling_window
    rolling_window = params.get('rolling_window')
    if rolling_window is not None:
        if rolling_window < 10:  # Слишком маленькое окно
            raise ValueError(f"rolling_window должен быть >= 10 для статистической значимости, получен: {rolling_window}")
        if rolling_window > 200:  # Слишком большое окно
            raise ValueError(f"rolling_window не должен превышать 200 для адаптивности, получен: {rolling_window}")

def _validate_cross_parameter_constraints(params: Dict[str, Any]) -> None:
    """Проверяет сложные межпараметрические ограничения.
    
    Args:
        params: Словарь параметров
        
    Raises:
        ValueError: При нарушении ограничений
    """
    # Проверяем соотношение стоп-лоссов
    stop_loss_mult = params.get('stop_loss_multiplier')
    time_stop_mult = params.get('time_stop_multiplier')
    
    if stop_loss_mult is not None and time_stop_mult is not None:
        if time_stop_mult < stop_loss_mult:
            raise ValueError(
                f"time_stop_multiplier ({time_stop_mult}) должен быть >= stop_loss_multiplier ({stop_loss_mult})"
            )
    
    # Проверяем риски и экспозицию
    risk_per_position = params.get('risk_per_position_pct')
    max_position_size = params.get('max_position_size_pct')
    max_positions = params.get('max_active_positions')
    
    if risk_per_position is not None and max_positions is not None:
        max_exposure = risk_per_position * max_positions
        if max_exposure > 1.0:
            raise ValueError(
                f"Максимальная экспозиция {max_exposure:.1%} превышает 100% капитала"
            )
    
    if max_position_size is not None and risk_per_position is not None:
        if max_position_size < risk_per_position:
            raise ValueError(
                f"max_position_size_pct ({max_position_size}) должен быть >= risk_per_position_pct ({risk_per_position})"
            )

def _validate_cost_parameters(params: Dict[str, Any]) -> None:
    """Валидирует параметры издержек и предотвращает двойной учет.

    Args:
        params: Словарь с параметрами (изменяется на месте)

    Raises:
        ValueError: Если обнаружен конфликт в параметрах издержек
    """
    # Проверяем наличие агрегированных параметров
    has_aggregate_commission = 'commission_pct' in params
    has_aggregate_slippage = 'slippage_pct' in params

    # Проверяем наличие детальных параметров
    has_detailed_commission = any(key in params for key in ['fee_maker', 'fee_taker'])
    has_detailed_slippage = any(key in params for key in ['slippage_bps', 'half_spread_bps'])

    # Предупреждаем о потенциальном двойном учете
    if has_aggregate_commission and has_detailed_commission:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Обнаружены и агрегированные (commission_pct) и детальные (fee_maker/fee_taker) параметры комиссий!")
        print("   Используем только агрегированные параметры для предотвращения двойного учета")
        # Удаляем детальные параметры
        for key in ['fee_maker', 'fee_taker']:
            params.pop(key, None)

    if has_aggregate_slippage and has_detailed_slippage:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Обнаружены и агрегированные (slippage_pct) и детальные (slippage_bps/half_spread_bps) параметры проскальзывания!")
        print("   Используем только агрегированные параметры для предотвращения двойного учета")
        # Удаляем детальные параметры
        for key in ['slippage_bps', 'half_spread_bps']:
            params.pop(key, None)

    # Валидируем значения
    if 'commission_pct' in params:
        commission = params['commission_pct']
        if commission < 0 or commission > 0.01:  # Максимум 1%
            raise ValueError(f"commission_pct должен быть в диапазоне [0, 0.01], получен: {commission}")

    if 'slippage_pct' in params:
        slippage = params['slippage_pct']
        if slippage < 0 or slippage > 0.01:  # Максимум 1%
            raise ValueError(f"slippage_pct должен быть в диапазоне [0, 0.01], получен: {slippage}")
