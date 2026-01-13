"""Тесты для утилит работы с метриками."""

import pytest
import numpy as np
from src.optimiser.metric_utils import extract_sharpe, normalize_params


class TestExtractSharpe:
    """Тесты для функции extract_sharpe."""

    @pytest.mark.unit
    def test_extract_sharpe_when_ratio_abs_present_then_returns_abs_value(self):
        """Тест извлечения sharpe_ratio_abs (приоритетная метрика)."""
        result = {
            "sharpe_ratio_abs": 1.5,
            "sharpe_ratio": 1.2,
            "total_trades": 100
        }
        assert extract_sharpe(result) == 1.5

    @pytest.mark.unit
    def test_extract_sharpe_when_only_ratio_present_then_returns_ratio(self):
        """Тест извлечения sharpe_ratio как fallback."""
        result = {
            "sharpe_ratio": 1.2,
            "total_trades": 100
        }
        assert extract_sharpe(result) == 1.2

    @pytest.mark.unit
    def test_extract_sharpe_when_metrics_missing_then_returns_none(self):
        """Тест возврата None при отсутствии метрик."""
        result = {
            "total_trades": 100,
            "max_drawdown": 0.1
        }
        assert extract_sharpe(result) is None

    @pytest.mark.unit
    def test_extract_sharpe_when_nan_value_then_returns_none(self):
        """Тест возврата None для NaN значений."""
        result = {
            "sharpe_ratio_abs": float('nan'),
            "total_trades": 100
        }
        assert extract_sharpe(result) is None

    @pytest.mark.unit
    def test_extract_sharpe_when_invalid_input_then_returns_none(self):
        """Тест возврата None для невалидного входа."""
        assert extract_sharpe(None) is None
        assert extract_sharpe("invalid") is None
        assert extract_sharpe([1, 2, 3]) is None
    
    @pytest.mark.unit
    def test_extract_sharpe_when_string_values_then_handles_gracefully(self):
        """Тест обработки строковых значений."""
        result = {
            "sharpe_ratio_abs": "not_a_number",
            "sharpe_ratio": 1.2
        }
        assert extract_sharpe(result) == 1.2


class TestNormalizeParams:
    """Тесты для функции normalize_params."""

    @pytest.mark.unit
    def test_normalize_params_when_short_names_then_maps_to_canonical(self):
        """Тест маппинга коротких имен на канонические."""
        params = {
            'z_entry': 2.0,
            'z_exit': 0.5,
            'sl_mult': 3.0,
            'time_stop_mult': 2.0,
            'risk_per_pos': 0.02,
            'max_pos_size': 0.05,
            'max_active_pos': 10
        }
        
        normalized = normalize_params(params)
        
        expected = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.05,
            'max_active_positions': 10
        }
        
        assert normalized == expected
    
    @pytest.mark.unit
    def test_normalize_params_when_canonical_names_then_preserves_them(self):
        """Тест сохранения канонических имен."""
        params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'other_param': 'value'
        }
        
        normalized = normalize_params(params)
        
        assert normalized == params
    
    @pytest.mark.unit
    def test_normalize_params_when_mixed_names_then_maps_correctly(self):
        """Тест смешанных коротких и канонических имен."""
        params = {
            'z_entry': 2.0,  # Короткое имя
            'zscore_exit': 0.5,  # Каноническое имя
            'risk_per_pos': 0.02,  # Короткое имя
            'other_param': 'value'  # Неизвестный параметр
        }
        
        normalized = normalize_params(params)
        
        expected = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'risk_per_position_pct': 0.02,
            'other_param': 'value'
        }
        
        assert normalized == expected
    
    @pytest.mark.unit
    def test_normalize_params_when_empty_dict_then_returns_empty(self):
        """Тест пустого словаря параметров."""
        params = {}
        normalized = normalize_params(params)
        assert normalized == {}
    
    @pytest.mark.unit
    def test_normalize_params_when_called_then_preserves_original_dict(self):
        """Тест, что оригинальный словарь не изменяется."""
        params = {'z_entry': 2.0, 'other': 'value'}
        original_params = params.copy()
        
        normalized = normalize_params(params)
        
        # Оригинальный словарь не должен измениться
        assert params == original_params
        # Нормализованный должен отличаться
        assert normalized != params
        assert normalized == {'zscore_threshold': 2.0, 'other': 'value'}