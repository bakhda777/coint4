"""
Автоматический валидатор результатов оптимизации.
Проверяет результаты на корректность и отсутствие overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации."""
    is_valid: bool
    confidence: float  # Уверенность в результате [0, 1]
    issues: List[str]  # Список проблем
    recommendations: List[str]  # Рекомендации
    metrics: Dict[str, float]  # Дополнительные метрики


class AutoValidator:
    """
    Автоматический валидатор результатов оптимизации.
    Проверяет на overfitting, нестабильность и другие проблемы.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Конфигурация валидатора
        """
        self.config = config or self._get_default_config()
        self.validation_history = []
        
    def validate_optimization_result(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        params: Dict[str, Any],
        historical_results: Optional[List[Dict]] = None
    ) -> ValidationResult:
        """
        Валидирует результат оптимизации.
        
        Args:
            train_metrics: Метрики на тренировочном периоде
            test_metrics: Метрики на тестовом периоде
            params: Параметры модели
            historical_results: История предыдущих результатов
            
        Returns:
            ValidationResult с результатами проверки
        """
        issues = []
        recommendations = []
        validation_metrics = {}
        
        # 1. Проверка на overfitting
        overfit_result = self._check_overfitting(train_metrics, test_metrics)
        if not overfit_result['is_ok']:
            issues.append(overfit_result['issue'])
            recommendations.append(overfit_result['recommendation'])
        validation_metrics['overfit_ratio'] = overfit_result['ratio']
        
        # 2. Проверка стабильности метрик
        stability_result = self._check_stability(test_metrics)
        if not stability_result['is_ok']:
            issues.append(stability_result['issue'])
            recommendations.append(stability_result['recommendation'])
        validation_metrics['stability_score'] = stability_result['score']
        
        # 3. Проверка минимального качества
        quality_result = self._check_minimum_quality(test_metrics)
        if not quality_result['is_ok']:
            issues.append(quality_result['issue'])
            recommendations.append(quality_result['recommendation'])
        validation_metrics['quality_score'] = quality_result['score']
        
        # 4. Проверка параметров на разумность
        params_result = self._check_parameter_sanity(params)
        if not params_result['is_ok']:
            issues.append(params_result['issue'])
            recommendations.append(params_result['recommendation'])
        validation_metrics['params_sanity'] = params_result['score']
        
        # 5. Проверка консистентности с историей
        if historical_results:
            consistency_result = self._check_historical_consistency(
                test_metrics, historical_results
            )
            if not consistency_result['is_ok']:
                issues.append(consistency_result['issue'])
                recommendations.append(consistency_result['recommendation'])
            validation_metrics['consistency_score'] = consistency_result['score']
        
        # Вычисляем общую уверенность
        confidence = self._calculate_confidence(validation_metrics)
        
        # Определяем валидность
        is_valid = len(issues) == 0 or (
            len(issues) <= 1 and confidence > 0.7
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            metrics=validation_metrics
        )
        
        # Сохраняем в историю
        self.validation_history.append(result)
        
        return result
    
    def _check_overfitting(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Проверяет на overfitting.
        
        Args:
            train_metrics: Метрики на train
            test_metrics: Метрики на test
            
        Returns:
            Результат проверки
        """
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        
        if train_sharpe <= 0:
            return {
                'is_ok': False,
                'issue': "Отрицательный Sharpe на train",
                'recommendation': "Параметры не подходят для данного периода",
                'ratio': 0.0
            }
        
        # Вычисляем соотношение
        ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0
        
        # Пороги overfitting
        severe_overfit = ratio < 0.3  # Test в 3+ раза хуже train
        moderate_overfit = ratio < 0.5  # Test в 2+ раза хуже train
        
        if severe_overfit:
            return {
                'is_ok': False,
                'issue': f"Сильный overfitting: test/train = {ratio:.2f}",
                'recommendation': "Уменьшите сложность модели или добавьте регуляризацию",
                'ratio': ratio
            }
        
        if moderate_overfit:
            return {
                'is_ok': False,
                'issue': f"Умеренный overfitting: test/train = {ratio:.2f}",
                'recommendation': "Попробуйте увеличить тренировочный период",
                'ratio': ratio
            }
        
        # Проверка на underfitting (test лучше train - подозрительно)
        if ratio > 1.5:
            return {
                'is_ok': False,
                'issue': f"Подозрительный результат: test лучше train в {ratio:.1f} раз",
                'recommendation': "Проверьте на lookahead bias",
                'ratio': ratio
            }
        
        return {
            'is_ok': True,
            'issue': "",
            'recommendation': "",
            'ratio': ratio
        }
    
    def _check_stability(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Проверяет стабильность метрик.
        
        Args:
            metrics: Метрики для проверки
            
        Returns:
            Результат проверки
        """
        # Проверяем волатильность
        volatility = metrics.get('volatility', 1.0)
        max_drawdown = metrics.get('max_drawdown', 1.0)
        
        # Высокая волатильность
        if volatility > 0.5:  # Годовая волатильность > 50%
            return {
                'is_ok': False,
                'issue': f"Высокая волатильность: {volatility:.1%}",
                'recommendation': "Стратегия слишком рискованная",
                'score': 1.0 - volatility
            }
        
        # Большая просадка
        if max_drawdown > 0.3:  # Просадка > 30%
            return {
                'is_ok': False,
                'issue': f"Большая максимальная просадка: {max_drawdown:.1%}",
                'recommendation': "Уменьшите размер позиций или добавьте стоп-лоссы",
                'score': 1.0 - max_drawdown
            }
        
        stability_score = 1.0 - (volatility + max_drawdown) / 2
        
        return {
            'is_ok': True,
            'issue': "",
            'recommendation': "",
            'score': stability_score
        }
    
    def _check_minimum_quality(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Проверяет минимальное качество.
        
        Args:
            metrics: Метрики для проверки
            
        Returns:
            Результат проверки
        """
        sharpe = metrics.get('sharpe_ratio', -10)
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Минимальные пороги
        min_sharpe = self.config['min_sharpe']
        min_win_rate = self.config['min_win_rate']
        min_trades = self.config['min_trades']
        
        issues = []
        
        if sharpe < min_sharpe:
            issues.append(f"Sharpe ratio < {min_sharpe}: {sharpe:.2f}")
        
        if win_rate < min_win_rate:
            issues.append(f"Win rate < {min_win_rate:.0%}: {win_rate:.1%}")
        
        if total_trades < min_trades:
            issues.append(f"Слишком мало сделок: {total_trades} < {min_trades}")
        
        if issues:
            return {
                'is_ok': False,
                'issue': "; ".join(issues),
                'recommendation': "Метрики не соответствуют минимальным требованиям",
                'score': 0.0
            }
        
        # Вычисляем качество
        quality_score = (
            np.clip(sharpe / 2.0, 0, 1) * 0.5 +
            win_rate * 0.3 +
            np.clip(total_trades / 100, 0, 1) * 0.2
        )
        
        return {
            'is_ok': True,
            'issue': "",
            'recommendation': "",
            'score': quality_score
        }
    
    def _check_parameter_sanity(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Проверяет параметры на разумность.
        
        Args:
            params: Параметры для проверки
            
        Returns:
            Результат проверки
        """
        issues = []
        
        # Проверяем zscore параметры
        zscore_threshold = params.get('zscore_threshold', 2.0)
        zscore_exit = params.get('zscore_exit', 0.5)
        
        if zscore_threshold < zscore_exit:
            issues.append("zscore_threshold < zscore_exit")
        
        if zscore_threshold - zscore_exit < 0.5:
            issues.append("Слишком маленький гистерезис между entry и exit")
        
        # Проверяем риск параметры
        stop_loss = params.get('stop_loss_multiplier', 3.0)
        if stop_loss < 1.5:
            issues.append(f"Слишком тайтный стоп-лосс: {stop_loss}")
        
        if stop_loss > 5.0:
            issues.append(f"Слишком широкий стоп-лосс: {stop_loss}")
        
        # Проверяем окно
        rolling_window = params.get('rolling_window', 60)
        if rolling_window < 20:
            issues.append(f"Слишком маленькое окно: {rolling_window}")
        
        if rolling_window > 200:
            issues.append(f"Слишком большое окно: {rolling_window}")
        
        if issues:
            return {
                'is_ok': False,
                'issue': "; ".join(issues),
                'recommendation': "Проверьте логику параметров",
                'score': 0.5
            }
        
        return {
            'is_ok': True,
            'issue': "",
            'recommendation': "",
            'score': 1.0
        }
    
    def _check_historical_consistency(
        self,
        current_metrics: Dict[str, float],
        historical_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Проверяет консистентность с историей.
        
        Args:
            current_metrics: Текущие метрики
            historical_results: Исторические результаты
            
        Returns:
            Результат проверки
        """
        if len(historical_results) < 3:
            return {
                'is_ok': True,
                'issue': "",
                'recommendation': "",
                'score': 1.0
            }
        
        # Извлекаем исторические Sharpe ratios
        historical_sharpes = [r.get('sharpe_ratio', 0) for r in historical_results[-10:]]
        current_sharpe = current_metrics.get('sharpe_ratio', 0)
        
        # Вычисляем статистики
        mean_sharpe = np.mean(historical_sharpes)
        std_sharpe = np.std(historical_sharpes)
        
        if std_sharpe == 0:
            return {
                'is_ok': True,
                'issue': "",
                'recommendation': "",
                'score': 1.0
            }
        
        # Z-score текущего результата
        z_score = (current_sharpe - mean_sharpe) / std_sharpe
        
        # Проверяем на выброс
        if abs(z_score) > 3:
            return {
                'is_ok': False,
                'issue': f"Результат сильно отличается от истории (z-score: {z_score:.2f})",
                'recommendation': "Проверьте данные и параметры",
                'score': 1.0 / (1.0 + abs(z_score))
            }
        
        # Проверяем на деградацию
        if current_sharpe < mean_sharpe - 2 * std_sharpe:
            return {
                'is_ok': False,
                'issue': "Результат хуже исторического среднего",
                'recommendation': "Возможна деградация модели или изменение рынка",
                'score': 0.5
            }
        
        consistency_score = 1.0 - min(abs(z_score) / 3.0, 1.0)
        
        return {
            'is_ok': True,
            'issue': "",
            'recommendation': "",
            'score': consistency_score
        }
    
    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Вычисляет общую уверенность в результате.
        
        Args:
            metrics: Метрики валидации
            
        Returns:
            Уверенность [0, 1]
        """
        scores = []
        weights = []
        
        if 'overfit_ratio' in metrics:
            # Преобразуем ratio в score (идеально около 0.8)
            overfit_score = 1.0 - abs(metrics['overfit_ratio'] - 0.8) / 0.8
            scores.append(np.clip(overfit_score, 0, 1))
            weights.append(0.3)
        
        if 'stability_score' in metrics:
            scores.append(metrics['stability_score'])
            weights.append(0.2)
        
        if 'quality_score' in metrics:
            scores.append(metrics['quality_score'])
            weights.append(0.3)
        
        if 'params_sanity' in metrics:
            scores.append(metrics['params_sanity'])
            weights.append(0.1)
        
        if 'consistency_score' in metrics:
            scores.append(metrics['consistency_score'])
            weights.append(0.1)
        
        if not scores:
            return 0.5
        
        # Взвешенное среднее
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(scores)
        
        confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return np.clip(confidence, 0, 1)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает дефолтную конфигурацию."""
        return {
            'min_sharpe': 0.5,
            'min_win_rate': 0.4,
            'min_trades': 10,
            'max_drawdown': 0.3,
            'max_volatility': 0.5
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по всем валидациям.
        
        Returns:
            Словарь со статистикой
        """
        if not self.validation_history:
            return {}
        
        valid_count = sum(1 for v in self.validation_history if v.is_valid)
        total_count = len(self.validation_history)
        
        avg_confidence = np.mean([v.confidence for v in self.validation_history])
        
        # Собираем все проблемы
        all_issues = []
        for v in self.validation_history:
            all_issues.extend(v.issues)
        
        # Частота проблем
        issue_frequency = {}
        for issue in all_issues:
            key = issue.split(':')[0]  # Берем только тип проблемы
            issue_frequency[key] = issue_frequency.get(key, 0) + 1
        
        return {
            'valid_count': valid_count,
            'total_count': total_count,
            'success_rate': valid_count / total_count if total_count > 0 else 0,
            'avg_confidence': avg_confidence,
            'common_issues': sorted(
                issue_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }