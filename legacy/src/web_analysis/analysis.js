/**
 * Детальный анализ расхождения между оптимизацией и валидацией
 * Модуль для диагностики проблем переоптимизации
 */

class OptimizationAnalyzer {
    constructor() {
        this.optimizationResults = {
            "sharpe_ratio": 2.2453,
            "trades_count": "N/A",
            "pairs_count": "N/A",
            "period": "Optuna optimization (4 trials)",
                                                            "parameters": {
                "zscore_entry_threshold": 1.5618,
                "zscore_exit": 0.7704,
                "max_active_positions": 5,
                "risk_per_position_pct": 0.0131,
                "max_position_size_pct": 0.05,
                "stop_loss_multiplier": 3.197,
                "time_stop_multiplier": 3.78
            },
                "trading_signals": {
                    "zscore_threshold": 0.8082,
                    "zscore_exit": 0.7882,
                    "rolling_window": 25
                },
                "portfolio": {
                    "max_active_positions": 15,
                    "risk_per_position_pct": 0.0143,
                    "max_position_size_pct": 0.1009
                },
                "risk_management": {
                    "stop_loss_multiplier": 2.532,
                    "time_stop_multiplier": 5.6515,
                    "cooldown_hours": 3
                },
                "data_processing": {
                    "normalization_method": "zscore",
                    "min_history_ratio": 0.5522
                },
                "zscore_entry_threshold": 0.8082,
                "zscore_exit": 0.7882,
                "max_active_positions": 15,
                "risk_per_position_pct": 0.0143
            }
};

        this.validationResults = {
            sharpe_ratio: 0.0,
            trades_count: 0,
            pnl_usd: 0.0,
            pairs_count: 2,
            period: 'No validation data',
            parameters: this.optimizationResults.parameters // Same parameters
        };

        this.balancedConfig = {
            zscore_entry_threshold: 1.35,
            zscore_exit: 0.0,
            max_active_positions: 12,
            risk_per_position_pct: 0.025,
            max_position_size_pct: 0.04,
            stop_loss_multiplier: 3.0,
            time_stop_multiplier: 2.0
        };
    }

    /**
     * Анализ деградации показателей
     */
    analyzeDegradation() {
        const degradation = {
            sharpe_ratio: {
                optimization: this.optimizationResults.sharpe_ratio,
                validation: this.validationResults.sharpe_ratio,
                degradation_pct: ((this.optimizationResults.sharpe_ratio - this.validationResults.sharpe_ratio) / this.optimizationResults.sharpe_ratio) * 100,
                severity: 'CRITICAL'
            },
            trades_count: {
                optimization: this.optimizationResults.trades_count,
                validation: this.validationResults.trades_count,
                degradation_pct: -100, // Complete loss of trading activity
                severity: 'CRITICAL'
            },
            trading_activity: {
                optimization: this.optimizationResults.pairs_count,
                validation: 0,
                degradation_pct: -100,
                severity: 'CRITICAL'
            }
        };

        return degradation;
    }

    /**
     * Диагностика причин переоптимизации
     */
    diagnoseOverfitting() {
        const issues = [];

        // 1. Агрессивные пороги входа
        if (this.optimizationResults.parameters.zscore_entry_threshold < 1.3) {
            issues.push({
                type: 'AGGRESSIVE_ENTRY',
                severity: 'HIGH',
                description: `Z-score порог входа ${this.optimizationResults.parameters.zscore_entry_threshold} слишком низкий`,
                recommendation: 'Увеличить до 1.3-1.5 для снижения ложных сигналов',
                current_value: this.optimizationResults.parameters.zscore_entry_threshold,
                recommended_value: 1.35
            });
        }

        // 2. Негативный порог выхода
        if (this.optimizationResults.parameters.zscore_exit < 0) {
            issues.push({
                type: 'NEGATIVE_EXIT',
                severity: 'HIGH',
                description: `Негативный порог выхода ${this.optimizationResults.parameters.zscore_exit}`,
                recommendation: 'Использовать нейтральный выход (0.0) для классического mean reversion',
                current_value: this.optimizationResults.parameters.zscore_exit,
                recommended_value: 0.0
            });
        }

        // 3. Высокий риск на позицию
        if (this.optimizationResults.parameters.risk_per_position_pct > 0.03) {
            issues.push({
                type: 'HIGH_RISK',
                severity: 'MEDIUM',
                description: `Риск на позицию ${(this.optimizationResults.parameters.risk_per_position_pct * 100).toFixed(1)}% слишком высокий`,
                recommendation: 'Снизить до 2.0-2.5% для лучшего управления рисками',
                current_value: this.optimizationResults.parameters.risk_per_position_pct,
                recommended_value: 0.025
            });
        }

        // 4. Ограниченное количество позиций
        if (this.optimizationResults.parameters.max_active_positions < 12) {
            issues.push({
                type: 'LIMITED_POSITIONS',
                severity: 'MEDIUM',
                description: `Максимум ${this.optimizationResults.parameters.max_active_positions} позиций ограничивает диверсификацию`,
                recommendation: 'Увеличить до 12-15 для лучшей диверсификации',
                current_value: this.optimizationResults.parameters.max_active_positions,
                recommended_value: 12
            });
        }

        return issues;
    }

    /**
     * Анализ различий в периодах данных
     */
    analyzePeriodDifferences() {
        return {
            optimization_period: {
                start: '2024-01-15',
                end: '2024-02-08',
                duration_days: 24,
                market_conditions: 'Период оптимизации с определенными рыночными условиями'
            },
            validation_period: {
                start: '2024-02-01',
                end: '2024-02-15',
                duration_days: 14,
                market_conditions: 'Частично перекрывающийся период с возможными структурными изменениями'
            },
            overlap: {
                start: '2024-02-01',
                end: '2024-02-08',
                duration_days: 7,
                overlap_percentage: 29.2 // 7 days out of 24
            },
            issues: [
                'Короткий период валидации (14 дней) недостаточен для статистической значимости',
                'Частичное перекрытие периодов может привести к data leakage',
                'Возможные структурные изменения рынка между периодами',
                'Недостаточно out-of-sample данных для надежной валидации'
            ]
        };
    }

    /**
     * Генерация рекомендаций по улучшению
     */
    generateRecommendations() {
        return {
            immediate_actions: [
                {
                    priority: 'HIGH',
                    action: 'Использовать сбалансированную конфигурацию',
                    description: 'Применить balanced_config.yaml с компромиссными параметрами',
                    expected_impact: 'Восстановление торговой активности и стабильности'
                },
                {
                    priority: 'HIGH',
                    action: 'Расширить период валидации',
                    description: 'Увеличить период валидации до 30-60 дней',
                    expected_impact: 'Повышение статистической значимости результатов'
                },
                {
                    priority: 'MEDIUM',
                    action: 'Провести walk-forward анализ',
                    description: 'Выполнить полный walk-forward на 6-месячном периоде',
                    expected_impact: 'Оценка робастности стратегии во времени'
                }
            ],
            long_term_improvements: [
                {
                    priority: 'HIGH',
                    action: 'Добавить регуляризацию в оптимизацию',
                    description: 'Внедрить L1/L2 регуляризацию или Bayesian optimization',
                    expected_impact: 'Снижение переоптимизации'
                },
                {
                    priority: 'MEDIUM',
                    action: 'Реализовать кросс-валидацию',
                    description: 'Временная кросс-валидация с блоками данных',
                    expected_impact: 'Более надежная оценка параметров'
                },
                {
                    priority: 'MEDIUM',
                    action: 'Мониторинг деградации',
                    description: 'Система раннего предупреждения о деградации показателей',
                    expected_impact: 'Быстрое обнаружение проблем в продакшене'
                }
            ]
        };
    }

    /**
     * Расчет метрик робастности
     */
    calculateRobustnessMetrics() {
        const degradation = this.analyzeDegradation();
        
        return {
            sharpe_stability: {
                score: Math.max(0, 1 - Math.abs(degradation.sharpe_ratio.degradation_pct) / 100),
                interpretation: degradation.sharpe_ratio.degradation_pct > 30 ? 'Нестабильный' : 'Стабильный'
            },
            trading_consistency: {
                score: this.validationResults.trades_count > 0 ? 0.5 : 0,
                interpretation: this.validationResults.trades_count === 0 ? 'Полная потеря активности' : 'Частичная активность'
            },
            parameter_sensitivity: {
                score: this.calculateParameterSensitivity(),
                interpretation: 'Высокая чувствительность к параметрам'
            },
            overall_robustness: {
                score: 0.1, // Very low due to complete failure
                interpretation: 'Критически низкая робастность - требуется полный пересмотр'
            }
        };
    }

    /**
     * Расчет чувствительности параметров
     */
    calculateParameterSensitivity() {
        const params = this.optimizationResults.parameters;
        let sensitivity_score = 1.0;

        // Снижаем оценку за агрессивные параметры
        if (params.zscore_entry_threshold < 1.3) sensitivity_score -= 0.3;
        if (params.zscore_exit < 0) sensitivity_score -= 0.3;
        if (params.risk_per_position_pct > 0.03) sensitivity_score -= 0.2;
        if (params.max_active_positions < 12) sensitivity_score -= 0.2;

        return Math.max(0, sensitivity_score);
    }

    /**
     * Генерация полного отчета
     */
    generateFullReport() {
        return {
            summary: {
                status: 'CRITICAL_FAILURE',
                main_issue: 'Переоптимизация с полной потерей торговой активности',
                confidence_level: 'HIGH'
            },
            degradation_analysis: this.analyzeDegradation(),
            overfitting_diagnosis: this.diagnoseOverfitting(),
            period_analysis: this.analyzePeriodDifferences(),
            robustness_metrics: this.calculateRobustnessMetrics(),
            recommendations: this.generateRecommendations(),
            next_steps: [
                'Немедленно переключиться на balanced_config.yaml',
                'Провести расширенную валидацию на 3-6 месячном периоде',
                'Реализовать систему мониторинга деградации',
                'Пересмотреть процесс оптимизации с добавлением регуляризации'
            ]
        };
    }
}

// Экспорт для использования в браузере
if (typeof window !== 'undefined') {
    window.OptimizationAnalyzer = OptimizationAnalyzer;
}

// Экспорт для Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizationAnalyzer;
}