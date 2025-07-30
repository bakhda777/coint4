/**
 * Главный JavaScript файл для веб-приложения анализа оптимизации
 * Обеспечивает интерактивность и визуализацию данных
 */

class OptimizationDashboard {
    constructor() {
        this.analyzer = new OptimizationAnalyzer();
        this.chartInstance = null;
        this.init();
    }

    /**
     * Инициализация приложения
     */
    init() {
        this.loadData();
        this.setupEventListeners();
        this.createDegradationChart();
        this.animateCounters();
    }

    /**
     * Загрузка и отображение данных
     */
    loadData() {
        const report = this.analyzer.generateFullReport();
        
        // Обновление карточек результатов
        this.updateResultCards(report);
        
        // Обновление анализа деградации
        this.updateDegradationAnalysis(report.degradation_analysis);
        
        // Обновление диагностики переоптимизации
        this.updateOverfittingDiagnosis(report.overfitting_diagnosis);
        
        // Обновление подобранных параметров
        this.updateOptimizedParameters();

        // Обновление сравнения параметров
        this.updateParametersComparison();

        // Обновление рекомендаций
        this.updateRecommendations(report.recommendations);
        
        // Обновление метрик робастности
        this.updateRobustnessMetrics(report.robustness_metrics);
    }

    /**
     * Обновление карточек результатов
     */
    updateResultCards(report) {
        const optimizationCard = document.querySelector('.optimization-card');
        const validationCard = document.querySelector('.validation-card');
        const degradationCard = document.querySelector('.degradation-card');

        // Карточка оптимизации
        optimizationCard.innerHTML = `
            <div class="card-header">
                <div class="card-icon">📈</div>
                <h3 class="card-title">Результаты оптимизации</h3>
                <div class="card-subtitle">Лучший Sharpe из Optuna</div>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value positive">${this.analyzer.optimizationResults.sharpe_ratio}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Количество сделок</span>
                <span class="metric-value neutral ${typeof this.analyzer.optimizationResults.trades_count === 'string' ? 'no-animate' : ''}">${typeof this.analyzer.optimizationResults.trades_count === 'string' ? this.analyzer.optimizationResults.trades_count : this.analyzer.optimizationResults.trades_count.toLocaleString()}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Торговых пар</span>
                <span class="metric-value neutral ${typeof this.analyzer.optimizationResults.pairs_count === 'string' ? 'no-animate' : ''}">${this.analyzer.optimizationResults.pairs_count}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Период</span>
                <span class="metric-value neutral no-animate">${this.analyzer.optimizationResults.period}</span>
            </div>
        `;

        // Карточка валидации
        validationCard.innerHTML = `
            <div class="card-header">
                <div class="card-icon">📉</div>
                <h3 class="card-title">Результаты валидации</h3>
                <div class="card-subtitle">Walk-forward тестирование</div>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value ${this.analyzer.validationResults.sharpe_ratio >= 0 ? 'positive' : 'negative'}">${this.analyzer.validationResults.sharpe_ratio}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Количество сделок</span>
                <span class="metric-value neutral">${this.analyzer.validationResults.trades_count.toLocaleString()}</span>
            </div>
            <div class="metric">
                <span class="metric-label">PnL (USD)</span>
                <span class="metric-value ${this.analyzer.validationResults.pnl_usd >= 0 ? 'positive' : 'negative'}">$${this.analyzer.validationResults.pnl_usd}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Период</span>
                <span class="metric-value neutral no-animate">${this.analyzer.validationResults.period}</span>
            </div>
        `;

        // Карточка деградации
        const sharpeDegradation = report.degradation_analysis.sharpe_ratio.degradation_pct;
        degradationCard.innerHTML = `
            <div class="card-header">
                <div class="card-icon">⚠️</div>
                <h3 class="card-title">Анализ деградации</h3>
            </div>
            <div class="metric">
                <span class="metric-label">Деградация Sharpe</span>
                <span class="metric-value negative">${sharpeDegradation.toFixed(1)}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Потеря активности</span>
                <span class="metric-value negative">100%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Статус</span>
                <span class="status-badge status-critical">Критический</span>
            </div>
            <div class="metric">
                <span class="metric-label">Робастность</span>
                <span class="metric-value negative">${(report.robustness_metrics.overall_robustness.score * 100).toFixed(1)}%</span>
            </div>
        `;
    }

    /**
     * Обновление анализа деградации
     */
    updateDegradationAnalysis(degradationData) {
        const container = document.getElementById('degradationDetails');
        
        container.innerHTML = `
            <div class="degradation-item">
                <h4>Sharpe Ratio</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.abs(degradationData.sharpe_ratio.degradation_pct)}%"></div>
                </div>
                <p>Деградация: ${degradationData.sharpe_ratio.degradation_pct.toFixed(1)}% (${degradationData.sharpe_ratio.severity})</p>
            </div>
            <div class="degradation-item">
                <h4>Торговая активность</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%; background: #e74c3c;"></div>
                </div>
                <p>Полная потеря торговой активности (${degradationData.trades_count.severity})</p>
            </div>
        `;
    }

    /**
     * Обновление диагностики переоптимизации
     */
    updateOverfittingDiagnosis(issues) {
        const container = document.getElementById('overfittingIssues');
        
        container.innerHTML = issues.map(issue => `
            <div class="cause-item priority-${issue.severity.toLowerCase()}">
                <div class="cause-title">${this.getIssueTitle(issue.type)}</div>
                <div class="cause-description">
                    <p><strong>Проблема:</strong> ${issue.description}</p>
                    <p><strong>Текущее значение:</strong> ${issue.current_value}</p>
                    <p><strong>Рекомендуемое:</strong> ${issue.recommended_value}</p>
                    <p><strong>Решение:</strong> ${issue.recommendation}</p>
                </div>
            </div>
        `).join('');
    }

    /**
     * Получение заголовка для типа проблемы
     */
    getIssueTitle(type) {
        const titles = {
            'AGGRESSIVE_ENTRY': 'Агрессивные пороги входа',
            'NEGATIVE_EXIT': 'Негативный порог выхода',
            'HIGH_RISK': 'Высокий риск на позицию',
            'LIMITED_POSITIONS': 'Ограниченная диверсификация'
        };
        return titles[type] || type;
    }

    /**
     * Обновление подобранных параметров оптимизации
     */
    updateOptimizedParameters() {
        const container = document.getElementById('optimizedParameters');
        const params = this.analyzer.optimizationResults.parameters;
        const optResults = this.analyzer.optimizationResults;

        // Отладочная информация
        console.log('Optimization results:', optResults);
        console.log('Parameters:', params);

        if (!params) {
            console.error('Parameters not found!');
            container.innerHTML = '<div class="error">Параметры не загружены</div>';
            return;
        }

        // Функция для рендеринга группы параметров
        const renderParamGroup = (title, icon, paramObj, descriptions = {}) => {
            if (!paramObj || Object.keys(paramObj).length === 0) return '';

            let html = `<div class="param-group">
                <h4 class="param-group-title">${icon} ${title}</h4>`;

            for (const [key, value] of Object.entries(paramObj)) {
                const description = descriptions[key] || '';
                html += `
                    <div class="param-item">
                        <span class="param-name">${key.replace(/_/g, ' ')}</span>
                        <span class="param-value highlight">${value}</span>
                        <span class="param-description">${description}</span>
                    </div>`;
            }

            html += '</div>';
            return html;
        };

        // Описания параметров
        const descriptions = {
            // Отбор пар
            'coint_pvalue_threshold': 'P-value для коинтеграции',
            'lookback_days': 'Период анализа истории',
            'max_hurst_exponent': 'Максимальный Hurst для mean reversion',
            'min_half_life_days': 'Минимальный half-life',
            'max_half_life_days': 'Максимальный half-life',
            'min_mean_crossings': 'Минимум пересечений среднего',
            'ssd_top_n': 'Количество пар по SSD',
            'pvalue_top_n': 'Количество пар по p-value',

            // Торговые сигналы
            'zscore_threshold': 'Порог входа в позицию',
            'zscore_exit': 'Порог выхода из позиции',
            'rolling_window': 'Окно для расчета z-score',

            // Портфель
            'max_active_positions': 'Максимум одновременных позиций',
            'risk_per_position_pct': 'Риск на позицию (%)',
            'max_position_size_pct': 'Максимальный размер позиции (%)',

            // Риск-менеджмент
            'stop_loss_multiplier': 'Множитель стоп-лосса',
            'time_stop_multiplier': 'Множитель тайм-стопа',
            'cooldown_hours': 'Кулдаун между сделками (часы)',

            // Исполнение
            'commission_pct': 'Комиссия (%)',
            'slippage_pct': 'Проскальзывание (%)',

            // Обработка данных
            'normalization_method': 'Метод нормализации',
            'min_history_ratio': 'Минимум истории для анализа',

            // Обратная совместимость
            'zscore_entry_threshold': 'Порог входа в позицию',
            'zscore_exit': 'Порог выхода из позиции'
        };

        let parametersHtml = '<div class="parameters-grid">';

        // Рендерим категории параметров
        if (params.pair_selection) {
            parametersHtml += renderParamGroup('Отбор пар', '📊', params.pair_selection, descriptions);
        }

        if (params.trading_signals) {
            parametersHtml += renderParamGroup('Торговые сигналы', '🎯', params.trading_signals, descriptions);
        }

        if (params.portfolio) {
            parametersHtml += renderParamGroup('Управление портфелем', '💰', params.portfolio, descriptions);
        }

        if (params.risk_management) {
            parametersHtml += renderParamGroup('Риск-менеджмент', '🛡️', params.risk_management, descriptions);
        }

        if (params.execution) {
            parametersHtml += renderParamGroup('Исполнение', '⚡', params.execution, descriptions);
        }

        if (params.data_processing) {
            parametersHtml += renderParamGroup('Обработка данных', '🔧', params.data_processing, descriptions);
        }

        // Если нет категорий, показываем основные параметры
        if (!params.pair_selection && !params.trading_signals) {
            parametersHtml += `
                <div class="param-group">
                    <h4 class="param-group-title">🎯 Основные параметры</h4>
                    <div class="param-item">
                        <span class="param-name">Z-score порог входа</span>
                        <span class="param-value highlight">${params.zscore_entry_threshold || params.zscore_threshold || 'N/A'}</span>
                        <span class="param-description">Порог для открытия позиций</span>
                    </div>
                    <div class="param-item">
                        <span class="param-name">Z-score выход</span>
                        <span class="param-value highlight">${params.zscore_exit || 'N/A'}</span>
                            <span class="param-description">Порог для закрытия позиций</span>
                        </div>
                    </div>

                    <div class="param-group">
                        <h4 class="param-group-title">💼 Управление портфелем</h4>
                        <div class="param-item">
                            <span class="param-name">Максимум позиций</span>
                            <span class="param-value highlight">${params.max_active_positions}</span>
                            <span class="param-description">Одновременных позиций</span>
                        </div>
                        <div class="param-item">
                            <span class="param-name">Риск на позицию</span>
                            <span class="param-value highlight">${(params.risk_per_position_pct * 100).toFixed(2)}%</span>
                            <span class="param-description">От общего капитала</span>
                        </div>
                        <div class="param-item">
                            <span class="param-name">Размер позиции</span>
                            <span class="param-value highlight">${(params.max_position_size_pct * 100).toFixed(2)}%</span>
                            <span class="param-description">Максимальный размер</span>
                        </div>
                    </div>

                    <div class="param-group">
                        <h4 class="param-group-title">🛡️ Управление рисками</h4>
                        <div class="param-item">
                            <span class="param-name">Стоп-лосс множитель</span>
                            <span class="param-value highlight">${params.stop_loss_multiplier}</span>
                            <span class="param-description">Относительно волатильности</span>
                        </div>
                        <div class="param-item">
                            <span class="param-name">Временной стоп</span>
                            <span class="param-value highlight">${params.time_stop_multiplier}</span>
                            <span class="param-description">Множитель времени удержания</span>
                        </div>
                    </div>
                </div>`;
        }

        parametersHtml += '</div>';

        container.innerHTML = `
            <div class="optimized-parameters">
                ${parametersHtml}

                <div class="optimization-summary">
                    <div class="summary-item">
                        <span class="summary-label">🏆 Достигнутый Sharpe Ratio</span>
                        <span class="summary-value positive">${optResults.sharpe_ratio}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">📊 Количество сделок</span>
                        <span class="summary-value">${typeof optResults.trades_count === 'string' ? optResults.trades_count : optResults.trades_count.toLocaleString()}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">🔗 Торговых пар</span>
                        <span class="summary-value">${optResults.pairs_count}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">⏱️ Период оптимизации</span>
                        <span class="summary-value">${optResults.period}</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Обновление сравнения параметров
     */
    updateParametersComparison() {
        const container = document.getElementById('parametersComparison');
        const params = this.analyzer.optimizationResults.parameters;
        const balanced = this.analyzer.balancedConfig;
        
        container.innerHTML = `
            <div class="parameters-comparison">
                <div class="param-item">
                    <span class="param-name">Z-score вход</span>
                    <div>
                        <span class="param-value" style="color: #e74c3c;">${params.zscore_entry_threshold}</span>
                        <span style="margin: 0 10px;">→</span>
                        <span class="param-value" style="color: #27ae60;">${balanced.zscore_entry_threshold}</span>
                    </div>
                </div>
                <div class="param-item">
                    <span class="param-name">Z-score выход</span>
                    <div>
                        <span class="param-value" style="color: #e74c3c;">${params.zscore_exit}</span>
                        <span style="margin: 0 10px;">→</span>
                        <span class="param-value" style="color: #27ae60;">${balanced.zscore_exit}</span>
                    </div>
                </div>
                <div class="param-item">
                    <span class="param-name">Макс. позиций</span>
                    <div>
                        <span class="param-value" style="color: #e74c3c;">${params.max_active_positions}</span>
                        <span style="margin: 0 10px;">→</span>
                        <span class="param-value" style="color: #27ae60;">${balanced.max_active_positions}</span>
                    </div>
                </div>
                <div class="param-item">
                    <span class="param-name">Риск на позицию</span>
                    <div>
                        <span class="param-value" style="color: #e74c3c;">${(params.risk_per_position_pct * 100).toFixed(1)}%</span>
                        <span style="margin: 0 10px;">→</span>
                        <span class="param-value" style="color: #27ae60;">${(balanced.risk_per_position_pct * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Обновление рекомендаций
     */
    updateRecommendations(recommendations) {
        const immediateContainer = document.getElementById('immediateActions');
        const longTermContainer = document.getElementById('longTermImprovements');
        
        immediateContainer.innerHTML = recommendations.immediate_actions.map(action => `
            <div class="recommendation-item priority-${action.priority.toLowerCase()}">
                <div class="rec-title">${action.action}</div>
                <div class="rec-description">
                    <p>${action.description}</p>
                    <p><strong>Ожидаемый эффект:</strong> ${action.expected_impact}</p>
                </div>
            </div>
        `).join('');
        
        longTermContainer.innerHTML = recommendations.long_term_improvements.map(improvement => `
            <div class="recommendation-item priority-${improvement.priority.toLowerCase()}">
                <div class="rec-title">${improvement.action}</div>
                <div class="rec-description">
                    <p>${improvement.description}</p>
                    <p><strong>Ожидаемый эффект:</strong> ${improvement.expected_impact}</p>
                </div>
            </div>
        `).join('');
    }

    /**
     * Обновление метрик робастности
     */
    updateRobustnessMetrics(metrics) {
        const container = document.getElementById('robustnessMetrics');
        
        container.innerHTML = `
            <div class="metric">
                <span class="metric-label">Стабильность Sharpe</span>
                <span class="metric-value ${metrics.sharpe_stability.score > 0.7 ? 'positive' : 'negative'}">
                    ${(metrics.sharpe_stability.score * 100).toFixed(1)}% (${metrics.sharpe_stability.interpretation})
                </span>
            </div>
            <div class="metric">
                <span class="metric-label">Консистентность торговли</span>
                <span class="metric-value ${metrics.trading_consistency.score > 0.5 ? 'positive' : 'negative'}">
                    ${(metrics.trading_consistency.score * 100).toFixed(1)}% (${metrics.trading_consistency.interpretation})
                </span>
            </div>
            <div class="metric">
                <span class="metric-label">Чувствительность параметров</span>
                <span class="metric-value ${metrics.parameter_sensitivity.score > 0.7 ? 'positive' : 'negative'}">
                    ${(metrics.parameter_sensitivity.score * 100).toFixed(1)}% (${metrics.parameter_sensitivity.interpretation})
                </span>
            </div>
            <div class="metric">
                <span class="metric-label">Общая робастность</span>
                <span class="metric-value ${metrics.overall_robustness.score > 0.5 ? 'positive' : 'negative'}">
                    ${(metrics.overall_robustness.score * 100).toFixed(1)}% (${metrics.overall_robustness.interpretation})
                </span>
            </div>
        `;
    }

    /**
     * Создание графика деградации
     */
    createDegradationChart() {
        const ctx = document.getElementById('degradationChart');
        if (!ctx) return;
        
        const degradationData = this.analyzer.analyzeDegradation();
        
        // Данные для графика
        const data = {
            labels: ['Sharpe Ratio', 'Количество сделок', 'Торговая активность'],
            datasets: [{
                label: 'Оптимизация',
                data: [
                    degradationData.sharpe_ratio.optimization,
                    degradationData.trades_count.optimization / 100, // Нормализация
                    degradationData.trading_activity.optimization / 10 // Нормализация
                ],
                backgroundColor: 'rgba(46, 204, 113, 0.8)',
                borderColor: 'rgba(46, 204, 113, 1)',
                borderWidth: 2
            }, {
                label: 'Валидация',
                data: [
                    degradationData.sharpe_ratio.validation,
                    degradationData.trades_count.validation / 100, // Нормализация
                    degradationData.trading_activity.validation / 10 // Нормализация
                ],
                backgroundColor: 'rgba(231, 76, 60, 0.8)',
                borderColor: 'rgba(231, 76, 60, 1)',
                borderWidth: 2
            }]
        };
        
        const config = {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Сравнение показателей: Оптимизация vs Валидация',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Нормализованные значения'
                        }
                    }
                }
            }
        };
        
        // Создание графика
        if (typeof Chart !== 'undefined') {
            this.chartInstance = new Chart(ctx, config);
        } else {
            // Fallback если Chart.js не загружен
            ctx.innerHTML = '<p class="text-center text-muted">График недоступен. Загрузите Chart.js для визуализации.</p>';
        }
    }

    /**
     * Анимация счетчиков
     */
    animateCounters() {
        const counters = document.querySelectorAll('.metric-value:not(.no-animate)');

        counters.forEach(counter => {
            const target = parseFloat(counter.textContent.replace(/[^0-9.-]/g, ''));
            if (isNaN(target)) return;
            
            let current = 0;
            const increment = target / 50;
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                
                // Форматирование в зависимости от типа значения
                if (counter.textContent.includes('%')) {
                    counter.textContent = current.toFixed(1) + '%';
                } else if (counter.textContent.includes('$')) {
                    counter.textContent = '$' + current.toFixed(2);
                } else if (target > 100) {
                    counter.textContent = Math.round(current).toLocaleString();
                } else {
                    counter.textContent = current.toFixed(4);
                }
            }, 20);
        });
    }

    /**
     * Настройка обработчиков событий
     */
    setupEventListeners() {
        // Обработчик для кнопки обновления данных
        const refreshButton = document.getElementById('refreshData');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.loadData();
                this.showNotification('Данные обновлены', 'success');
            });
        }
        
        // Обработчик для экспорта отчета
        const exportButton = document.getElementById('exportReport');
        if (exportButton) {
            exportButton.addEventListener('click', () => {
                this.exportReport();
            });
        }
        
        // Обработчики для интерактивных элементов
        document.querySelectorAll('.result-card').forEach(card => {
            card.addEventListener('click', () => {
                card.style.transform = 'scale(1.02)';
                setTimeout(() => {
                    card.style.transform = '';
                }, 200);
            });
        });
    }

    /**
     * Показ уведомлений
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        if (type === 'success') {
            notification.style.backgroundColor = '#27ae60';
        } else if (type === 'error') {
            notification.style.backgroundColor = '#e74c3c';
        } else {
            notification.style.backgroundColor = '#3498db';
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    /**
     * Экспорт отчета
     */
    exportReport() {
        const report = this.analyzer.generateFullReport();
        const dataStr = JSON.stringify(report, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `optimization_analysis_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        this.showNotification('Отчет экспортирован', 'success');
    }
}

// Инициализация приложения после загрузки DOM
document.addEventListener('DOMContentLoaded', () => {
    new OptimizationDashboard();
});

// Экспорт для глобального использования
if (typeof window !== 'undefined') {
    window.OptimizationDashboard = OptimizationDashboard;
}