"""Генератор детализированного отчета для бэктест-системы.

Этот модуль создает комплексный отчет с секциями:
- CONFIG: Конфигурация и параметры
- METRICS_TEST: Тестовые метрики и статистики
- TURNOVER: Анализ оборачиваемости
- COST_BREAKDOWN: Детализация издержек
- DRAWDOWN_TABLE: Таблица просадок
- SENSITIVITY_FEES: Анализ чувствительности к комиссиям
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import json

from ..utils.config_loader import BacktestConfig
from ..engine.backtest_engine import PairBacktester


class DetailedReportGenerator:
    """Генератор детализированного отчета."""
    
    def __init__(self, config: BacktestConfig):
        """Инициализация генератора отчета.
        
        Args:
            config: Конфигурация бэктеста
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Настройка стиля графиков
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_full_report(self, 
                           backtest_results: Dict[str, Any],
                           pair_data: pd.DataFrame,
                           output_dir: str = "reports") -> str:
        """Генерирует полный отчет.
        
        Args:
            backtest_results: Результаты бэктеста
            pair_data: Исходные данные пары
            output_dir: Директория для сохранения отчета
            
        Returns:
            str: Путь к сгенерированному отчету
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(output_dir) / f"backtest_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Генерация отчета в: {report_dir}")
        
        # Генерируем все секции
        sections = {}
        
        if self.config.reporting.sections.config:
            sections['CONFIG'] = self._generate_config_section()
        
        if self.config.reporting.sections.metrics_test:
            sections['METRICS_TEST'] = self._generate_metrics_section(backtest_results)
        
        if self.config.reporting.sections.turnover:
            sections['TURNOVER'] = self._generate_turnover_section(backtest_results)
        
        if self.config.reporting.sections.cost_breakdown:
            sections['COST_BREAKDOWN'] = self._generate_cost_breakdown_section(backtest_results)
        
        if self.config.reporting.sections.drawdown_table:
            sections['DRAWDOWN_TABLE'] = self._generate_drawdown_section(backtest_results)
        
        if self.config.reporting.sections.sensitivity_fees:
            sections['SENSITIVITY_FEES'] = self._generate_sensitivity_section(pair_data)
        
        # Создаем HTML отчет
        html_report = self._create_html_report(sections, timestamp)
        
        # Сохраняем отчет
        report_path = report_dir / "detailed_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Сохраняем данные в JSON для дальнейшего анализа
        json_path = report_dir / "report_data.json"
        
        # Функция для сериализации сложных объектов
        def json_serializer(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        # Очищаем данные от несериализуемых объектов
        clean_sections = self._clean_for_json(sections)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_sections, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        # Генерируем графики если включено
        if self.config.reporting.generate_plots:
            self._generate_plots(backtest_results, report_dir)
        
        self.logger.info(f"Отчет сгенерирован: {report_path}")
        return str(report_path)
    
    def _generate_config_section(self) -> Dict[str, Any]:
        """Генерирует секцию CONFIG."""
        return {
            'title': 'Конфигурация бэктеста',
            'walk_forward': {
                'Обучающее окно': f"{self.config.walk_forward.train_window} баров",
                'Тестовое окно': f"{self.config.walk_forward.test_window} баров",
                'Скользящее окно': f"{self.config.walk_forward.rolling_window} баров",
                'Шаг сдвига': f"{self.config.walk_forward.step_size} баров"
            },
            'signals': {
                'Порог входа (Z-score)': self.config.signals.z_entry,
                'Порог выхода (Z-score)': self.config.signals.z_exit,
                'Стоп-лосс (Z-score)': self.config.signals.z_stop_loss,
                'Мин. период полураспада': f"{self.config.signals.min_half_life} дней",
                'Макс. период полураспада': f"{self.config.signals.max_half_life} дней",
                'Мин. корреляция': self.config.signals.min_correlation
            },
            'risk_management': {
                'Капитал под риском': f"${self.config.risk_management.capital_at_risk:,.0f}",
                'Макс. активных позиций': self.config.risk_management.max_active_positions,
                'Лимит плеча': f"{self.config.risk_management.leverage_limit}x",
                'Макс. доля капитала (f_max)': f"{self.config.risk_management.f_max:.1%}",
                'Множитель стоп-лосса': self.config.risk_management.stop_loss_multiplier,
                'Множитель тайм-стопа': self.config.risk_management.time_stop_multiplier
            },
            'costs': {
                'Комиссия': f"{self.config.costs.commission_pct:.3%}",
                'Проскальзывание': f"{self.config.costs.slippage_pct:.3%}",
                'Спред S1': f"{self.config.costs.bid_ask_spread_pct_s1:.4%}",
                'Спред S2': f"{self.config.costs.bid_ask_spread_pct_s2:.4%}",
                'Ставка финансирования': f"{self.config.costs.funding_rate_pct:.4%}"
            },
            'metadata': {
                'Версия конфигурации': self.config.version,
                'Random seed': self.config.random_seed,
                'Время генерации': datetime.now().isoformat()
            }
        }
    
    def _generate_metrics_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию METRICS_TEST."""
        metrics = results.get('performance_metrics', {})
        
        return {
            'title': 'Тестовые метрики и статистики',
            'returns': {
                'Общая доходность': f"{metrics.get('total_return', 0):.2%}",
                'Годовая доходность': f"{metrics.get('annual_return', 0):.2%}",
                'Волатильность': f"{metrics.get('volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.3f}",
                'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.3f}"
            },
            'risk_metrics': {
                'Максимальная просадка': f"{metrics.get('max_drawdown', 0):.2%}",
                'VaR (95%)': f"{metrics.get('var_95', 0):.2%}",
                'CVaR (95%)': f"{metrics.get('cvar_95', 0):.2%}",
                'Скос доходности': f"{metrics.get('skewness', 0):.3f}",
                'Эксцесс доходности': f"{metrics.get('kurtosis', 0):.3f}"
            },
            'trading_stats': {
                'Количество сделок': metrics.get('num_trades', 0),
                'Процент прибыльных': f"{metrics.get('win_rate', 0):.1%}",
                'Средняя прибыль': f"{metrics.get('avg_win', 0):.2%}",
                'Средний убыток': f"{metrics.get('avg_loss', 0):.2%}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
                'Средняя длительность позиции': f"{metrics.get('avg_holding_period', 0):.1f} баров"
            },
            'statistical_tests': {
                'Тест Жарка-Бера (p-value)': f"{metrics.get('jarque_bera_pvalue', 0):.4f}",
                'Тест на автокорреляцию (p-value)': f"{metrics.get('ljung_box_pvalue', 0):.4f}",
                'Тест на стационарность (p-value)': f"{metrics.get('adf_pvalue', 0):.4f}"
            }
        }
    
    def _generate_turnover_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию TURNOVER."""
        df = results.get('results_df', pd.DataFrame())
        
        if df.empty:
            return {'title': 'Анализ оборачиваемости', 'error': 'Нет данных для анализа'}
        
        # Расчет оборачиваемости
        position_changes = df['position'].diff().abs().fillna(0)
        total_turnover = position_changes.sum()
        
        # Периодизация
        trading_days = len(df) / (252 * 24 * 4)  # Предполагаем 15-минутные бары
        annual_turnover = total_turnover / trading_days if trading_days > 0 else 0
        
        # Анализ по периодам
        df['date'] = pd.to_datetime(df.index)
        df['month'] = df['date'].dt.to_period('M')
        monthly_turnover = df.groupby('month')['position'].apply(
            lambda x: x.diff().abs().sum()
        )
        
        return {
            'title': 'Анализ оборачиваемости',
            'overall': {
                'Общий оборот': f"{total_turnover:,.0f}",
                'Годовой оборот': f"{annual_turnover:,.0f}",
                'Средний дневной оборот': f"{total_turnover / len(df) * 252 * 24 * 4:,.0f}",
                'Оборот на сделку': f"{total_turnover / max(1, results.get('performance_metrics', {}).get('num_trades', 1)):,.0f}"
            },
            'monthly_stats': {
                'Средний месячный оборот': f"{monthly_turnover.mean():,.0f}",
                'Медианный месячный оборот': f"{monthly_turnover.median():,.0f}",
                'Стд. откл. месячного оборота': f"{monthly_turnover.std():,.0f}",
                'Макс. месячный оборот': f"{monthly_turnover.max():,.0f}",
                'Мин. месячный оборот': f"{monthly_turnover.min():,.0f}"
            },
            'monthly_data': monthly_turnover.to_dict()
        }
    
    def _generate_cost_breakdown_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию COST_BREAKDOWN."""
        df = results.get('results_df', pd.DataFrame())
        
        if df.empty:
            return {'title': 'Детализация издержек', 'error': 'Нет данных для анализа'}
        
        # Расчет различных типов издержек
        cost_columns = {
            'commission_costs': 'Комиссии',
            'slippage_costs': 'Проскальзывание',
            'bid_ask_costs': 'Bid-Ask спреды',
            'costs': 'Общие издержки'
        }
        
        cost_breakdown = {}
        total_costs = 0
        
        for col, name in cost_columns.items():
            if col in df.columns:
                cost_sum = df[col].sum()
                cost_breakdown[name] = {
                    'total': f"${cost_sum:,.2f}",
                    'percentage': 0,  # Будет рассчитано позже
                    'per_trade': f"${cost_sum / max(1, results.get('performance_metrics', {}).get('num_trades', 1)):,.2f}",
                    'annualized': f"${cost_sum * 252 / max(1, len(df) / (252 * 24 * 4)):,.2f}"
                }
                if col == 'costs':
                    total_costs = cost_sum
        
        # Расчет процентов
        if total_costs > 0:
            for name, data in cost_breakdown.items():
                if name != 'Общие издержки':
                    cost_val = float(data['total'].replace('$', '').replace(',', ''))
                    data['percentage'] = f"{cost_val / total_costs * 100:.1f}%"
        
        # Анализ влияния на доходность
        gross_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
        net_pnl = gross_pnl - total_costs
        cost_impact = (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        
        return {
            'title': 'Детализация издержек',
            'breakdown': cost_breakdown,
            'impact_analysis': {
                'Валовая прибыль': f"${gross_pnl:,.2f}",
                'Общие издержки': f"${total_costs:,.2f}",
                'Чистая прибыль': f"${net_pnl:,.2f}",
                'Влияние издержек': f"{cost_impact:.1f}% от валовой прибыли",
                'Издержки к капиталу': f"{total_costs / self.config.risk_management.capital_at_risk * 100:.2f}%"
            },
            'cost_efficiency': {
                'Издержки на $1 оборота': f"${total_costs / max(1, df['position'].diff().abs().sum()):,.4f}",
                'Издержки на $1 прибыли': f"${total_costs / max(1, abs(gross_pnl)):,.4f}",
                'Breakeven оборот': f"{total_costs / max(0.001, self.config.costs.commission_pct):,.0f}"
            }
        }
    
    def _generate_drawdown_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию DRAWDOWN_TABLE."""
        df = results.get('results_df', pd.DataFrame())
        
        if df.empty or 'equity' not in df.columns:
            return {'title': 'Таблица просадок', 'error': 'Нет данных для анализа'}
        
        # Расчет просадок
        equity = df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        # Поиск периодов просадок
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001 and not in_drawdown:  # Начало просадки (>0.1%)
                in_drawdown = True
                start_idx = i
            elif dd >= -0.001 and in_drawdown:  # Конец просадки
                in_drawdown = False
                if start_idx is not None:
                    period_dd = drawdown[start_idx:i+1]
                    max_dd = period_dd.min()
                    duration = i - start_idx + 1
                    
                    drawdown_periods.append({
                        'start_date': df.index[start_idx].strftime('%Y-%m-%d %H:%M'),
                        'end_date': df.index[i].strftime('%Y-%m-%d %H:%M'),
                        'duration_bars': duration,
                        'duration_days': duration / (24 * 4),  # 15-минутные бары
                        'max_drawdown': f"{max_dd:.2%}",
                        'start_equity': f"${equity.iloc[start_idx]:,.2f}",
                        'min_equity': f"${equity.iloc[start_idx:i+1].min():,.2f}",
                        'recovery_equity': f"${equity.iloc[i]:,.2f}"
                    })
        
        # Сортируем по размеру просадки
        drawdown_periods.sort(key=lambda x: float(x['max_drawdown'].replace('%', '')), reverse=True)
        
        # Статистики просадок
        if drawdown_periods:
            durations = [p['duration_days'] for p in drawdown_periods]
            max_dds = [float(p['max_drawdown'].replace('%', '')) for p in drawdown_periods]
            
            stats = {
                'Количество просадок': len(drawdown_periods),
                'Средняя просадка': f"{np.mean(max_dds):.2f}%",
                'Медианная просадка': f"{np.median(max_dds):.2f}%",
                'Средняя длительность': f"{np.mean(durations):.1f} дней",
                'Медианная длительность': f"{np.median(durations):.1f} дней",
                'Макс. длительность': f"{max(durations):.1f} дней",
                'Время в просадке': f"{sum(durations) / (len(df) / (24 * 4)) * 100:.1f}%"
            }
        else:
            stats = {'Просадки': 'Не обнаружены'}
        
        return {
            'title': 'Таблица просадок',
            'statistics': stats,
            'top_drawdowns': drawdown_periods[:10],  # Топ-10 просадок
            'all_drawdowns': drawdown_periods
        }
    
    def _generate_sensitivity_section(self, pair_data: pd.DataFrame) -> Dict[str, Any]:
        """Генерирует секцию SENSITIVITY_FEES."""
        sensitivity_results = []
        
        base_params = {
            'rolling_window': self.config.walk_forward.rolling_window,
            'z_threshold': self.config.signals.z_entry,
            'z_exit': self.config.signals.z_exit,
            'stop_loss_multiplier': self.config.risk_management.stop_loss_multiplier,
            'capital_at_risk': self.config.risk_management.capital_at_risk,
            'commission_pct': self.config.costs.commission_pct,
            'slippage_pct': self.config.costs.slippage_pct,
            'bid_ask_spread_pct_s1': self.config.costs.bid_ask_spread_pct_s1,
            'bid_ask_spread_pct_s2': self.config.costs.bid_ask_spread_pct_s2,
            'half_life': 10.0,
            'time_stop_multiplier': self.config.risk_management.time_stop_multiplier,
            'cooldown_periods': self.config.risk_management.cooldown_periods
        }
        
        # Тестируем различные множители комиссий
        for multiplier in self.config.reporting.sensitivity.fee_multipliers:
            try:
                # Создаем параметры с измененными комиссиями
                test_params = base_params.copy()
                test_params['commission_pct'] *= multiplier
                test_params['slippage_pct'] *= multiplier
                test_params['bid_ask_spread_pct_s1'] *= multiplier
                test_params['bid_ask_spread_pct_s2'] *= multiplier
                
                # Запускаем бэктест
                engine = PairBacktester(pair_data=pair_data, **test_params)
                engine.run()
                metrics = engine.get_performance_metrics()
                
                sensitivity_results.append({
                    'fee_multiplier': multiplier,
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'num_trades': metrics.get('num_trades', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'commission_pct': f"{test_params['commission_pct']:.4%}",
                    'total_costs': engine.results['costs'].sum() if 'costs' in engine.results.columns else 0
                })
                
            except Exception as e:
                self.logger.warning(f"Ошибка в анализе чувствительности для множителя {multiplier}: {e}")
                sensitivity_results.append({
                    'fee_multiplier': multiplier,
                    'error': str(e)
                })
        
        # Анализ результатов
        valid_results = [r for r in sensitivity_results if 'error' not in r]
        
        if valid_results:
            base_result = next((r for r in valid_results if r['fee_multiplier'] == 1.0), valid_results[0])
            
            analysis = {
                'Базовый Sharpe': f"{base_result['sharpe_ratio']:.3f}",
                'Базовая доходность': f"{base_result['total_return']:.2%}",
                'Эластичность Sharpe': 0,  # Будет рассчитано
                'Эластичность доходности': 0,  # Будет рассчитано
                'Критический множитель': 'N/A'  # Множитель при котором Sharpe становится отрицательным
            }
            
            # Расчет эластичности (приблизительно)
            if len(valid_results) >= 2:
                sorted_results = sorted(valid_results, key=lambda x: x['fee_multiplier'])
                
                # Эластичность между первым и последним результатом
                first, last = sorted_results[0], sorted_results[-1]
                
                if first['sharpe_ratio'] != 0 and first['fee_multiplier'] != last['fee_multiplier']:
                    sharpe_change = (last['sharpe_ratio'] - first['sharpe_ratio']) / first['sharpe_ratio']
                    fee_change = (last['fee_multiplier'] - first['fee_multiplier']) / first['fee_multiplier']
                    analysis['Эластичность Sharpe'] = f"{sharpe_change / fee_change:.2f}"
                
                if first['total_return'] != 0:
                    return_change = (last['total_return'] - first['total_return']) / first['total_return']
                    analysis['Эластичность доходности'] = f"{return_change / fee_change:.2f}"
                
                # Поиск критического множителя
                for result in sorted_results:
                    if result['sharpe_ratio'] <= 0:
                        analysis['Критический множитель'] = f"{result['fee_multiplier']:.1f}x"
                        break
        else:
            analysis = {'error': 'Не удалось провести анализ чувствительности'}
        
        return {
            'title': 'Анализ чувствительности к комиссиям',
            'results_table': sensitivity_results,
            'analysis': analysis,
            'interpretation': {
                'Низкая чувствительность': 'Эластичность < 0.5 - стратегия устойчива к издержкам',
                'Средняя чувствительность': 'Эластичность 0.5-2.0 - умеренное влияние издержек',
                'Высокая чувствительность': 'Эластичность > 2.0 - стратегия очень чувствительна к издержкам',
                'Критический уровень': 'Множитель при котором стратегия становится убыточной'
            }
        }
    
    def _create_html_report(self, sections: Dict[str, Any], timestamp: str) -> str:
        """Создает HTML отчет."""
        html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Детализированный отчет бэктеста - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ecf0f1;
            border-radius: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric-item {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}
        .metric-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value {{
            color: #27ae60;
            font-size: 1.1em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .warning {{
            color: #f39c12;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Детализированный отчет бэктеста</h1>
        <div class="timestamp">Сгенерирован: {timestamp}</div>
"""
        
        # Добавляем каждую секцию
        for section_name, section_data in sections.items():
            html += self._render_section(section_name, section_data)
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def _render_section(self, section_name: str, section_data: Dict[str, Any]) -> str:
        """Рендерит секцию отчета."""
        if 'error' in section_data:
            return f"""
        <div class="section">
            <h2>{section_data.get('title', section_name)}</h2>
            <p class="warning">Ошибка: {section_data['error']}</p>
        </div>
        """
        
        html = f'<div class="section"><h2>{section_data.get("title", section_name)}</h2>'
        
        if section_name == 'CONFIG':
            html += self._render_config_section(section_data)
        elif section_name == 'METRICS_TEST':
            html += self._render_metrics_section(section_data)
        elif section_name == 'TURNOVER':
            html += self._render_turnover_section(section_data)
        elif section_name == 'COST_BREAKDOWN':
            html += self._render_cost_breakdown_section(section_data)
        elif section_name == 'DRAWDOWN_TABLE':
            html += self._render_drawdown_section(section_data)
        elif section_name == 'SENSITIVITY_FEES':
            html += self._render_sensitivity_section(section_data)
        
        html += '</div>'
        return html
    
    def _render_config_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию конфигурации."""
        html = ""
        
        for category, items in data.items():
            if category == 'title':
                continue
            
            html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in items.items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        return html
    
    def _render_metrics_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию метрик."""
        html = ""
        
        for category, items in data.items():
            if category == 'title':
                continue
            
            html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in items.items():
                # Определяем класс для раскраски
                css_class = ""
                if 'return' in key.lower() or 'ratio' in key.lower():
                    try:
                        num_val = float(value.replace('%', '').replace(',', ''))
                        css_class = 'positive' if num_val > 0 else 'negative'
                    except:
                        pass
                
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value {css_class}">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        return html
    
    def _render_turnover_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию оборачиваемости."""
        html = ""
        
        for category, items in data.items():
            if category in ['title', 'monthly_data']:
                continue
            
            html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in items.items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        return html
    
    def _render_cost_breakdown_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию издержек."""
        html = ""
        
        # Таблица разбивки издержек
        if 'breakdown' in data:
            html += "<h3>Разбивка издержек</h3>"
            html += """
            <table>
                <tr>
                    <th>Тип издержек</th>
                    <th>Общая сумма</th>
                    <th>Процент</th>
                    <th>На сделку</th>
                    <th>Годовые</th>
                </tr>
            """
            
            for cost_type, cost_data in data['breakdown'].items():
                html += f"""
                <tr>
                    <td>{cost_type}</td>
                    <td>{cost_data['total']}</td>
                    <td>{cost_data['percentage']}</td>
                    <td>{cost_data['per_trade']}</td>
                    <td>{cost_data['annualized']}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Остальные секции как метрики
        for category, items in data.items():
            if category in ['title', 'breakdown']:
                continue
            
            html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in items.items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        return html
    
    def _render_drawdown_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию просадок."""
        html = ""
        
        # Статистики
        if 'statistics' in data:
            html += "<h3>Статистики просадок</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in data['statistics'].items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        # Таблица топ просадок
        if 'top_drawdowns' in data and data['top_drawdowns']:
            html += "<h3>Топ-10 просадок</h3>"
            html += """
            <table>
                <tr>
                    <th>Начало</th>
                    <th>Конец</th>
                    <th>Длительность (дни)</th>
                    <th>Макс. просадка</th>
                    <th>Начальный капитал</th>
                    <th>Мин. капитал</th>
                </tr>
            """
            
            for dd in data['top_drawdowns']:
                html += f"""
                <tr>
                    <td>{dd['start_date']}</td>
                    <td>{dd['end_date']}</td>
                    <td>{dd['duration_days']:.1f}</td>
                    <td class="negative">{dd['max_drawdown']}</td>
                    <td>{dd['start_equity']}</td>
                    <td>{dd['min_equity']}</td>
                </tr>
                """
            
            html += "</table>"
        
        return html
    
    def _render_sensitivity_section(self, data: Dict[str, Any]) -> str:
        """Рендерит секцию анализа чувствительности."""
        html = ""
        
        # Таблица результатов
        if 'results_table' in data:
            html += "<h3>Результаты анализа чувствительности</h3>"
            html += """
            <table>
                <tr>
                    <th>Множитель комиссий</th>
                    <th>Комиссия</th>
                    <th>Доходность</th>
                    <th>Sharpe Ratio</th>
                    <th>Макс. просадка</th>
                    <th>Количество сделок</th>
                    <th>Общие издержки</th>
                </tr>
            """
            
            for result in data['results_table']:
                if 'error' in result:
                    html += f"""
                    <tr>
                        <td>{result['fee_multiplier']}</td>
                        <td colspan="6" class="warning">Ошибка: {result['error']}</td>
                    </tr>
                    """
                else:
                    return_class = 'positive' if result['total_return'] > 0 else 'negative'
                    sharpe_class = 'positive' if result['sharpe_ratio'] > 0 else 'negative'
                    
                    html += f"""
                    <tr>
                        <td>{result['fee_multiplier']}x</td>
                        <td>{result['commission_pct']}</td>
                        <td class="{return_class}">{result['total_return']:.2%}</td>
                        <td class="{sharpe_class}">{result['sharpe_ratio']:.3f}</td>
                        <td class="negative">{result['max_drawdown']:.2%}</td>
                        <td>{result['num_trades']}</td>
                        <td>${result['total_costs']:,.2f}</td>
                    </tr>
                    """
            
            html += "</table>"
        
        # Анализ
        if 'analysis' in data:
            html += "<h3>Анализ результатов</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in data['analysis'].items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        # Интерпретация
        if 'interpretation' in data:
            html += "<h3>Интерпретация результатов</h3>"
            html += '<div class="metric-grid">'
            
            for key, value in data['interpretation'].items():
                html += f"""
                <div class="metric-item">
                    <div class="metric-label">{key}:</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += '</div>'
        
        return html
    
    def _generate_plots(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Генерирует графики для отчета."""
        df = results.get('results_df', pd.DataFrame())
        
        if df.empty:
            return
        
        # График кривой капитала
        if 'equity' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['equity'])
            plt.title('Кривая капитала')
            plt.xlabel('Время')
            plt.ylabel('Капитал ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # График просадок
        if 'equity' in df.columns:
            equity = df['equity']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
            plt.plot(df.index, drawdown, color='red')
            plt.title('Просадки')
            plt.xlabel('Время')
            plt.ylabel('Просадка (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'drawdowns.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # График позиций
        if 'position' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['position'])
            plt.title('Позиции во времени')
            plt.xlabel('Время')
            plt.ylabel('Размер позиции')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'positions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Графики сохранены в: {output_dir}")
    
    def _clean_for_json(self, obj):
        """Очищает объект от несериализуемых элементов для JSON."""
        if isinstance(obj, dict):
            return {str(k): self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return self._clean_for_json(obj.to_dict())
        elif isinstance(obj, (pd.Period, pd.Timestamp)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj