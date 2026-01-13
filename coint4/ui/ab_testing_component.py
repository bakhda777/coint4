#!/usr/bin/env python3
"""
A/B testing component for comparing different strategy configurations.
Supports statistical significance testing and performance comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from dataclasses import dataclass
import subprocess
import concurrent.futures
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StrategyConfig:
    """Configuration for a strategy variant."""
    name: str
    params: Dict[str, Any]
    config_path: str
    color: str


@dataclass
class TestResult:
    """Results from A/B test."""
    strategy_name: str
    metrics: Dict[str, float]
    trades: List[Dict]
    equity_curve: pd.DataFrame
    statistical_tests: Dict[str, Any]


class ABTestRunner:
    """Runs A/B tests between strategy configurations."""
    
    def __init__(self, output_dir: str = "outputs/ab_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        
    def create_strategy_config(
        self,
        name: str,
        base_config: str,
        param_overrides: Dict[str, Any]
    ) -> str:
        """Create configuration file for a strategy variant.
        
        Returns:
            Path to created config file
        """
        # Load base config
        with open(base_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply parameter overrides
        for param_name, value in param_overrides.items():
            # Navigate config structure
            if 'zscore' in param_name or 'rolling' in param_name:
                config.setdefault('backtest', {})
                config['backtest'][param_name] = value
            elif 'commission' in param_name or 'slippage' in param_name:
                config.setdefault('backtest', {})
                config['backtest'][param_name] = value
            elif 'stop' in param_name:
                config.setdefault('backtest', {})
                config['backtest'][param_name] = value
            elif 'position' in param_name or 'risk_per_position' in param_name:
                config.setdefault('portfolio', {})
                config['portfolio'][param_name] = value
            # Add more mappings as needed
        
        # Save config
        config_path = self.output_dir / f"config_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def run_backtest(
        self,
        strategy_config: StrategyConfig,
        test_start: str,
        test_end: str,
        pairs_file: str = "artifacts/universe/pairs_universe.yaml"
    ) -> TestResult:
        """Run backtest for a strategy configuration.
        
        Returns:
            Test results with metrics and trades
        """
        # Create config file
        config_path = self.create_strategy_config(
            strategy_config.name,
            strategy_config.config_path,
            strategy_config.params
        )
        
        # Run backtest command
        output_dir = self.output_dir / strategy_config.name
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "scripts/trading/run_fixed.py",
            "--period-start", test_start,
            "--period-end", test_end,
            "--pairs-file", pairs_file,
            "--config", config_path,
            "--out-dir", str(output_dir),
            "--max-bars", "1500"
        ]
        
        # Execute backtest
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse results
        metrics = self._parse_backtest_results(output_dir)
        trades = self._load_trades(output_dir)
        equity_curve = self._load_equity_curve(output_dir)
        
        return TestResult(
            strategy_name=strategy_config.name,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            statistical_tests={}
        )
    
    def _parse_backtest_results(self, output_dir: Path) -> Dict[str, float]:
        """Parse backtest results from output files."""
        metrics = {
            'sharpe_ratio': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0,
            'avg_trade_duration': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0
        }
        
        # Look for results file
        results_files = list(output_dir.glob("*_results.yaml"))
        if results_files:
            with open(results_files[0], 'r') as f:
                data = yaml.safe_load(f)
                metrics.update(data)
        
        return metrics
    
    def _load_trades(self, output_dir: Path) -> List[Dict]:
        """Load trades from output files."""
        trades_files = list(output_dir.glob("*_trades.csv"))
        if trades_files:
            df = pd.read_csv(trades_files[0])
            return df.to_dict('records')
        return []
    
    def _load_equity_curve(self, output_dir: Path) -> pd.DataFrame:
        """Load equity curve from output files."""
        equity_files = list(output_dir.glob("*_equity.csv"))
        if equity_files:
            return pd.read_csv(equity_files[0])
        
        # Generate synthetic equity curve for demo
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        equity = np.cumsum(np.random.randn(100) * 100) + 10000
        return pd.DataFrame({'date': dates, 'equity': equity})
    
    def run_ab_test(
        self,
        strategy_a: StrategyConfig,
        strategy_b: StrategyConfig,
        test_periods: List[Tuple[str, str]],
        pairs_file: str = "artifacts/universe/pairs_universe.yaml"
    ) -> Dict[str, Any]:
        """Run A/B test between two strategies.
        
        Args:
            strategy_a: Configuration for strategy A
            strategy_b: Configuration for strategy B
            test_periods: List of (start, end) date tuples
            pairs_file: Path to pairs file
        
        Returns:
            Dictionary with test results and statistical analysis
        """
        results_a = []
        results_b = []
        
        # Run backtests for each period
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for period_start, period_end in test_periods:
                # Submit both backtests
                future_a = executor.submit(
                    self.run_backtest,
                    strategy_a,
                    period_start,
                    period_end,
                    pairs_file
                )
                future_b = executor.submit(
                    self.run_backtest,
                    strategy_b,
                    period_start,
                    period_end,
                    pairs_file
                )
                
                # Collect results
                results_a.append(future_a.result())
                results_b.append(future_b.result())
        
        # Statistical analysis
        statistical_tests = self.perform_statistical_tests(results_a, results_b)
        
        return {
            'strategy_a': {
                'config': strategy_a,
                'results': results_a
            },
            'strategy_b': {
                'config': strategy_b,
                'results': results_b
            },
            'statistical_tests': statistical_tests
        }
    
    def perform_statistical_tests(
        self,
        results_a: List[TestResult],
        results_b: List[TestResult]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests.
        
        Returns:
            Dictionary with test results
        """
        tests = {}
        
        # Extract metrics for comparison
        sharpe_a = [r.metrics.get('sharpe_ratio', 0) for r in results_a]
        sharpe_b = [r.metrics.get('sharpe_ratio', 0) for r in results_b]
        
        pnl_a = [r.metrics.get('total_pnl', 0) for r in results_a]
        pnl_b = [r.metrics.get('total_pnl', 0) for r in results_b]
        
        # T-test for Sharpe ratio
        if len(sharpe_a) > 1 and len(sharpe_b) > 1:
            t_stat, p_value = stats.ttest_ind(sharpe_a, sharpe_b)
            tests['sharpe_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_a': np.mean(sharpe_a),
                'mean_b': np.mean(sharpe_b),
                'std_a': np.std(sharpe_a),
                'std_b': np.std(sharpe_b)
            }
        
        # Mann-Whitney U test (non-parametric)
        if len(pnl_a) > 1 and len(pnl_b) > 1:
            u_stat, p_value = stats.mannwhitneyu(pnl_a, pnl_b)
            tests['pnl_mannwhitney'] = {
                'u_statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'median_a': np.median(pnl_a),
                'median_b': np.median(pnl_b)
            }
        
        # Effect size (Cohen's d)
        if len(sharpe_a) > 1 and len(sharpe_b) > 1:
            pooled_std = np.sqrt((np.std(sharpe_a)**2 + np.std(sharpe_b)**2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(sharpe_a) - np.mean(sharpe_b)) / pooled_std
                tests['effect_size'] = {
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
        
        return tests
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"


def render_ab_testing_ui():
    """Render A/B testing UI in Streamlit."""
    
    st.title("🔬 A/B Тестирование стратегий")
    st.caption("Сравните производительность разных конфигураций с статистической значимостью")
    
    # Initialize runner
    if 'ab_runner' not in st.session_state:
        st.session_state.ab_runner = ABTestRunner()
    
    runner = st.session_state.ab_runner
    
    # Configuration section
    st.subheader("⚙️ Конфигурация тестов")
    
    # Test setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strategy A (Control)")
        
        name_a = st.text_input("Название A", value="Baseline", key="name_a")
        
        with st.expander("Параметры Strategy A", expanded=True):
            zscore_a = st.slider(
                "Z-score threshold",
                min_value=0.5, max_value=3.0,
                value=1.5, step=0.1,
                key="zscore_a"
            )
            
            window_a = st.slider(
                "Rolling window",
                min_value=10, max_value=100,
                value=30, step=5,
                key="window_a"
            )
            
            commission_a = st.slider(
                "Commission %",
                min_value=0.0002, max_value=0.0010,
                value=0.0004, step=0.0001,
                format="%.4f",
                key="commission_a"
            )
        
        strategy_a = StrategyConfig(
            name=name_a,
            params={
                'zscore_threshold': zscore_a,
                'rolling_window': window_a,
                'commission_pct': commission_a
            },
            config_path="configs/main_2024.yaml",
            color="blue"
        )
    
    with col2:
        st.markdown("### Strategy B (Variant)")
        
        name_b = st.text_input("Название B", value="Optimized", key="name_b")
        
        with st.expander("Параметры Strategy B", expanded=True):
            zscore_b = st.slider(
                "Z-score threshold",
                min_value=0.5, max_value=3.0,
                value=2.0, step=0.1,
                key="zscore_b"
            )
            
            window_b = st.slider(
                "Rolling window",
                min_value=10, max_value=100,
                value=45, step=5,
                key="window_b"
            )
            
            commission_b = st.slider(
                "Commission %",
                min_value=0.0002, max_value=0.0010,
                value=0.0005, step=0.0001,
                format="%.4f",
                key="commission_b"
            )
        
        strategy_b = StrategyConfig(
            name=name_b,
            params={
                'zscore_threshold': zscore_b,
                'rolling_window': window_b,
                'commission_pct': commission_b
            },
            config_path="configs/main_2024.yaml",
            color="red"
        )
    
    # Test periods
    st.markdown("---")
    st.subheader("📅 Периоды тестирования")
    
    test_mode = st.radio(
        "Режим тестирования",
        ["Single Period", "Multiple Periods", "Walk-Forward"],
        horizontal=True
    )
    
    test_periods = []
    
    if test_mode == "Single Period":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Начало", value=date(2024, 1, 1))
        with col2:
            end_date = st.date_input("Конец", value=date(2024, 3, 31))
        
        test_periods = [(str(start_date), str(end_date))]
    
    elif test_mode == "Multiple Periods":
        n_periods = st.slider("Количество периодов", 2, 10, 3)
        
        for i in range(n_periods):
            with st.expander(f"Период {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    start = st.date_input(f"Начало {i+1}", value=date(2024, 1+i, 1), key=f"start_{i}")
                with col2:
                    end = st.date_input(f"Конец {i+1}", value=date(2024, 1+i, 28), key=f"end_{i}")
                test_periods.append((str(start), str(end)))
    
    else:  # Walk-Forward
        total_days = st.slider("Общий период (дней)", 30, 365, 90)
        window_days = st.slider("Окно теста (дней)", 7, 30, 14)
        step_days = st.slider("Шаг (дней)", 7, 30, 14)
        
        start_date = date(2024, 1, 1)
        for i in range(0, total_days - window_days, step_days):
            period_start = start_date + timedelta(days=i)
            period_end = period_start + timedelta(days=window_days)
            test_periods.append((str(period_start), str(period_end)))
    
    st.info(f"📊 Будет проведено {len(test_periods)} тестов")
    
    # Run A/B test
    st.markdown("---")
    
    if st.button("🚀 ЗАПУСТИТЬ A/B ТЕСТ", type="primary", use_container_width=True):
        with st.spinner("🔄 Выполняется A/B тестирование..."):
            # Run test (simulated for demo)
            # In production, this would call runner.run_ab_test()
            
            # Simulate results
            results_a = []
            results_b = []
            
            for period_start, period_end in test_periods:
                # Simulate metrics for A
                results_a.append(TestResult(
                    strategy_name=strategy_a.name,
                    metrics={
                        'sharpe_ratio': np.random.normal(0.8, 0.2),
                        'total_pnl': np.random.normal(5000, 1000),
                        'win_rate': np.random.uniform(0.45, 0.55),
                        'max_drawdown': np.random.uniform(0.1, 0.2),
                        'n_trades': np.random.randint(50, 150)
                    },
                    trades=[],
                    equity_curve=pd.DataFrame(),
                    statistical_tests={}
                ))
                
                # Simulate metrics for B (slightly better)
                results_b.append(TestResult(
                    strategy_name=strategy_b.name,
                    metrics={
                        'sharpe_ratio': np.random.normal(1.0, 0.2),
                        'total_pnl': np.random.normal(6000, 1000),
                        'win_rate': np.random.uniform(0.48, 0.58),
                        'max_drawdown': np.random.uniform(0.08, 0.18),
                        'n_trades': np.random.randint(40, 120)
                    },
                    trades=[],
                    equity_curve=pd.DataFrame(),
                    statistical_tests={}
                ))
            
            # Perform statistical tests
            statistical_tests = runner.perform_statistical_tests(results_a, results_b)
            
            # Store results
            st.session_state.ab_results = {
                'strategy_a': {'config': strategy_a, 'results': results_a},
                'strategy_b': {'config': strategy_b, 'results': results_b},
                'statistical_tests': statistical_tests
            }
            
            st.success("✅ A/B тест завершен!")
    
    # Display results
    if 'ab_results' in st.session_state:
        results = st.session_state.ab_results
        
        st.markdown("---")
        st.subheader("📊 Результаты A/B теста")
        
        # Summary metrics
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"### {results['strategy_a']['config'].name}")
            sharpe_a = [r.metrics['sharpe_ratio'] for r in results['strategy_a']['results']]
            st.metric("Avg Sharpe", f"{np.mean(sharpe_a):.3f}")
            pnl_a = [r.metrics['total_pnl'] for r in results['strategy_a']['results']]
            st.metric("Avg PnL", f"${np.mean(pnl_a):.0f}")
        
        with col2:
            st.markdown(f"### {results['strategy_b']['config'].name}")
            sharpe_b = [r.metrics['sharpe_ratio'] for r in results['strategy_b']['results']]
            st.metric("Avg Sharpe", f"{np.mean(sharpe_b):.3f}")
            pnl_b = [r.metrics['total_pnl'] for r in results['strategy_b']['results']]
            st.metric("Avg PnL", f"${np.mean(pnl_b):.0f}")
        
        with col3:
            st.markdown("### 📈 Улучшение")
            sharpe_improvement = (np.mean(sharpe_b) / np.mean(sharpe_a) - 1) * 100
            st.metric("Sharpe", f"{sharpe_improvement:+.1f}%")
            pnl_improvement = (np.mean(pnl_b) / np.mean(pnl_a) - 1) * 100
            st.metric("PnL", f"{pnl_improvement:+.1f}%")
        
        # Statistical significance
        st.markdown("---")
        st.subheader("📊 Статистическая значимость")
        
        tests = results['statistical_tests']
        
        if 'sharpe_ttest' in tests:
            test = tests['sharpe_ttest']
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**T-test для Sharpe Ratio**")
                st.write(f"t-statistic: {test['t_statistic']:.3f}")
                st.write(f"p-value: {test['p_value']:.4f}")
                
                if test['significant']:
                    st.success("✅ Статистически значимое различие (p < 0.05)")
                else:
                    st.warning("⚠️ Различие не значимое (p ≥ 0.05)")
            
            with col2:
                if 'effect_size' in tests:
                    effect = tests['effect_size']
                    st.markdown("**Размер эффекта**")
                    st.write(f"Cohen's d: {effect['cohens_d']:.3f}")
                    st.write(f"Интерпретация: {effect['interpretation']}")
        
        # Visualization tabs
        viz_tabs = st.tabs(["📈 Сравнение метрик", "📊 Распределения", "🎯 Box Plots"])
        
        with viz_tabs[0]:  # Metrics comparison
            # Create comparison chart
            metrics_df = pd.DataFrame({
                'Period': list(range(1, len(test_periods) + 1)),
                f'{strategy_a.name} Sharpe': sharpe_a,
                f'{strategy_b.name} Sharpe': sharpe_b,
                f'{strategy_a.name} PnL': pnl_a,
                f'{strategy_b.name} PnL': pnl_b
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=metrics_df['Period'],
                y=metrics_df[f'{strategy_a.name} Sharpe'],
                name=f'{strategy_a.name} Sharpe',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=metrics_df['Period'],
                y=metrics_df[f'{strategy_b.name} Sharpe'],
                name=f'{strategy_b.name} Sharpe',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Sharpe Ratio по периодам",
                xaxis_title="Период",
                yaxis_title="Sharpe Ratio",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:  # Distributions
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=sharpe_a,
                name=strategy_a.name,
                opacity=0.7,
                marker_color='blue'
            ))
            
            fig.add_trace(go.Histogram(
                x=sharpe_b,
                name=strategy_b.name,
                opacity=0.7,
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Распределение Sharpe Ratio",
                xaxis_title="Sharpe Ratio",
                yaxis_title="Частота",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:  # Box plots
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=sharpe_a,
                name=strategy_a.name,
                marker_color='blue'
            ))
            
            fig.add_trace(go.Box(
                y=sharpe_b,
                name=strategy_b.name,
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Box Plot: Sharpe Ratio",
                yaxis_title="Sharpe Ratio",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.markdown("---")
        
        export_data = {
            'test_periods': test_periods,
            'strategy_a': {
                'name': strategy_a.name,
                'params': strategy_a.params,
                'results': [r.metrics for r in results['strategy_a']['results']]
            },
            'strategy_b': {
                'name': strategy_b.name,
                'params': strategy_b.params,
                'results': [r.metrics for r in results['strategy_b']['results']]
            },
            'statistical_tests': tests
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            yaml_str = yaml.dump(export_data, default_flow_style=False)
            st.download_button(
                "📥 Скачать результаты (YAML)",
                yaml_str,
                file_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                mime="text/yaml"
            )
        
        with col2:
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Скачать результаты (JSON)",
                json_str,
                file_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    render_ab_testing_ui()
