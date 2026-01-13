#!/usr/bin/env python3
"""
Out-of-sample validation component for Streamlit UI.
Validates optimized parameters on unseen data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationRunner:
    """Runs out-of-sample validation of optimized parameters."""
    
    def __init__(self, output_dir: str = "outputs/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_validation_config(
        self,
        best_params: Dict[str, Any],
        base_config_path: str = "configs/main_2024.yaml"
    ) -> str:
        """Prepare configuration with best parameters.
        
        Returns:
            Path to the prepared config file
        """
        # Load base config
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        config.setdefault('backtest', {})
        config.setdefault('portfolio', {})
        config.setdefault('pair_selection', {})
        
        # Update with best parameters
        for param_name, value in best_params.items():
            # Map parameters to config structure
            if 'zscore_threshold' in param_name:
                config['backtest']['zscore_threshold'] = value
            elif 'zscore_exit' in param_name:
                config['backtest']['zscore_exit'] = value
            elif 'rolling_window' in param_name:
                config['backtest']['rolling_window'] = int(value)
            elif 'stop_loss' in param_name:
                config['backtest']['stop_loss_multiplier'] = value
            elif 'time_stop' in param_name:
                config['backtest']['time_stop_multiplier'] = value
            elif 'position_size' in param_name:
                config['portfolio']['max_position_size_pct'] = value
            elif 'coint_pvalue' in param_name:
                config['pair_selection']['coint_pvalue_threshold'] = value
            elif 'hurst' in param_name:
                config['pair_selection']['max_hurst_exponent'] = value
            elif 'half_life' in param_name:
                if 'min' in param_name:
                    config['pair_selection']['min_half_life_days'] = value
                else:
                    config['pair_selection']['max_half_life_days'] = value
            elif 'commission' in param_name:
                config['backtest']['commission_pct'] = value
            elif 'slippage' in param_name:
                config['backtest']['slippage_pct'] = value
        
        # Save validation config
        validation_config_path = self.output_dir / f"validation_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(validation_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(validation_config_path)
    
    def run_validation(
        self,
        best_params: Dict[str, Any],
        test_start: str,
        test_end: str,
        pairs_file: str = "artifacts/universe/pairs_universe.yaml",
        base_config: str = "configs/main_2024.yaml",
        max_bars: int = 1500
    ) -> Dict[str, Any]:
        """Run validation on out-of-sample period.
        
        Returns:
            Dictionary with validation results
        """
        # Prepare config
        config_path = self.prepare_validation_config(best_params, base_config)
        
        # Build command
        cmd = [
            sys.executable,
            "scripts/trading/run_fixed.py",
            "--period-start", test_start,
            "--period-end", test_end,
            "--pairs-file", pairs_file,
            "--config", config_path,
            "--out-dir", str(self.output_dir),
            "--max-bars", str(max_bars)
        ]
        
        # Run validation
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.rstrip())
        
        process.wait()
        
        # Parse results
        results = self._parse_validation_results()
        results['output'] = output_lines
        results['config_path'] = config_path
        
        return results
    
    def _parse_validation_results(self) -> Dict[str, Any]:
        """Parse validation results from output files.
        
        Returns:
            Dictionary with parsed metrics
        """
        results = {
            'sharpe_ratio': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0,
            'avg_trade_pnl': 0.0,
            'equity_curve': [],
            'trades': []
        }
        
        # Look for results files
        results_files = list(self.output_dir.glob("*_results.yaml"))
        if results_files:
            latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
            with open(latest_results, 'r') as f:
                data = yaml.safe_load(f)
                results.update(data)
        
        # Look for equity curve
        equity_files = list(self.output_dir.glob("*_equity.csv"))
        if equity_files:
            latest_equity = max(equity_files, key=lambda x: x.stat().st_mtime)
            equity_df = pd.read_csv(latest_equity)
            results['equity_curve'] = equity_df.to_dict('records')
        
        # Look for trades
        trades_files = list(self.output_dir.glob("*_trades.csv"))
        if trades_files:
            latest_trades = max(trades_files, key=lambda x: x.stat().st_mtime)
            trades_df = pd.read_csv(latest_trades)
            results['trades'] = trades_df.to_dict('records')
            results['n_trades'] = len(trades_df)
            if len(trades_df) > 0:
                results['win_rate'] = (trades_df['pnl'] > 0).mean()
                results['avg_trade_pnl'] = trades_df['pnl'].mean()
        
        return results
    
    def compare_in_vs_out_sample(
        self,
        in_sample_metrics: Dict[str, float],
        out_sample_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Compare in-sample vs out-of-sample performance.
        
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        metrics = ['sharpe_ratio', 'total_pnl', 'win_rate', 'max_drawdown', 'n_trades']
        
        for metric in metrics:
            in_value = in_sample_metrics.get(metric, 0)
            out_value = out_sample_metrics.get(metric, 0)
            
            if in_value != 0:
                change = (out_value / in_value - 1) * 100
            else:
                change = 0
            
            comparison.append({
                'Metric': metric.replace('_', ' ').title(),
                'In-Sample': in_value,
                'Out-of-Sample': out_value,
                'Change %': change,
                'Status': '✅' if abs(change) < 50 else '⚠️'
            })
        
        return pd.DataFrame(comparison)


def render_validation_ui(best_params: Dict[str, Any] = None):
    """Render validation UI in Streamlit."""
    
    st.title("🧪 Валидация на Out-of-Sample данных")
    st.caption("Проверьте робастность оптимизированных параметров на новых данных")
    
    # Initialize runner
    runner = ValidationRunner()
    
    # Settings columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📅 Период валидации")
        
        # Period selection
        period_preset = st.radio(
            "Выберите период",
            ["Q2 2024", "Q3 2024", "Q4 2024", "Custom"],
            horizontal=True
        )
        
        if period_preset == "Custom":
            test_start = st.date_input("Начало", value=date(2024, 5, 1))
            test_end = st.date_input("Конец", value=date(2024, 5, 31))
        else:
            if period_preset == "Q2 2024":
                test_start = date(2024, 4, 1)
                test_end = date(2024, 6, 30)
            elif period_preset == "Q3 2024":
                test_start = date(2024, 7, 1)
                test_end = date(2024, 9, 30)
            else:  # Q4 2024
                test_start = date(2024, 10, 1)
                test_end = date(2024, 12, 31)
        
        st.info(f"📅 {test_start} → {test_end}")
        
        # Additional settings
        st.markdown("---")
        pairs_file = st.text_input(
            "📁 Файл с парами",
            value="artifacts/universe/pairs_universe.yaml"
        )
        
        max_bars = st.number_input(
            "📊 Max bars",
            min_value=100,
            max_value=10000,
            value=1500,
            step=100
        )
    
    with col2:
        st.subheader("🎯 Параметры для валидации")
        
        # Load or input parameters
        param_source = st.radio(
            "Источник параметров",
            ["Из оптимизации", "Ввести вручную", "Загрузить из файла"],
            horizontal=True
        )
        
        if param_source == "Из оптимизации":
            if best_params:
                st.success("✅ Параметры загружены из оптимизации")
                validation_params = best_params
            else:
                st.warning("⚠️ Нет результатов оптимизации")
                validation_params = {}
        
        elif param_source == "Загрузить из файла":
            uploaded_file = st.file_uploader(
                "Выберите файл",
                type=['yaml', 'json']
            )
            if uploaded_file:
                if uploaded_file.name.endswith('.yaml'):
                    validation_params = yaml.safe_load(uploaded_file)
                else:
                    validation_params = json.load(uploaded_file)
                st.success(f"✅ Загружено {len(validation_params)} параметров")
            else:
                validation_params = {}
        
        else:  # Manual input
            validation_params = {}
            
            with st.expander("Ввести параметры", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    validation_params['zscore_threshold'] = st.number_input(
                        "Z-score threshold",
                        value=1.5,
                        min_value=0.5,
                        max_value=3.0,
                        step=0.1
                    )
                    validation_params['rolling_window'] = st.number_input(
                        "Rolling window",
                        value=30,
                        min_value=10,
                        max_value=100,
                        step=5
                    )
                
                with col2:
                    validation_params['zscore_exit'] = st.number_input(
                        "Z-score exit",
                        value=0.0,
                        min_value=-0.5,
                        max_value=0.5,
                        step=0.05
                    )
                    validation_params['commission_pct'] = st.number_input(
                        "Commission %",
                        value=0.0004,
                        min_value=0.0002,
                        max_value=0.0010,
                        step=0.0001,
                        format="%.4f"
                    )
        
        # Show parameters
        if validation_params:
            st.markdown("---")
            st.markdown("**Параметры для валидации:**")
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in validation_params.items()
            ])
            st.dataframe(params_df, use_container_width=True, height=200)
    
    # Run validation button
    st.markdown("---")
    
    if st.button("🚀 ЗАПУСТИТЬ ВАЛИДАЦИЮ", type="primary", use_container_width=True):
        if not validation_params:
            st.error("❌ Нет параметров для валидации")
            return
        
        with st.spinner("🔄 Выполняется валидация..."):
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                # Run validation
                results = runner.run_validation(
                    best_params=validation_params,
                    test_start=str(test_start),
                    test_end=str(test_end),
                    pairs_file=pairs_file,
                    max_bars=max_bars
                )
                
                # Show results
                st.success("✅ Валидация завершена!")
                
                # Metrics
                st.markdown("---")
                st.subheader("📊 Результаты валидации")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results.get('sharpe_ratio', 0):.3f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Total PnL",
                        f"${results.get('total_pnl', 0):.2f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Win Rate",
                        f"{results.get('win_rate', 0):.1%}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{results.get('max_drawdown', 0):.1%}",
                        delta=None
                    )
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trades", results.get('n_trades', 0))
                
                with col2:
                    st.metric(
                        "Avg Trade PnL",
                        f"${results.get('avg_trade_pnl', 0):.2f}"
                    )
                
                with col3:
                    period_days = (test_end - test_start).days
                    st.metric("Period", f"{period_days} days")
                
                # Visualization tabs
                viz_tabs = st.tabs(["📈 Equity Curve", "📊 Trades", "📝 Logs"])
                
                with viz_tabs[0]:  # Equity Curve
                    if results.get('equity_curve'):
                        equity_df = pd.DataFrame(results['equity_curve'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=equity_df.index,
                            y=equity_df['equity'],
                            mode='lines',
                            name='Equity',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Equity Curve",
                            xaxis_title="Time",
                            yaxis_title="Equity ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных для отображения")
                
                with viz_tabs[1]:  # Trades
                    if results.get('trades'):
                        trades_df = pd.DataFrame(results['trades'])
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Trade distribution
                        fig = px.histogram(
                            trades_df,
                            x='pnl',
                            nbins=30,
                            title='Trade PnL Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет сделок для отображения")
                
                with viz_tabs[2]:  # Logs
                    if results.get('output'):
                        log_text = "\n".join(results['output'][-100:])
                        st.text_area(
                            "Последние логи",
                            value=log_text,
                            height=300
                        )
                
                # Export results
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export validation results
                    validation_data = {
                        'parameters': validation_params,
                        'metrics': {
                            'sharpe_ratio': results.get('sharpe_ratio', 0),
                            'total_pnl': results.get('total_pnl', 0),
                            'win_rate': results.get('win_rate', 0),
                            'max_drawdown': results.get('max_drawdown', 0),
                            'n_trades': results.get('n_trades', 0)
                        },
                        'period': {
                            'start': str(test_start),
                            'end': str(test_end)
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    yaml_str = yaml.dump(validation_data, default_flow_style=False)
                    st.download_button(
                        "📥 Скачать результаты (YAML)",
                        yaml_str,
                        file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                        mime="text/yaml"
                    )
                
                with col2:
                    json_str = json.dumps(validation_data, indent=2)
                    st.download_button(
                        "📥 Скачать результаты (JSON)",
                        json_str,
                        file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    # Test the validation UI
    test_params = {
        'zscore_threshold': 1.5,
        'zscore_exit': 0.0,
        'rolling_window': 30,
        'commission_pct': 0.0004,
        'slippage_pct': 0.0005
    }
    
    render_validation_ui(test_params)
