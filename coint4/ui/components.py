"""
UI components for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


def render_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       delta_color: str = "normal", icon: str = "📊"):
    """Render a metric card with optional delta."""
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<h1 style='text-align: center'>{icon}</h1>", unsafe_allow_html=True)
        with col2:
            st.metric(title, value, delta, delta_color=delta_color)


def render_status_indicator(status: str, message: str = ""):
    """Render a status indicator with color coding."""
    colors = {
        "ready": "#28a745",
        "running": "#ffc107", 
        "error": "#dc3545",
        "success": "#17a2b8"
    }
    
    icons = {
        "ready": "✅",
        "running": "🔄",
        "error": "❌",
        "success": "🎉"
    }
    
    color = colors.get(status, "#6c757d")
    icon = icons.get(status, "ℹ️")
    
    st.markdown(
        f"""
        <div style='padding: 10px; background-color: {color}20; border-left: 4px solid {color}; border-radius: 5px;'>
            <span style='font-size: 20px'>{icon}</span> <strong>{status.upper()}</strong>
            {f': {message}' if message else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_progress_stage(stage_name: str, current: int, total: int, status: str = "running"):
    """Render a progress indicator for a stage."""
    progress = current / total if total > 0 else 0
    
    status_colors = {
        "pending": "#6c757d",
        "running": "#007bff",
        "success": "#28a745",
        "error": "#dc3545"
    }
    
    color = status_colors.get(status, "#6c757d")
    
    col1, col2, col3 = st.columns([3, 4, 1])
    
    with col1:
        st.markdown(f"**{stage_name}**")
    
    with col2:
        st.progress(progress)
    
    with col3:
        if status == "success":
            st.success("✓")
        elif status == "error":
            st.error("✗")
        elif status == "running":
            st.info("⟳")
        else:
            st.empty()
    
    return progress


def create_sharpe_chart(results: pd.DataFrame) -> go.Figure:
    """Create Sharpe Ratio chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=results.index,
        y=results['sharpe_ratio'],
        marker_color=['green' if x > 0 else 'red' for x in results['sharpe_ratio']],
        text=results['sharpe_ratio'].round(2),
        textposition='outside',
        name='Sharpe Ratio'
    ))
    
    fig.update_layout(
        title="Sharpe Ratio по парам",
        xaxis_title="Пары",
        yaxis_title="Sharpe Ratio",
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig


def create_pnl_chart(pnl_data: List[float], timestamps: Optional[List] = None) -> go.Figure:
    """Create cumulative PnL chart."""
    if timestamps is None:
        timestamps = list(range(len(pnl_data)))
    
    cumulative_pnl = pd.Series(pnl_data).cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cumulative_pnl,
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.1)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Color areas
    fig.add_hrect(y0=0, y1=cumulative_pnl.max() if cumulative_pnl.max() > 0 else 1,
                  fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hrect(y0=cumulative_pnl.min() if cumulative_pnl.min() < 0 else -1, y1=0,
                  fillcolor="red", opacity=0.05, line_width=0)
    
    fig.update_layout(
        title="Cumulative PnL",
        xaxis_title="Время",
        yaxis_title="PnL ($)",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_drawdown_chart(equity_curve: List[float]) -> go.Figure:
    """Create drawdown chart."""
    equity = pd.Series(equity_curve)
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(drawdown))),
        y=drawdown,
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Время",
        yaxis_title="Drawdown (%)",
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def create_pairs_heatmap(pairs_data: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for pairs."""
    # Sample correlation matrix (in real app, calculate from pairs_data)
    corr_matrix = pairs_data.select_dtypes(include=['float64', 'int64']).corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Корреляционная матрица параметров",
        height=500,
        xaxis=dict(side="bottom"),
        yaxis=dict(side="left")
    )
    
    return fig


def render_run_history_table(history: List[Dict]) -> pd.DataFrame:
    """Render run history as an interactive table."""
    if not history:
        st.info("История запусков пуста")
        return pd.DataFrame()
    
    df = pd.DataFrame(history)
    
    # Format columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'sharpe_ratio' in df.columns:
        df['sharpe_ratio'] = df['sharpe_ratio'].round(3)
    
    if 'total_pnl' in df.columns:
        df['total_pnl'] = df['total_pnl'].round(2)
    
    # Add status column with emojis
    if 'status' in df.columns:
        df['status'] = df['status'].map({
            'success': '✅ Success',
            'error': '❌ Error',
            'running': '🔄 Running',
            'pending': '⏳ Pending'
        })
    
    return df


def save_run_results(run_id: str, results: Dict):
    """Save run results to history."""
    history_file = Path("artifacts/run_history.json")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing history
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new run
    results['run_id'] = run_id
    results['timestamp'] = datetime.now().isoformat()
    history.append(results)
    
    # Keep only last 100 runs
    history = history[-100:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def load_run_history() -> List[Dict]:
    """Load run history from file."""
    history_file = Path("artifacts/run_history.json")
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    
    return []


def render_dashboard_header():
    """Render dashboard header with system status."""
    col1, col2, col3, col4 = st.columns(4)
    
    # Get system status (mock data for now)
    with col1:
        render_metric_card("📊 Доступно данных", "43 файлов", "↑ 2 новых", "normal")
    
    with col2:
        history = load_run_history()
        last_run = history[-1] if history else None
        last_run_time = last_run['timestamp'][:16] if last_run else "Нет запусков"
        render_metric_card("🕐 Последний запуск", last_run_time)
    
    with col3:
        success_runs = len([r for r in history if r.get('status') == 'success'])
        total_runs = len(history)
        success_rate = f"{success_runs}/{total_runs}" if total_runs > 0 else "0/0"
        render_metric_card("✅ Успешность", success_rate, 
                          f"{success_runs/total_runs*100:.0f}%" if total_runs > 0 else None)
    
    with col4:
        # Check if any process is running (mock)
        status = "ready"  # Could be: ready, running, error
        render_status_indicator(status, "Система готова к работе")


def render_best_worst_pairs(pairs_df: pd.DataFrame, metric: str = 'sharpe_ratio'):
    """Render best and worst performing pairs."""
    if pairs_df.empty or metric not in pairs_df.columns:
        st.info("Нет данных для отображения")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Топ-5 лучших пар")
        best = pairs_df.nlargest(5, metric)[['pair', metric, 'total_pnl']]
        st.dataframe(best, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📉 Топ-5 худших пар")
        worst = pairs_df.nsmallest(5, metric)[['pair', metric, 'total_pnl']]
        st.dataframe(worst, use_container_width=True, hide_index=True)


def create_performance_timeline(history: List[Dict]) -> go.Figure:
    """Create performance timeline chart."""
    if not history:
        return go.Figure()
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Add Sharpe ratio line
    if 'sharpe_ratio' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sharpe_ratio'],
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
    
    # Add PnL bars
    if 'total_pnl' in df.columns:
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['total_pnl'],
            name='Total PnL',
            marker_color=['green' if x > 0 else 'red' for x in df['total_pnl']],
            yaxis='y2',
            opacity=0.5
        ))
    
    fig.update_layout(
        title="История производительности",
        xaxis_title="Дата",
        yaxis=dict(title="Sharpe Ratio", side="left"),
        yaxis2=dict(title="PnL ($)", overlaying="y", side="right"),
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig