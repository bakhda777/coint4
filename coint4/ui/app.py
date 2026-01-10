import streamlit as st
import subprocess, sys, yaml, re
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, date
import zipfile
import io
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly не установлен. Установите: pip install plotly")

try:
    from components import (
        render_metric_card, render_status_indicator, render_progress_stage,
        create_sharpe_chart, create_pnl_chart, create_drawdown_chart,
        create_pairs_heatmap, render_run_history_table, save_run_results,
        load_run_history, render_dashboard_header, render_best_worst_pairs,
        create_performance_timeline
    )
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    # Define stub functions if components not available
    def render_metric_card(*args, **kwargs): pass
    def render_status_indicator(*args, **kwargs): pass
    def render_progress_stage(*args, **kwargs): return 0
    def create_sharpe_chart(*args, **kwargs): return None
    def create_pnl_chart(*args, **kwargs): return None
    def create_drawdown_chart(*args, **kwargs): return None
    def create_pairs_heatmap(*args, **kwargs): return None
    def render_run_history_table(*args, **kwargs): return pd.DataFrame()
    def save_run_results(*args, **kwargs): pass
    def load_run_history(*args, **kwargs): return []
    def render_dashboard_header(*args, **kwargs): pass
    def render_best_worst_pairs(*args, **kwargs): pass
    def create_performance_timeline(*args, **kwargs): return None

st.set_page_config(
    page_title="🚀 Cointegration Pairs Trading Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Профессиональная платформа для парного трейдинга на основе коинтеграции"
    }
)

def ts_folder(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def run_cmd(cmd, workdir="."):
    st.write("**Command:**", " ".join(cmd))
    st.divider()
    with subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        log_lines = []
        log_box = st.empty()
        for line in p.stdout:
            log_lines.append(line.rstrip("\n"))
            log_box.code("\n".join(log_lines[-400:]))
        ret = p.wait()
    st.success(f"Exit code: {ret}")
    return ret

def pairs_df(path: Path):
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    pairs = data.get("pairs", [])
    return pd.DataFrame(pairs) if pairs else None

# Sidebar for navigation and status
with st.sidebar:
    st.title("🎛️ Навигация")
    st.markdown("---")
    
    # System status
    st.subheader("📊 Статус системы")
    
    # Check data availability
    data_path = Path("data_downloaded")
    if data_path.exists():
        parquet_files = list(data_path.rglob("*.parquet"))
        st.success(f"✅ Данные: {len(parquet_files)} файлов")
    else:
        st.error("❌ Данные не найдены")
    
    # Check last run
    history = load_run_history()
    if history:
        last_run = history[-1]
        st.info(f"🕐 Последний запуск: {last_run['timestamp'][:16]}")
    else:
        st.warning("⏳ Запусков еще не было")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Настройки")
    theme = st.selectbox("Тема", ["Светлая", "Темная"], index=0)
    auto_refresh = st.checkbox("Автообновление", value=False)
    
    if auto_refresh:
        st.info("🔄 Автообновление каждые 30 сек")
        st.rerun()

# Main content area with tabs
tab_dashboard, tab_quick, tab_optimize, tab_sel, tab_merge, tab_bt, tab_history, tab_runs = st.tabs(
    ["📊 Dashboard", "🚀 Quick Start", "🔬 Оптимизация", "🔍 Selector", "🔀 Merge", "📈 Backtest", "📜 История", "⚙️ Runs"]
)

# Dashboard tab
with tab_dashboard:
    st.title("📊 Dashboard - Ключевые метрики")
    st.markdown("---")
    
    # Header metrics - simplified version
    col1, col2, col3, col4 = st.columns(4)
    
    # Get basic stats
    data_path = Path("data_downloaded")
    parquet_files = list(data_path.rglob("*.parquet")) if data_path.exists() else []
    history = load_run_history() if COMPONENTS_AVAILABLE else []
    
    with col1:
        st.metric("📁 Доступно данных", f"{len(parquet_files)} файлов")
    
    with col2:
        last_run_time = history[-1]['timestamp'][:16] if history else "Нет запусков"
        st.metric("🕐 Последний запуск", last_run_time)
    
    with col3:
        if history:
            success_runs = len([r for r in history if r.get('status') == 'success'])
            total_runs = len(history)
            st.metric("✅ Успешность", f"{success_runs}/{total_runs}")
        else:
            st.metric("✅ Успешность", "0/0")
    
    with col4:
        status = "ready" if parquet_files else "no_data"
        status_text = "Готов к работе" if status == "ready" else "Нет данных"
        if status == "ready":
            st.success(f"✅ {status_text}")
        else:
            st.error(f"❌ {status_text}")
    
    st.markdown("---")
    
    # Main content area
    if PLOTLY_AVAILABLE and COMPONENTS_AVAILABLE:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance timeline
            st.subheader("📈 История производительности")
            if history:
                fig = create_performance_timeline(history)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных для отображения. Запустите анализ в Quick Start.")
        
        with col2:
            # Recent runs
            st.subheader("🕐 Последние запуски")
            if history:
                try:
                    recent = pd.DataFrame(history[-5:])
                    if 'timestamp' in recent.columns:
                        recent = recent[['timestamp', 'mode', 'status']]
                        recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%m/%d %H:%M')
                        st.dataframe(recent, use_container_width=True, hide_index=True)
                    else:
                        st.info("Нет данных")
                except:
                    st.info("Нет данных")
            else:
                st.info("Нет запусков")
    else:
        # Simplified view without plotly
        st.info("📈 Для полной функциональности Dashboard установите plotly: `pip install plotly`")
        
        if history:
            st.subheader("📊 Последние запуски")
            try:
                recent_df = pd.DataFrame(history[-10:])
                st.dataframe(recent_df, use_container_width=True)
            except:
                st.write("Нет данных для отображения")
        else:
            st.info("Нет истории запусков. Запустите анализ в Quick Start.")

# -------- Quick Start --------
with tab_quick:
    # Header with description
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 Quick Start - Анализ пар за 3 клика")
        st.caption("Простой режим для быстрого старта. Выберите период и режим - остальное сделаем автоматически!")
    with col2:
        # Quick status
        if 'analysis_running' in st.session_state and st.session_state.analysis_running:
            render_status_indicator("running", "Анализ выполняется...")
        else:
            render_status_indicator("ready", "Готов к запуску")
    
    st.markdown("---")
    
    # Step 1: Period selection with Train/Test/Validation split
    st.markdown("### 📅 Шаг 1: Выберите период для анализа")
    col1, col2, col3, col4 = st.columns(4)
    
    period_choice = None
    if col1.button("📊 Q1 2024", key="qs_q1", use_container_width=True):
        period_choice = "q1_2024"
    if col2.button("📈 Q2 2024", key="qs_q2", use_container_width=True):
        period_choice = "q2_2024"
    if col3.button("📉 Q3 2024", key="qs_q3", use_container_width=True):
        period_choice = "q3_2024"
    if col4.button("🎯 Свой период", key="qs_custom", use_container_width=True):
        period_choice = "custom"
    
    # Store period choice in session state
    if period_choice:
        st.session_state.qs_period = period_choice
    
    # Define periods with proper train/test/validation split
    if st.session_state.get('qs_period') == 'custom':
        col1, col2, col3 = st.columns(3)
        train_start = col1.date_input("Train начало", value=date(2024, 1, 1))
        train_end = col2.date_input("Train конец", value=date(2024, 2, 29))
        test_days = col3.number_input("Test дней", value=30, min_value=7, max_value=90)
        
        # Calculate test and validation periods
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days)
        val_start = test_end + timedelta(days=1)
        val_end = val_start + timedelta(days=test_days)
    else:
        # Predefined quarters with automatic split
        if st.session_state.get('qs_period') == 'q1_2024':
            train_start = date(2024, 1, 1)
            train_end = date(2024, 2, 15)  # 45 days train
            test_start = date(2024, 2, 16)
            test_end = date(2024, 3, 15)   # 30 days test
            val_start = date(2024, 3, 16)
            val_end = date(2024, 3, 31)    # 15 days validation
        elif st.session_state.get('qs_period') == 'q2_2024':
            train_start = date(2024, 4, 1)
            train_end = date(2024, 5, 15)
            test_start = date(2024, 5, 16)
            test_end = date(2024, 6, 15)
            val_start = date(2024, 6, 16)
            val_end = date(2024, 6, 30)
        elif st.session_state.get('qs_period') == 'q3_2024':
            train_start = date(2024, 7, 1)
            train_end = date(2024, 8, 15)
            test_start = date(2024, 8, 16)
            test_end = date(2024, 9, 15)
            val_start = date(2024, 9, 16)
            val_end = date(2024, 9, 30)
        else:
            # Default to Q1 2024
            train_start = date(2024, 1, 1)
            train_end = date(2024, 2, 15)
            test_start = date(2024, 2, 16)
            test_end = date(2024, 3, 15)
            val_start = date(2024, 3, 16)
            val_end = date(2024, 3, 31)
    
    # Display selected periods
    if 'qs_period' in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.info(f"🏋️ **Train:** {train_start} до {train_end}")
        col2.info(f"🧪 **Test:** {test_start} до {test_end}")
        col3.info(f"✅ **Validation:** {val_start} до {val_end}")
    
    st.markdown("---")
    
    # Step 2: Mode selection
    st.markdown("### ⚙️ Шаг 2: Выберите режим анализа")
    
    mode_col1, mode_col2, mode_col3 = st.columns(3)
    
    with mode_col1:
        if st.button("🏃 **Быстрый тест**\n\n• 50 пар\n• Мягкие фильтры\n• ~2 минуты", 
                     key="qs_mode_fast", use_container_width=True):
            st.session_state.qs_mode = "fast"
    
    with mode_col2:
        if st.button("⚖️ **Стандартный**\n\n• 200 пар\n• Средние фильтры\n• ~5 минут", 
                     key="qs_mode_standard", use_container_width=True):
            st.session_state.qs_mode = "standard"
    
    with mode_col3:
        if st.button("🔬 **Глубокий**\n\n• 500 пар\n• Строгие фильтры\n• ~10 минут", 
                     key="qs_mode_deep", use_container_width=True):
            st.session_state.qs_mode = "deep"
    
    # Mode presets
    mode_presets = {
        "fast": {
            "limit_pairs": 50,
            "top_n": 10,
            "criteria": """coint_pvalue_max: 0.20
hl_min: 3
hl_max: 500
hurst_min: 0.10
hurst_max: 0.70
min_cross: 5
beta_drift_max: 0.30"""
        },
        "standard": {
            "limit_pairs": 200,
            "top_n": 30,
            "criteria": """coint_pvalue_max: 0.10
hl_min: 5
hl_max: 300
hurst_min: 0.15
hurst_max: 0.65
min_cross: 8
beta_drift_max: 0.20"""
        },
        "deep": {
            "limit_pairs": 500,
            "top_n": 50,
            "criteria": """coint_pvalue_max: 0.05
hl_min: 7
hl_max: 200
hurst_min: 0.20
hurst_max: 0.60
min_cross: 10
beta_drift_max: 0.15"""
        }
    }
    
    # Display selected mode
    if 'qs_mode' in st.session_state:
        mode = st.session_state.qs_mode
        mode_names = {"fast": "🏃 Быстрый тест", "standard": "⚖️ Стандартный", "deep": "🔬 Глубокий"}
        st.success(f"Выбран режим: **{mode_names[mode]}**")
    
    st.markdown("---")
    
    # Step 3: Run analysis
    st.markdown("### 🎯 Шаг 3: Запустите анализ")
    
    # Check if both period and mode are selected
    ready_to_run = 'qs_period' in st.session_state and 'qs_mode' in st.session_state
    
    if not ready_to_run:
        st.warning("⚠️ Выберите период и режим анализа выше")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 **ЗАПУСТИТЬ ПОЛНЫЙ АНАЛИЗ**", 
                        type="primary", use_container_width=True, key="qs_run"):
                
                # Get mode settings
                mode = st.session_state.qs_mode
                preset = mode_presets[mode]
                
                # Create timestamped output directory
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = f"artifacts/universe/quickstart_{mode}_{ts}"
                
                # Save criteria
                criteria_path = Path(out_dir) / "criteria_snapshot.yaml"
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                criteria_path.write_text(preset["criteria"])
                
                # Build selector command для train периода
                cmd_selector = [
                    sys.executable, "scripts/universe/select_pairs.py",
                    "--data-root", "./data_downloaded",
                    "--timeframe", "15T",
                    "--period-start", str(train_start),
                    "--period-end", str(train_end),
                    "--criteria-config", str(criteria_path),
                    "--limit-pairs", str(preset["limit_pairs"]),
                    "--out-dir", out_dir,
                    "--top-n", str(preset["top_n"]),
                    "--log-every", "50",
                    "--diversify-by-base", "--max-per-base", "5"
                ]
                
                # Initialize progress tracking
                st.session_state.analysis_running = True
                progress_container = st.container()
                
                with progress_container:
                    st.info("🔄 **Анализ запущен**")
                    
                    # Progress indicators
                    prog_col1, prog_col2, prog_col3 = st.columns(3)
                    with prog_col1:
                        train_progress = st.empty()
                    with prog_col2:
                        test_progress = st.empty()
                    with prog_col3:
                        val_progress = st.empty()
                    
                    train_progress.info("🔄 Этап 1/3: Train...")
                    test_progress.empty()
                    val_progress.empty()
                
                # Run selector on train data with progress
                with st.spinner(f"Анализируем данные {train_start} - {train_end}..."):
                    # Create expander for logs
                    with st.expander("📋 Логи Train этапа", expanded=False):
                        log_box = st.empty()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    proc = subprocess.Popen(cmd_selector, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1)
                    
                    log_lines = []
                    pairs_found = 0
                    for i, line in enumerate(proc.stdout):
                        log_lines.append(line.rstrip("\n"))
                        log_box.code("\n".join(log_lines[-10:]))
                        
                        # Update progress based on log content
                        if "pairs found" in line.lower():
                            try:
                                pairs_found = int(re.search(r'\d+', line).group())
                                status_text.text(f"Найдено пар: {pairs_found}")
                            except:
                                pass
                        
                        # Simulate progress
                        progress_bar.progress(min(i / 100, 0.9))
                    
                    ret = proc.wait()
                    progress_bar.progress(1.0)
                
                if ret == 0:
                    train_progress.success("✅ Этап 1/3: Train завершен")
                    
                    # Show metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Найдено пар", pairs_found if pairs_found else "N/A")
                    
                    # Check results
                    pairs_file = Path(out_dir) / "pairs_universe.yaml"
                    if pairs_file.exists():
                        df_pairs = pairs_df(pairs_file)
                        if df_pairs is not None and len(df_pairs) > 0:
                            st.metric("Найдено пар", len(df_pairs))
                            
                            # Run backtest on TEST period
                            test_progress.info("🔄 Этап 2/3: Test...")
                            
                            # Build backtest command for TEST
                            bt_test_dir = f"outputs/quickstart_{mode}_{ts}_test"
                            cmd_test = [
                                sys.executable, "scripts/trading/run_fixed.py",
                                "--period-start", str(test_start),
                                "--period-end", str(test_end),
                                "--pairs-file", str(pairs_file),
                                "--config", "configs/main_2024.yaml",
                                "--out-dir", bt_test_dir,
                                "--max-bars", "1500"
                            ]
                            
                            with st.spinner(f"Тестируем на периоде {test_start} - {test_end}..."):
                                Path(bt_test_dir).mkdir(parents=True, exist_ok=True)
                                
                                # Create expander for test logs
                                with st.expander("📋 Логи Test этапа", expanded=False):
                                    test_log_box = st.empty()
                                
                                test_progress_bar = st.progress(0)
                                
                                proc_test = subprocess.Popen(cmd_test, stdout=subprocess.PIPE,
                                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                                
                                log_lines_test = []
                                for i, line in enumerate(proc_test.stdout):
                                    log_lines_test.append(line.rstrip("\n"))
                                    test_log_box.code("\n".join(log_lines_test[-10:]))
                                    test_progress_bar.progress(min(i / 50, 0.9))
                                
                                ret_test = proc_test.wait()
                                test_progress_bar.progress(1.0)
                            
                            if ret_test == 0:
                                test_progress.success("✅ Этап 2/3: Test завершен")
                                
                                # Run VALIDATION
                                val_progress.info("🔄 Этап 3/3: Validation...")
                                
                                bt_val_dir = f"outputs/quickstart_{mode}_{ts}_validation"
                                cmd_val = [
                                    sys.executable, "scripts/trading/run_fixed.py",
                                    "--period-start", str(val_start),
                                    "--period-end", str(val_end),
                                    "--pairs-file", str(pairs_file),
                                    "--config", "configs/main_2024.yaml",
                                    "--out-dir", bt_val_dir,
                                    "--max-bars", "1500"
                                ]
                                
                                with st.spinner(f"Валидация на {val_start} - {val_end}..."):
                                    Path(bt_val_dir).mkdir(parents=True, exist_ok=True)
                                    
                                    # Create expander for validation logs
                                    with st.expander("📋 Логи Validation этапа", expanded=False):
                                        val_log_box = st.empty()
                                    
                                    val_progress_bar = st.progress(0)
                                    
                                    proc_val = subprocess.Popen(cmd_val, stdout=subprocess.PIPE,
                                                             stderr=subprocess.STDOUT, text=True, bufsize=1)
                                    
                                    log_lines_val = []
                                    for i, line in enumerate(proc_val.stdout):
                                        log_lines_val.append(line.rstrip("\n"))
                                        val_log_box.code("\n".join(log_lines_val[-10:]))
                                        val_progress_bar.progress(min(i / 50, 0.9))
                                        log_box.code("\n".join(log_lines_val[-10:]))
                                    
                                    ret_val = proc_val.wait()
                                
                                if ret_val == 0:
                                    st.balloons()
                                    st.success("🎉 **Полный цикл анализа завершен успешно!**")
                                    
                                    # Show comprehensive results
                                    st.markdown("### 📊 Сводные результаты")
                                    
                                    # Load all results
                                    test_summary = Path(bt_test_dir) / "QUICK_SUMMARY.csv"
                                    val_summary = Path(bt_val_dir) / "QUICK_SUMMARY.csv"
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    # Train results
                                    col1.markdown("**🏋️ Train**")
                                    col1.metric("Пар отобрано", len(df_pairs))
                                    col1.caption(f"{train_start} - {train_end}")
                                    
                                    # Test results
                                    col2.markdown("**🧪 Test**")
                                    if test_summary.exists():
                                        test_df = pd.read_csv(test_summary)
                                        if 'sharpe_ratio' in test_df.columns:
                                            col2.metric("Sharpe", f"{test_df['sharpe_ratio'].iloc[0]:.2f}")
                                        if 'num_trades' in test_df.columns:
                                            col2.metric("Сделок", int(test_df['num_trades'].iloc[0]))
                                    col2.caption(f"{test_start} - {test_end}")
                                    
                                    # Validation results
                                    col3.markdown("**✅ Validation**")
                                    if val_summary.exists():
                                        val_df = pd.read_csv(val_summary)
                                        if 'sharpe_ratio' in val_df.columns:
                                            col3.metric("Sharpe", f"{val_df['sharpe_ratio'].iloc[0]:.2f}")
                                        if 'num_trades' in val_df.columns:
                                            col3.metric("Сделок", int(val_df['num_trades'].iloc[0]))
                                    col3.caption(f"{val_start} - {val_end}")
                                
                                    # Show top pairs
                                    st.markdown("#### 🏆 Топ-5 найденных пар")
                                    st.dataframe(df_pairs.head(5), use_container_width=True)
                                    
                                    # Links to detailed results
                                    st.markdown("#### 📁 Детальные результаты")
                                    st.info(f"""
                                    **Отобранные пары:** `{out_dir}/`
                                    **Test результаты:** `{bt_test_dir}/`
                                    **Validation результаты:** `{bt_val_dir}/`
                                    """)
                                    
                                    # Option to go to advanced tabs
                                    st.markdown("---")
                                    st.caption("💡 Для детального анализа используйте вкладки **Selector**, **Backtest** и **Runs**")
                                else:
                                    st.error("❌ Ошибка при валидации")
                            else:
                                st.error("❌ Ошибка при тестировании")
                        else:
                            st.warning("⚠️ Не найдено подходящих пар. Попробуйте изменить период или режим.")
                else:
                    st.error("❌ Ошибка при поиске пар")
        
        with col2:
            with st.expander("ℹ️ Что будет сделано"):
                st.caption("""
                1. **Загрузка данных** за выбранный период
                2. **Анализ** всех возможных пар
                3. **Отбор** лучших по критериям
                4. **Тестирование** стратегии
                5. **Отчет** с результатами
                """)

# -------- Optimization Tab --------
with tab_optimize:
    st.title("🔬 Оптимизация параметров")
    st.caption("Найдите оптимальные параметры для live торговли с помощью Optuna")
    
    # Check if Optuna is available
    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False
        st.error("⚠️ Optuna не установлен. Установите: `pip install optuna`")
    
    if OPTUNA_AVAILABLE:
        # Create columns for layout
        col_settings, col_params = st.columns([1, 2])
        
        with col_settings:
            st.subheader("⚙️ Настройки оптимизации")
            
            # Optimization preset
            preset = st.selectbox(
                "🎯 Preset",
                ["🏃 Fast (10 trials)", "⚖️ Balanced (50 trials)", "🔬 Deep (100 trials)", "🎯 Custom"],
                help="Выберите предустановку или настройте вручную"
            )
            
            # Map preset to settings
            if "🏃 Fast" in preset:
                n_trials = 10
                sampler = "TPE"
                pruner = True
            elif "⚖️ Balanced" in preset:
                n_trials = 50
                sampler = "TPE"
                pruner = True
            elif "🔬 Deep" in preset:
                n_trials = 100
                sampler = "TPE"
                pruner = False
            else:  # Custom
                n_trials = st.number_input("🔢 Количество trials", min_value=5, max_value=1000, value=50)
                sampler = st.selectbox("🎲 Sampler", ["TPE", "Random", "Grid"])
                pruner = st.checkbox("✂️ Использовать pruner", value=True, help="Останавливать неперспективные trials")
            
            st.markdown("---")
            
            # Target metric
            target_metric = st.selectbox(
                "🎯 Целевая метрика",
                ["Sharpe Ratio", "Total PnL", "Win Rate", "Calmar Ratio"],
                help="Метрика для оптимизации"
            )
            
            # Data period
            st.markdown("---")
            st.subheader("📅 Период данных")
            
            period_preset = st.radio(
                "Выберите период",
                ["Q1 2024", "Q2 2024", "Q3 2024", "Custom"],
                horizontal=True
            )
            
            if period_preset == "Custom":
                train_start = st.date_input("Train начало", value=date(2024, 1, 1))
                train_end = st.date_input("Train конец", value=date(2024, 2, 29))
                test_days = st.number_input("Test дней", value=30, min_value=7, max_value=90)
            else:
                # Predefined periods
                if period_preset == "Q1 2024":
                    train_start, train_end = date(2024, 1, 1), date(2024, 2, 15)
                    test_days = 30
                elif period_preset == "Q2 2024":
                    train_start, train_end = date(2024, 4, 1), date(2024, 5, 15)
                    test_days = 30
                else:  # Q3 2024
                    train_start, train_end = date(2024, 7, 1), date(2024, 8, 15)
                    test_days = 30
            
            # Walk-forward settings
            st.markdown("---")
            with st.expander("🔄 Walk-forward настройки", expanded=False):
                n_splits = st.slider("Количество splits", min_value=1, max_value=10, value=3)
                purge_days = st.number_input("Purge дней", value=2, min_value=0, max_value=7)
        
        with col_params:
            st.subheader("🎯 Параметры для оптимизации")
            
            # Create tabs for different parameter groups
            param_tabs = st.tabs(["📈 Сигналы", "🛡️ Риски", "🔍 Фильтры", "💰 Издержки"])
            
            # Store parameter ranges
            param_ranges = {}
            
            with param_tabs[0]:  # Signals
                st.markdown("Параметры входа/выхода")
                
                col1, col2 = st.columns(2)
                with col1:
                    zscore_range = st.slider(
                        "Z-score threshold",
                        min_value=0.5, max_value=3.0, value=(0.8, 2.0), step=0.1,
                        help="Порог z-score для входа в позицию"
                    )
                    param_ranges['zscore_threshold'] = zscore_range
                    
                    rolling_range = st.slider(
                        "Rolling window",
                        min_value=10, max_value=100, value=(20, 50), step=5,
                        help="Окно для расчета статистик"
                    )
                    param_ranges['rolling_window'] = rolling_range
                
                with col2:
                    exit_range = st.slider(
                        "Z-score exit",
                        min_value=-0.5, max_value=0.5, value=(-0.2, 0.2), step=0.1,
                        help="Порог z-score для выхода из позиции"
                    )
                    param_ranges['zscore_exit'] = exit_range
            
            with param_tabs[1]:  # Risk
                st.markdown("Управление рисками")
                
                col1, col2 = st.columns(2)
                with col1:
                    stop_loss_range = st.slider(
                        "Stop loss multiplier",
                        min_value=2.0, max_value=5.0, value=(2.5, 3.5), step=0.5,
                        help="Стоп-лосс как множитель std"
                    )
                    param_ranges['stop_loss_multiplier'] = stop_loss_range
                    
                    position_size_range = st.slider(
                        "Max position size %",
                        min_value=0.5, max_value=5.0, value=(1.0, 3.0), step=0.5,
                        help="Максимальный размер позиции"
                    )
                    param_ranges['max_position_size_pct'] = tuple(x/100 for x in position_size_range)
                
                with col2:
                    time_stop_range = st.slider(
                        "Time stop multiplier",
                        min_value=1.0, max_value=5.0, value=(1.5, 2.5), step=0.5,
                        help="Временной стоп как множитель half-life"
                    )
                    param_ranges['time_stop_multiplier'] = time_stop_range
            
            with param_tabs[2]:  # Filters
                st.markdown("Фильтры отбора пар")
                
                col1, col2 = st.columns(2)
                with col1:
                    pvalue_range = st.slider(
                        "Cointegration p-value",
                        min_value=0.01, max_value=0.25, value=(0.05, 0.15), step=0.01,
                        help="Порог p-value для коинтеграции"
                    )
                    param_ranges['coint_pvalue_threshold'] = pvalue_range
                    
                    hurst_range = st.slider(
                        "Max Hurst exponent",
                        min_value=0.3, max_value=0.7, value=(0.4, 0.6), step=0.05,
                        help="Максимальный Hurst (<0.5 = mean-reverting)"
                    )
                    param_ranges['max_hurst_exponent'] = hurst_range
                
                with col2:
                    halflife_range = st.slider(
                        "Half-life days",
                        min_value=0.5, max_value=10.0, value=(1.0, 5.0), step=0.5,
                        help="Диапазон half-life в днях"
                    )
                    param_ranges['min_half_life_days'] = (halflife_range[0], halflife_range[0])
                    param_ranges['max_half_life_days'] = (halflife_range[1], halflife_range[1]*10)
            
            with param_tabs[3]:  # Costs
                st.markdown("Торговые издержки")
                
                col1, col2 = st.columns(2)
                with col1:
                    commission_range = st.slider(
                        "Commission %",
                        min_value=0.02, max_value=0.1, value=(0.03, 0.05), step=0.01,
                        help="Комиссия в %"
                    )
                    param_ranges['commission_pct'] = tuple(x/100 for x in commission_range)
                
                with col2:
                    slippage_range = st.slider(
                        "Slippage %",
                        min_value=0.01, max_value=0.2, value=(0.02, 0.05), step=0.01,
                        help="Проскальзывание в %"
                    )
                    param_ranges['slippage_pct'] = tuple(x/100 for x in slippage_range)
        
        # Run optimization button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            run_real_optimization = st.checkbox("🎯 Использовать реальную Optuna оптимизацию", value=False, 
                                  help="Если выключено, будет использоваться симуляция")
            
            if st.button("🚀 **ЗАПУСТИТЬ ОПТИМИЗАЦИЮ**", type="primary", use_container_width=True):
                
                if run_real_optimization:
                    # Use real optimization component
                    try:
                        from optimizer_component import run_optimization_with_ui, validate_on_out_of_sample
                        
                        # Run real optimization
                        study = run_optimization_with_ui(
                            n_trials=n_trials,
                            param_ranges=param_ranges,
                            config_path="configs/main_2024.yaml"
                        )
                        
                        if study:
                            # Get best params
                            best_params = study.best_params
                            best_value = study.best_value
                            
                            # Validation section
                            st.markdown("---")
                            st.subheader("🧪 Валидация на out-of-sample")
                            
                            # Calculate test period
                            test_start = train_end + timedelta(days=1)
                            test_end = test_start + timedelta(days=test_days)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"📅 Период валидации: {test_start} - {test_end}")
                            with col2:
                                if st.button("🧪 Запустить валидацию", use_container_width=True):
                                    validation_metrics = validate_on_out_of_sample(
                                        best_params,
                                        str(test_start),
                                        str(test_end)
                                    )
                            
                            # Export section
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                yaml_content = yaml.dump(best_params, default_flow_style=False)
                                st.download_button(
                                    label="📥 Экспорт YAML",
                                    data=yaml_content,
                                    file_name="optimized_params.yaml",
                                    mime="text/yaml",
                                    use_container_width=True
                                )
                            
                            with col2:
                                json_content = json.dumps(best_params, indent=2)
                                st.download_button(
                                    label="📥 Экспорт JSON",
                                    data=json_content,
                                    file_name="optimized_params.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            
                            with col3:
                                if st.button("🔄 Применить в Quick Start", use_container_width=True):
                                    st.session_state.optimized_params = best_params
                                    st.success("Параметры применены!")
                                    st.info("Перейдите в Quick Start для запуска")
                    
                    except ImportError as e:
                        st.error(f"Не удалось загрузить компонент оптимизации: {e}")
                        st.info("Используется симуляция")
                        run_real_optimization = False
                    except Exception as e:
                        st.error(f"Ошибка оптимизации: {e}")
                        run_real_optimization = False
                
                if not run_real_optimization:
                    # Fallback to simulation
                    st.session_state.optimization_running = True
                    
                    # Create progress container
                    progress_container = st.container()
                    
                    with progress_container:
                        st.info("🔄 Оптимизация запущена (симуляция)...")
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Metrics placeholders
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            trials_metric = st.empty()
                        with col2:
                            best_metric = st.empty()
                        with col3:
                            time_metric = st.empty()
                        with col4:
                            improvement_metric = st.empty()
                        
                        # Results container
                        results_container = st.container()
                        
                        # Simulate optimization
                        import time
                        import random
                        
                        best_value = 0
                        start_time = time.time()
                        
                        for i in range(n_trials):
                            # Update progress
                            progress = (i + 1) / n_trials
                            progress_bar.progress(progress)
                            status_text.text(f"Trial {i+1}/{n_trials}")
                            
                            # Simulate trial
                            time.sleep(0.05)  # Faster simulation
                            trial_value = random.gauss(0.5, 0.3)
                            
                            if trial_value > best_value:
                                best_value = trial_value
                            
                            # Update metrics
                            trials_metric.metric("🔄 Trials", f"{i+1}/{n_trials}")
                            best_metric.metric("🏆 Best Sharpe", f"{best_value:.3f}")
                            elapsed = time.time() - start_time
                            time_metric.metric("⏱️ Время", f"{elapsed:.1f}s")
                            
                            if i > 0:
                                improvement = (best_value / 0.5 - 1) * 100
                                improvement_metric.metric("📈 Улучшение", f"{improvement:+.1f}%")
                        
                        # Show results
                        st.session_state.optimization_running = False
                        st.balloons()
                        
                        with results_container:
                            st.success(f"✅ Оптимизация завершена! Лучший Sharpe: {best_value:.3f}")
                            
                            # Best parameters (simulated)
                            st.markdown("---")
                            st.subheader("🏆 Лучшие параметры (симуляция)")
                            
                            best_params = {
                                "zscore_threshold": random.uniform(param_ranges['zscore_threshold'][0], param_ranges['zscore_threshold'][1]),
                                "zscore_exit": random.uniform(param_ranges['zscore_exit'][0], param_ranges['zscore_exit'][1]),
                                "rolling_window": random.randint(int(param_ranges['rolling_window'][0]), int(param_ranges['rolling_window'][1])),
                                "stop_loss_multiplier": random.uniform(param_ranges['stop_loss_multiplier'][0], param_ranges['stop_loss_multiplier'][1]),
                                "commission_pct": random.uniform(param_ranges['commission_pct'][0], param_ranges['commission_pct'][1]),
                                "slippage_pct": random.uniform(param_ranges['slippage_pct'][0], param_ranges['slippage_pct'][1])
                            }
                            
                            col1, col2 = st.columns(2)
                            params_items = list(best_params.items())
                            
                            with col1:
                                for key, value in params_items[:len(params_items)//2]:
                                    st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
                            
                            with col2:
                                for key, value in params_items[len(params_items)//2:]:
                                    st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
                            
                            # Export button
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                yaml_content = yaml.dump(best_params, default_flow_style=False)
                                st.download_button(
                                    label="📥 Экспорт YAML",
                                    data=yaml_content,
                                    file_name="optimized_params.yaml",
                                    mime="text/yaml",
                                    use_container_width=True
                                )
                            
                            with col2:
                                json_content = json.dumps(best_params, indent=2)
                                st.download_button(
                                    label="📥 Экспорт JSON",
                                    data=json_content,
                                    file_name="optimized_params.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            
                            with col3:
                                if st.button("🔄 Применить в Quick Start", use_container_width=True):
                                    st.session_state.optimized_params = best_params
                                    st.success("Параметры применены!")
                                    st.info("Перейдите в Quick Start для запуска с оптимизированными параметрами")
        
        # Information section
        with st.expander("📚 Как это работает", expanded=False):
            st.markdown("""
            **Оптимизация с Optuna:**
            
            1. **Bayesian Optimization** - использует TPE (Tree-structured Parzen Estimator) для эффективного поиска
            2. **Walk-forward validation** - тестирует на нескольких периодах для устойчивости
            3. **Pruning** - останавливает неперспективные trials
            4. **Multi-objective** - может оптимизировать несколько метрик одновременно
            
            **Советы:**
            - Начните с Fast preset для быстрой проверки
            - Используйте Deep preset для финальной оптимизации
            - Проверяйте результаты на out-of-sample данных
            """)
    
    else:
        st.info("💡 Установите Optuna для использования оптимизации")
        st.code("pip install optuna plotly")

# -------- Selector --------
with tab_sel:
    st.subheader("📊 Universe Selector - Расширенный поиск пар")
    
    # Mode selection
    mode_col1, mode_col2 = st.columns([1, 3])
    with mode_col1:
        selector_mode = st.radio("Режим", ["🎯 Basic", "⚙️ Advanced"], key="sel_mode")
    
    if selector_mode == "🎯 Basic":
        st.caption("Простой режим с основными параметрами")
        
        # Basic mode - only essential fields
        col1, col2 = st.columns(2)
        p_start = col1.date_input("📅 Начало периода", 
                                  value=datetime.now().date() - timedelta(days=30),
                                  key="sel_basic_start")
        p_end = col2.date_input("📅 Конец периода", 
                                value=datetime.now().date() - timedelta(days=1),
                                key="sel_basic_end")
        
        # Strictness slider
        strictness = st.slider("🎚️ Строгость фильтров", 
                              min_value=1, max_value=5, value=3,
                              help="1 = очень мягкие (больше пар), 5 = очень строгие (меньше пар)")
        
        # Auto-calculate parameters based on strictness
        strictness_presets = {
            1: {"limit_pairs": 100, "top_n": 30, "pvalue": 0.20, "min_cross": 5},
            2: {"limit_pairs": 200, "top_n": 40, "pvalue": 0.15, "min_cross": 6},
            3: {"limit_pairs": 400, "top_n": 50, "pvalue": 0.10, "min_cross": 8},
            4: {"limit_pairs": 600, "top_n": 60, "pvalue": 0.05, "min_cross": 10},
            5: {"limit_pairs": 800, "top_n": 70, "pvalue": 0.02, "min_cross": 12}
        }
        
        preset = strictness_presets[strictness]
        
        # Hidden parameters for basic mode
        data_root = "./data_downloaded"
        timeframe = "15T"
        out_dir = f"artifacts/universe/{ts_folder('ui')}"
        limit_pairs = preset["limit_pairs"]
        top_n = preset["top_n"]
        log_every = 100
        diversify = True
        max_per = 5
        
        # Auto-generate criteria based on strictness
        crit_area = f"""coint_pvalue_max: {preset["pvalue"]}
hl_min: 5
hl_max: 300
hurst_min: 0.15
hurst_max: 0.65
min_cross: {preset["min_cross"]}
beta_drift_max: 0.20"""
        
        # Show what will be done
        with st.expander("📋 Параметры поиска"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Протестировать пар", limit_pairs)
            col2.metric("Отобрать топ", top_n)
            col3.metric("P-value порог", preset["pvalue"])
    
    else:  # Advanced mode
        st.caption("Полный контроль над всеми параметрами")
        
        col1, col2, col3 = st.columns(3)
        data_root = col1.text_input("📁 Data root", "./data_downloaded", 
                                   help="Путь к данным")
        timeframe = col2.text_input("⏱️ Timeframe", "15T",
                                   help="Временной интервал (15T = 15 минут)")
        out_dir = col3.text_input("📂 Output dir", f"artifacts/universe/{ts_folder('ui')}",
                                 help="Куда сохранить результаты")

        col1, col2, col3 = st.columns(3)
        p_start = col1.text_input("📅 Period start (YYYY-MM-DD)", "2024-01-01",
                                 help="Начало периода для анализа")
        p_end = col2.text_input("📅 Period end (YYYY-MM-DD)", "2024-01-15",
                               help="Конец периода для анализа")
        log_every = col3.number_input("📝 Log every", min_value=50, max_value=5000, value=500, step=50,
                                     help="Частота логирования прогресса")

        col1, col2, col3 = st.columns(3)
        limit_pairs = col1.number_input("🔍 Limit pairs", min_value=10, max_value=200000, value=400, step=10,
                                       help="Сколько пар протестировать")
        top_n = col2.number_input("🏆 Top N", min_value=1, max_value=5000, value=40, step=1,
                                 help="Сколько лучших пар отобрать")
        diversify = col3.checkbox("🔀 Diversify by base", value=True,
                                 help="Диверсификация по базовым активам")
        max_per = col3.number_input("📊 Max per base", min_value=1, max_value=50, value=5, step=1,
                                   help="Макс. пар на один базовый актив")

        st.write("**🎛️ Criteria YAML** (настройки фильтров)")
        crit_area = st.text_area("criteria", height=160, value="""coint_pvalue_max: 0.10
hl_min: 5
hl_max: 300
hurst_min: 0.15
hurst_max: 0.65
min_cross: 8
beta_drift_max: 0.20
""", help="YAML конфигурация критериев отбора")

    # Build command
    criteria_path = Path(out_dir) / "criteria_snapshot.yaml"
    
    # Convert dates to strings if they are date objects
    if isinstance(p_start, date):
        p_start = str(p_start)
    if isinstance(p_end, date):
        p_end = str(p_end)
    
    cmd = [
        sys.executable, "scripts/universe/select_pairs.py",
        "--data-root", data_root,
        "--timeframe", timeframe,
        "--period-start", p_start,
        "--period-end", p_end,
        "--criteria-config", str(criteria_path),
        "--limit-pairs", str(int(limit_pairs)),
        "--out-dir", out_dir,
        "--top-n", str(int(top_n)),
        "--log-every", str(int(log_every)),
    ]
    if diversify:
        cmd += ["--diversify-by-base", "--max-per-base", str(int(max_per))]

    c1, c2 = st.columns(2)
    if c1.button("Dry run (show command)"):
        st.code(" ".join(cmd))

    if c2.button("Run selector"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "_cmd.txt").write_text(" ".join(cmd))
        criteria_path.write_text(crit_area)
        ret = run_cmd(cmd)

        st.divider()
        pairs_path = Path(out_dir) / "pairs_universe.yaml"
        rej_path = Path(out_dir) / "REJECTION_BREAKDOWN.yaml"
        metrics = Path(out_dir) / "universe_metrics.csv"

        if pairs_path.exists():
            df = pairs_df(pairs_path)
            if df is not None:
                st.success(f"Loaded {len(df)} pairs")
                st.dataframe(df.head(50), use_container_width=True)
        if rej_path.exists():
            rej = yaml.safe_load(rej_path.read_text())
            st.json(rej)
        if metrics.exists():
            st.download_button("Download universe_metrics.csv",
                               metrics.read_bytes(),
                               file_name="universe_metrics.csv")

# -------- Merge --------
with tab_merge:
    st.subheader("Merge pairs")
    glob_pat = st.text_input("glob", "artifacts/universe/*/pairs_universe.yaml")
    out_file = st.text_input("out", "bench/pairs_merged.yaml")
    colA, colB = st.columns(2)
    top_k = colA.number_input("top_k (optional)", min_value=1, value=200)
    min_score = colB.text_input("min_score", "-inf")

    cmd = [
        sys.executable, "scripts/universe/merge_pairs.py",
        "--glob", glob_pat, "--out", out_file,
        "--top-k", str(int(top_k)), "--min-score", str(min_score)
    ]
    c1, c2 = st.columns(2)
    if c1.button("Dry run (show command)", key="merge_dry"):
        st.code(" ".join(cmd))
    if c2.button("Run merge", key="merge_run"):
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(Path(out_file).parent, "_cmd_merge.txt").write_text(" ".join(cmd))
        ret = run_cmd(cmd)
        outp = Path(out_file)
        if outp.exists():
            merged = yaml.safe_load(outp.read_text()).get("pairs", [])
            st.success(f"Merged {len(merged)} pairs")
            st.dataframe(pd.DataFrame(merged).head(50), use_container_width=True)

# -------- Backtest (fixed) --------
with tab_bt:
    st.subheader("Backtest (fixed)")
    col1, col2, col3 = st.columns(3)
    p_start = col1.text_input("period_start", "2024-04-01", key="bt_start")
    p_end   = col2.text_input("period_end", "2024-04-30", key="bt_end")
    out_dir_bt = col3.text_input("out_dir", f"outputs/{ts_folder('paper_ui')}", key="bt_out")
    pairs_file = st.text_input("pairs_file", "bench/pairs_merged.yaml")
    config = st.text_input("config", "configs/main_2024.yaml")
    max_bars = st.number_input("max_bars", min_value=100, max_value=50000, value=1500, step=100)

    st.write("**Config delta (YAML overlay, optional)**")
    delta = st.text_area("config_delta", height=120, value="")

    cmd = [
        sys.executable, "scripts/trading/run_fixed.py",
        "--period-start", p_start, "--period-end", p_end,
        "--pairs-file", pairs_file, "--config", config,
        "--out-dir", out_dir_bt, "--max-bars", str(int(max_bars))
    ]
    if delta.strip():
        delta_path = Path(out_dir_bt) / "config_delta_snapshot.yaml"
        cmd += ["--config-delta", str(delta_path)]

    c1, c2, c3 = st.columns(3)
    if c1.button("Dry run (show command)", key="bt_dry"):
        st.code(" ".join(cmd))
    
    # Stop button state management
    if 'bt_process' not in st.session_state:
        st.session_state.bt_process = None
    
    stop_disabled = st.session_state.bt_process is None
    if c3.button("🛑 Stop run", disabled=stop_disabled, key="bt_stop"):
        if st.session_state.bt_process:
            try:
                st.session_state.bt_process.terminate()
                st.warning("⚠️ Process terminated by user")
                st.session_state.bt_process = None
            except:
                pass

    if c2.button("Run backtest", key="bt_run"):
        Path(out_dir_bt).mkdir(parents=True, exist_ok=True)
        Path(out_dir_bt, "_cmd_backtest.txt").write_text(" ".join(cmd))
        if delta.strip():
            (Path(out_dir_bt) / "config_delta_snapshot.yaml").write_text(delta)
        if not Path(pairs_file).exists():
            st.error(f"pairs_file not found: {pairs_file}")
        elif not Path(config).exists():
            st.error(f"config not found: {config}")
        else:
            # Run with progress tracking
            progress_placeholder = st.empty()
            log_box = st.empty()
            
            proc = subprocess.Popen(cmd, cwd=".", stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
            st.session_state.bt_process = proc
            
            log_lines = []
            current_pair = 0
            total_pairs = 0
            
            for line in proc.stdout:
                log_lines.append(line.rstrip("\n"))
                log_box.code("\n".join(log_lines[-400:]))
                
                # Parse progress from log messages
                if "Processing pair" in line:
                    match = re.search(r'Processing pair (\d+)/(\d+)', line)
                    if match:
                        current_pair = int(match.group(1))
                        total_pairs = int(match.group(2))
                        progress = current_pair / total_pairs if total_pairs > 0 else 0
                        progress_placeholder.progress(progress, text=f"Processing pair {current_pair}/{total_pairs}")
                elif "pairs to process" in line:
                    match = re.search(r'(\d+) pairs to process', line)
                    if match:
                        total_pairs = int(match.group(1))
            
            ret = proc.wait()
            st.session_state.bt_process = None
            progress_placeholder.empty()
            
            if ret == 0:
                st.success(f"✅ Exit code: {ret}")
                quick = Path(out_dir_bt) / "QUICK_SUMMARY.csv"
                if quick.exists():
                    st.success("Loaded QUICK_SUMMARY.csv")
                    st.dataframe(pd.read_csv(quick), use_container_width=True)
                
                # Download all results button
                if st.button("📥 Download all results", key="bt_download"):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for file_path in Path(out_dir_bt).rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(Path(out_dir_bt).parent)
                                zf.write(file_path, arcname)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        "💾 Download results.zip",
                        data=zip_buffer,
                        file_name=f"{Path(out_dir_bt).name}.zip",
                        mime="application/zip"
                    )
            else:
                st.error(f"Process exited with code {ret}")

# -------- Runs (history browser) --------
with tab_runs:
    st.subheader("Saved runs browser")

    import glob, os, hashlib, json

    def list_dirs(base_glob, limit=50):
        dirs = [Path(p) for p in glob.glob(base_glob) if Path(p).is_dir()]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[:limit]
    
    def parse_train_from_cmd(cmd_path):
        """Extract train period and params from _cmd.txt"""
        if not cmd_path or not cmd_path.exists():
            return None, None, None
        
        cmd_text = cmd_path.read_text()
        
        # Extract period
        start_match = re.search(r'--period-start\s+(\S+)', cmd_text)
        end_match = re.search(r'--period-end\s+(\S+)', cmd_text)
        timeframe_match = re.search(r'--timeframe\s+(\S+)', cmd_text)
        
        train_start = start_match.group(1) if start_match else None
        train_end = end_match.group(1) if end_match else None
        timeframe = timeframe_match.group(1) if timeframe_match else '15T'
        
        return train_start, train_end, timeframe
    
    def suggest_next_oos_period(train_start, train_end):
        """Suggest next period of same length"""
        if not train_start or not train_end:
            return "", ""
        
        try:
            start = date.fromisoformat(train_start)
            end = date.fromisoformat(train_end)
            duration = end - start
            
            # Next period starts where train ended
            oos_start = end
            oos_end = end + duration
            
            return oos_start.isoformat(), oos_end.isoformat()
        except Exception:
            return "", ""
    
    def validate_oos_period(train_start, train_end, oos_start, oos_end):
        """Validate OOS period against train period"""
        if not all([train_start, train_end, oos_start, oos_end]):
            return "ok", ""
        
        try:
            ts = date.fromisoformat(train_start)
            te = date.fromisoformat(train_end)
            os = date.fromisoformat(oos_start)
            oe = date.fromisoformat(oos_end)
            
            # Check for overlap
            if os < te:
                if os >= ts:
                    return "warning", f"⚠️ OOS period overlaps with train period ({train_start} to {train_end})"
                else:
                    return "error", f"❌ OOS period starts before train period"
            
            # Check if gap is too large
            gap_days = (os - te).days
            if gap_days > 30:
                return "warning", f"⚠️ Large gap ({gap_days} days) between train and OOS periods"
            
            return "ok", ""
        except Exception as e:
            return "error", f"❌ Invalid date format: {e}"
    
    def create_manifest(cmd, params, universe_dir, out_dir):
        """Create manifest.json for reproducibility"""
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "command": cmd,
            "parameters": params
        }
        
        # Add git commit if available
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                manifest["git_commit"] = result.stdout.strip()
        except:
            pass
        
        # Add pairs file hash
        if params.get("pairs_file") and Path(params["pairs_file"]).exists():
            with open(params["pairs_file"], 'rb') as f:
                manifest["pairs_hash"] = hashlib.md5(f.read()).hexdigest()
        
        # Add train period info if available
        if universe_dir and Path(universe_dir).exists():
            cmd_path = Path(universe_dir) / "_cmd.txt"
            train_start, train_end, timeframe = parse_train_from_cmd(cmd_path)
            if train_start and train_end:
                manifest["train_period"] = {
                    "start": train_start,
                    "end": train_end,
                    "timeframe": timeframe
                }
        
        # Save manifest
        manifest_path = Path(out_dir) / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
    
    def parse_cmd_file(cmd_path, report_path=None):
        """Parse _cmd.txt and extract parameters, fallback to UNIVERSE_REPORT.md"""
        params = {}
        
        # Try to parse _cmd.txt first
        if cmd_path and cmd_path.exists():
            cmd_text = cmd_path.read_text()
            
            # Extract period
            start_match = re.search(r'--period-start\s+(\S+)', cmd_text)
            end_match = re.search(r'--period-end\s+(\S+)', cmd_text)
            if start_match:
                params['period_start'] = start_match.group(1)
            if end_match:
                params['period_end'] = end_match.group(1)
            
            # Extract other params
            timeframe_match = re.search(r'--timeframe\s+(\S+)', cmd_text)
            if timeframe_match:
                params['timeframe'] = timeframe_match.group(1)
            
            limit_match = re.search(r'--limit-pairs\s+(\d+)', cmd_text)
            if limit_match:
                params['limit_pairs'] = limit_match.group(1)
            
            top_match = re.search(r'--top-n\s+(\d+)', cmd_text)
            if top_match:
                params['top_n'] = top_match.group(1)
            
            # Check for diversification
            params['diversify'] = '--diversify-by-base' in cmd_text
            if params['diversify']:
                max_per_match = re.search(r'--max-per-base\s+(\d+)', cmd_text)
                if max_per_match:
                    params['max_per_base'] = max_per_match.group(1)
            
            # For backtest commands
            pairs_match = re.search(r'--pairs-file\s+(\S+)', cmd_text)
            if pairs_match:
                params['pairs_file'] = pairs_match.group(1)
            
            config_match = re.search(r'--config\s+(\S+)', cmd_text)
            if config_match:
                params['config'] = config_match.group(1)
            
            max_bars_match = re.search(r'--max-bars\s+(\d+)', cmd_text)
            if max_bars_match:
                params['max_bars'] = max_bars_match.group(1)
        
        # Fallback to UNIVERSE_REPORT.md if no cmd file
        elif report_path and report_path.exists():
            report_text = report_path.read_text()
            
            # Try to extract period from report (format: **Period**: 2024-01-01 to 2024-01-15)
            period_match = re.search(r'\*\*Period\*\*:\s*(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', report_text)
            if period_match:
                params['period_start'] = period_match.group(1)
                params['period_end'] = period_match.group(2)
            
            # Extract total pairs tested
            tested_match = re.search(r'Total pairs tested\*\*:\s*(\d+)', report_text)
            if tested_match:
                params['limit_pairs'] = tested_match.group(1)
            
            # Extract pairs selected
            selected_match = re.search(r'Pairs selected\*\*:\s*(\d+)', report_text)
            if selected_match:
                params['top_n'] = selected_match.group(1)
            
            # Default timeframe to 15T if not found (common default)
            params['timeframe'] = '15T'
        
        return params

    uni_dirs = list_dirs("artifacts/universe/*", limit=100)
    out_dirs = list_dirs("outputs/*", limit=100)

    colA, colB = st.columns(2)
    colA.write("**Universe runs** (artifacts/universe)")
    sel_uni = colA.selectbox(
        "pick universe run",
        options=["—"] + [str(d) for d in uni_dirs],
        key="runs_uni"
    )
    colB.write("**Backtest runs** (outputs)")
    sel_bt = colB.selectbox(
        "pick backtest run",
        options=["—"] + [str(d) for d in out_dirs],
        key="runs_bt"
    )

    st.divider()

    def show_universe_run(run_dir: Path):
        st.write(f"**Run:** {run_dir.name}")
        
        # Parse command file to get parameters, with fallback to report
        cmd = run_dir / "_cmd.txt"
        report = run_dir / "UNIVERSE_REPORT.md"
        params = parse_cmd_file(cmd, report)
        
        # Show key parameters
        if params:
            col1, col2, col3 = st.columns(3)
            with col1:
                period = f"{params.get('period_start', '?')} to {params.get('period_end', '?')}"
                st.info(f"📅 **Period:** {period}")
            with col2:
                st.info(f"⏱️ **Timeframe:** {params.get('timeframe', 'N/A')}")
            with col3:
                tested = params.get('limit_pairs', 'N/A')
                selected = params.get('top_n', 'N/A')
                st.info(f"📊 **Pairs:** {tested} tested / {selected} selected")
            
            if params.get('diversify'):
                st.caption(f"✅ Diversification enabled (max {params.get('max_per_base', 'N/A')} per base)")
        
        pairs = run_dir / "pairs_universe.yaml"
        rej   = run_dir / "REJECTION_BREAKDOWN.yaml"
        metrics = run_dir / "universe_metrics.csv"
        crit = run_dir / "criteria_snapshot.yaml"
        log = run_dir / "run.log"
        
        # Check if run completed successfully
        if not pairs.exists():
            st.warning("⚠️ **Incomplete run:** `pairs_universe.yaml` not found. The run may have been interrupted or failed.")
            if cmd.exists():
                st.caption("You can re-run with the same parameters using the command below:")
                with st.expander("Show command to re-run"):
                    st.code(cmd.read_text())
        
        if pairs.exists():
            df = pairs_df(pairs)
            if df is not None:
                st.success(f"{len(df)} pairs")
                st.dataframe(df.head(100), use_container_width=True)
                st.download_button("Download pairs_universe.yaml",
                                   pairs.read_bytes(), file_name=pairs.name)
        if rej.exists():
            st.caption("REJECTION_BREAKDOWN.yaml")
            st.json(yaml.safe_load(rej.read_text()))
        if metrics.exists():
            st.download_button("Download universe_metrics.csv",
                               metrics.read_bytes(), file_name=metrics.name)
        if crit.exists():
            with st.expander("Selection criteria"):
                st.code(crit.read_text())
        if cmd.exists():
            with st.expander("Full command"):
                st.code(cmd.read_text())
        if log.exists():
            with st.expander("run.log (tail)"):
                tail = "\n".join(log.read_text().splitlines()[-400:])
                st.code(tail)

    def show_backtest_run(run_dir: Path):
        st.write(f"**Run:** {run_dir.name}")
        
        # Parse command file to get parameters
        cmd = run_dir / "_cmd_backtest.txt"
        params = parse_cmd_file(cmd)
        
        # Show key parameters
        if params:
            col1, col2 = st.columns(2)
            with col1:
                period = f"{params.get('period_start', '?')} to {params.get('period_end', '?')}"
                st.info(f"📅 **Test Period:** {period}")
            with col2:
                max_bars = params.get('max_bars', 'N/A')
                st.info(f"📈 **Max bars:** {max_bars}")
            
            if params.get('pairs_file'):
                st.caption(f"📁 Pairs: {params.get('pairs_file')}")
            if params.get('config'):
                st.caption(f"⚙️ Config: {params.get('config')}")
        
        quick = run_dir / "QUICK_SUMMARY.csv"
        delta = run_dir / "config_delta_snapshot.yaml"

        if quick.exists():
            st.caption("QUICK_SUMMARY.csv")
            st.dataframe(pd.read_csv(quick), use_container_width=True)
            st.download_button("Download QUICK_SUMMARY.csv",
                               quick.read_bytes(), file_name=quick.name)
        if delta.exists():
            with st.expander("Config delta"):
                st.code(delta.read_text())
        if cmd.exists():
            with st.expander("Full command"):
                st.code(cmd.read_text())

    left, right = st.columns(2)

    if sel_uni != "—":
        with left:
            show_universe_run(Path(sel_uni))
    if sel_bt != "—":
        with right:
            show_backtest_run(Path(sel_bt))
    
    # === Quick OOS backtest ===
    st.markdown("---")
    st.subheader("Quick OOS backtest")
    
    # Show help message if no universe selected
    if not sel_uni or sel_uni == "—":
        st.info("💡 **Tip:** Select a universe run from the left dropdown to auto-fill OOS period and pairs file")
    
    # Auto-fill OOS period from selected universe
    train_start, train_end, train_timeframe = None, None, None
    default_oos_start, default_oos_end = "2024-04-01", "2024-04-30"
    
    if sel_uni and sel_uni != "—":
        cmd_path = Path(sel_uni) / "_cmd.txt"
        train_start, train_end, train_timeframe = parse_train_from_cmd(cmd_path)
        
        if train_start and train_end:
            # Show train period info
            st.info(f"📊 **Train period:** {train_start} to {train_end} (timeframe: {train_timeframe})")
            
            # Suggest next period
            suggested_start, suggested_end = suggest_next_oos_period(train_start, train_end)
            if suggested_start and suggested_end:
                default_oos_start = suggested_start
                default_oos_end = suggested_end
    
    col_a, col_b = st.columns(2)
    with col_a:
        oos_start = st.text_input("OOS period_start (YYYY-MM-DD)", value=default_oos_start, key="oos_start")
        oos_end   = st.text_input("OOS period_end (YYYY-MM-DD)", value=default_oos_end, key="oos_end")
        
        # Config presets
        config_files = sorted(glob.glob("configs/*.yaml"))
        config_files = [c for c in config_files if not c.endswith('_snapshot.yaml')]  # Exclude snapshots
        
        default_config = "configs/main_2024.yaml"
        if default_config in config_files:
            default_idx = config_files.index(default_config)
        else:
            default_idx = 0
        
        config_preset = st.selectbox("Config preset", config_files, index=default_idx, key="oos_preset")
        cfg_path = st.text_input("Config path (custom)", value=config_preset, key="oos_cfg")
    
    with col_b:
        # Using sel_uni as the selected universe directory
        uni_name = Path(sel_uni).name if sel_uni and sel_uni != "—" else "run"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = f"outputs/{uni_name}_OOS_{ts}"
        oos_out_dir = st.text_input("out_dir", value=default_out, key="oos_out")
        
        # Quick period actions
        st.caption("Quick period actions:")
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            if st.button("Next 2 weeks", key="q2w"):
                if train_end:
                    try:
                        te = date.fromisoformat(train_end)
                        st.session_state.oos_start = te.isoformat()
                        st.session_state.oos_end = (te + timedelta(days=14)).isoformat()
                        st.rerun()
                    except:
                        pass
        with qcol2:
            if st.button("Next month", key="q1m"):
                if train_end:
                    try:
                        te = date.fromisoformat(train_end)
                        st.session_state.oos_start = te.isoformat()
                        st.session_state.oos_end = (te + timedelta(days=30)).isoformat()
                        st.rerun()
                    except:
                        pass
        with qcol3:
            if st.button("Next quarter", key="q3m"):
                if train_end:
                    try:
                        te = date.fromisoformat(train_end)
                        st.session_state.oos_start = te.isoformat()
                        st.session_state.oos_end = (te + timedelta(days=90)).isoformat()
                        st.rerun()
                    except:
                        pass
        
        oos_max_bars = st.number_input("max_bars (optional)", min_value=0, value=0, step=1000, key="oos_bars")
    
    # Pairs override section
    with st.expander("Advanced: Pairs override", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            default_pairs = str(Path(sel_uni) / "pairs_universe.yaml") if sel_uni and sel_uni != "—" else ""
            custom_pairs = st.text_input("Custom pairs file (optional)", value="", 
                                        placeholder=default_pairs,
                                        key="oos_custom_pairs")
        with col2:
            use_top_k = st.checkbox("Use only top-K pairs", value=False, key="oos_topk")
            top_k = st.number_input("Top K", min_value=1, value=50, disabled=not use_top_k, key="oos_topk_val")
    
    cfg_delta = st.text_area("config delta (YAML, optional)", height=120,
                             placeholder="# e.g.\nbacktest:\n  zscore_threshold: 2.2\n  commission_pct: 0.0005",
                             key="oos_delta")
    
    # Config preview
    with st.expander("Config preview", expanded=False):
        if Path(cfg_path).exists():
            try:
                with open(cfg_path, 'r') as f:
                    st.code(f.read()[:2000], language="yaml")  # First 2000 chars
            except Exception as e:
                st.warning(f"Cannot read config: {e}")
    
    # Dry run checkbox and validation
    col1, col2 = st.columns(2)
    with col1:
        dry_run = st.checkbox("Dry run (show command only)", value=False, key="oos_dry")
    
    # Validate OOS period if we have train period
    if train_start and train_end and oos_start and oos_end:
        status, msg = validate_oos_period(train_start, train_end, oos_start, oos_end)
        if status == "warning":
            st.warning(msg)
        elif status == "error":
            st.error(msg)
    
    run_btn = st.button("Run OOS backtest" if not dry_run else "Show command", 
                        type="primary" if not dry_run else "secondary",
                        disabled=(sel_uni == "—" or not sel_uni),
                        key="oos_run")
    
    if run_btn:
        # Validate inputs
        ok = True
        error_msgs = []
        
        if not sel_uni or sel_uni == "—":
            error_msgs.append("Please select a universe run on the left")
            ok = False
        
        # Determine pairs file - custom or default
        if custom_pairs and custom_pairs.strip():
            pairs_file = custom_pairs
        else:
            if sel_uni and sel_uni != "—":
                pairs_file = str(Path(sel_uni) / "pairs_universe.yaml")
                # Check if this is an incomplete run
                if not Path(pairs_file).exists():
                    error_msgs.append(f"Selected universe run is incomplete - pairs_universe.yaml not found in {Path(sel_uni).name}")
                    error_msgs.append("Please select a completed universe run or provide a custom pairs file")
                    ok = False
            else:
                pairs_file = ""
        
        # Create output directory early for subset creation
        if ok and pairs_file:
            Path(oos_out_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle top-K subset if requested
        if use_top_k and pairs_file and Path(pairs_file).exists():
            try:
                # Load original pairs
                with open(pairs_file, 'r') as f:
                    pairs_data = yaml.safe_load(f)
                
                # Get top-K pairs
                original_pairs = pairs_data.get('pairs', [])
                subset_pairs = original_pairs[:top_k]
                
                # Save subset
                subset_path = Path(oos_out_dir) / "pairs_subset.yaml"
                subset_data = {'pairs': subset_pairs, 'original_count': len(original_pairs), 'subset_count': len(subset_pairs)}
                with open(subset_path, 'w') as f:
                    yaml.safe_dump(subset_data, f)
                
                pairs_file = str(subset_path)
                st.info(f"📊 Using top-{top_k} pairs from {len(original_pairs)} total")
            except Exception as e:
                error_msgs.append(f"Failed to create pairs subset: {e}")
                ok = False
        
        if not oos_start or not oos_end:
            error_msgs.append("Please fill in OOS period")
            ok = False
        
        if not pairs_file:
            error_msgs.append("No pairs file specified. Please select a universe run or provide a custom pairs file.")
            ok = False
        elif not Path(pairs_file).exists():
            error_msgs.append(f"Pairs file not found: {pairs_file}")
            ok = False
        
        if not Path(cfg_path).exists():
            error_msgs.append(f"Config not found: {cfg_path}")
            ok = False
        
        # Show errors if any
        if error_msgs:
            for msg in error_msgs:
                st.error(msg)
            ok = False
        
        if ok:
            # Create output directory
            Path(oos_out_dir).mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd = [
                sys.executable, "scripts/trading/run_fixed.py",
                "--period-start", oos_start,
                "--period-end", oos_end,
                "--pairs-file", pairs_file,
                "--config", cfg_path,
                "--out-dir", oos_out_dir,
            ]
            
            if oos_max_bars and int(oos_max_bars) > 0:
                cmd += ["--max-bars", str(int(oos_max_bars))]
            
            # Handle config delta
            if cfg_delta.strip():
                delta_path = str(Path(oos_out_dir) / "config_delta_snapshot.yaml")
                try:
                    # Validate YAML
                    delta_data = yaml.safe_load(cfg_delta)
                    with open(delta_path, "w") as f:
                        yaml.safe_dump(delta_data, f, sort_keys=False)
                    cmd += ["--config-delta", delta_path]
                except Exception as e:
                    st.error(f"Config delta YAML error: {e}")
                    ok = False
            
            if ok:
                # Save command
                cmd_path = Path(oos_out_dir) / "_cmd_backtest.txt"
                cmd_path.write_text(" ".join(cmd))
                
                # Show command
                st.code(" ".join(cmd), language="bash")
                
                # Create manifest for reproducibility
                params = {
                    "period_start": oos_start,
                    "period_end": oos_end,
                    "pairs_file": pairs_file,
                    "config": cfg_path,
                    "max_bars": oos_max_bars if oos_max_bars > 0 else None,
                    "config_delta": yaml.safe_load(cfg_delta) if cfg_delta.strip() else None
                }
                
                if not dry_run:
                    manifest_path = create_manifest(cmd, params, sel_uni if sel_uni != "—" else None, oos_out_dir)
                    st.caption(f"📝 Manifest saved: {manifest_path}")
                
                # If dry run, stop here
                if dry_run:
                    st.info("🔍 Dry run mode - command shown above. Click 'Run OOS backtest' with Dry run unchecked to execute.")
                else:
                    # Run with streaming output
                    st.write("**Running OOS backtest...**")
                    log_box = st.empty()
                    
                    try:
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                              stderr=subprocess.STDOUT, text=True, bufsize=1)
                        buf = []
                        for line in proc.stdout:
                            buf.append(line.rstrip("\n"))
                            log_box.code("\n".join(buf[-400:]))  # Last 400 lines
                        
                        ret = proc.wait()
                        if ret == 0:
                            st.success(f"✅ Complete! Results saved to: {oos_out_dir}")
                            
                            # Show results in columns
                            rcol1, rcol2, rcol3 = st.columns(3)
                            
                            # Try to show quick summary if exists
                            quick_summary = Path(oos_out_dir) / "QUICK_SUMMARY.csv"
                            metrics_file = Path(oos_out_dir) / "metrics.yaml"
                            
                            if quick_summary.exists():
                                df = pd.read_csv(quick_summary)
                                st.caption("📊 Quick Summary:")
                                st.dataframe(df)
                                
                                # Show key metrics
                                if 'num_trades' in df.columns:
                                    rcol1.metric("Trades", int(df['num_trades'].iloc[0]))
                                if 'sharpe_ratio' in df.columns:
                                    rcol2.metric("Sharpe", f"{df['sharpe_ratio'].iloc[0]:.3f}")
                                if 'total_pnl' in df.columns:
                                    rcol3.metric("Total PnL", f"${df['total_pnl'].iloc[0]:.2f}")
                            
                            elif metrics_file.exists():
                                # Fallback to metrics.yaml
                                with open(metrics_file, 'r') as f:
                                    metrics = yaml.safe_load(f)
                                if metrics:
                                    if 'num_trades' in metrics:
                                        rcol1.metric("Trades", metrics['num_trades'])
                                    if 'sharpe_ratio' in metrics:
                                        rcol2.metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
                                    if 'total_return' in metrics:
                                        rcol3.metric("Return", f"{metrics['total_return']:.2%}")
                            
                            # Link to output directory
                            st.caption(f"📁 [Open results folder]({oos_out_dir})")
                        else:
                            st.error(f"Process exited with code {ret}. Check logs above.")
                            
                    except FileNotFoundError:
                        st.error("Script not found: scripts/trading/run_fixed.py")
                    except Exception as e:
                        st.error(f"Execution error: {e}")