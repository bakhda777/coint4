document.addEventListener('DOMContentLoaded', () => {
    const app = new TaskRunnerApp();
    app.init();
    // Expose app instance for inline onclick handlers
    document.querySelector('.container')._app = app;
});

class TaskRunnerApp {
    constructor() {
        this.runBtn = document.getElementById('runBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.logsEl = document.getElementById('logs');
        this.statusBadge = document.getElementById('statusBadge');
        this.historyList = document.getElementById('historyList');
        this.summaryBox = document.getElementById('resultSummary');
        this.modeRadios = document.querySelectorAll('input[name="mode"]');
        this.refreshHistBtn = document.getElementById('refreshHistory');

        this.baseConfigSelect = document.getElementById('base_config');
        this.searchSpaceSelect = document.getElementById('search_space');

        this.previewConfigBtn = document.getElementById('previewConfigBtn');
        this.configPreviewModal = document.getElementById('configPreviewModal');
        this.closePreviewBtn = document.getElementById('closePreviewBtn');
        this.saveConfigBtn = document.getElementById('saveConfigBtn');
        this.previewContent = document.getElementById('previewContent');
        this.previewTitle = document.getElementById('previewTitle');
        this.saveStatus = document.getElementById('saveStatus');

        // Analysis Modal Elements
        this.analysisModal = document.getElementById('analysisModal');
        this.closeAnalysisBtn = document.getElementById('closeAnalysisBtn');
        this.analysisTitle = document.getElementById('analysisTitle');
        this.analysisSummary = document.getElementById('analysisSummary');
        this.analysisParams = document.getElementById('analysisParams');
        this.pairsStats = document.getElementById('pairsStats');
        this.topPairsBody = document.getElementById('topPairsBody');
        this.bottomPairsBody = document.getElementById('bottomPairsBody');
        this.equityChartContainer = document.getElementById('equityChartContainer');
        this.tradesTableBody = document.querySelector('#tradesTable tbody');

        this.currentTaskId = null;
        this.eventSource = null;
        this.pollInterval = null;

        // Pagination state
        this.currentPage = 1;
        this.itemsPerPage = 10;
        this.totalItems = 0;
    }

    init() {
        this.setupEventListeners();
        this.updateFormVisibility();
        this.loadOptions();
        this.loadHistory();

        setInterval(() => this.loadHistory(true), 5000); // Pass true to indicate background refresh
    }

    setupEventListeners() {
        this.runBtn.addEventListener('click', () => this.runTask());
        this.stopBtn.addEventListener('click', () => this.stopTask());

        this.modeRadios.forEach(radio => {
            radio.addEventListener('change', () => this.updateFormVisibility());
        });

        this.refreshHistBtn.addEventListener('click', () => this.loadHistory());

        document.getElementById('prevPage').addEventListener('click', () => this.changePage(-1));
        document.getElementById('nextPage').addEventListener('click', () => this.changePage(1));

        this.previewConfigBtn.addEventListener('click', () => this.showConfigPreview());
        this.closePreviewBtn.addEventListener('click', () => this.closeConfigPreview());
        this.saveConfigBtn.addEventListener('click', () => this.saveConfig());

        // Close modal on outside click
        this.configPreviewModal.addEventListener('click', (e) => {
            if (e.target === this.configPreviewModal) {
                this.closeConfigPreview();
            }
        });

        this.closeAnalysisBtn.addEventListener('click', () => this.closeAnalysis());
        this.analysisModal.addEventListener('click', (e) => {
            if (e.target === this.analysisModal) {
                this.closeAnalysis();
            }
        });
    }

    closeAnalysis() {
        this.analysisModal.classList.add('hidden');
    }

    async showAnalysis(taskId) {
        this.analysisModal.classList.remove('hidden');
        this.analysisSummary.innerHTML = 'Loading...';
        this.analysisParams.innerHTML = 'Loading...';
        this.pairsStats.innerHTML = 'Loading...';
        this.topPairsBody.innerHTML = '';
        this.bottomPairsBody.innerHTML = '';
        this.equityChartContainer.innerHTML = 'Loading chart...';
        this.tradesTableBody.innerHTML = '';

        try {
            // Get task details
            const resp = await fetch(`/api/status?task_id=${taskId}`);
            const data = await resp.json();
            const task = data.task;

            if (!task || !task.result) {
                this.analysisSummary.innerHTML = 'No results available for this task.';
                return;
            }

            const res = task.result;

            // 1. Summary
            this.analysisSummary.innerHTML = `
                <strong>Total PnL:</strong> $${res.total_pnl?.toFixed(2) || '0.00'}<br>
                <strong>Total Trades:</strong> ${res.total_trades || 0}<br>
                <strong>Sharpe Ratio:</strong> ${res.sharpe_ratio_abs?.toFixed(4) || '0.0000'}<br>
                <strong>Status:</strong> ${task.exit_code === 0 ? '<span style="color:green">Success</span>' : '<span style="color:red">Failed</span>'}
            `;

            // 2. Parameters
            if (res.config) {
                // Filter and format config for display
                const relevantConfig = {};
                if (res.config.backtest) relevantConfig.backtest = res.config.backtest;
                if (res.config.portfolio) relevantConfig.portfolio = res.config.portfolio;
                if (res.config.pair_selection) relevantConfig.pair_selection = res.config.pair_selection;

                this.analysisParams.innerHTML = JSON.stringify(relevantConfig, null, 2);
            } else {
                this.analysisParams.innerHTML = `
                    <em>Parameters not stored in task history.</em><br>
                    Check log file.
                `;
            }

            // 3. Pair Statistics
            if (res.trade_stat && Array.isArray(res.trade_stat)) {
                const stats = res.trade_stat;

                // Aggregate by pair across periods
                const pairAgg = {};
                stats.forEach(s => {
                    if (!pairAgg[s.pair]) {
                        pairAgg[s.pair] = {
                            pair: s.pair,
                            pnl: 0,
                            trades: 0,
                            wins: 0,
                            losses: 0
                        };
                    }
                    pairAgg[s.pair].pnl += s.total_pnl;
                    pairAgg[s.pair].trades += s.trade_count;
                    pairAgg[s.pair].wins += s.win_days || 0; // Approximate wins
                    pairAgg[s.pair].losses += s.lose_days || 0;
                });

                const aggregatedPairs = Object.values(pairAgg);
                const tradedPairs = aggregatedPairs.filter(p => p.trades > 0);
                const profitablePairs = tradedPairs.filter(p => p.pnl > 0);

                this.pairsStats.innerHTML = `
                    <div><strong>Total Pairs Scanned:</strong> ${aggregatedPairs.length}</div>
                    <div><strong>Active Pairs:</strong> ${tradedPairs.length}</div>
                    <div><strong>Profitable Pairs:</strong> ${profitablePairs.length} (${tradedPairs.length > 0 ? (profitablePairs.length / tradedPairs.length * 100).toFixed(1) : 0}%)</div>
                    <div><strong>Avg PnL per Active Pair:</strong> $${tradedPairs.length > 0 ? (res.total_pnl / tradedPairs.length).toFixed(2) : '0.00'}</div>
                `;

                // Sort for Top/Bottom
                aggregatedPairs.sort((a, b) => b.pnl - a.pnl);

                const top5 = aggregatedPairs.slice(0, 5);
                const bottom5 = aggregatedPairs.slice(-5).reverse();

                const renderRow = (p) => `
                    <tr>
                        <td>${p.pair}</td>
                        <td style="color:${p.pnl > 0 ? 'green' : 'red'}">${p.pnl.toFixed(2)}</td>
                        <td>${p.trades}</td>
                    </tr>
                `;

                this.topPairsBody.innerHTML = top5.map(renderRow).join('');
                this.bottomPairsBody.innerHTML = bottom5.map(renderRow).join('');

            } else {
                this.pairsStats.innerHTML = 'No pair statistics available.';
            }

            // 4. Equity Curve
            if (res.pnl_series) {
                this.renderEquityChart(res.pnl_series);
            } else {
                this.equityChartContainer.innerHTML = 'No PnL data available for chart.';
            }

            // 5. Trades
            if (res.trades && Array.isArray(res.trades)) {
                const tradesHtml = res.trades.map(t => {
                    if (typeof t === 'string') {
                        return `<tr><td>${t}</td><td colspan="4">Detail unavailable</td></tr>`;
                    } else {
                        return `
                            <tr>
                                <td>${t.entry_time || '-'}</td>
                                <td>${t.pair || '-'}</td>
                                <td>${t.type || '-'}</td>
                                <td style="color:${t.pnl > 0 ? 'green' : 'red'}">${t.pnl?.toFixed(2) || '0.00'}</td>
                                <td>${t.duration || '-'}</td>
                            </tr>
                        `;
                    }
                }).join('');
                this.tradesTableBody.innerHTML = tradesHtml;
            } else {
                this.tradesTableBody.innerHTML = '<tr><td colspan="5">No trades recorded.</td></tr>';
            }

        } catch (e) {
            this.analysisSummary.innerHTML = `Error loading analysis: ${e.message}`;
            console.error(e);
        }
    }

    renderEquityChart(pnlSeries) {
        // pnlSeries is dict {date: pnl}
        const dates = Object.keys(pnlSeries).sort();
        const values = dates.map(d => pnlSeries[d]);

        // Calculate cumulative equity
        let equity = 0;
        const equityCurve = values.map(v => {
            equity += v;
            return equity;
        });

        if (equityCurve.length === 0) {
            this.equityChartContainer.innerHTML = 'Not enough data to chart.';
            return;
        }

        // Simple SVG Chart
        const width = this.equityChartContainer.clientWidth;
        const height = this.equityChartContainer.clientHeight;
        const padding = 40;

        const minVal = Math.min(...equityCurve);
        const maxVal = Math.max(...equityCurve);
        const range = maxVal - minVal || 1;

        // Scale functions
        const xScale = (i) => padding + (i / (equityCurve.length - 1)) * (width - 2 * padding);
        const yScale = (v) => height - padding - ((v - minVal) / range) * (height - 2 * padding);

        // Generate path
        let d = `M ${xScale(0)} ${yScale(equityCurve[0])}`;
        for (let i = 1; i < equityCurve.length; i++) {
            d += ` L ${xScale(i)} ${yScale(equityCurve[i])}`;
        }

        const svg = `
            <svg width="${width}" height="${height}" style="overflow: visible;">
                <!-- Grid lines -->
                <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#eee" />
                <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#eee" />
                
                <!-- Zero line if visible -->
                ${(minVal < 0 && maxVal > 0) ? `<line x1="${padding}" y1="${yScale(0)}" x2="${width - padding}" y2="${yScale(0)}" stroke="#ccc" stroke-dasharray="4" />` : ''}
                
                <!-- Path -->
                <path d="${d}" fill="none" stroke="#3498db" stroke-width="2" />
                
                <!-- Labels -->
                <text x="${padding - 10}" y="${yScale(maxVal)}" text-anchor="end" font-size="10" fill="#666">${maxVal.toFixed(0)}</text>
                <text x="${padding - 10}" y="${yScale(minVal)}" text-anchor="end" font-size="10" fill="#666">${minVal.toFixed(0)}</text>
                <text x="${padding}" y="${height - 10}" font-size="10" fill="#666">${dates[0].split('T')[0]}</text>
                <text x="${width - padding}" y="${height - 10}" text-anchor="end" font-size="10" fill="#666">${dates[dates.length - 1].split('T')[0]}</text>
            </svg>
        `;

        this.equityChartContainer.innerHTML = svg;
    }

    async showConfigPreview() {
        const configPath = this.baseConfigSelect.value;
        if (!configPath) return;

        this.previewContent.value = 'Loading...';
        this.previewContent.disabled = true;
        this.previewTitle.textContent = `Editor: ${configPath}`;
        this.configPreviewModal.classList.remove('hidden');

        try {
            const resp = await fetch(`/api/config?path=${encodeURIComponent(configPath)}`);
            const data = await resp.json();

            if (data.content) {
                this.previewContent.value = data.content;
                this.previewContent.disabled = false;
            } else {
                this.previewContent.value = `Error: ${data.error || 'Unknown error'}`;
            }
        } catch (e) {
            this.previewContent.value = `Network Error: ${e.message}`;
        }
    }

    async saveConfig() {
        const configPath = this.baseConfigSelect.value;
        const content = this.previewContent.value;
        if (!configPath) return;

        this.saveConfigBtn.disabled = true;
        this.saveConfigBtn.textContent = 'Saving...';

        try {
            const resp = await fetch('/api/config/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: configPath, content: content })
            });
            const data = await resp.json();

            if (data.status === 'ok') {
                this.showSaveStatus('Saved successfully!');
            } else {
                this.showSaveStatus(`Error: ${data.error}`, true);
            }
        } catch (e) {
            this.showSaveStatus(`Network Error: ${e.message}`, true);
        } finally {
            this.saveConfigBtn.disabled = false;
            this.saveConfigBtn.textContent = '💾 Save Changes';
        }
    }

    showSaveStatus(msg, isError = false) {
        this.saveStatus.textContent = msg;
        this.saveStatus.style.color = isError ? '#e74c3c' : '#27ae60';
        this.saveStatus.style.opacity = '1';
        setTimeout(() => {
            this.saveStatus.style.opacity = '0';
        }, 3000);
    }

    closeConfigPreview() {
        this.configPreviewModal.classList.add('hidden');
    }

    async loadOptions() {
        try {
            const resp = await fetch('/api/options');
            const data = await resp.json();

            if (data.configs && data.configs.length > 0) {
                this.fillSelect(this.baseConfigSelect, data.configs, 'configs/main_2024.yaml');
            }
            if (data.search_spaces && data.search_spaces.length > 0) {
                this.fillSelect(this.searchSpaceSelect, data.search_spaces, 'configs/search_space_fast.yaml');
            }
        } catch (e) {
            console.error('Options load failed', e);
        }
    }

    fillSelect(el, items, defaultVal) {
        if (!el) return;
        el.innerHTML = items.map(it => `<option value="${it}">${it}</option>`).join('');
        // Try to select default if exists, otherwise first item
        if (defaultVal && items.includes(defaultVal)) {
            el.value = defaultVal;
        } else if (items.length > 0) {
            el.value = items[0];
        }
    }

    updateFormVisibility() {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        document.querySelectorAll('.control-group').forEach(group => {
            const modes = group.dataset.modes.split(' ');
            if (modes.includes(mode)) {
                group.classList.remove('hidden');
            } else {
                group.classList.add('hidden');
            }
        });
    }

    async runTask() {
        if (this.currentTaskId) {
            // Should not happen usually because button is hidden/disabled, but safety check
            if (!confirm('Task is running. Start new one?')) return;
            this.stopMonitoring();
        }

        const mode = document.querySelector('input[name="mode"]:checked').value;
        const payload = { mode };

        document.querySelectorAll('.control-group:not(.hidden) input, .control-group:not(.hidden) select').forEach(input => {
            if (input.type === 'number') {
                payload[input.id] = parseInt(input.value, 10);
            } else {
                payload[input.id] = input.value;
            }
        });

        // Process Walk-Forward overrides
        const wfOverrides = {};
        if (payload['wf_start_date']) {
            wfOverrides.start_date = payload['wf_start_date'];
            delete payload['wf_start_date'];
        }
        // Check for num_steps (could be 0 or NaN if empty)
        if (payload['wf_num_steps'] !== undefined && !isNaN(payload['wf_num_steps'])) {
            wfOverrides.num_steps = payload['wf_num_steps'];
            delete payload['wf_num_steps'];
        }

        if (Object.keys(wfOverrides).length > 0) {
            payload.wf_overrides = wfOverrides;
        }

        console.log("RUN_TASK payload:", payload);

        // Handle empty study name for auto-generation
        if (mode === 'optimization' && !payload['study_name']) {
            delete payload['study_name']; // Backend will handle generation
        }

        this.setUIState('running');
        this.logsEl.textContent = 'Starting task...\n';
        this.summaryBox.classList.add('hidden');

        try {
            const resp = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();

            if (data.task_id) {
                this.currentTaskId = data.task_id;
                this.startMonitoring(data.task_id);
            } else {
                this.logsEl.textContent += 'Error: No task ID returned.\n';
                this.setUIState('error');
            }
        } catch (e) {
            this.logsEl.textContent += `Network Error: ${e.message}\n`;
            this.setUIState('error');
        }
    }

    async stopTask() {
        if (!this.currentTaskId) return;

        if (!confirm('Are you sure you want to stop the current task?')) return;

        this.stopBtn.disabled = true;
        this.stopBtn.textContent = 'Stopping...';

        try {
            const resp = await fetch('/api/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: this.currentTaskId })
            });
            const data = await resp.json();

            if (data.status && (data.status === 'stopped' || data.status === 'stopped_stale')) {
                this.logsEl.textContent += '\n=== STOPPED BY USER ===\n';
                // Monitoring loop will pick up the status change or we force it
            } else {
                alert('Failed to stop: ' + (data.error || 'Unknown error'));
                this.stopBtn.disabled = false;
                this.stopBtn.textContent = '⏹ Stop Task';
            }
        } catch (e) {
            alert('Network error stopping task: ' + e.message);
            this.stopBtn.disabled = false;
            this.stopBtn.textContent = '⏹ Stop Task';
        }
    }

    startMonitoring(taskId) {
        this.eventSource = new EventSource(`/api/stream?task_id=${taskId}`);
        this.eventSource.onmessage = (e) => {
            this.logsEl.textContent += e.data + '\n';
            this.logsEl.scrollTop = this.logsEl.scrollHeight;
        };
        this.eventSource.onerror = () => {
            this.eventSource.close();
        };

        this.pollInterval = setInterval(async () => {
            try {
                const resp = await fetch(`/api/status?task_id=${taskId}`);
                const data = await resp.json();
                const task = data.task;

                if (task) {
                    if (task.status === 'done') {
                        this.stopMonitoring();
                        this.setUIState(task.exit_code === 0 ? 'done' : 'error');
                        this.showSummary(task);
                        this.loadHistory();
                    }
                }
            } catch (e) {
                console.error('Poll error', e);
            }
        }, 1000);
    }

    stopMonitoring() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        this.currentTaskId = null;
    }

    setUIState(state) {
        this.statusBadge.className = 'status-badge';

        if (state === 'running') {
            this.runBtn.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');
            this.stopBtn.disabled = false;
            this.stopBtn.textContent = '⏹ Stop Task';

            this.statusBadge.textContent = 'Running';
            this.statusBadge.classList.add('status-running');
        } else {
            this.runBtn.classList.remove('hidden');
            this.stopBtn.classList.add('hidden');
            this.runBtn.disabled = false;

            if (state === 'done') {
                this.statusBadge.textContent = 'Done';
                this.statusBadge.classList.add('status-done');
            } else if (state === 'error') {
                this.statusBadge.textContent = 'Error';
                this.statusBadge.classList.add('status-error');
            } else {
                this.statusBadge.textContent = 'Ready';
                this.statusBadge.classList.add('status-ready');
            }
        }
    }

    showSummary(task) {
        let html = '';
        if (task.mode === 'backtest' && task.result) {
            const r = task.result;
            html = `<strong>Backtest Result:</strong><br>
                    Trades: ${r.total_trades}<br>
                    PnL: ${r.total_pnl?.toFixed(2)}<br>
                    Sharpe: ${r.sharpe_ratio_abs?.toFixed(4)}`;
        } else if (task.mode === 'tests' && task.junit) {
            const j = task.junit;
            html = `<strong>Tests Result:</strong><br>
                    Total: ${j.tests}, Failed: ${j.failures}, Errors: ${j.errors}, Skipped: ${j.skipped}`;
            if (j.failed && j.failed.length) {
                html += '<br><span style="color:#c0392b">Failures:</span><ul>';
                j.failed.forEach(f => html += `<li>${f.name}: ${f.message}</li>`);
                html += '</ul>';
            }
        } else {
            html = `<strong>Task Finished</strong> (Exit Code: ${task.exit_code})`;
        }

        this.summaryBox.innerHTML = html;
        this.summaryBox.classList.remove('hidden');
    }

    changePage(delta) {
        const maxPage = Math.ceil(this.totalItems / this.itemsPerPage) || 1;
        const newPage = this.currentPage + delta;

        if (newPage >= 1 && newPage <= maxPage) {
            this.currentPage = newPage;
            this.loadHistory();
        }
    }

    async loadHistory(isBackground = false) {
        try {
            // Add timestamp to prevent caching
            const resp = await fetch('/api/tasks?t=' + Date.now());
            if (!resp.ok) throw new Error(`HTTP status ${resp.status}`);

            const data = await resp.json();
            const tasks = data.tasks || [];
            this.totalItems = tasks.length;

            // Reset to page 1 on manual refresh if items changed significantly? 
            // Actually, let's keep current page unless out of bounds
            const maxPage = Math.ceil(this.totalItems / this.itemsPerPage) || 1;
            if (this.currentPage > maxPage) this.currentPage = maxPage;

            this.updatePaginationControls(maxPage);

            // Check if we have a running task to restore state
            if (!this.currentTaskId) {
                const runningTask = tasks.find(t => t.status === 'running');
                if (runningTask) {
                    console.log('Restoring running task', runningTask.task_id);
                    this.currentTaskId = runningTask.task_id;
                    this.setUIState('running');
                    this.startMonitoring(runningTask.task_id);
                } else if (tasks.length > 0 && !isBackground) {
                    // Only auto-load first log on initial load or manual refresh
                    if (this.logsEl.textContent.trim() === '' || this.logsEl.textContent.trim() === 'Select a task and click Run...') {
                        this.viewLog(tasks[0].task_id);
                    }
                }
            }

            if (tasks.length === 0) {
                this.historyList.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: #666;">No tasks found</td></tr>';
                return;
            }

            // Slice for pagination
            const startIdx = (this.currentPage - 1) * this.itemsPerPage;
            const endIdx = startIdx + this.itemsPerPage;
            const pageTasks = tasks.slice(startIdx, endIdx);

            this.historyList.innerHTML = pageTasks.map(t => {
                const date = t.started_at ? new Date(t.started_at * 1000).toLocaleString() : '-';
                let resultStr = '';

                if (t.exit_code !== null) {
                    resultStr = t.exit_code === 0 ? '<span style="color:green">Success</span>' : (t.exit_code === -1 || t.exit_code === 255 ? '<span style="color:red">Stopped/Error</span>' : `<span style="color:red">Exit ${t.exit_code}</span>`);
                } else {
                    resultStr = '<span style="color:orange">Running...</span>';
                }

                // Extract start date
                let startDate = '-';
                if (t.wf_overrides && t.wf_overrides.start_date) {
                    startDate = t.wf_overrides.start_date;
                } else if (t.result && t.result.config && t.result.config.walk_forward) {
                    // Try to get from result config if available
                    startDate = t.result.config.walk_forward.start_date;
                }

                // Extract steps
                let steps = '-';
                if (t.result && t.result.wf_steps !== undefined) {
                    steps = t.result.wf_steps;
                } else if (t.wf_overrides && t.wf_overrides.num_steps) {
                    steps = t.wf_overrides.num_steps;
                } else if (t.result && t.result.config && t.result.config.walk_forward && t.result.config.walk_forward.max_steps) {
                    steps = t.result.config.walk_forward.max_steps;
                }

                // Extract metrics
                let totalPnL = '-';
                let totalTrades = '-';
                let sharpeRatio = '-';

                if (t.result) {
                     if (t.result.total_pnl !== undefined) totalPnL = '$' + t.result.total_pnl.toFixed(2);
                     if (t.result.total_trades !== undefined) totalTrades = t.result.total_trades;
                     if (t.result.sharpe_ratio_abs !== undefined) sharpeRatio = t.result.sharpe_ratio_abs.toFixed(4);
                }

                return `
                <tr>
                  <td>${date}</td>
                  <td>${startDate}</td>
                  <td>${steps}</td>
                  <td><strong>${t.mode}</strong></td>
                  <td>${totalPnL}</td>
                  <td>${totalTrades}</td>
                  <td>${sharpeRatio}</td>
                  <td>${t.status}</td>
                  <td>${resultStr}</td>
                  <td>
                    <div style="display: flex; gap: 5px;">
                        <button onclick="document.querySelector('.container')._app.viewLog('${t.task_id}')" class="btn" style="padding: 4px 8px; font-size: 12px; background: #3498db;">👁️ View</button>
                        <a href="/api/logs/download?task_id=${t.task_id}" target="_blank" class="btn" style="padding: 4px 8px; font-size: 12px; background: #95a5a6; text-decoration: none; color: white;">⬇️ Log</a>
                        ${t.mode === 'backtest' && t.exit_code === 0 ? `<button onclick="document.querySelector('.container')._app.showAnalysis('${t.task_id}')" class="btn" style="padding: 4px 8px; font-size: 12px; background: #9b59b6;">📊 Analysis</button>` : ''}
                    </div>
                  </td>
                </tr>
              `;
            }).join('');
        } catch (e) {
            console.error('History load failed', e);
            this.historyList.innerHTML = `<tr><td colspan="10" style="color:red; text-align:center; padding: 20px;">Error loading history: ${e.message}</td></tr>`;
        }
    }

    updatePaginationControls(maxPage) {
        document.getElementById('pageInfo').textContent = `Page ${this.currentPage} of ${maxPage}`;
        document.getElementById('prevPage').disabled = this.currentPage <= 1;
        document.getElementById('nextPage').disabled = this.currentPage >= maxPage;
    }

    async viewLog(taskId) {
        this.logsEl.textContent = 'Loading log...';
        try {
            const resp = await fetch(`/api/logs/download?task_id=${taskId}`);
            if (resp.ok) {
                const text = await resp.text();
                this.logsEl.textContent = text;
                this.logsEl.scrollTop = this.logsEl.scrollHeight;
            } else {
                this.logsEl.textContent = 'Error loading log file.';
            }
        } catch (e) {
            this.logsEl.textContent = 'Network error loading log.';
        }
    }
}
