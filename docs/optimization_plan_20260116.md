# План оптимизации параметров (2026-01-16)

Цель: добиться `sharpe_ratio_abs > 1.0` при контролируемой просадке и стабильном количестве сделок/пар.

## Критерии отбора
- Основная метрика: `sharpe_ratio_abs` по WFA (макс 5 шагов).
- Минимумы: `total_trades >= 200`, `total_pairs_traded >= 50`.
- Ограничение по риску: `max_drawdown_abs >= -200` (жестче — по результатам лидеров).
- Фильтрация пар: параллельная (n_jobs = `backtest.n_jobs`, backend `threads`; `COINT_FILTER_BACKEND=processes` для override) для полной загрузки CPU.

## Файл состояния
- `docs/optimization_state.md` — краткое состояние и следующий шаг; обновлять после каждого блока прогонов.
- В `scripts/optimization/on_done_codex_prompt.txt` обязательно держать фразу "Прогон завершён, продолжай выполнение плана" и инструкции: headless‑работа + запись причины в `docs/optimization_state.md` при сбое.

## План (13 шагов)
1. Завершить SSD sweep (6 значений) через `run_queue.csv` с CPU‑watcher `scripts/optimization/watch_wfa_queue.sh` и обновить журнал `docs/optimization_runs_20260116.md`.
2. Завершить SSD sweep (3 значения) и зафиксировать результаты/фильтрацию пар.
3. Прогнать strict PV grid через очередь и зафиксировать лидеров по Sharpe/DD.
4. Прогнать selection grid (оставшиеся конфиги) и обновить сводную таблицу кандидатов.
5. Обновлять rollup индекс (`artifacts/wfa/aggregate/rollup/`) после каждого блока прогонов.
6. Запустить SSD refine (20k/25k/30k/40k) на фильтрах лидера strict PV.
7. Запустить signal grid (z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1) на `ssd=25000` с parallel=16, n_jobs=1.
8. Запустить piogoga grid (leader filters, zscore sweep) на `ssd=25000` с parallel=16, n_jobs=1.
9. Запустить risk sweep (stop/time/cooldown/rolling_window) для снижения DD на лидере.
10. Пересмотреть tradeability и quality‑universe 200k/250k с мягкими фильтрами для увеличения пар.
11. Сформировать shortlist из 4–5 конфигов: лидер strict PV, лидер SSD, лидер quality‑universe, лучший по PnL, лучший по Sharpe.
12. Провести WFA‑валидацию (5 шагов) на shortlist и отсеять нестабильные профили (начать с лидера SSD: `configs/leader_validation_20260116/leader_validate_20260116_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000.yaml`).
13. Проверить holdout fixed backtest (2024‑H1, top‑200) для top‑1/2 кандидатов и зафиксировать результаты.

## Команды (из `coint4/`)
Возобновление очередей:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv
```

Возобновление очередей с headless Codex по завершении:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --on-done-prompt-file scripts/optimization/on_done_codex_prompt.txt \
  --on-done-log artifacts/wfa/aggregate/20260115_ssd_topn_sweep/codex_on_done.log
```

Rollup индекс:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```

Piogoga grid:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv \
  --parallel 16
```

Leader validation:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv
```
