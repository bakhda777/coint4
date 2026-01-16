# План оптимизации параметров (2026-01-16)

Цель: добиться `sharpe_ratio_abs > 1.0` при контролируемой просадке и стабильном количестве сделок/пар.

## Критерии отбора
- Основная метрика: `sharpe_ratio_abs` по WFA (макс 5 шагов).
- Минимумы: `total_trades >= 200`, `total_pairs_traded >= 50`.
- Ограничение по риску: `max_drawdown_abs >= -200` (жестче — по результатам лидеров).

## План (10 шагов)
1. Возобновить все прерванные очереди (SSD sweep 6/3, strict PV grid, selection grid) и обновить журнал `docs/optimization_runs_20260116.md`.
2. Сформировать rollup индекс (`artifacts/wfa/aggregate/rollup/`) и зафиксировать лидеров по Sharpe и PnL.
3. Локально уточнить SSD вокруг лидера `ssd=25000` (например 20k/25k/30k/40k) с лучшими фильтрами из strict PV grid.
4. Протестировать зону сигналов `zscore_entry 0.75–0.85` и `zscore_exit 0.04–0.08` на лучших SSD/фильтрах.
5. Добавить риск‑контроли (time_stop/stop_loss/cooldown/rolling_window) на лидерах, чтобы снизить DD без потери Sharpe.
6. Пересмотреть tradeability и quality‑universe 200k/250k с мягкими фильтрами для увеличения пар при сохранении качества.
7. Сформировать shortlist из 4–5 конфигов: лидер strict PV, лидер SSD, лидер quality‑universe, лучший по PnL, лучший по Sharpe в широкой сетке.
8. Провести WFA‑валидацию (5 шагов) на shortlist и отсеять нестабильные профили.
9. Проверить holdout fixed backtest (2024‑H1, top‑200) для top‑1/2 кандидатов.
10. Зафиксировать финальные результаты и фильтрацию пар в `docs/optimization_runs_20260116.md` и rollup‑индексе.

## Команды (из `coint4/`)
Возобновление очередей:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv
```

Rollup индекс:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```
