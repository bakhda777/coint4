# План paper/forward теста (2026-01-31)

Цель: проверить устойчивость кандидата на реальном потоке данных после того, как extended OOS и turnover-grid не прошли стресс-издержки.

## Кандидаты
- Основной: `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
- Запасной: `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`

## Период и режим
- Длительность: 4-6 недель.
- WFA отключен, результаты пишутся в `coint4/artifacts/live`.
- Модель издержек: агрегированная (`commission_pct` + `slippage_pct`), `enable_realistic_costs: false`.

## KPI (оценка по итогам периода)
- cumulative PnL > 0.
- cost_ratio <= 0.5.
- max_drawdown_abs >= -500 (<= 5% от 10k).
- total_trades >= 300, total_pairs_traded >= 40.
- концентрация PnL: top10 пар <= 0.7 от total PnL.

## Stop-условия (ранний выход)
- 14-дневный cost_ratio > 1.0.
- max_drawdown_abs < -800 (8% от 10k).
- 20 торговых дней подряд с отрицательным cumulative PnL и Sharpe < 0.
- Проблемы данных/логов более 24 часов.

## Мониторинг и отчетность
- Ежедневный контроль метрик в `coint4/artifacts/live/logs/metrics.jsonl`.
- Еженедельный снимок: `PYTHONPATH=src ./.venv/bin/python scripts/extract_live_snapshot.py`.
- Обновлять `coint4/artifacts/live/LIVE_DASHBOARD.md` раз в неделю.
