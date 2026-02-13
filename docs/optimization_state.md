# Optimization state

Last updated: 2026-02-13

Current stage: Max-Sharpe mode for `$1000`: текущий лидер — `tp15` (testing_period_days=15) поверх `ms0p1` (min_spread_move_sigma=0.1), `ts1p5` (time_stop_multiplier=1.5) и `vm10055` (max_var_multiplier=1.0055), holdout/stress Sharpe `5.142/4.899` (robust `4.899`).

Recent updates (2026-02-13):
- Signal sprint19 (hold/cooldown sweep under `ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint19: максимум по robust-метрике остаётся на baseline `hold300/cd300` (Sharpe `4.424/4.119`); `hold60` уходит в отрицательный Sharpe, `hold600/900` резко режут PnL и ухудшают cost_ratio.
- Signal sprint20 (min_spread_move_sigma sweep under `ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint20: новый лидер `ms0p1` (min_spread_move_sigma=0.1) — Sharpe `4.572/4.277` (лучше baseline `ms0p2` = `4.424/4.119`).
- Signal sprint21 (corr/pvalue sweep under `ms0p1`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint21: loosen/tighten `corr/pvalue` ухудшает Sharpe; лидер остаётся `ms0p1` (baseline `corr0p34_pv0p35`).
- Signal sprint22 (time_stop_multiplier sweep under `ms0p1`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint22: `ts1p5` остаётся лучшим (чуть лучше `ts1p0`); новый лидер не найден.
- Signal sprint23 (pair_stop_loss_zscore sweep under `ms0p1+ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint23: лидер остаётся `slz3p0` (=3.0); `2.0–2.5` ломает Sharpe через churn/издержки, `3.5–4.0` ухудшает Sharpe и раздувает DD.
- Signal sprint24 (protections toggles sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint24: все защиты должны оставаться включенными; `market_regime_detection=false` особенно разрушает Sharpe и DD; лидер остаётся `ms0p1`.
- Signal sprint25 (rolling_window sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint25: `rolling_window=96` остаётся явным максимумом Sharpe; `48/144/192/288` резко ухудшают метрики, вплоть до отрицательного PnL в stress.
- Signal sprint26 (z-entry sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint26: `z=1.15` остаётся локальным максимумом; `z=0.9–1.0` ухудшает Sharpe через churn, `z=1.30–1.45` режет PnL и снижает Sharpe.
- Signal sprint27 (structural-break intensity sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint27: baseline `sb_base` (= прежние Numba константы) остаётся лучшим; изменение min_correlation или мультипликаторов ухудшает robust Sharpe.
- Signal sprint28 (market-regime clamp sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint28: базовый clamp `rg0p5to1p5` остаётся лучшим; расширение верхней границы или сужение диапазона резко ухудшает Sharpe.
- Signal sprint29 (max_pairs sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint29: `max_pairs=24` остаётся лучшим; уменьшение/увеличение числа торгуемых пар ухудшает Sharpe через потерю диверсификации или рост издержек.
- Signal sprint30 (training_period_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint30: лидер остаётся baseline `tr90` (training=90d); `tr60` ломает стратегию (отрицательные Sharpe/PnL), а `120–240d` ухудшают robust Sharpe и повышают stress cost_ratio.
- Signal sprint31 (max_active_positions sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint31: лидер остаётся baseline `ap18` (max_active_positions=18); уменьшение лимита (`12/16`) ухудшает Sharpe, увеличение (`20/24`) не улучшает (почти идентичные метрики и чуть хуже robust Sharpe).
- Signal sprint32 (lookback_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint32: все варианты дали идентичные метрики → `pair_selection.lookback_days` сейчас не влияет на WFA (окно данных задаётся `training_start..testing_end`).
- Signal sprint33 (testing_period_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint33: новый лидер `tp15` — Sharpe `5.142/4.899` (robust `4.899`), но горизонт теста при `max_steps=5` становится короче (≈75 дней) → нужно подтверждение на сопоставимом горизонте (например, `tp15 + max_steps=10`).

Recent updates (2026-02-12):
- Проверена целостность последних `$1000` прогонов: для очередей `20260131_budget1000_*` обязательные артефакты присутствуют; Sharpe consistency check пройден.
- Подготовлена новая очередь из 10 конфигов: `coint4/artifacts/wfa/aggregate/20260212_budget1000_tlow_extended_sharpe_recover10/run_queue.csv`.
- Новые конфиги: `coint4/configs/budget_20260212_1000_tlow_extended_sharpe_recover10/*.yaml` (варианты r1-r5, holdout/stress).
- Гипотеза: улучшить Sharpe через более широкий пул пар (`corr/pvalue`, `max_pairs`) и снижение нелинейности sizing (`min/max_notional`).
- Запуск на `85.198.90.128` завершён: `10/10 completed`.
- Лучший вариант: `r4` (min Sharpe holdout/stress = `1.482`), пары выросли до `53`, но DD остался за гейтом (`~ -356`) и stress cost_ratio `0.62 > 0.5`.
- Дополнительный sweep без ограничений по рисковым гейтам: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_unbounded_minnot/run_queue.csv` (`12/12 completed`).
- Лучший вариант в Max-Sharpe режиме: `u6` (holdout/stress Sharpe `2.775/2.425`, pairs `58`, stress PnL `668.69`).
- Наблюдение по min_notional: в вариантах `u1-u3` (`min_notional` 0.5/1/2) метрики идентичны — в этой зоне параметр не лимитирует; рост Sharpe получен за счёт комбинированной смены режима (`z/ms/corr/pvalue/max_pairs`).
- Дополнительный signal sprint around `u6`: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint1/run_queue.csv` (`10/10 completed`).
- Новый лидер по robust-метрике `min(Sharpe_holdout, Sharpe_stress)`: `v1` (`3.338/3.007`), выше `u6` (`2.775/2.425`).
- Целостность результатов `signal_sprint1`: `Sharpe consistency OK (10 run(s))`, обязательные артефакты есть в `10/10`, в `run.log` нет `Traceback/ERROR`.
- Signal sprint2 (local search around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint2/run_queue.csv` (`10/10 completed`).
- Итог sprint2: ни один `s1-s5` не улучшил `v1`; лучший robust `min_sharpe` у `s5` = `2.067` (хуже `v1` = `3.007`). Понижение `z/ms` относительно `v1` ухудшает Sharpe.
- Infra: устранена коррупция memory-mapped кэша при параллельных WFA (lock + atomic replace + range-keyed cache filename для consolidated parquet).
- Signal sprint3 (z fine sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint3/run_queue.csv` (`10/10 completed`).
- Итог sprint3: лучший `zf4` (z=1.15) совпал с `v1` (Sharpe `3.338/3.007`), остальные `z=1.12-1.16` хуже → по `z` достигнут локальный максимум.
- Signal sprint4 (exit sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint4/run_queue.csv` (`10/10 completed`).
- Итог sprint4: лучший `ex3` (exit=0.08) совпал с `v1` (Sharpe `3.338/3.007`), остальные `exit=0.06-0.10` хуже → по `exit` достигнут локальный максимум.
- Signal sprint5 (ms sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint5/run_queue.csv` (`10/10 completed`).
- Итог sprint5: лучший `ms3` (ms=0.20) совпал с `v1` (Sharpe `3.338/3.007`), остальные `ms=0.16-0.24` хуже → по `ms` достигнут локальный максимум.
- Signal sprint6 (max_pairs sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint6/run_queue.csv` (`10/10 completed`).
- Итог sprint6: лучший `mp4` (max_pairs=24) совпал с `v1` (Sharpe `3.338/3.007`), остальные max_pairs хуже → по `max_pairs` локальный максимум на `24` (для max-Sharpe режима).
- Signal sprint7 (stop_loss_zscore sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint7/run_queue.csv` (`10/10 completed`).
- Итог sprint7: лучший `slz3p0` (stop_loss_z=3.0) совпал с `v1`; `2.0-2.5` убивает edge через churn/издержки, `3.5-4.0` раздувает DD → локальный максимум по stop_loss на `3.0`.
- Signal sprint8 (max_active_positions sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint8/run_queue.csv` (`10/10 completed`).
- Итог sprint8: лучший `ap18` (max_active_positions=18) совпал с `v1`; `ap24` почти идентичен, но чуть хуже; `ap6-ap14` дают просадку Sharpe.
- Signal sprint9 (max_var_multiplier sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint9/run_queue.csv` (`10/10 completed`).
- Итог sprint9: Sharpe резко растёт при уменьшении `max_var_multiplier`; лучший `vm1` (1.10) уже лучше `v1` (Sharpe `3.650/3.306` vs `3.338/3.007`).
- Signal sprint10 (max_var_multiplier fine sweep) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint10/run_queue.csv` (`10/10 completed`).
- Итог sprint10: новый лидер `vmf101` (max_var_multiplier=1.01) — Sharpe `4.348/4.043`, PnL `2153.76/1908.57`.
- Signal sprint11 (adaptive+regime+struct toggles) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint11/run_queue.csv` (`10/10 completed`).
- Итог sprint11: `market_regime_detection` и `structural_break_protection` должны оставаться включенными; `adaptive_thresholds=false` почти не хуже, но лидер всё равно `at1vm101` (= `vmf101`).
- Signal sprint12 (z sweep under max_var_multiplier=1.01) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint12/run_queue.csv` (`10/10 completed`).
- Итог sprint12: по `z` локальный максимум остался на `z=1.15` (`z1p15` совпадает с лидером `vmf101`).
- Signal sprint13 (exit sweep under `vmf101`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint13/run_queue.csv` (`10/10 completed`).
- Итог sprint13: по `exit` локальный максимум остался на `exit=0.08` (`ex08` совпадает с лидером `vmf101`).
- Signal sprint14 (stop_loss_zscore sweep under `vmf101`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint14/run_queue.csv` (`10/10 completed`).
- Итог sprint14: по `pair_stop_loss_zscore` локальный максимум остался на `3.0` (`slz3p0` совпадает с лидером `vmf101`).
- Signal sprint15 (max_var_multiplier ultra-fine sweep) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint15/run_queue.csv` (`10/10 completed`).
- Итог sprint15: новый лидер `vm1005` (max_var_multiplier=1.005) — Sharpe `4.378/4.074`, PnL `2188.25/1940.54`.
- Signal sprint16 (max_var_multiplier refine around 1.005) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint16/run_queue.csv` (`10/10 completed`).
- Итог sprint16: новый лидер `vm10055` (max_var_multiplier=1.0055) — Sharpe `4.380/4.076`, PnL `2190.02/1942.05`.
- Signal sprint17 (z micro-sweep under `vm10055`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint17/run_queue.csv` (`10/10 completed`).
- Итог sprint17: `z=1.15` остаётся локальным максимумом, новый лидер не найден.
- Signal sprint18 (time_stop_multiplier sweep under `vm10055`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint18/run_queue.csv` (`10/10 completed`).
- Итог sprint18: новый лидер `ts1p5` (`time_stop_multiplier=1.5`) — Sharpe `4.424/4.119`, PnL `2230.75/1978.76`.
- Детали: `docs/optimization_runs_20260212.md`.

Recent updates (2026-01-31):
- Extended OOS (2023-05-01 → 2024-04-30) для top20/top30 завершён: stress cost_ratio > 1.0, слабый Sharpe (см. `docs/optimization_runs_20260130.md`).
- Turnover-grid extended OOS (top10/top15, ms0.25/0.30, hold/cd 240) завершён: провал по парам и стресс-издержкам (см. `docs/optimization_runs_20260131.md`).
- Stop-condition: extended OOS stress cost_ratio > 0.5 и пары < 50 → оптимизацию в этом направлении останавливаем.
- Сформирован обзор по $1000: `docs/budget1000_overview_20260131.md`.
- Очередь tlow extended OOS для $1000 выполнена: risk0p0175 Sharpe 1.70/1.36, cost_ratio 0.26/0.60, DD ~-32%, pairs 36; risk0p015 отрицательный (см. `docs/optimization_runs_20260131.md`).
- Refine‑очередь (z=1.25/1.30, ms=0.30/0.35, hold/cd=300) выполнена: DD снизился до ~-17…-18%, но stress cost_ratio 0.53–0.56 и пары 36 (см. `docs/optimization_runs_20260131.md`).
- Refine2‑очередь (z=1.35/1.40, ms=0.35/0.40, hold/cd=360) выполнена: PnL/Sharpe ухудшились, stress cost_ratio 1.85–1.86, пары 36 — направление закрываем (см. `docs/optimization_runs_20260131.md`).
- Tradeability+basecap3 очередь (corr 0.35/0.40, pv 0.25/0.20, пары basecap3=102) выполнена: Sharpe < 0, PnL отрицательный, pairs 26–27; tradeM/tradeS совпали → ветку закрываем (см. `docs/optimization_runs_20260131.md`).
- Paper configs: `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` (primary) и `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` (fallback).
- План paper/forward: `docs/paper_forward_plan_20260131.md`.

Next steps:
- Paper/forward тест кандидата: см. `docs/paper_forward_plan_20260131.md`.
- Если extended OOS обязателен для $1000: текущие попытки (tradeability+basecap3) дали отрицательные метрики → целесообразнее фиксировать stop‑condition и переходить к paper/forward.
- Исправление `total_costs` для Numba-бэктеста выполнено; метрики обновлены (done).
- Baseline WFA (5 шагов) выполнен и зафиксирован (done).
- Turnover sweep завершён, лучшая комбинация entry 0.95 / exit 0.10 (done).
- Quality sweep завершён, лучший Sharpe при corr 0.65 (done).
- Risk sweep завершён, существенных отличий не выявлено (done).
- Shortlist WFA (5 шагов) завершён на 85.198.90.128; топ Sharpe: baseline `5.7560`, corr0.7 `5.7302`, turnover `4.9631` (см. rollup).
- Holdout WFA (2024-05-01 → 2024-12-31, фактический тест до 2024-09-28) завершён: Sharpe `-3.41/-3.27`, PnL `-324/-307` (baseline/corr0.7).
- Stress costs WFA завершён: Sharpe `4.66/4.57`, PnL `616/543` (baseline/corr0.7).
- Диагностика holdout завершена: см. `docs/holdout_diagnostics_20260118.md` + CSV в `coint4/results/holdout_20260118_*`.
- Добавлен фильтр стабильности пар (pair_stability_window_steps/min_steps) в WFA.
- Stability shortlist (20260119) завершён: Sharpe 2.41–5.92, но total_pairs_traded 4–34 (ниже порога 100).
- Stability_relaxed WFA завершён: Sharpe `3.99/2.59`, total_pairs_traded `78/41` (ещё ниже порога 100).
- Stability_relaxed2 WFA завершён: Sharpe `2.13`, total_pairs_traded `52`, total_trades `1010`, PnL `84.13`.
- Stability_relaxed3 WFA завершён: лучший Sharpe `2.04`, total_pairs_traded `51`; 2 конфига дали 0 пар из-за KPSS фильтра.
- Stability_relaxed4 WFA завершён: Sharpe `3.07–3.09`, total_pairs_traded `159–272`; лучший вариант corr0.45 + ssd50000 + kpss0.03.
- Holdout relaxed4 завершён: Sharpe `-1.32/-1.08`, PnL `-98.19/-78.69` (corr0.45/corr0.5).
- Stress relaxed4 завершён: Sharpe `2.386/2.382`, PnL `426/416`.
- Диагностика holdout relaxed4 завершена: пересечение пар 4 (Jaccard ~0.0048), доминирует pvalue; см. `docs/holdout_diagnostics_20260119_relaxed4.md`.
- Stability_relaxed5 WFA завершён: w3m2 Sharpe `5.81`, pairs `383`, PnL `958.72` (corr0.45, ssd50000, kpss0.03, window=3/min=2).
- Holdout relaxed5 (w3m2) завершён: Sharpe `-1.77`, PnL `-220.63`, pairs `1018`.
- Stress relaxed5 (w3m2) завершён: Sharpe `4.76`, PnL `783.53`.
- Диагностика holdout w3m2 завершена: пересечение пар `16` (Jaccard ~0.0116), доминирует pvalue; см. `docs/holdout_diagnostics_20260119_relaxed5.md`.
- Подготовлен фиксированный universe из WFA w3m2 (383 пары) для повторного holdout; очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_holdout_fixed/run_queue.csv`.
- Holdout relaxed5 fixed universe (w3m2) завершён: Sharpe `1.25`, PnL `13.94`, pairs `18`, total_trades `352` (низкая статистика).
- Holdout fixed universe (window=1/min=1) завершён: Sharpe `-0.02`, PnL `-0.20`, pairs `11`, total_trades `253`.
- Stability_relaxed6 WFA завершён: Sharpe `5.36`, pairs `303`, PnL `894.35` (pvalue 0.12, kpss 0.05, hurst 0.70).
- Holdout relaxed6 (w3m2) завершён: Sharpe `-2.20`, PnL `-267.02`, pairs `779`, total_trades `13268`.
- Диагностика holdout relaxed6 завершена: пересечение пар `6` (Jaccard ~0.0056); см. `docs/holdout_diagnostics_20260119_relaxed6.md`.
- Holdout relaxed6 fixed universe завершён: Sharpe `3.77`, PnL `24.98`, pairs `7`, total_trades `141`.
- Stability_relaxed7 WFA завершён: Sharpe `5.25`, pairs `177`, PnL `914.61` (train=90d).
- Holdout relaxed7 (train=90d) завершён: Sharpe `-0.20`, PnL `-20.95`, pairs `339`, total_trades `6509`.
- Собран новый universe (relaxed8 строгий, 110 пар) для пред‑holdout периода: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/pairs_universe.yaml`.
- Собран expanded universe (relaxed8 strict pre‑holdout v2, 250 пар) с limit_symbols=300: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`.
- Подготовлен более мягкий criteria-файл для возможного следующего расширения universe: `coint4/configs/criteria_relaxed8_nokpss_universe.yaml`.
- WFA relaxed8 fixed universe завершён: 0 торгуемых пар, Sharpe `0.00` (фильтры режут до нуля).
- WFA relaxed8_loose завершён: 0 торгуемых пар (KPSS режет до нуля даже при kpss=0.1).
- WFA relaxed8_nokpss завершён: Sharpe `1.86`, pairs `35`, PnL `104.11` (kpss=1.0).
- Holdout relaxed8_nokpss завершён: Sharpe `3.21`, PnL `145.84`, pairs `64`, total_trades `2252`.
- Stress relaxed8_nokpss holdout завершён: Sharpe `2.17`, PnL `98.35`, pairs `64`, total_trades `2252`, costs `108.65`.
- Holdout relaxed8_nokpss_u250 завершён: Sharpe `4.20`, PnL `421.60`, pairs `168`, total_trades `6572`.
- Stress holdout u250 завершён: Sharpe `2.89`, PnL `289.30`, pairs `168`, total_trades `6572`, costs `309.95`.
- Итоговый кандидат: `docs/candidate_relaxed8_u250_20260119.md`.
- Запланирован turnover stress grid (u250) для снижения числа сделок: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_turnover_stress/run_queue.csv`.
- Top-k лимит пар (max_pairs=10/20) выполнен для u250 holdout + stress: top10 Sharpe `2.74/2.08`, trades `798`; top20 Sharpe `4.56/3.65`, trades `1693`, costs `46.37/82.43` (holdout/stress).
- Turnover grid поверх top20 завершён: лучший вариант z1.05/exit0.08/hold120/cd120 → holdout Sharpe `3.85`, stress Sharpe `3.44`, trades `630`, costs `18.04/32.07`.
- Baseline u250 turnover grid завершён: лучший вариант z0.95/exit0.08/hold120/cd120 → holdout Sharpe `4.52`, stress Sharpe `3.70`, PnL `447.87/366.56`, trades `3936`.
- Candidate sweep по риск-параметрам завершён: метрики почти не меняются, max_active_positions даёт минимальные отличия; оставляем baseline z0.95/0.08/120/120.
- Sharpe annualization: WFA использует `annualizing_factor * (24*60/bar_minutes)`; base_engine приведён к динамическому periods_per_year по шагу данных.
- Micro-grid u250 (entry/exit/hold/cd + max_pairs 50/100/150) завершён: лучший min‑Sharpe у z0.95/exit0.06/hold120/cd120 (4.54/3.73); exit0.10 практически идентичен. Очередь: `coint4/artifacts/wfa/aggregate/20260121_relaxed8_nokpss_u250_search/run_queue.csv`.
- Кандидат обновлён: `docs/candidate_relaxed8_u250_20260121.md`.
- Numba: включены cooldown/min_hold/min_spread_move/stop-loss + выход по |z|<=z_exit; портфельная симуляция использует позиции вместо PnL-сигналов.
- Очередь churnfix micro-grid (u250) подготовлена: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix/run_queue.csv` (результаты 0 сделок из-за адаптивных порогов).
- Churnfix v2 (после фикса адаптивных порогов) завершён: 0 сделок в holdout/stress, требуется диагностика порогов/volatility (`coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v2/run_queue.csv`).
- Sanity no-adapt завершён: 0 сделок даже при entry 0.75 (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity/run_queue.csv`).
- Sanity v2 (current-bar signals) завершён: 0 сделок, Sharpe 0.00 (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v2/run_queue.csv`).
- Numba выравнен с базовой логикой по std (guard 1e-6, без min_volatility clamp в z-score) и принимает beta/mu/sigma напрямую.
- Sanity v3 завершён: 0 сделок (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v3/run_queue.csv`).
- Найдено: min_spread_move_sigma блокировал входы из-за NaN в last_flat_spread при fastmath; исправлено через last_flat_valid.
- Sanity v4 завершён: сделки восстановились, но turnover очень высокий (26k+), stress Sharpe отрицательный (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v4/run_queue.csv`).
- Churnfix v3 (holdout+stress) завершён: Sharpe 5.0–8.0, PnL 583–1072, trades 20k–28k.
- Churnfix top‑k завершён: top20 Sharpe 7.74/6.84, PnL 898/793, trades 4841; top50 Sharpe 7.63/6.43, PnL 1083/914, trades 11823 (holdout/stress).
- Churnfix msgrid завершён: ms0p2/ms0p3 на hold180 дают метрики близкие к ms0p1; hold240 снижает Sharpe/PNL.
- Alt holdout (2022-09-01 → 2023-04-30) завершён: top50/full идентичны (≈47 пар), Sharpe 7.96/6.72, PnL 941/794; top20 чуть хуже.
- Sensitivity top50 завершён: лучший Sharpe у z1.00/exit0.06 (9.01/7.64) при PnL 1115/946; z0.95/exit0.08 даёт максимум PnL (1180/997).
- Basecap3 завершён: Sharpe 4.87/3.80, PnL 674/526, pairs 71 — слишком жёстко.
- Новый лучший компромисс (robust): top50/z1.00/exit0.06/hold180/cd180/ms0.1 → Sharpe 9.01/7.64, PnL 1115/946, trades 11414.
- Кандидат обновлён: `docs/candidate_relaxed8_u250_20260122.md`.
- Канонический конфиг: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml` (PnL‑альтернатива рядом).
- Alt-holdout top50 sens завершён: z1.00/exit0.06 Sharpe 8.65/7.48, PnL 1049/907; z0.95/exit0.08 чуть ниже (см. `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/run_queue.csv`).
- OOS 2023-05 → 2023-12: лучше z0.95/exit0.08 (Sharpe 4.24/2.94, PnL 810/561); z1.00 ниже (2.63/1.55, PnL 556/326).
- OOS 2025-01 → 2025-06: лучше z1.00/exit0.06 (Sharpe 3.83/2.61, PnL 400/271); z0.95 ниже.
- OOS top30/top40 (z1.00/ms0p1): top30 лучше top40, но уступает top50 по Sharpe/PNL; turnover снижается ~34% (2025H1), поэтому оставляем top50 как primary.
- По шагам WFA (daily_pnl срезы) есть отрицательные минимумы на обоих OOS периодах; детали в `docs/optimization_runs_20260122.md`.
- Концентрация на новых OOS умеренная: top10 ≈ 44–50%, top20 ≈ 63–68%; отрицательных пар 50–69 из 141–145.
- Churngrid min_spread_move_sigma завершён: ms0p2 лучше ms0p15 в базовом holdout, но OOS 2025H1 для ms0p2 хуже (Sharpe 2.66/1.44, PnL 278/149), поэтому оставляем ms0p1 как primary.
- Decision matrix (summary):
  - Primary (z1.00/ms0p1): лучший на OOS 2025H1, сильный базовый holdout, стабильнее ms0p2.
  - PnL alt (z0.95/exit0.08): лучший на OOS 2023H2, но слабее на 2025H1.
- Sharpe sanity: скрипт `scripts/optimization/check_sharpe_consistency.py` прошёл для ключевых очередей (32 прогона).
- Base cap: текущий pairs_universe уже max_per_base=4; basecap3 ухудшал метрики, поэтому отдельный cap 5/8 не даёт нового эффекта.
- Следующий шаг: финальная проверка концентрации/устойчивости и решение о paper/live.

Legacy context:
Current stage: Leader holdout WFA (2024-05-01 → 2024-12-31, max_steps=5) via artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv (parallel=1, n_jobs=-1). Additional: next5_fast WFA (manual sequential runs; queue file artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv used for status, backtest.n_jobs=-1, COINT_FILTER_BACKEND=threads). Current next5_fast run: none (latest best by Sharpe: pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000); queued: none.

Progress:
- ssd5000 completed
- ssd10000 completed
- ssd25000 completed
- ssd50000 active (WF step 3/3, step 2: 122 pairs, P&L +355.58)
- leader_validation completed (Sharpe 0.5255, PnL 1388.71, DD -199.31)
- leader_holdout active: coint4/configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml → artifacts/wfa/runs/20260116_leader_holdout/best_config__leader_holdout_ssd25000__20260116_211943 (COINT_FILTER_BACKEND=processes)

Parallel stage:
- Piogoga grid (leader filters, zscore sweep) via artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv (parallel=16, n_jobs=1).
- Signal grid (16 configs, z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1) via artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv (parallel=16, n_jobs=1).
- SSD sweep (6 values) via artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv (queue has running statuses; check worker activity before resume).
- Leader validation (post-analysis, single run) completed: artifacts/wfa/runs/20260116_leader_validation/.
- Патч: фильтрация пар теперь параллельная (n_jobs из backtest, backend threads; `COINT_FILTER_BACKEND=processes` с spawn для OpenMP‑безопасности) — цель полной загрузки CPU.

After ssd50000 DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for ssd50000).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start SSD sweep (3 values) queue: artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv.
4) Update this file with the new stage and next steps.

After signal grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for signal grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.

After piogoga grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for piogoga grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start leader validation queue: artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv.

After leader holdout DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for leader holdout).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Decide next stage (risk sweep vs signal grid refinement) based on holdout Sharpe/PnL/DD.

Notes:
- NOTE: Sharpe в записях/артефактах до фикса annualization (2026-01-18) занижен примерно в √96 раз для 15m; для актуальных значений используйте `coint4/artifacts/wfa/aggregate/rollup/run_index.*`.
- 2026-01-19: normalized_backtester Sharpe приведён к annualization 365*96 и учитывает нулевые доходности (см. `coint4/src/coint2/core/numba_kernels_v2.py`).
- 2026-01-18: shortlist WFA completed на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-18: holdout + stress WFA завершены на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-17: smoke WFA для проверки логирования команд (config main_2024_smoke.yaml, results artifacts/wfa/runs/logging_smoke_20260117_072821).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260116_stop2p5_time3p5 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p08_ssd25000 (PnL 741.27, Sharpe 0.5908, DD -146.05).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p95_exit0p08_ssd25000 (PnL 685.02, Sharpe 0.5665, DD -114.16).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop2p5_time2p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p0_time2p5_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p5_time3p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p1_ssd25000 (PnL 821.86, Sharpe 0.6434, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p06_ssd25000 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p12_ssd25000 (PnL 855.78, Sharpe 0.6789, DD -95.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p09_ssd25000 (PnL 822.87, Sharpe 0.6441, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p1_ssd25000 (PnL 780.90, Sharpe 0.5965, DD -139.51).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 789.56, Sharpe 0.6637, DD -98.47).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 744.46, Sharpe 0.6037, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000 (PnL 404.72, Sharpe 0.6275, DD -44.95).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 621.19, Sharpe 0.7190, DD -91.63).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 555.34, Sharpe 0.6492, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 946.25, Sharpe 0.5384, DD -193.91).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p02_top500_z0p85_exit0p12_ssd25000 (PnL 456.81, Sharpe 0.6618, DD -78.36).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p65_hurst0p5_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 620.39, Sharpe 0.5028, DD -126.15).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd8000_pv0p03_top600_z0p85_exit0p12_ssd25000 (PnL 247.31, Sharpe 0.4533, DD -37.73).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000 (PnL 547.88, Sharpe 0.6830, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p75_pv0p015_top300_hurst0p48_kpss0p03_z0p85_exit0p12 (PnL 95.37, Sharpe 0.1718, DD -105.85).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p01_top200_kpss0p02_hl0p2_20_z0p85_exit0p12 (PnL 623.38, Sharpe 0.4579, DD -230.50).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd4000_corr0p65_pv0p02_top400_z0p85_exit0p12 (PnL 61.89, Sharpe 0.2783, DD -25.44).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_hurst0p5_kpss0p02_cross1_z0p85_exit0p12 (PnL 747.15, Sharpe 0.5444, DD -130.58).
