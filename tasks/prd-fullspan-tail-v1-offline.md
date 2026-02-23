# PRD: Робастный отбор fullspan и защита от tail-концентрации (v1 offline)

## 1. Overview
Текущий отбор кандидатов переоценивает короткий OOS по среднему Sharpe и недостаточно штрафует хвостовые провалы на длинном горизонте fullspan.  
Цель эпика: объединить три направления в одном процессе отбора:
- Робастный ранжирующий score вместо “лучший по mean Sharpe”.
- Прокси-защита tail-risk (концентрация убытка по паре/периоду) на offline-этапе.
- Tradeability/universe-стабильность (гистерезис) как offline-сигнал качества.

Граница `v1`: только offline pipeline (без runtime-изменений в live-движке).

## 2. Goals
- Ввести hard-gate по drawdown для fullspan: fail при `max_drawdown_on_equity < -0.50`.
- Ввести soft-score на основе `PSR + DSR` и штрафов за худший WF-период и tail-концентрацию в одной паре.
- Расширить `run_index` новыми полями робастности.
- Добавить `robust_ranking.csv` в `coint4/artifacts/wfa/aggregate/<run_group>/`.
- Валидировать решение replay-методом на существующих артефактах без тяжёлых новых прогонов.

## 3. Quality Gates
Эти команды должны проходить для каждой user story:
- `make ci`

## 4. User Stories

### US-001: Извлечение робастных метрик на уровне run
**Description:** Как инженер оптимизации, я хочу получать PSR/DSR и tail-диагностику для каждого run, чтобы ранжирование учитывало устойчивость, а не только средний Sharpe.

**Acceptance Criteria:**
- [ ] В `run_index` добавлены поля минимум: `psr`, `dsr`, `wf_worst_period`, `wf_worst_period_pnl`, `wf_worst_pair`, `wf_worst_pair_pnl`, `tail_loss_concentration`, `pair_stability_hysteresis`, `robust_metrics_status`.
- [ ] Метрики считаются из существующих артефактов run (`equity_curve.csv`, `daily_pnl.csv`, `trade_statistics.csv`) без изменения формата исходных артефактов.
- [ ] При частично отсутствующих входных файлах ранкер не падает: поля получают `null`, а `robust_metrics_status` отражает причину.

### US-002: Hard-gate по DD для fullspan
**Description:** Как инженер оптимизации, я хочу автоматически отбрасывать кандидаты с недопустимым DD, чтобы исключать катастрофические fullspan-профили.

**Acceptance Criteria:**
- [ ] Реализован hard-gate: вариант не проходит, если `worst_dd_pct > 0.50`, где `worst_dd_pct = max_window(max(abs(dd_holdout), abs(dd_stress)))`.
- [ ] Порог `0.50` зафиксирован как дефолт `v1` и параметризуем через CLI.
- [ ] Для каждого исключённого варианта сохраняется явная причина (`fail_reason=dd_gate`).

### US-003: Soft-score (PSR/DSR + worst-period + concentration)
**Description:** Как инженер оптимизации, я хочу ранжировать варианты по робастному score, чтобы снизить шанс выбора хрупких конфигов.

**Acceptance Criteria:**
- [ ] `PSR` и `DSR` используются как компоненты score (не как отдельный hard-fail).
- [ ] В score входят штрафы за `worst_wf_period_pnl` и `tail_loss_concentration`.
- [ ] Формула score детерминирована и документирована, включая нормализацию, знаки и tie-breaker.
- [ ] Итоговое ранжирование строится только среди кандидатов, прошедших DD gate.

### US-004: Offline tradeability/universe-гистерезис
**Description:** Как инженер оптимизации, я хочу видеть стабильность пар между периодами, чтобы не продвигать конфиги, завязанные на случайный всплеск одной пары.

**Acceptance Criteria:**
- [ ] Рассчитывается `pair_stability_hysteresis` по повторяемости/устойчивости пар между WF-периодами.
- [ ] Метрика публикуется в выходных артефактах как минимум диагностически в `v1`.
- [ ] Реализация не требует новых тяжёлых прогонов и работает на существующих run-артефактах.

### US-005: Артефакты вывода (вариант 8C)
**Description:** Как инженер оптимизации, я хочу получать единые артефакты rollup и ranking, чтобы downstream-процессы и отчёты работали без ручной склейки.

**Acceptance Criteria:**
- [ ] `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` расширен новыми полями ранжирования/робастности.
- [ ] `coint4/artifacts/wfa/aggregate/rollup/run_index.json` расширен теми же полями.
- [ ] `coint4/artifacts/wfa/aggregate/rollup/run_index.md` содержит краткую сводку по робастным метрикам и DD-gate.
- [ ] Для каждого обработанного `run_group` создаётся `coint4/artifacts/wfa/aggregate/<run_group>/robust_ranking.csv`.

### US-006: Валидация через replay существующих артефактов
**Description:** Как инженер оптимизации, я хочу доказать корректность нового отбора без тяжёлых запусков на текущем сервере.

**Acceptance Criteria:**
- [ ] Добавлены unit/integration проверки для извлечения метрик, DD gate и score.
- [ ] Выполняется replay на существующих артефактах (без новых long WFA).
- [ ] На replay подтверждается expected behavior: fullspan-кейсы с DD хуже `-0.50` отбрасываются, OOS-кейсы с приемлемым DD остаются в ранжировании.

### US-007: Документация состояния и прогонов
**Description:** Как владелец процесса оптимизации, я хочу зафиксировать новую схему отбора в состоянии проекта и дневнике, чтобы решения были трассируемыми.

**Acceptance Criteria:**
- [ ] Обновлён `docs/optimization_state.md` с описанием нового gate/score и порядка запуска.
- [ ] Добавлена запись в `docs/optimization_runs_YYYYMMDD.md` по replay-валидации и итогам ранжирования.
- [ ] В документации явно зафиксировано, что `v1` не включает runtime-защиты в live-исполнении.

## 5. Functional Requirements
1. FR-1: Система должна вычислять `PSR` и `DSR` для каждого run на основе доходностей `equity_curve.csv`.
2. FR-2: Система должна извлекать худший WF-период (`wf_worst_period`) и его PnL из run-артефактов.
3. FR-3: Система должна вычислять tail-концентрацию убытка в одной паре (`tail_loss_concentration`) из `trade_statistics.csv`.
4. FR-4: Система должна вычислять метрику гистерезиса стабильности пар (`pair_stability_hysteresis`) по периодам.
5. FR-5: Система должна применять hard-gate `worst_dd_pct <= 0.50` (дефолт `v1`).
6. FR-6: Система должна строить soft-score, где `PSR` и `DSR` добавляются, а `worst_wf_period_pnl` и `tail_loss_concentration` штрафуются.
7. FR-7: Система должна сохранять причины отклонения кандидатов (`fail_reason`) в ranking-выводе.
8. FR-8: Система должна расширять `run_index.(csv|json|md)` новыми полями без ломки обратной совместимости для старых run.
9. FR-9: Система должна создавать `robust_ranking.csv` для каждого `run_group`.
10. FR-10: Система должна работать в режиме replay по уже существующим данным без запуска новых тяжёлых WFA.
11. FR-11: Система должна быть детерминированной: одинаковый вход даёт одинаковый ranking.
12. FR-12: Система должна поддерживать параметризацию порогов/весов через CLI (с фиксированными дефолтами `v1`).

## 6. Non-Goals (Out of Scope for v1)
- Runtime-ограничения в live-исполнении (quarantine/deleverage/stop) не внедряются в `v1`.
- Запуск тяжёлых fullspan/WFA на текущем сервере не входит в `v1`.
- Изменение live/cutover-потока и торгового исполнения не входит в `v1`.
- Автоматическое внедрение новых market-data источников (спред/фандинг/ликвидность) не является обязательным для `v1`.

## 7. Technical Considerations
- Предпочтительные точки интеграции:
- `coint4/src/coint2/ops/run_index.py`
- `coint4/scripts/optimization/build_run_index.py`
- `coint4/scripts/optimization/rank_multiwindow_robust_runs.py`
- `coint4/scripts/optimization/postprocess_queue.py`
- Выходные файлы:
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.json`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.md`
- `coint4/artifacts/wfa/aggregate/<run_group>/robust_ranking.csv`
- Для PSR/DSR использовать существующие функции в `coint4/src/coint2/core/performance.py`.
- При отсутствии части входов использовать graceful degradation (`null` + статус), а не падение пайплайна.

## 8. Success Metrics
- 100% обработанных `run_group` получают `robust_ranking.csv` без ручных правок.
- Для replay-набора все варианты с `worst_dd_pct > 0.50` автоматически исключаются.
- Рейтинг становится чувствительным к tail-концентрации: кандидаты с высоким убытком в одной паре опускаются относительно baseline.
- Все изменения проходят `make ci`.

## 9. Open Questions
- Какие точные дефолт-веса у компонентов soft-score (`PSR`, `DSR`, `worst_period`, `concentration`) фиксируем в `v1`?
- Какой `benchmark_sr` использовать для PSR по умолчанию?
- Какое значение `trials` использовать для DSR в разных режимах отбора?
- Включать ли `pair_stability_hysteresis` в итоговый score уже в `v1` или оставить только диагностикой до `v2`?