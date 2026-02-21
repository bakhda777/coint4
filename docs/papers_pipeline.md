# Papers Pipeline

## Что является входом
- Входной корпус: `papers/text/**/*.txt`.
- Это уже извлеченный текст статей; PDF/HTML парсинг в этот пайплайн не входит.
- Каждый файл читается устойчиво: сначала UTF-8, при ошибке fallback на latin-1 с заменой ошибок.

## Какие артефакты создаются
- Каталог: `papers/db.sqlite`
  - Таблица `papers` с метаданными, digest, тегами, приоритетом и дедуп-группами.
  - Таблица `runs` с историей запусков и статистикой.
  - FTS5 таблица `papers_fts` для полнотекстового поиска по title/digest/excerpt.
- Digest-файлы: `papers/digests/<paper_id>.txt`
- Приоритетная очередь: `papers/synthesis/priority_queue.jsonl`
- Отчет ingest: `papers/synthesis/ingest_report.md`
- Опциональные карточки LLM: `papers/cards/<paper_id>.json`
- Опциональный синтез из карточек:
  - `papers/synthesis/methods_overview.md`
  - `papers/synthesis/pitfalls.md`
  - `papers/synthesis/backlog_experiments.md`
  - `papers/synthesis/executive_summary.md`

## Инкрементальность и кэш
- Идентификация текста: `text_sha256` от нормализованного текста.
- Базовый `paper_id`: первые 12 символов `text_sha256`.
- На повторном запуске `index_texts.py` файл пропускается (`skipped`), если:
  - `text_sha256` не изменился,
  - версия схемы digest/tags совпадает,
  - digest-файл существует.
- Digest/tags пересчитываются только при изменении текста или версии схемы (`DIGEST_SCHEMA_VERSION`).
- Карточки (`make_cards.py`) кэшируются по `text_sha256 + schema_version`.
- Версия схемы карточек берётся из `papers/cards/card.schema.json` (`$id`), изменение `$id` форсирует пересчёт карточек.

## SQLite устойчивость
- Подключение к SQLite выполняется только через `connect_db()` из `scripts/papers/lib_pipeline.py`.
- Единые PRAGMA:
  - `journal_mode=WAL`
  - `synchronous=NORMAL`
  - `busy_timeout=10000`
  - `temp_store=MEMORY`
- Для write-секций используется lock-файл `papers/.locks/sqlite-writer.lock`.
- Записи в БД выполняются короткими транзакциями (`BEGIN IMMEDIATE` на короткий участок), чтобы не держать долгие блокировки.

## Скрипты
- `scripts/papers/index_texts.py`
  - Индексация txt-файлов в SQLite.
  - Эвристики title/year/doi/url.
  - Генерация digest и keyword hits.
  - Проставление тегов и обновление FTS.
- `scripts/papers/dedup.py`
  - Жесткий дедуп по `text_sha256`.
  - Мягкий дедуп по `normalized_title + year`.
  - Проставление `dup_group_id` и причин дублей.
- `scripts/papers/prioritize.py`
  - Считает `priority_score` 0..100.
  - Учитывает granular-сигналы: costs/funding/perpetual/borrow/fees/slippage, OOS/WFA/rolling/robustness, метрики и implementation density.
  - Штрафует review-like материалы без экспериментального сигнала.
  - Штрафует неканонические дубли.
  - Пишет `priority_queue.jsonl` с `sort_key` и подсчетом подскорингов.
- `scripts/papers/report.py`
  - Генерирует markdown-отчет с counts, top tags, top-10 приоритета и проблемными файлами.
- `scripts/papers/make_cards.py`
  - Опционально вызывает `codex exec` для top-N работ из очереди.
  - По умолчанию: `--top 30 --batch-size 5`.
  - Контекст на статью ограничен: `digest + excerpt + method/results/conclusion snippets` (не весь текст).
  - Валидирует JSON карточки по `papers/cards/card.schema.json`.
  - В конце печатает `generated/skipped/failed` и top-5 причин ошибок.
- `scripts/papers/synthesize.py`
  - Строит итоговые документы methods/pitfalls/backlog/executive summary из карточек.

## Рекомендуемый порядок запуска
1. `python3 scripts/papers/index_texts.py`
2. `python3 scripts/papers/dedup.py`
3. `python3 scripts/papers/prioritize.py`
4. `python3 scripts/papers/report.py`

## Опциональный этап LLM
1. `python3 scripts/papers/make_cards.py --top 30 --batch-size 5`
2. `python3 scripts/papers/synthesize.py`

## Устойчивость
- Пустые и слишком короткие файлы не ломают прогон; попадают в `parse_status` и в раздел проблемных.
- Ошибки конкретного файла логируются в `papers.parse_error`; конвейер продолжает обработку остальных файлов.
- Статистика каждого шага фиксируется в `runs.stats_json`.
