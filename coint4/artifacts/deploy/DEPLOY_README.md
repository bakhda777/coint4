# Coint2 Deployment Guide v0.1.1

Полный пакет для развертывания системы коинтеграционной торговли в production-окружении.

## Быстрый Старт

### 1. Docker Deployment (Рекомендуется)

```bash
# Клонировать репозиторий
git clone <repository-url>
cd coint2

# Настроить переменные окружения
cp .env.example .env
# Отредактировать .env для вашей среды

# Запустить в paper режиме
docker-compose up -d coint2-trader

# Проверить статус
docker-compose ps
docker-compose logs -f coint2-trader
```

### 2. Systemd Deployment (Linux)

```bash
# Создать пользователя
sudo useradd -r -s /bin/false -d /opt/coint2 coint2
sudo mkdir -p /opt/coint2
sudo chown coint2:coint2 /opt/coint2

# Скопировать файлы
sudo cp -r . /opt/coint2/
sudo chown -R coint2:coint2 /opt/coint2

# Установить systemd unit
sudo cp deploy/coint2.service /etc/systemd/system/
sudo systemctl daemon-reload

# Настроить logrotate
sudo cp deploy/coint2.logrotate /etc/logrotate.d/coint2

# Запустить сервис
sudo systemctl enable coint2
sudo systemctl start coint2
```

## Архитектура Deployment

### Компоненты Системы

1. **coint2-trader** - Основной торговый сервис
2. **postgres** - База данных для Optuna (опционально)
3. **redis** - Кэш для данных (опционально)
4. **monitor** - Система мониторинга (Prometheus)
5. **canary** - Paper trading rehearsal (по требованию)

### Файловая Структура

```
/opt/coint2/                    # Базовая директория
├── src/                        # Исходный код
├── configs/                    # Конфигурации
├── scripts/                    # Скрипты управления
├── data_downloaded/            # Исторические данные (RO)
├── artifacts/                  # Артефакты времени исполнения
│   ├── live/                   # Live торговля
│   │   ├── logs/              # Логи (ротируются)
│   │   ├── metrics/           # Метрики (ротируются)
│   │   └── trades/            # История сделок
│   ├── state/                 # Состояние системы
│   └── deploy/                # Deployment артефакты
└── .env                       # Переменные окружения
```

## Режимы Работы

### Paper Trading (По умолчанию)
```bash
# Docker
docker-compose up -d
# или
TRADING_MODE=paper docker-compose up -d

# Systemd
sudo systemctl start coint2
```

### Live Trading (Production)
```bash
# Убедиться что API ключи настроены в .env
TRADING_MODE=live docker-compose up -d

# Проверить preflight checks
docker-compose exec coint2-trader python scripts/run_preflight.py
```

### Dry Run (Тестирование)
```bash
TRADING_MODE=dry-run docker-compose up -d
```

## Preflight Проверки

Перед запуском live торговли **ОБЯЗАТЕЛЬНО** выполните preflight checks:

```bash
# В контейнере
docker-compose exec coint2-trader python scripts/run_preflight.py

# В systemd
sudo -u coint2 /opt/coint2/.venv/bin/python /opt/coint2/scripts/run_preflight.py

# Проверить отчет
cat artifacts/live/PREFLIGHT_REPORT.md
```

### Критерии Готовности

- ✅ Конфигурации валидны
- ✅ Данные доступны и свежие
- ✅ API соединения работают
- ✅ Риск-лимиты настроены
- ✅ Системные ресурсы достаточны
- ✅ Логирование функционирует

## Paper Canary Rehearsal

60-90 минутная репетиция перед live запуском:

```bash
# Запуск canary (отдельным профилем)
docker-compose --profile canary up canary

# Или прямой вызов
docker-compose exec coint2-trader python scripts/run_paper_canary.py \
  --duration-minutes 90 \
  --pair BTC/USDT

# Проверить результаты
cat artifacts/live/LIVE_DASHBOARD.md
```

## Мониторинг и Observability

### Логи
```bash
# Просмотр в реальном времени
docker-compose logs -f coint2-trader

# Systemd logs
sudo journalctl -u coint2 -f

# Структурированные логи
tail -f /opt/coint2/artifacts/live/logs/main.jsonl | jq
```

### Метрики
```bash
# Live snapshot
python scripts/extract_live_snapshot.py

# Trades index
cat artifacts/live/TRADES_INDEX.csv
```

### Health Checks
```bash
# Docker health
docker-compose exec coint2-trader python scripts/run_preflight.py

# HTTP endpoint (если доступен)
curl http://localhost:8080/health
```

## Конфигурация

### Основные Файлы

- **configs/prod.yaml** - Production конфигурация торговли
- **configs/risk.yaml** - Параметры риск-менеджмента
- **.env** - Переменные окружения и секреты

### Ключевые Параметры

```yaml
# configs/prod.yaml
data:
  timeframe: "15T"                    # 15-минутные бары
  
backtesting:
  normalization_method: "rolling_zscore"  # Production-safe
  commission_pct: 0.0008              # 0.08% комиссия
  slippage_pct: 0.0002               # 0.02% slippage

walk_forward:
  train_days: 90                      # 3 месяца тренировки
  test_days: 30                      # 1 месяц тестирования
  gap_minutes: 15                    # Минимальный разрыв
```

```yaml
# configs/risk.yaml
max_daily_loss_pct: 3.0             # Максимальная дневная потеря
max_drawdown_pct: 25.0              # Максимальная просадка
position_size_usd: 100              # Размер позиции
max_positions: 10                   # Максимум позиций
```

## Безопасность и Риски

### Risk Management
- **Stop Loss**: Автоматические стопы по просадке
- **Position Limits**: Ограничение размеров позиций  
- **Daily Limits**: Дневные лимиты потерь
- **Emergency Stop**: Аварийная остановка системы

### Security
- **Non-Root User**: Контейнер работает под непривилегированным пользователем
- **Read-Only Configs**: Конфигурации доступны только на чтение
- **Secret Management**: API ключи в переменных окружения
- **Network Isolation**: Изолированная Docker сеть

### Monitoring
- **Health Checks**: Автоматическая проверка работоспособности
- **Alerts**: Уведомления при критичных событиях
- **Log Rotation**: Автоматическая ротация логов
- **Backup**: Автоматическое резервное копирование состояния

## Troubleshooting

### Общие Проблемы

1. **Container не запускается**
   ```bash
   docker-compose logs coint2-trader
   # Проверить .env и конфигурации
   ```

2. **Preflight checks не проходят**
   ```bash
   # Проверить данные
   ls -la data_downloaded/
   # Проверить конфигурации
   python -c "from coint2.utils.config import load_config; print(load_config('configs/prod.yaml'))"
   ```

3. **No trades generated**
   ```bash
   # Проверить фильтры пар
   cat artifacts/live/logs/main.jsonl | jq '.msg' | grep -i pair
   # Снизить пороги фильтрации
   ```

4. **High memory usage**
   ```bash
   # Проверить кэш
   docker-compose exec coint2-trader python -c "import psutil; print(psutil.virtual_memory())"
   # Очистить кэш
   rm -rf artifacts/live/cache/*
   ```

### Логи и Диагностика

```bash
# Полная диагностика
python scripts/extract_live_snapshot.py --logs 200 --trades 20

# Проверка производительности
python scripts/benchmark_strategies.py

# Анализ сделок
python scripts/analyze_trades.py artifacts/live/TRADES_INDEX.csv
```

## Scaling и Production

### Horizontal Scaling
```yaml
# docker-compose.yml
deploy:
  replicas: 3
  resources:
    limits:
      cpus: '2.0'
      memory: 2G
```

### Monitoring Integration
```yaml
# Prometheus targets
- job_name: 'coint2'
  static_configs:
    - targets: ['coint2-trader:8080']
```

### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
tar -czf "backup-$(date +%Y%m%d_%H%M%S).tar.gz" \
  artifacts/live/ \
  configs/ \
  .env
```

## Support и Maintenance

### Обновление Системы
```bash
# Graceful restart
docker-compose exec coint2-trader python scripts/graceful_shutdown.py
docker-compose pull
docker-compose up -d

# Rollback if needed
docker-compose down
git checkout previous-version
docker-compose up -d
```

### Мониторинг Здоровья
```bash
# Ежедневная проверка
0 8 * * * /opt/coint2/scripts/daily_health_check.sh
```

### Log Analysis
```bash
# Анализ ошибок
cat artifacts/live/logs/main.jsonl | jq 'select(.level=="ERROR")'

# Анализ производительности
cat artifacts/live/logs/metrics.jsonl | jq '.value' | sort -n | tail -10
```

---

## Контакты и Поддержка

- **Documentation**: См. CLAUDE.md в корне репозитория
- **Issues**: GitHub Issues для багов и feature requests  
- **Monitoring**: Prometheus/Grafana dashboard на порту 9090
- **Alerts**: Настроить уведомления в .env (Slack/Telegram)

**Версия**: v0.1.1  
**Последнее обновление**: 2025-08-10  
**Режим**: Production Ready 🚀