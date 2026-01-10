#!/bin/bash
# Скрипт для настройки PostgreSQL для Optuna оптимизации

echo "🐘 Настройка PostgreSQL для Optuna"

# Проверяем установлен ли PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL не установлен. Установите его командой:"
    echo "   macOS: brew install postgresql"
    echo "   Ubuntu: sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

# Параметры подключения
DB_NAME="optuna_coint"
DB_USER="optuna_user"
DB_PASSWORD="optuna_pass"
DB_HOST="localhost"
DB_PORT="5432"

echo "📝 Создание базы данных и пользователя..."

# Создаем пользователя и базу данных
sudo -u postgres psql <<EOF
-- Создаем пользователя если не существует
DO
\$do\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_user
      WHERE usename = '${DB_USER}') THEN
      CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
   END IF;
END
\$do\$;

-- Создаем базу данных если не существует
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec

-- Даем все права пользователю
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
EOF

echo "⚙️ Оптимизация параметров PostgreSQL для Optuna..."

# Настройка оптимальных параметров для Optuna
sudo -u postgres psql -d ${DB_NAME} <<EOF
-- Оптимизация для параллельной работы
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- Оптимизация для SSD
ALTER SYSTEM SET random_page_cost = 1.0;

-- Включаем параллельные запросы
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;
EOF

echo "🔄 Перезапуск PostgreSQL для применения настроек..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew services restart postgresql
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo systemctl restart postgresql
fi

echo "✅ PostgreSQL настроен!"
echo ""
echo "📌 Параметры подключения:"
echo "   Database: ${DB_NAME}"
echo "   User: ${DB_USER}"
echo "   Password: ${DB_PASSWORD}"
echo "   Host: ${DB_HOST}"
echo "   Port: ${DB_PORT}"
echo ""
echo "🔗 Connection string для Optuna:"
echo "   postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo ""
echo "💡 Добавьте в .env файл:"
echo "   OPTUNA_DB_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"