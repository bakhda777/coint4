#!/bin/bash
# Скрипт для настройки PostgreSQL для Optuna оптимизации

set -e

echo "🚀 Настройка PostgreSQL для Optuna оптимизации..."

# Добавляем PostgreSQL в PATH
export PATH="/usr/local/opt/postgresql@15/bin:$PATH"

# Проверяем, установлен ли PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL не найден. Устанавливаем..."
    brew install postgresql@15
    export PATH="/usr/local/opt/postgresql@15/bin:$PATH"
fi

# Запускаем PostgreSQL сервис
echo "📊 Запускаем PostgreSQL сервис..."
brew services start postgresql@15

# Ждем запуска сервиса
sleep 5

# Создаем базу данных для Optuna
echo "🗄️ Создаем базу данных optuna_studies..."
createdb optuna_studies 2>/dev/null || echo "База данных optuna_studies уже существует"

# Создаем пользователя для оптимизации (опционально)
echo "👤 Настраиваем пользователя..."
psql -d optuna_studies -c "CREATE USER optuna_user WITH PASSWORD 'optuna_password';" 2>/dev/null || echo "Пользователь optuna_user уже существует"
psql -d optuna_studies -c "GRANT ALL PRIVILEGES ON DATABASE optuna_studies TO optuna_user;" 2>/dev/null || true

# Проверяем подключение
echo "🔍 Проверяем подключение к базе данных..."
psql -d optuna_studies -c "SELECT version();" > /dev/null

echo "✅ PostgreSQL успешно настроен!"
echo "📝 Строка подключения: postgresql://optuna_user:optuna_password@localhost:5432/optuna_studies"
echo "📝 Альтернативная строка: postgresql://localhost:5432/optuna_studies"

# Устанавливаем необходимые Python пакеты
echo "📦 Устанавливаем psycopg2-binary для Python..."
pip install psycopg2-binary

echo "🎉 Настройка завершена! Теперь можно запускать параллельную оптимизацию."
