# Dockerfile для coint2 production deployment
FROM python:3.11-slim

# Метаданные
LABEL maintainer="coint2-team"
LABEL version="0.1.1"
LABEL description="Cointegration pairs trading system"

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Poetry установка
ENV POETRY_HOME="/opt/poetry" \
    POETRY_PATH="/opt/poetry/bin/poetry" \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Копируем исходный код
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Устанавливаем проект
RUN poetry install --without dev

# Создаем директории для данных и артефактов
RUN mkdir -p data_downloaded artifacts/live/logs artifacts/live/metrics

# Пользователь для безопасности (не root)
RUN useradd -r -s /bin/false coint2user && \
    chown -R coint2user:coint2user /app
USER coint2user

# Порты (если будет веб-интерфейс)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=5m --retries=3 \
  CMD python scripts/run_preflight.py || exit 1

# Переменные окружения по умолчанию
ENV PYTHONPATH=/app/src
ENV COINT2_ENV=production
ENV COINT2_LOG_LEVEL=INFO

# Точка входа
ENTRYPOINT ["poetry", "run"]
CMD ["python", "scripts/run_live.py", "--mode", "paper", "--config", "configs/prod.yaml"]