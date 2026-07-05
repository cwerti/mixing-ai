FROM python:3.12-slim

# Установка системных зависимостей для аудио
RUN apt-get update && apt-get install -y \
    libresample1-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка зависимостей
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Копируем код
COPY app /app/app

# Открываем порт для API (будущего)
EXPOSE 8000

# Пока просто держим контейнер запущенным
CMD ["python", "-m", "app.core.analyze_task"]
