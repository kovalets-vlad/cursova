# Використовуємо стабільний образ Bookworm
FROM python:3.10-slim-bookworm

# Забороняємо Python створювати .pyc файли та вмикаємо миттєвий вивід логів
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Встановлюємо лише необхідні системні пакети для збірки bcrypt/pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо залежності
COPY requirements.txt .

# Оновлюємо pip та встановлюємо бібліотеки
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код проєкту
COPY . .

# Відкриваємо порт
EXPOSE 8000

# Запуск бекенду
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]