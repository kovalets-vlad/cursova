#!/bin/bash
export LANG=en_US.UTF-8

# Перевірка, чи доступний docker compose (V2) чи docker-compose (V1)
if docker compose version >/dev/null 2>&1; then
    DOCKER_CMD="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    DOCKER_CMD="docker-compose"
else
    echo "❌ Помилка: Docker не знайдено. Встановіть Docker Desktop."
    exit 1
fi

echo "------------------------------------------"
echo "🚀 Використовується команда: $DOCKER_CMD"
echo "------------------------------------------"

$DOCKER_CMD up -d --build

echo ""
echo "⏳ Очікування завантаження..."
sleep 5

# Відкриття браузера
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:5173
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:5173 2>/dev/null || echo "Відкрийте http://localhost:5173 вручну"
fi

echo "✅ Готово!"