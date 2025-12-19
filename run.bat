@echo off
chcp 65001 > nul
echo [1/3] Запуск Docker контейнерів...
docker-compose up -d --build

echo [2/3] Очікування готовності сервісів...
timeout /t 5 /nobreak > nul

echo [3/3] Відкриття застосунку у браузері...
start http://localhost:5173

echo Усе готово! Натисніть будь-яку клавішу, щоб закрити це вікно.
pause