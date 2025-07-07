@echo off
setlocal

REM — 1. Устанавливаем зависимости
echo [1/4] Installing requirements...
pip install -r requirements.txt

REM — 2. Переменные окружения для Flask
echo [2/4] Setting environment variables...
set FLASK_APP=app.py
set FLASK_ENV=development

REM — 3. Убиваем старый Flask, если слушает порт 5000
echo [3/4] Checking for existing Flask on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":5000" ^| find "LISTENING"') do (
    echo   Killing PID %%a...
    taskkill /PID %%a /F >nul 2>&1
)

REM — 4. Инициализируем/обновляем миграции и накат
echo [4/4] Running migrations...
python -m flask db init       2>nul
python -m flask db migrate -m "Auto migration"
python -m flask db upgrade

REM — 5. Запускаем сервер Flask в новом окне
echo Starting Flask server...
start "FlaskServer" cmd /k "python -m flask run"

endlocal
pause
