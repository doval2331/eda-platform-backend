@echo off
setlocal
cd /d "%~dp0.."

if not exist ".env" (
    copy /Y ".env.docker.example" ".env" >nul
    echo Creado .env desde .env.docker.example
)

docker compose up -d
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" scripts\wait_for_db.py
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" scripts\seed_user.py
if errorlevel 1 exit /b 1

echo.
echo PostgreSQL en localhost:5432 - DBeaver: usuario eda, BD eda_platform, password eda_local_dev
echo API: .venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
endlocal
