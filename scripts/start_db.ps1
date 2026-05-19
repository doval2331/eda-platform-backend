# Levanta Postgres y prepara tablas + usuario demo (Windows / PowerShell)
# Si falla la política de ejecución, usa: scripts\start_db.bat
# o: powershell -ExecutionPolicy Bypass -File .\scripts\start_db.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.docker.example" ".env"
    Write-Host "Creado .env desde .env.docker.example"
}

docker compose up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe scripts\wait_for_db.py
.\.venv\Scripts\python.exe scripts\seed_user.py

Write-Host ""
Write-Host "PostgreSQL en localhost:5432 — conecta DBeaver con usuario eda / DB eda_platform"
Write-Host "Arranca la API: .\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000"
