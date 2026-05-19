#!/usr/bin/env bash
# Levanta Postgres y prepara tablas + usuario demo (Linux / macOS)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.docker.example .env
  echo "Creado .env desde .env.docker.example"
fi

docker compose up -d
python scripts/wait_for_db.py
python scripts/seed_user.py

echo ""
echo "PostgreSQL en localhost:5432 — DBeaver: usuario eda, DB eda_platform"
echo "API: uvicorn app.main:app --reload --port 8000"
