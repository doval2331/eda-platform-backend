"""Espera a que PostgreSQL acepte conexiones (tras docker compose up -d)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text

from app.config import get_settings


def main(max_attempts: int = 30, interval_sec: float = 2.0) -> None:
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite"):
        print("SQLite: no hace falta esperar al contenedor.")
        return

    engine = create_engine(url)
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("PostgreSQL listo.")
            return
        except Exception as exc:
            print(f"Esperando PostgreSQL ({attempt}/{max_attempts}): {exc}")
            time.sleep(interval_sec)

    raise SystemExit(
        "PostgreSQL no respondió a tiempo. ¿Está `docker compose up -d` en ejecución?"
    )


if __name__ == "__main__":
    main()
