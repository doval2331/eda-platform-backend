"""Vacia el historial de ejecuciones (PostgreSQL/SQLite + DuckDB)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.db import SessionLocal, init_db
from app.services.runs.run_reset import reset_all_runs


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        result = reset_all_runs(db)
    finally:
        db.close()

    deleted = result["deleted_runs"]
    print(f"Ejecuciones eliminadas: {deleted}")
    print("DuckDB:")
    for table, count in (result.get("duckdb_tables_cleared") or {}).items():
        print(f"  - {table}: {count} filas")
    bi = result.get("bi_tables_cleared")
    if bi is not None:
        print("Tablas BI:")
        for table, count in bi.items():
            print(f"  - {table}: {count} filas")
    print("Listo. Reinicia el frontend y ejecuta un pipeline nuevo.")


if __name__ == "__main__":
    main()
