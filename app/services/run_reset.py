from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import AnalysisRun
from app.services.duckdb_store import clear_all_run_data, clear_run_data


BI_TABLES = (
    "bi_selected_insights",
    "bi_service_risk",
    "bi_sla_by_category",
    "bi_cluster_summary",
    "bi_evidences",
    "bi_runs",
)


def _clear_bi_tables_if_enabled() -> dict[str, int] | None:
    settings = get_settings()
    if not settings.bi_sync_enabled:
        return None
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.bi_database_url)
        cleared: dict[str, int] = {}
        with engine.begin() as conn:
            for table in BI_TABLES:
                try:
                    row = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    count = int(row[0]) if row else 0
                    conn.execute(text(f"DELETE FROM {table}"))
                    cleared[table] = count
                except Exception:
                    cleared[table] = 0
        return cleared
    except Exception:
        return None


def _delete_bi_run_if_enabled(run_id: str) -> dict[str, int] | None:
    settings = get_settings()
    if not settings.bi_sync_enabled:
        return None
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.bi_database_url)
        cleared: dict[str, int] = {}
        with engine.begin() as conn:
            for table in BI_TABLES:
                try:
                    row = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE run_id = :run_id"),
                        {"run_id": run_id},
                    ).fetchone()
                    count = int(row[0]) if row else 0
                    conn.execute(
                        text(f"DELETE FROM {table} WHERE run_id = :run_id"),
                        {"run_id": run_id},
                    )
                    cleared[table] = count
                except Exception:
                    cleared[table] = 0
        return cleared
    except Exception:
        return None


def delete_run(db: Session, run_id: str) -> dict[str, object]:
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise LookupError("Ejecucion no encontrada")
    db.delete(row)
    db.commit()
    duckdb_cleared = clear_run_data(run_id)
    bi_cleared = _delete_bi_run_if_enabled(run_id)
    return {
        "run_id": run_id,
        "duckdb_tables_cleared": duckdb_cleared,
        "bi_tables_cleared": bi_cleared,
    }


def reset_all_runs(db: Session) -> dict[str, object]:
    deleted_runs = db.query(AnalysisRun).count()
    db.query(AnalysisRun).delete()
    db.commit()
    duckdb_cleared = clear_all_run_data()
    bi_cleared = _clear_bi_tables_if_enabled()
    return {
        "deleted_runs": deleted_runs,
        "duckdb_tables_cleared": duckdb_cleared,
        "bi_tables_cleared": bi_cleared,
    }
