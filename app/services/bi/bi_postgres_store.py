from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from app.config import get_settings
from app.services.bi.metabase_embed import embedding_is_configured
from app.services.runs.duckdb_store import _connect, init_duckdb


@dataclass
class BiSyncResult:
    status: str
    message: str
    tables: dict[str, int]


DEFAULT_BI_INSERT_CHUNK_SIZE = 1_000
MAX_POSTGRES_BIND_PARAMETERS = 60_000


def _settings():
    return get_settings()


def is_bi_sync_enabled() -> bool:
    return bool(_settings().bi_sync_enabled)


def _engine() -> Engine:
    return create_engine(_settings().bi_database_url, pool_pre_ping=True)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.where(pd.notnull(df), None)


def _bi_insert_chunksize(df: pd.DataFrame) -> int:
    column_count = max(len(df.columns), 1)
    safe_chunk = MAX_POSTGRES_BIND_PARAMETERS // column_count
    return max(1, min(DEFAULT_BI_INSERT_CHUNK_SIZE, safe_chunk))


def _read_duckdb(query: str, params: list[Any] | None = None) -> pd.DataFrame:
    init_duckdb()
    with _connect() as con:
        return _clean_df(con.execute(query, params or []).df())


def _align_df_to_table(df: pd.DataFrame, con: Connection, table_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    table_columns = _existing_columns(con, table_name)
    columns = [column for column in df.columns if column in table_columns]
    return df[columns] if columns else df.iloc[0:0]


def _existing_columns(con: Connection, table_name: str) -> set[str]:
    rows = con.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    )
    return {str(row[0]) for row in rows}


def init_bi_schema(engine: Engine | None = None) -> None:
    engine = engine or _engine()
    statements = [
        """
        CREATE TABLE IF NOT EXISTS bi_runs (
            run_id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            modality TEXT,
            reduction_method TEXT,
            seed INTEGER,
            n_samples INTEGER,
            outliers_count INTEGER,
            silhouette DOUBLE PRECISION,
            davies_bouldin DOUBLE PRECISION,
            n_clusters INTEGER,
            calinski_harabasz DOUBLE PRECISION,
            ari DOUBLE PRECISION,
            nmi DOUBLE PRECISION,
            cluster_stability DOUBLE PRECISION,
            noise_pct DOUBLE PRECISION
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bi_evidences (
            run_id TEXT,
            evidence_index INTEGER,
            evidence_id TEXT,
            preview TEXT,
            source TEXT,
            x DOUBLE PRECISION,
            y DOUBLE PRECISION,
            cluster_label INTEGER,
            incident_id TEXT,
            categoria TEXT,
            subcategoria TEXT,
            prioridad TEXT,
            servicio_afectado TEXT,
            canal_entrada TEXT,
            tiempo_resolucion_horas DOUBLE PRECISION,
            sla_incumplido BOOLEAN,
            reaperturas DOUBLE PRECISION,
            escalados DOUBLE PRECISION,
            satisfaccion_usuario DOUBLE PRECISION,
            coste_estimado DOUBLE PRECISION,
            descripcion_corta TEXT,
            causa_raiz_simulada TEXT,
            synthetic_segment TEXT,
            sector TEXT,
            service_line TEXT,
            support_channel TEXT,
            segment TEXT,
            category TEXT,
            severity TEXT,
            status TEXT,
            assignment_group TEXT,
            affected_service TEXT,
            monthly_tickets DOUBLE PRECISION,
            critical_incidents DOUBLE PRECISION,
            avg_resolution_hours DOUBLE PRECISION,
            resolution_minutes DOUBLE PRECISION,
            sla_breach_rate DOUBLE PRECISION,
            sla_breached BOOLEAN,
            operational_risk_score DOUBLE PRECISION,
            business_impact_score DOUBLE PRECISION,
            security_incidents DOUBLE PRECISION,
            downtime_hours DOUBLE PRECISION,
            customer_satisfaction DOUBLE PRECISION,
            priority TEXT,
            channel TEXT,
            reopen_count DOUBLE PRECISION,
            escalation_level DOUBLE PRECISION,
            customer_wait_minutes DOUBLE PRECISION,
            related_incidents_count DOUBLE PRECISION,
            short_description TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bi_cluster_summary (
            run_id TEXT,
            cluster_label INTEGER,
            evidence_count INTEGER,
            avg_monthly_tickets DOUBLE PRECISION,
            avg_sla_breach_rate DOUBLE PRECISION,
            avg_resolution_hours DOUBLE PRECISION,
            avg_risk DOUBLE PRECISION
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bi_sla_by_category (
            run_id TEXT,
            category TEXT,
            evidence_count INTEGER,
            sla_breached_count INTEGER,
            avg_sla_breach_rate DOUBLE PRECISION,
            avg_resolution_hours DOUBLE PRECISION,
            avg_risk DOUBLE PRECISION
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bi_service_risk (
            run_id TEXT,
            affected_service TEXT,
            evidence_count INTEGER,
            avg_sla_breach_rate DOUBLE PRECISION,
            avg_resolution_hours DOUBLE PRECISION,
            avg_risk DOUBLE PRECISION,
            avg_business_impact DOUBLE PRECISION,
            total_security_incidents DOUBLE PRECISION,
            total_downtime_hours DOUBLE PRECISION
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bi_selected_insights (
            selected_key TEXT PRIMARY KEY,
            run_id TEXT,
            user_id TEXT,
            insight_id TEXT,
            title TEXT,
            description TEXT,
            metric_label TEXT,
            metric_value DOUBLE PRECISION,
            dimension TEXT,
            filter_kind TEXT,
            filter_value TEXT,
            selected_at TIMESTAMP
        )
        """,
    ]
    with engine.begin() as con:
        for statement in statements:
            con.execute(text(statement))
        evidence_extra_columns = {
            "incident_id": "TEXT",
            "categoria": "TEXT",
            "subcategoria": "TEXT",
            "prioridad": "TEXT",
            "servicio_afectado": "TEXT",
            "canal_entrada": "TEXT",
            "tiempo_resolucion_horas": "DOUBLE PRECISION",
            "sla_incumplido": "BOOLEAN",
            "reaperturas": "DOUBLE PRECISION",
            "escalados": "DOUBLE PRECISION",
            "satisfaccion_usuario": "DOUBLE PRECISION",
            "coste_estimado": "DOUBLE PRECISION",
            "descripcion_corta": "TEXT",
            "causa_raiz_simulada": "TEXT",
            "synthetic_segment": "TEXT",
            "priority": "TEXT",
            "channel": "TEXT",
            "reopen_count": "DOUBLE PRECISION",
            "escalation_level": "DOUBLE PRECISION",
            "customer_wait_minutes": "DOUBLE PRECISION",
            "related_incidents_count": "DOUBLE PRECISION",
            "short_description": "TEXT",
        }
        evidence_columns = _existing_columns(con, "bi_evidences")
        for column, dtype in evidence_extra_columns.items():
            if column not in evidence_columns:
                con.execute(
                    text(
                        f"ALTER TABLE bi_evidences ADD COLUMN {column} {dtype}"
                    )
                )
        con.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_bi_evidences_run "
                "ON bi_evidences(run_id)"
            )
        )
        con.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_bi_evidences_cluster "
                "ON bi_evidences(run_id, cluster_label)"
            )
        )
        con.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_bi_selected_run "
                "ON bi_selected_insights(run_id)"
            )
        )
        run_columns = _existing_columns(con, "bi_runs")
        for column, sql_type in (
            ("calinski_harabasz", "DOUBLE PRECISION"),
            ("ari", "DOUBLE PRECISION"),
            ("nmi", "DOUBLE PRECISION"),
            ("cluster_stability", "DOUBLE PRECISION"),
            ("noise_pct", "DOUBLE PRECISION"),
        ):
            if column not in run_columns:
                con.execute(
                    text(
                        f"ALTER TABLE bi_runs ADD COLUMN {column} {sql_type}"
                    )
                )


def _run_filter(column: str, run_id: str | None) -> tuple[str, list[Any]]:
    if not run_id:
        return "", []
    return f"WHERE {column} = ?", [run_id]


def _load_bi_frames(run_id: str | None = None) -> dict[str, pd.DataFrame]:
    run_where, params = _run_filter("run_id", run_id)
    evidence_where, evidence_params = _run_filter("run_id", run_id)

    frames = {
        "bi_runs": _read_duckdb(
            f"""
            SELECT *
            FROM run_registry
            {run_where}
            """,
            params,
        ),
        "bi_evidences": _read_duckdb(
            f"""
            SELECT *
            FROM run_evidences
            {evidence_where}
            """,
            evidence_params,
        ),
        "bi_cluster_summary": _read_duckdb(
            f"""
            SELECT
                run_id,
                cluster_label,
                CAST(evidence_count AS INTEGER) AS evidence_count,
                avg_monthly_tickets,
                avg_sla_breach_rate,
                avg_resolution_hours,
                avg_risk
            FROM vw_cluster_summary
            {run_where}
            """,
            params,
        ),
        "bi_sla_by_category": _read_duckdb(
            f"""
            SELECT
                run_id,
                COALESCE(categoria, category, sector, 'Sin categoria') AS category,
                CAST(COUNT(*) AS INTEGER) AS evidence_count,
                CAST(SUM(CASE WHEN COALESCE(sla_incumplido, sla_breached) THEN 1 ELSE 0 END) AS INTEGER)
                    AS sla_breached_count,
                AVG(
                    COALESCE(
                        sla_breach_rate,
                        CASE
                            WHEN sla_incumplido IS NULL THEN NULL
                            WHEN sla_incumplido THEN 1.0
                            ELSE 0.0
                        END
                    )
                ) AS avg_sla_breach_rate,
                AVG(COALESCE(tiempo_resolucion_horas, avg_resolution_hours)) AS avg_resolution_hours,
                AVG(operational_risk_score) AS avg_risk
            FROM run_evidences
            {evidence_where}
            GROUP BY run_id, COALESCE(categoria, category, sector, 'Sin categoria')
            """,
            evidence_params,
        ),
        "bi_service_risk": _read_duckdb(
            f"""
            SELECT
                run_id,
                COALESCE(servicio_afectado, affected_service, service_line, 'Sin servicio') AS affected_service,
                CAST(COUNT(*) AS INTEGER) AS evidence_count,
                AVG(
                    COALESCE(
                        sla_breach_rate,
                        CASE
                            WHEN sla_incumplido IS NULL THEN NULL
                            WHEN sla_incumplido THEN 1.0
                            ELSE 0.0
                        END
                    )
                ) AS avg_sla_breach_rate,
                AVG(COALESCE(tiempo_resolucion_horas, avg_resolution_hours)) AS avg_resolution_hours,
                AVG(operational_risk_score) AS avg_risk,
                AVG(business_impact_score) AS avg_business_impact,
                SUM(security_incidents) AS total_security_incidents,
                SUM(downtime_hours) AS total_downtime_hours
            FROM run_evidences
            {evidence_where}
            GROUP BY run_id, COALESCE(servicio_afectado, affected_service, service_line, 'Sin servicio')
            """,
            evidence_params,
        ),
    }

    selected_where, selected_params = _run_filter("run_id", run_id)
    selected = _read_duckdb(
        f"""
        SELECT
            run_id,
            user_id,
            insight_id,
            title,
            description,
            metric_label,
            metric_value,
            dimension,
            filter_kind,
            filter_value,
            selected_at
        FROM selected_insights
        {selected_where}
        """,
        selected_params,
    )
    if not selected.empty:
        selected["selected_key"] = selected.apply(
            lambda row: (
                f"{row.get('run_id')}:"
                f"{row.get('user_id') or 'anon'}:"
                f"{row.get('insight_id')}"
            ),
            axis=1,
        )
        selected = selected[
            [
                "selected_key",
                "run_id",
                "user_id",
                "insight_id",
                "title",
                "description",
                "metric_label",
                "metric_value",
                "dimension",
                "filter_kind",
                "filter_value",
                "selected_at",
            ]
        ]
    frames["bi_selected_insights"] = selected
    return frames


def sync_bi_tables(run_id: str | None = None, *, force: bool = False) -> BiSyncResult:
    if not force and not is_bi_sync_enabled():
        return BiSyncResult(
            status="disabled",
            message="BI sync deshabilitado. Activar BI_SYNC_ENABLED=true.",
            tables={},
        )

    engine = _engine()
    init_bi_schema(engine)
    frames = _load_bi_frames(run_id=run_id)

    with engine.begin() as con:
        if run_id:
            for table in (
                "bi_selected_insights",
                "bi_service_risk",
                "bi_sla_by_category",
                "bi_cluster_summary",
                "bi_evidences",
                "bi_runs",
            ):
                con.execute(text(f"DELETE FROM {table} WHERE run_id = :run_id"), {"run_id": run_id})
        else:
            for table in (
                "bi_selected_insights",
                "bi_service_risk",
                "bi_sla_by_category",
                "bi_cluster_summary",
                "bi_evidences",
                "bi_runs",
            ):
                con.execute(text(f"DELETE FROM {table}"))

        counts: dict[str, int] = {}
        for table, df in frames.items():
            counts[table] = len(df)
            if not df.empty:
                df = _align_df_to_table(df, con, table)
                df.to_sql(
                    table,
                    con,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=_bi_insert_chunksize(df),
                )

    return BiSyncResult(
        status="ok",
        message="Tablas BI sincronizadas en PostgreSQL para Metabase.",
        tables=counts,
    )


def try_sync_bi_tables(run_id: str | None = None) -> BiSyncResult:
    try:
        return sync_bi_tables(run_id=run_id)
    except Exception as exc:  # pragma: no cover - evita romper el pipeline local.
        return BiSyncResult(
            status="error",
            message=f"No se pudo sincronizar PostgreSQL BI: {exc}",
            tables={},
        )


def _bi_table_counts(con) -> dict[str, int]:
    tables = (
        "bi_runs",
        "bi_evidences",
        "bi_cluster_summary",
        "bi_sla_by_category",
        "bi_service_risk",
        "bi_selected_insights",
    )
    counts: dict[str, int] = {}
    for table in tables:
        exists = con.execute(text("SELECT to_regclass(:table)"), {"table": table}).scalar()
        if not exists:
            counts[table] = 0
            continue
        counts[table] = int(con.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0)
    return counts


def get_bi_status() -> dict[str, Any]:
    settings = _settings()
    status: dict[str, Any] = {
        "enabled": settings.bi_sync_enabled,
        "metabase_url": settings.metabase_url,
        "dashboard_url": settings.metabase_dashboard_url or None,
        "postgres_status": "disabled" if not settings.bi_sync_enabled else "unknown",
        "detail": None,
        "tables": {},
    }
    if not settings.bi_sync_enabled:
        status["detail"] = "BI_SYNC_ENABLED=false"
        status["embedding_configured"] = embedding_is_configured()
        return status

    try:
        engine = _engine()
        with engine.begin() as con:
            con.execute(text("SET LOCAL lock_timeout = '750ms'"))
            con.execute(text("SELECT 1"))
            status["tables"] = _bi_table_counts(con)
        status["postgres_status"] = "ok"
        status["detail"] = "PostgreSQL BI disponible"
    except Exception as exc:  # pragma: no cover - depende de servicio externo.
        status["postgres_status"] = "error"
        status["detail"] = str(exc)
    status["embedding_configured"] = embedding_is_configured()
    return status
