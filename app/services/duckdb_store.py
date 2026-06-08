from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.config import get_settings


def _db_path() -> Path:
    path = Path(get_settings().duckdb_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_db_path()))


def init_duckdb() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS run_registry (
                run_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                modality VARCHAR,
                reduction_method VARCHAR,
                seed INTEGER,
                n_samples INTEGER,
                outliers_count INTEGER,
                silhouette DOUBLE,
                davies_bouldin DOUBLE,
                n_clusters INTEGER,
                calinski_harabasz DOUBLE,
                ari DOUBLE,
                nmi DOUBLE,
                cluster_stability DOUBLE,
                noise_pct DOUBLE
            )
            """
        )
        registry_columns = {
            row[1]
            for row in con.execute("PRAGMA table_info('run_registry')").fetchall()
        }
        for column, sql_type in (
            ("calinski_harabasz", "DOUBLE"),
            ("ari", "DOUBLE"),
            ("nmi", "DOUBLE"),
            ("cluster_stability", "DOUBLE"),
            ("noise_pct", "DOUBLE"),
        ):
            if column not in registry_columns:
                con.execute(f"ALTER TABLE run_registry ADD COLUMN {column} {sql_type}")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS run_evidences (
                run_id VARCHAR,
                evidence_index INTEGER,
                evidence_id VARCHAR,
                preview VARCHAR,
                source VARCHAR,
                x DOUBLE,
                y DOUBLE,
                cluster_label INTEGER,
                incident_id VARCHAR,
                categoria VARCHAR,
                subcategoria VARCHAR,
                prioridad VARCHAR,
                servicio_afectado VARCHAR,
                canal_entrada VARCHAR,
                tiempo_resolucion_horas DOUBLE,
                sla_incumplido BOOLEAN,
                reaperturas DOUBLE,
                escalados DOUBLE,
                satisfaccion_usuario DOUBLE,
                coste_estimado DOUBLE,
                descripcion_corta VARCHAR,
                causa_raiz_simulada VARCHAR,
                synthetic_segment VARCHAR,
                sector VARCHAR,
                service_line VARCHAR,
                support_channel VARCHAR,
                segment VARCHAR,
                category VARCHAR,
                severity VARCHAR,
                status VARCHAR,
                assignment_group VARCHAR,
                affected_service VARCHAR,
                monthly_tickets DOUBLE,
                critical_incidents DOUBLE,
                avg_resolution_hours DOUBLE,
                resolution_minutes DOUBLE,
                sla_breach_rate DOUBLE,
                sla_breached BOOLEAN,
                operational_risk_score DOUBLE,
                business_impact_score DOUBLE,
                security_incidents DOUBLE,
                downtime_hours DOUBLE,
                customer_satisfaction DOUBLE
            )
            """
        )
        evidence_columns = {
            row[1]
            for row in con.execute("PRAGMA table_info('run_evidences')").fetchall()
        }
        incident_columns = {
            "incident_id": "VARCHAR",
            "categoria": "VARCHAR",
            "subcategoria": "VARCHAR",
            "prioridad": "VARCHAR",
            "servicio_afectado": "VARCHAR",
            "canal_entrada": "VARCHAR",
            "tiempo_resolucion_horas": "DOUBLE",
            "sla_incumplido": "BOOLEAN",
            "reaperturas": "DOUBLE",
            "escalados": "DOUBLE",
            "satisfaccion_usuario": "DOUBLE",
            "coste_estimado": "DOUBLE",
            "descripcion_corta": "VARCHAR",
            "causa_raiz_simulada": "VARCHAR",
            "synthetic_segment": "VARCHAR",
        }
        for column, dtype in incident_columns.items():
            if column not in evidence_columns:
                con.execute(f"ALTER TABLE run_evidences ADD COLUMN {column} {dtype}")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS selected_insights (
                run_id VARCHAR,
                user_id VARCHAR,
                insight_id VARCHAR,
                title VARCHAR,
                description VARCHAR,
                metric_label VARCHAR,
                metric_value DOUBLE,
                dimension VARCHAR,
                filter_kind VARCHAR,
                filter_value VARCHAR,
                selected_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_strategy_recommendations (
                run_id VARCHAR,
                strategy_id VARCHAR,
                strategy_type VARCHAR,
                recommendation VARCHAR,
                justification VARCHAR,
                variables_used VARCHAR,
                metric_or_criterion VARCHAR,
                priority VARCHAR,
                created_at VARCHAR,
                trace_id VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_cluster_samples (
                run_id VARCHAR,
                evidence_index INTEGER,
                evidence_id VARCHAR,
                preview VARCHAR,
                cluster_label INTEGER,
                category VARCHAR,
                categoria VARCHAR,
                severity VARCHAR,
                prioridad VARCHAR,
                affected_service VARCHAR,
                servicio_afectado VARCHAR,
                assignment_group VARCHAR,
                support_channel VARCHAR,
                sla_breached BOOLEAN,
                sla_incumplido BOOLEAN,
                sla_breach_rate DOUBLE,
                resolution_minutes DOUBLE,
                avg_resolution_hours DOUBLE,
                business_impact_score DOUBLE,
                operational_risk_score DOUBLE,
                causa_raiz_simulada VARCHAR,
                selection_rank INTEGER,
                selection_score DOUBLE,
                selection_criteria VARCHAR,
                sample_size_config INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_cluster_insights (
                run_id VARCHAR,
                cluster_insight_id VARCHAR,
                cluster_label INTEGER,
                cluster_name VARCHAR,
                summary VARCHAR,
                main_characteristics VARCHAR,
                highlighted_variables VARCHAR,
                possible_causes VARCHAR,
                recommendations VARCHAR,
                business_conclusion VARCHAR,
                sample_evidence_ids VARCHAR,
                sample_size INTEGER,
                risk_level VARCHAR,
                generated_at VARCHAR,
                trace_id VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_decisions (
                trace_id VARCHAR,
                run_id VARCHAR,
                agent_name VARCHAR,
                decision_type VARCHAR,
                model_name VARCHAR,
                parameters VARCHAR,
                variables_used VARCHAR,
                input_artifacts VARCHAR,
                prompt VARCHAR,
                response VARCHAR,
                created_at VARCHAR
            )
            """
        )
        existing_columns = {
            row[1]
            for row in con.execute("PRAGMA table_info('selected_insights')").fetchall()
        }
        if "user_id" not in existing_columns:
            con.execute("ALTER TABLE selected_insights ADD COLUMN user_id VARCHAR")
        con.execute(
            """
            CREATE OR REPLACE VIEW vw_run_kpis AS
            SELECT
                run_id,
                COUNT(*) AS evidence_count,
                AVG(monthly_tickets) AS avg_monthly_tickets,
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
                SUM(CASE WHEN cluster_label = -1 THEN 1 ELSE 0 END) AS outlier_count
            FROM run_evidences
            GROUP BY run_id
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW vw_cluster_summary AS
            SELECT
                run_id,
                cluster_label,
                COUNT(*) AS evidence_count,
                AVG(monthly_tickets) AS avg_monthly_tickets,
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
            GROUP BY run_id, cluster_label
            """
        )


REGISTRY_COLUMNS = [
    "run_id",
    "created_at",
    "modality",
    "reduction_method",
    "seed",
    "n_samples",
    "outliers_count",
    "silhouette",
    "davies_bouldin",
    "n_clusters",
    "calinski_harabasz",
    "ari",
    "nmi",
    "cluster_stability",
    "noise_pct",
]

EVIDENCE_COLUMNS = [
    "run_id",
    "evidence_index",
    "evidence_id",
    "preview",
    "source",
    "x",
    "y",
    "cluster_label",
    "sector",
    "service_line",
    "support_channel",
    "segment",
    "category",
    "severity",
    "status",
    "assignment_group",
    "affected_service",
    "monthly_tickets",
    "critical_incidents",
    "avg_resolution_hours",
    "resolution_minutes",
    "sla_breach_rate",
    "sla_breached",
    "operational_risk_score",
    "business_impact_score",
    "security_incidents",
    "downtime_hours",
    "customer_satisfaction",
]


def _insert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    columns: list[str],
    register_name: str,
) -> None:
    if df.empty:
        return
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para {table}: {', '.join(missing)}")
    subset = df[columns]
    column_list = ", ".join(columns)
    con.register(register_name, subset)
    con.execute(
        f"INSERT INTO {table} ({column_list}) SELECT {column_list} FROM {register_name}"
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc).replace(tzinfo=None)


def persist_run_detail(detail: dict[str, Any]) -> None:
    init_duckdb()
    result = detail["result"]
    metrics = result.get("metrics", {})
    run_id = detail["id"]

    registry_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at": _created_at(detail.get("created_at")),
                "modality": detail.get("modality"),
                "reduction_method": detail.get("reduction_method"),
                "seed": detail.get("seed"),
                "n_samples": detail.get("n_samples"),
                "outliers_count": detail.get("outliers_count"),
                "silhouette": _metric_value(metrics, "silhouette"),
                "davies_bouldin": _metric_value(metrics, "davies_bouldin"),
                "n_clusters": (
                    int(metrics["n_clusters"])
                    if metrics.get("n_clusters") is not None
                    else None
                ),
                "calinski_harabasz": _metric_value(metrics, "calinski_harabasz"),
                "ari": _metric_value(metrics, "ari"),
                "nmi": _metric_value(metrics, "nmi"),
                "cluster_stability": _metric_value(metrics, "cluster_stability"),
                "noise_pct": _metric_value(metrics, "noise_pct"),
            }
        ]
    )

    points = result.get("X_2d", [])
    labels = result.get("cluster_labels", [])
    metadata = result.get("metadata", [])
    evidence_rows: list[dict[str, Any]] = []
    for index, item in enumerate(metadata):
        point = points[index] if index < len(points) else [None, None]
        label = labels[index] if index < len(labels) else None
        evidence_rows.append(
            {
                "run_id": run_id,
                "evidence_index": index,
                "evidence_id": item.get("id"),
                "preview": item.get("preview"),
                "source": item.get("source"),
                "x": point[0] if point else None,
                "y": point[1] if point and len(point) > 1 else None,
                "cluster_label": label,
                "incident_id": item.get("incident_id"),
                "categoria": item.get("categoria"),
                "subcategoria": item.get("subcategoria"),
                "prioridad": item.get("prioridad"),
                "servicio_afectado": item.get("servicio_afectado"),
                "canal_entrada": item.get("canal_entrada"),
                "tiempo_resolucion_horas": item.get("tiempo_resolucion_horas"),
                "sla_incumplido": item.get("sla_incumplido"),
                "reaperturas": item.get("reaperturas"),
                "escalados": item.get("escalados"),
                "satisfaccion_usuario": item.get("satisfaccion_usuario"),
                "coste_estimado": item.get("coste_estimado"),
                "descripcion_corta": item.get("descripcion_corta"),
                "causa_raiz_simulada": item.get("causa_raiz_simulada"),
                "synthetic_segment": item.get("synthetic_segment"),
                "sector": item.get("sector"),
                "service_line": item.get("service_line"),
                "support_channel": item.get("support_channel"),
                "segment": item.get("segment"),
                "category": item.get("category"),
                "severity": item.get("severity"),
                "status": item.get("status"),
                "assignment_group": item.get("assignment_group"),
                "affected_service": item.get("affected_service"),
                "monthly_tickets": item.get("monthly_tickets"),
                "critical_incidents": item.get("critical_incidents"),
                "avg_resolution_hours": item.get("avg_resolution_hours"),
                "resolution_minutes": item.get("resolution_minutes"),
                "sla_breach_rate": item.get("sla_breach_rate"),
                "sla_breached": item.get("sla_breached"),
                "operational_risk_score": item.get("operational_risk_score"),
                "business_impact_score": item.get("business_impact_score"),
                "security_incidents": item.get("security_incidents"),
                "downtime_hours": item.get("downtime_hours"),
                "customer_satisfaction": item.get("customer_satisfaction"),
            }
        )
    evidences_df = pd.DataFrame(evidence_rows)

    with _connect() as con:
        con.execute("DELETE FROM run_registry WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM run_evidences WHERE run_id = ?", [run_id])
        con.register("registry_df", registry_df)
        con.execute("INSERT INTO run_registry SELECT * FROM registry_df")
        if not evidences_df.empty:
            con.register("evidences_df", evidences_df)
            evidence_columns_sql = ", ".join(evidences_df.columns)
            con.execute(
                f"""
                INSERT INTO run_evidences ({evidence_columns_sql})
                SELECT {evidence_columns_sql}
                FROM evidences_df
                """
            )
        _insert_dataframe(con, "run_registry", registry_df, REGISTRY_COLUMNS, "registry_df")
        _insert_dataframe(
            con, "run_evidences", evidences_df, EVIDENCE_COLUMNS, "evidences_df"
        )


def run_exists(run_id: str) -> bool:
    init_duckdb()
    with _connect() as con:
        count = con.execute(
            "SELECT COUNT(*) FROM run_evidences WHERE run_id = ?", [run_id]
        ).fetchone()[0]
    return bool(count)


def load_run_evidences(run_id: str) -> pd.DataFrame:
    init_duckdb()
    with _connect() as con:
        return con.execute(
            """
            SELECT *
            FROM run_evidences
            WHERE run_id = ?
            ORDER BY evidence_index
            """,
            [run_id],
        ).df()


def save_selected_insight(
    run_id: str, insight: dict[str, Any], *, user_id: str | None = None
) -> None:
    init_duckdb()
    row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "user_id": user_id,
                "insight_id": insight.get("id"),
                "title": insight.get("title"),
                "description": insight.get("description"),
                "metric_label": insight.get("metric_label"),
                "metric_value": insight.get("metric_value"),
                "dimension": insight.get("dimension"),
                "filter_kind": insight.get("filter_kind"),
                "filter_value": insight.get("filter_value"),
                "selected_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        ]
    )
    with _connect() as con:
        con.register("selected_df", row)
        con.execute(
            """
            DELETE FROM selected_insights
            WHERE run_id = ?
              AND insight_id = ?
              AND (user_id = ? OR user_id IS NULL)
            """,
            [run_id, insight.get("id"), user_id],
        )
        con.execute(
            """
            INSERT INTO selected_insights (
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
            )
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
            FROM selected_df
            """
        )


def list_selected_insights(
    *, run_id: str | None = None, user_id: str | None = None
) -> list[dict[str, Any]]:
    init_duckdb()
    filters = []
    params: list[Any] = []
    if run_id:
        filters.append("si.run_id = ?")
        params.append(run_id)
    if user_id:
        filters.append("(si.user_id = ? OR si.user_id IS NULL)")
        params.append(user_id)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with _connect() as con:
        df = con.execute(
            f"""
            SELECT
                si.run_id,
                si.insight_id AS id,
                si.title,
                si.description,
                si.metric_label,
                si.metric_value,
                si.dimension,
                si.filter_kind,
                si.filter_value,
                si.selected_at,
                rr.created_at AS run_created_at,
                rr.modality,
                rr.reduction_method,
                CAST(vk.evidence_count AS INTEGER) AS evidence_count,
                vk.avg_sla_breach_rate,
                vk.avg_resolution_hours,
                vk.avg_risk
            FROM selected_insights si
            LEFT JOIN run_registry rr ON rr.run_id = si.run_id
            LEFT JOIN vw_run_kpis vk ON vk.run_id = si.run_id
            {where_clause}
            ORDER BY si.selected_at DESC
            """,
            params,
        ).df()

    if df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def save_agent_recommendations(run_id: str, recommendations: pd.DataFrame) -> None:
    init_duckdb()
    _replace_run_rows(
        "agent_strategy_recommendations",
        run_id,
        recommendations,
        [
            "run_id",
            "strategy_id",
            "strategy_type",
            "recommendation",
            "justification",
            "variables_used",
            "metric_or_criterion",
            "priority",
            "created_at",
            "trace_id",
        ],
    )


def save_agent_cluster_samples(run_id: str, samples: pd.DataFrame) -> None:
    init_duckdb()
    _replace_run_rows(
        "agent_cluster_samples",
        run_id,
        samples,
        [
            "run_id",
            "evidence_index",
            "evidence_id",
            "preview",
            "cluster_label",
            "category",
            "categoria",
            "severity",
            "prioridad",
            "affected_service",
            "servicio_afectado",
            "assignment_group",
            "support_channel",
            "sla_breached",
            "sla_incumplido",
            "sla_breach_rate",
            "resolution_minutes",
            "avg_resolution_hours",
            "business_impact_score",
            "operational_risk_score",
            "causa_raiz_simulada",
            "selection_rank",
            "selection_score",
            "selection_criteria",
            "sample_size_config",
        ],
    )


def save_agent_cluster_insights(run_id: str, insights: pd.DataFrame) -> None:
    init_duckdb()
    _replace_run_rows(
        "agent_cluster_insights",
        run_id,
        insights,
        [
            "run_id",
            "cluster_insight_id",
            "cluster_label",
            "cluster_name",
            "summary",
            "main_characteristics",
            "highlighted_variables",
            "possible_causes",
            "recommendations",
            "business_conclusion",
            "sample_evidence_ids",
            "sample_size",
            "risk_level",
            "generated_at",
            "trace_id",
        ],
    )


def append_agent_decisions(decisions: pd.DataFrame) -> None:
    init_duckdb()
    _append_rows(
        "agent_decisions",
        decisions,
        [
            "trace_id",
            "run_id",
            "agent_name",
            "decision_type",
            "model_name",
            "parameters",
            "variables_used",
            "input_artifacts",
            "prompt",
            "response",
            "created_at",
        ],
    )


def list_agent_decisions(run_id: str) -> list[dict[str, Any]]:
    init_duckdb()
    with _connect() as con:
        df = con.execute(
            """
            SELECT *
            FROM agent_decisions
            WHERE run_id = ?
            ORDER BY created_at
            """,
            [run_id],
        ).df()
    if df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _replace_run_rows(
    table: str,
    run_id: str,
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    with _connect() as con:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
        _append_rows_with_connection(con, table, df, columns)


def _append_rows(table: str, df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        return
    with _connect() as con:
        _append_rows_with_connection(con, table, df, columns)


def _append_rows_with_connection(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    if df.empty:
        return
    work = df.copy()
    for column in columns:
        if column not in work:
            work[column] = None
    work = work[columns]
    register_name = f"{table}_df"
    con.register(register_name, work)
    column_list = ", ".join(columns)
    con.execute(
        f"INSERT INTO {table} ({column_list}) SELECT {column_list} FROM {register_name}"
    )
