import json
from datetime import datetime, timezone
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import get_settings
from app.db import AnalysisRun, SelectedInsight, User, get_db, run_to_detail, save_run
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDashboardResponse,
    DatasetProfileResponse,
    HealthResponse,
    InsightDashboardItem,
    InsightPayload,
    InsightSelectRequest,
    InsightSelectResponse,
    PipelineMetrics,
    RunCreateBody,
    RunDetail,
    RunSummary,
)
from app.services.dataset_store import get_dataset_meta, save_upload
from app.services.pipeline import run_pipeline

router = APIRouter()


def _metrics_from_row(row: AnalysisRun) -> PipelineMetrics:
    sil = float(row.silhouette) if row.silhouette else None
    dbi = float(row.davies_bouldin) if row.davies_bouldin else None
    n_clusters = None
    try:
        payload = json.loads(row.result_json)
        n_clusters = payload.get("metrics", {}).get("n_clusters")
        if n_clusters is not None:
            n_clusters = int(n_clusters)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if n_clusters is None:
        try:
            labels = json.loads(row.result_json).get("cluster_labels", [])
            n_clusters = len({int(x) for x in labels if int(x) >= 0})
        except (json.JSONDecodeError, TypeError, ValueError):
            n_clusters = None
    return PipelineMetrics(
        silhouette=sil,
        davies_bouldin=dbi,
        n_clusters=n_clusters,
    )


def _run_result(row: AnalysisRun) -> dict:
    try:
        return json.loads(row.result_json)
    except json.JSONDecodeError:
        return {}


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values) -> float | None:
    numbers = [_number(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _top_value(items: list[dict], *fields: str) -> str | None:
    counts: dict[str, int] = {}
    for item in items:
        for field in fields:
            value = item.get(field)
            if value:
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
                break
    if not counts:
        return None
    return sorted(counts.items(), key=lambda entry: entry[1], reverse=True)[0][0]


def _filtered_metadata(row: AnalysisRun, insight: InsightPayload) -> list[dict]:
    result = _run_result(row)
    metadata = result.get("metadata") or []
    labels = result.get("cluster_labels") or []

    if insight.filter_kind == "cluster_label" and insight.filter_value is not None:
        try:
            target = int(insight.filter_value)
        except ValueError:
            return []
        return [
            item
            for index, item in enumerate(metadata)
            if index < len(labels) and int(labels[index]) == target
        ]

    return [item for item in metadata if isinstance(item, dict)]


def _insight_stats(row: AnalysisRun, insight: InsightPayload) -> dict:
    items = _filtered_metadata(row, insight)
    return {
        "evidence_count": len(items) if items else None,
        "avg_sla_breach_rate": _avg(item.get("sla_breach_rate") for item in items),
        "avg_risk": _avg(item.get("operational_risk_score") for item in items),
    }


def _selected_to_dashboard_item(
    selected: SelectedInsight,
    run: AnalysisRun,
) -> InsightDashboardItem:
    insight = InsightPayload(
        id=selected.id,
        title=selected.title,
        description=selected.description,
        metric_label=selected.metric_label,
        metric_value=_number(selected.metric_value),
        dimension=selected.dimension,
        filter_kind=selected.filter_kind,
        filter_value=selected.filter_value,
    )
    return InsightDashboardItem(
        **insight.model_dump(),
        run_id=run.id,
        selected_at=selected.selected_at,
        run_created_at=run.created_at,
        modality=run.modality,  # type: ignore[arg-type]
        reduction_method=run.reduction_method,  # type: ignore[arg-type]
        **_insight_stats(run, insight),
    )


def _cluster_summaries(row: AnalysisRun) -> list[dict]:
    result = _run_result(row)
    labels = result.get("cluster_labels") or []
    metadata = result.get("metadata") or []
    grouped: dict[int, list[dict]] = {}
    for index, label in enumerate(labels):
        if index >= len(metadata):
            continue
        grouped.setdefault(int(label), []).append(metadata[index])

    summaries = []
    for label, items in grouped.items():
        avg_sla = _avg(item.get("sla_breach_rate") for item in items)
        avg_risk = _avg(item.get("operational_risk_score") for item in items)
        service = _top_value(items, "affected_service", "service_line", "sector", "category")
        name = "Casos atipicos" if label == -1 else f"Cluster {label}"
        if service:
            name = f"{name} - {service}"
        score = (avg_sla or 0) * 100 + (avg_risk or 0) + len(items) * 0.05
        summaries.append(
            {
                "cluster_label": label,
                "name": name,
                "count": len(items),
                "avg_sla": avg_sla,
                "avg_risk": avg_risk,
                "service": service,
                "score": score,
            }
        )
    return summaries


def _summary_to_insight(run_id: str, summary: dict, kind: str) -> InsightPayload:
    if kind == "sla":
        metric_label = "sla_breach_rate"
        metric_value = summary.get("avg_sla")
        title = f"SLA destacado en {summary['name']}"
    elif kind == "risk":
        metric_label = "operational_risk_score"
        metric_value = summary.get("avg_risk")
        title = f"Riesgo destacado en {summary['name']}"
    elif kind == "volume":
        metric_label = "evidence_count"
        metric_value = float(summary.get("count") or 0)
        title = f"Volumen destacado en {summary['name']}"
    else:
        metric_label = "cluster_priority_score"
        metric_value = summary.get("score")
        title = f"Cluster prioritario: {summary['name']}"

    sla = summary.get("avg_sla")
    risk = summary.get("avg_risk")
    if sla is not None:
        description = (
            f"{summary['name']} concentra {summary['count']} evidencias, "
            f"SLA promedio {sla * 100:.1f}%"
        )
    else:
        description = f"{summary['name']} concentra {summary['count']} evidencias"
    if risk is not None:
        description += f" y riesgo promedio {risk:.1f}."
    else:
        description += "."

    return InsightPayload(
        id=f"chat-{run_id}-{kind}-{summary['cluster_label']}",
        title=title,
        description=description,
        metric_label=metric_label,
        metric_value=metric_value,
        dimension="cluster_label",
        filter_kind="cluster_label",
        filter_value=str(summary["cluster_label"]),
    )


BI_TABLES = {
    "bi_runs": "SELECT COUNT(*) FROM bi_runs",
    "bi_evidences": "SELECT COUNT(*) FROM bi_evidences",
    "bi_cluster_summary": "SELECT COUNT(*) FROM bi_cluster_summary",
    "bi_sla_by_category": "SELECT COUNT(*) FROM bi_sla_by_category",
    "bi_service_risk": "SELECT COUNT(*) FROM bi_service_risk",
    "bi_selected_insights": "SELECT COUNT(*) FROM bi_selected_insights",
}

METABASE_CARD_DEFINITIONS = [
    {
        "name": "TFM - SLA por categoria",
        "description": "Porcentaje de tickets con SLA incumplido por categoria.",
        "display": "bar",
        "query": """
            SELECT category, ticket_count, ROUND(sla_breach_rate::numeric, 2) AS sla_breach_rate
            FROM bi_sla_by_category
            ORDER BY sla_breach_rate DESC
        """,
        "visualization_settings": {
            "graph.dimensions": ["category"],
            "graph.metrics": ["sla_breach_rate"],
        },
    },
    {
        "name": "TFM - Riesgo por servicio",
        "description": "Servicios con mayor impacto de negocio y volumen de tickets.",
        "display": "bar",
        "query": """
            SELECT
                affected_service,
                ticket_count,
                ROUND(avg_business_impact::numeric, 2) AS avg_business_impact
            FROM bi_service_risk
            ORDER BY avg_business_impact DESC, ticket_count DESC
            LIMIT 10
        """,
        "visualization_settings": {
            "graph.dimensions": ["affected_service"],
            "graph.metrics": ["avg_business_impact"],
        },
    },
    {
        "name": "TFM - Clusters prioritarios",
        "description": "Resumen de clusters con SLA, impacto y servicio dominante.",
        "display": "table",
        "query": """
            SELECT
                cluster_label,
                cluster_name,
                size,
                ROUND(sla_breach_rate::numeric, 2) AS sla_breach_rate,
                ROUND(avg_business_impact::numeric, 2) AS avg_business_impact,
                top_category,
                top_service
            FROM bi_cluster_summary
            ORDER BY sla_breach_rate DESC, avg_business_impact DESC
        """,
        "visualization_settings": {},
    },
    {
        "name": "TFM - Tickets por severidad",
        "description": "Distribucion de evidencias por severidad.",
        "display": "bar",
        "query": """
            SELECT severity, COUNT(*)::integer AS ticket_count
            FROM bi_evidences
            GROUP BY severity
            ORDER BY ticket_count DESC
        """,
        "visualization_settings": {
            "graph.dimensions": ["severity"],
            "graph.metrics": ["ticket_count"],
        },
    },
    {
        "name": "TFM - Insights seleccionados",
        "description": "Insights guardados desde la interpretacion guiada.",
        "display": "table",
        "query": """
            SELECT selected_at, title, metric_label, metric_value, dimension, filter_value
            FROM bi_selected_insights
            ORDER BY selected_at DESC
            LIMIT 25
        """,
        "visualization_settings": {},
    },
]


def _bi_engine():
    settings = get_settings()
    return create_engine(settings.bi_database_url, pool_pre_ping=True)


def _metabase_health(url: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/api/health", timeout=3) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def _metabase_api(settings, method: str, path: str, payload: dict | None = None, token: str | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Metabase-Session"] = token

    request = Request(
        f"{settings.metabase_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Metabase {method} {path} respondio {exc.code}: {detail}") from exc
    except (OSError, URLError) as exc:
        raise RuntimeError(f"No se pudo consultar Metabase {method} {path}: {exc}") from exc

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Metabase {method} {path} devolvio JSON invalido") from exc


def _metabase_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _metabase_login(settings) -> str:
    if not settings.metabase_username or not settings.metabase_password:
        raise RuntimeError(
            "Faltan METABASE_USERNAME y METABASE_PASSWORD en la configuracion del backend"
        )
    result = _metabase_api(
        settings,
        "POST",
        "/api/session",
        {
            "username": settings.metabase_username,
            "password": settings.metabase_password,
        },
    )
    token = result.get("id") if isinstance(result, dict) else None
    if not token:
        raise RuntimeError("Metabase no devolvio una sesion valida")
    return str(token)


def _metabase_ensure_database(settings, token: str) -> int:
    databases = _metabase_items(_metabase_api(settings, "GET", "/api/database", token=token))
    for database in databases:
        if database.get("name") == settings.metabase_database_name and not database.get("archived"):
            return int(database["id"])

    created = _metabase_api(
        settings,
        "POST",
        "/api/database",
        {
            "engine": "postgres",
            "name": settings.metabase_database_name,
            "details": {
                "host": settings.metabase_pg_host,
                "port": settings.metabase_pg_port,
                "dbname": settings.metabase_pg_dbname,
                "user": settings.metabase_pg_user,
                "password": settings.metabase_pg_password,
                "ssl": False,
                "tunnel_enabled": False,
            },
            "is_full_sync": True,
            "is_on_demand": False,
            "auto_run_queries": True,
        },
        token=token,
    )
    database_id = int(created["id"])
    try:
        _metabase_api(settings, "POST", f"/api/database/{database_id}/sync_schema", token=token)
    except RuntimeError:
        pass
    return database_id


def _metabase_ensure_dashboard(settings, token: str) -> dict:
    dashboard = _metabase_find_dashboard(settings, token)
    if dashboard:
        return dashboard

    created = _metabase_api(
        settings,
        "POST",
        "/api/dashboard",
        {
            "name": settings.metabase_dashboard_name,
            "description": "Dashboard BI generado desde Plataforma EDA.",
            "parameters": [],
        },
        token=token,
    )
    if not isinstance(created, dict) or not created.get("id"):
        raise RuntimeError("Metabase no devolvio el dashboard creado")
    return created


def _metabase_find_dashboard(settings, token: str) -> dict | None:
    dashboards = _metabase_items(_metabase_api(settings, "GET", "/api/dashboard", token=token))
    for dashboard in dashboards:
        if dashboard.get("name") == settings.metabase_dashboard_name and not dashboard.get("archived"):
            return dashboard
    return None


def _metabase_dashboard_url(settings, token: str) -> str | None:
    dashboard = _metabase_find_dashboard(settings, token)
    if dashboard:
        return f"{settings.metabase_url.rstrip('/')}/dashboard/{dashboard['id']}"
    return None


def _metabase_ensure_cards(settings, token: str, database_id: int) -> list[dict]:
    existing_cards = _metabase_items(_metabase_api(settings, "GET", "/api/card", token=token))
    by_name = {
        card.get("name"): card
        for card in existing_cards
        if card.get("name") and not card.get("archived")
    }
    cards = []
    for definition in METABASE_CARD_DEFINITIONS:
        card = by_name.get(definition["name"])
        if card and int(card.get("database_id") or 0) == database_id:
            cards.append(card)
            continue

        created = _metabase_api(
            settings,
            "POST",
            "/api/card",
            {
                "name": definition["name"],
                "description": definition["description"],
                "display": definition["display"],
                "dataset_query": {
                    "database": database_id,
                    "type": "native",
                    "native": {"query": " ".join(definition["query"].split())},
                },
                "visualization_settings": definition["visualization_settings"],
                "parameters": [],
            },
            token=token,
        )
        if not isinstance(created, dict) or not created.get("id"):
            raise RuntimeError(f"Metabase no devolvio la tarjeta creada: {definition['name']}")
        cards.append(created)
    return cards


def _metabase_layout_cards(cards: list[dict]) -> list[dict]:
    positions = [
        {"row": 0, "col": 0, "size_x": 12, "size_y": 7},
        {"row": 0, "col": 12, "size_x": 12, "size_y": 7},
        {"row": 7, "col": 0, "size_x": 24, "size_y": 8},
        {"row": 15, "col": 0, "size_x": 12, "size_y": 7},
        {"row": 15, "col": 12, "size_x": 12, "size_y": 7},
    ]
    layout = []
    for index, card in enumerate(cards):
        position = positions[index]
        layout.append(
            {
                "id": -(index + 1),
                "card_id": int(card["id"]),
                "row": position["row"],
                "col": position["col"],
                "size_x": position["size_x"],
                "size_y": position["size_y"],
                "parameter_mappings": [],
                "series": [],
            }
        )
    return layout


def _metabase_attach_cards(settings, token: str, dashboard_id: int, cards: list[dict]) -> None:
    _metabase_api(
        settings,
        "PUT",
        f"/api/dashboard/{dashboard_id}/cards",
        {"cards": _metabase_layout_cards(cards)},
        token=token,
    )


def _count_bi_tables(conn) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for table, sql in BI_TABLES.items():
        try:
            counts[table] = int(conn.execute(text(sql)).scalar_one())
        except Exception:
            counts[table] = None
    return counts


def _recreate_bi_tables(db: Session, run_id: str | None = None) -> dict[str, int | None]:
    engine = _bi_engine()
    runs = db.query(AnalysisRun)
    if run_id:
        runs = runs.filter(AnalysisRun.id == run_id)
    run_rows = runs.order_by(AnalysisRun.created_at.desc()).all()

    selected = db.query(SelectedInsight)
    if run_id:
        selected = selected.filter(SelectedInsight.run_id == run_id)
    selected_rows = selected.order_by(SelectedInsight.selected_at.desc()).all()

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS bi_selected_insights"))
        conn.execute(text("DROP TABLE IF EXISTS bi_service_risk"))
        conn.execute(text("DROP TABLE IF EXISTS bi_sla_by_category"))
        conn.execute(text("DROP TABLE IF EXISTS bi_cluster_summary"))
        conn.execute(text("DROP TABLE IF EXISTS bi_evidences"))
        conn.execute(text("DROP TABLE IF EXISTS bi_runs"))

        conn.execute(
            text(
                """
                CREATE TABLE bi_runs (
                    run_id text PRIMARY KEY,
                    run_created_at timestamp with time zone,
                    modality text,
                    reduction_method text,
                    seed integer,
                    n_samples integer,
                    outliers_count integer,
                    silhouette double precision,
                    davies_bouldin double precision,
                    n_clusters integer
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_runs (
                    run_id, run_created_at, modality, reduction_method, seed,
                    n_samples, outliers_count, silhouette, davies_bouldin, n_clusters
                )
                VALUES (
                    'imported_tfm', NOW(), 'it_ops', 'UMAP', 42,
                    NULL, NULL, NULL, NULL, NULL
                )
                """
            )
        )
        if run_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO bi_runs (
                        run_id, run_created_at, modality, reduction_method, seed,
                        n_samples, outliers_count, silhouette, davies_bouldin, n_clusters
                    )
                    VALUES (
                        :run_id, :run_created_at, :modality, :reduction_method, :seed,
                        :n_samples, :outliers_count, :silhouette, :davies_bouldin,
                        :n_clusters
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        run_created_at = EXCLUDED.run_created_at,
                        modality = EXCLUDED.modality,
                        reduction_method = EXCLUDED.reduction_method,
                        seed = EXCLUDED.seed,
                        n_samples = EXCLUDED.n_samples,
                        outliers_count = EXCLUDED.outliers_count,
                        silhouette = EXCLUDED.silhouette,
                        davies_bouldin = EXCLUDED.davies_bouldin,
                        n_clusters = EXCLUDED.n_clusters
                    """
                ),
                [
                    {
                        "run_id": row.id,
                        "run_created_at": row.created_at,
                        "modality": row.modality,
                        "reduction_method": row.reduction_method,
                        "seed": row.seed,
                        "n_samples": row.n_samples,
                        "outliers_count": row.outliers_count,
                        "silhouette": _number(row.silhouette),
                        "davies_bouldin": _number(row.davies_bouldin),
                        "n_clusters": _metrics_from_row(row).n_clusters,
                    }
                    for row in run_rows
                ],
            )

        conn.execute(
            text(
                """
                CREATE TABLE bi_evidences AS
                SELECT
                    'imported_tfm'::text AS run_id,
                    e.evidence_id,
                    c.cluster_id::integer AS cluster_label,
                    COALESCE(c.is_noise, false) AS is_noise,
                    e.opened_at,
                    e.severity,
                    e.category,
                    e.status,
                    e.assignment_group,
                    e.affected_service,
                    e.resolution_minutes,
                    e.sla_breached,
                    e.business_impact_score,
                    e.evidence_text,
                    e.source_dataset,
                    e.ticket_type,
                    e.requester_department,
                    e.region,
                    e.channel,
                    e.summary,
                    e.description,
                    e.root_cause,
                    e.sla_target_minutes,
                    e.cost_impact_usd,
                    e.urgency_score
                FROM tfm_evidences e
                LEFT JOIN tfm_clusters c
                    ON c.evidence_id = e.evidence_id
                    AND c.algorithm = 'hdbscan'
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_cluster_summary AS
                SELECT
                    'imported_tfm'::text AS run_id,
                    cluster_id::integer AS cluster_label,
                    cluster_name,
                    size,
                    sla_breach_rate,
                    avg_business_impact,
                    critical_or_high_count,
                    top_category,
                    top_service,
                    top_terms,
                    silhouette,
                    stability,
                    noise_rate
                FROM vw_it_cluster_overview
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_sla_by_category AS
                SELECT 'imported_tfm'::text AS run_id, *
                FROM vw_it_sla_by_category
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_service_risk AS
                SELECT
                    'imported_tfm'::text AS run_id,
                    affected_service,
                    COUNT(*)::integer AS ticket_count,
                    AVG(business_impact_score) AS avg_business_impact,
                    AVG(CASE WHEN sla_breached THEN 1.0 ELSE 0.0 END) * 100 AS sla_breach_rate,
                    AVG(resolution_minutes) AS avg_resolution_minutes,
                    COUNT(*) FILTER (WHERE severity IN ('Critical', 'High'))::integer
                        AS critical_or_high_count
                FROM tfm_evidences
                GROUP BY affected_service
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_selected_insights (
                    id text,
                    run_id text,
                    selected_at timestamp with time zone,
                    title text,
                    description text,
                    metric_label text,
                    metric_value double precision,
                    dimension text,
                    filter_kind text,
                    filter_value text
                )
                """
            )
        )
        if selected_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO bi_selected_insights (
                        id, run_id, selected_at, title, description, metric_label,
                        metric_value, dimension, filter_kind, filter_value
                    )
                    VALUES (
                        :id, :run_id, :selected_at, :title, :description, :metric_label,
                        :metric_value, :dimension, :filter_kind, :filter_value
                    )
                    """
                ),
                [
                    {
                        "id": row.id,
                        "run_id": row.run_id,
                        "selected_at": row.selected_at,
                        "title": row.title,
                        "description": row.description,
                        "metric_label": row.metric_label,
                        "metric_value": _number(row.metric_value),
                        "dimension": row.dimension,
                        "filter_kind": row.filter_kind,
                        "filter_value": row.filter_value,
                    }
                    for row in selected_rows
                ],
            )

        return _count_bi_tables(conn)


@router.get("/health", response_model=HealthResponse)
def health(db: Annotated[Session, Depends(get_db)]):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", database=db_status)


@router.get("/api/metabase/status")
def metabase_status(_user: Annotated[User, Depends(get_current_user)]):
    settings = get_settings()
    if not settings.bi_sync_enabled:
        return {
            "enabled": False,
            "postgres_status": "disabled",
            "metabase_url": settings.metabase_url,
            "dashboard_url": settings.metabase_dashboard_url,
            "detail": "Sincronizacion BI desactivada por configuracion.",
            "tables": {},
        }

    try:
        engine = _bi_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            table_counts = _count_bi_tables(conn)
        postgres_status = "ok"
        detail = "PostgreSQL BI disponible."
    except Exception as exc:
        table_counts = {}
        postgres_status = "error"
        detail = f"PostgreSQL BI no disponible: {exc}"

    metabase_ok = _metabase_health(settings.metabase_url)
    if postgres_status == "ok" and not metabase_ok:
        detail = "PostgreSQL BI disponible; Metabase no responde todavia."
    dashboard_url = settings.metabase_dashboard_url
    if metabase_ok:
        try:
            dashboard_url = _metabase_dashboard_url(settings, _metabase_login(settings)) or dashboard_url
        except RuntimeError:
            pass

    return {
        "enabled": postgres_status == "ok",
        "postgres_status": postgres_status,
        "metabase_status": "ok" if metabase_ok else "error",
        "metabase_url": settings.metabase_url,
        "dashboard_url": dashboard_url,
        "detail": detail,
        "tables": table_counts,
    }


@router.post("/api/bi-sync")
def sync_all_bi_tables(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    if not settings.bi_sync_enabled:
        raise HTTPException(status_code=400, detail="Sincronizacion BI desactivada")
    try:
        counts = _recreate_bi_tables(db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo publicar BI: {exc}") from exc
    return {
        "status": "ok",
        "message": "Tablas BI publicadas en PostgreSQL.",
        "tables": counts,
    }


@router.post("/api/runs/{run_id}/bi-sync")
def sync_run_bi_tables(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    if db.get(AnalysisRun, run_id) is None:
        raise HTTPException(status_code=404, detail="Ejecucion no encontrada")
    settings = get_settings()
    if not settings.bi_sync_enabled:
        raise HTTPException(status_code=400, detail="Sincronizacion BI desactivada")
    try:
        counts = _recreate_bi_tables(db, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo publicar BI: {exc}") from exc
    return {
        "status": "ok",
        "message": "Tablas BI publicadas para la ejecucion seleccionada.",
        "tables": counts,
    }


@router.post("/api/metabase/dashboard")
def create_metabase_dashboard(_user: Annotated[User, Depends(get_current_user)]):
    settings = get_settings()
    if not settings.bi_sync_enabled:
        raise HTTPException(status_code=400, detail="Sincronizacion BI desactivada")
    if not _metabase_health(settings.metabase_url):
        raise HTTPException(status_code=503, detail="Metabase no responde en la URL configurada")
    try:
        token = _metabase_login(settings)
        database_id = _metabase_ensure_database(settings, token)
        dashboard = _metabase_ensure_dashboard(settings, token)
        cards = _metabase_ensure_cards(settings, token, database_id)
        _metabase_attach_cards(settings, token, int(dashboard["id"]), cards)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    metabase_url = settings.metabase_url.rstrip("/")
    dashboard_url = f"{metabase_url}/dashboard/{dashboard['id']}"
    return {
        "status": "ok",
        "message": (
            "Dashboard creado en Metabase con preguntas sobre las tablas BI publicadas."
        ),
        "dashboard_url": dashboard_url,
        "cards": [
            {
                "id": card["id"],
                "name": card["name"],
                "url": f"{metabase_url}/question/{card['id']}",
            }
            for card in cards
        ],
    }


@router.post("/api/datasets/upload", response_model=DatasetProfileResponse, status_code=201)
async def upload_dataset(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    del db  # reservado para futura persistencia en BD
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
    content = await file.read()
    try:
        meta = save_upload(
            user_id=user.id,
            filename=file.filename,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error al guardar el dataset") from exc
    return DatasetProfileResponse(**meta)


@router.get("/api/datasets/{dataset_id}", response_model=DatasetProfileResponse)
def get_dataset_profile(
    dataset_id: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        meta = get_dataset_meta(dataset_id, user_id=user.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset no encontrado") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return DatasetProfileResponse(**meta)


@router.post("/api/runs", response_model=RunDetail, status_code=201)
def create_run(
    body: RunCreateBody,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    settings = get_settings()
    seed = body.seed if body.seed is not None else settings.default_seed
    n_samples = body.n_samples or settings.default_n_samples

    if body.modality == "tabular" and not body.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Sube un CSV y proporciona dataset_id para modalidad tabular",
        )

    try:
        result = run_pipeline(
            modality=body.modality,
            reduction_method=body.reduction_method,
            seed=seed,
            n_samples=n_samples,
            dataset_path=settings.it_ops_dataset_path,
            dataset_id=body.dataset_id,
            user_id=user.id,
            id_column=body.id_column,
            exclude_columns=body.exclude_columns or None,
            numeric_columns=body.numeric_columns,
            categorical_columns=body.categorical_columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    payload = {
        "modality": body.modality,
        "reduction_method": body.reduction_method,
        "seed": seed,
        "n_samples": n_samples,
        "result": result.model_dump(),
    }
    row = save_run(db, payload=payload)
    return RunDetail(**run_to_detail(row))


@router.post("/api/runs/{run_id}/chat", response_model=ChatResponse)
def chat_run(
    run_id: str,
    body: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="EjecuciÃ³n no encontrada")

    summaries = _cluster_summaries(row)
    if not summaries:
        return ChatResponse(
            answer="No encontre clusters persistidos para esta ejecucion.",
            suggested_questions=[
                "Que puedo analizar con esta data?",
                "Que grupos incumplen SLA?",
            ],
        )

    question = body.question.lower()
    if "sla" in question:
        selected = sorted(
            summaries,
            key=lambda item: item.get("avg_sla") or 0,
            reverse=True,
        )[:3]
        kind = "sla"
        answer = "Los grupos con mayor incumplimiento de SLA aparecen primero."
    elif "riesgo" in question or "risk" in question:
        selected = sorted(
            summaries,
            key=lambda item: item.get("avg_risk") or 0,
            reverse=True,
        )[:3]
        kind = "risk"
        answer = "Estos clusters concentran el mayor riesgo operativo promedio."
    elif "volumen" in question or "servicio" in question or "tickets" in question:
        selected = sorted(
            summaries,
            key=lambda item: item.get("count") or 0,
            reverse=True,
        )[:3]
        kind = "volume"
        answer = "Estos grupos concentran mas evidencias y ayudan a priorizar volumen."
    else:
        selected = sorted(
            summaries,
            key=lambda item: item.get("score") or 0,
            reverse=True,
        )[:3]
        kind = "priority"
        answer = "Te propongo empezar por estos clusters por combinacion de SLA, riesgo y volumen."

    return ChatResponse(
        answer=answer,
        insights=[_summary_to_insight(run_id, summary, kind) for summary in selected],
        suggested_questions=[
            "Que grupos incumplen SLA?",
            "Que servicios concentran mas volumen?",
            "Que clusters tienen mayor riesgo?",
        ],
    )


@router.post(
    "/api/runs/{run_id}/insights/select",
    response_model=InsightSelectResponse,
)
def select_insight(
    run_id: str,
    body: InsightSelectRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="EjecuciÃ³n no encontrada")

    insight = body.insight
    selected_id = f"{user.id}:{run_id}:{insight.id}"
    selected = db.get(SelectedInsight, selected_id)
    metric_value = str(insight.metric_value) if insight.metric_value is not None else None

    if selected is None:
        selected = SelectedInsight(
            selected_id=selected_id,
            id=insight.id,
            run_id=run_id,
            user_id=user.id,
            selected_at=datetime.now(timezone.utc),
            title=insight.title,
            description=insight.description,
            metric_label=insight.metric_label,
            metric_value=metric_value,
            dimension=insight.dimension,
            filter_kind=insight.filter_kind,
            filter_value=insight.filter_value,
        )
        db.add(selected)
    else:
        selected.selected_at = datetime.now(timezone.utc)
        selected.title = insight.title
        selected.description = insight.description
        selected.metric_label = insight.metric_label
        selected.metric_value = metric_value
        selected.dimension = insight.dimension
        selected.filter_kind = insight.filter_kind
        selected.filter_value = insight.filter_value

    db.commit()
    db.refresh(selected)
    return InsightSelectResponse(insight=_selected_to_dashboard_item(selected, row))


@router.get("/api/conversation-dashboard", response_model=ConversationDashboardResponse)
def conversation_dashboard(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    run_id: str | None = None,
):
    query = db.query(SelectedInsight).filter(SelectedInsight.user_id == user.id)
    if run_id:
        query = query.filter(SelectedInsight.run_id == run_id)
    selected_rows = query.order_by(SelectedInsight.selected_at.desc()).all()

    items: list[InsightDashboardItem] = []
    for selected in selected_rows:
        run = db.get(AnalysisRun, selected.run_id)
        if run is None:
            continue
        items.append(_selected_to_dashboard_item(selected, run))

    return ConversationDashboardResponse(total=len(items), insights=items)


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
    limit: int = 20,
):
    limit = min(max(1, limit), 100)
    rows = (
        db.query(AnalysisRun)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RunSummary(
            id=r.id,
            created_at=r.created_at,
            modality=r.modality,  # type: ignore[arg-type]
            reduction_method=r.reduction_method,  # type: ignore[arg-type]
            seed=r.seed,
            n_samples=r.n_samples,
            outliers_count=r.outliers_count,
            metrics=_metrics_from_row(r),
        )
        for r in rows
    ]


@router.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
):
    row = db.get(AnalysisRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return RunDetail(**run_to_detail(row))
