from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from sqlalchemy import create_engine, text

from app.config import get_settings


class MetabaseDashboardError(RuntimeError):
    pass


@dataclass(frozen=True)
class DashboardCardSpec:
    name: str
    description: str
    display: str
    query: str
    col: int
    row: int
    size_x: int
    size_y: int
    dimensions: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()


def _settings():
    return get_settings()


def _metabase_base_url() -> str:
    return _settings().metabase_url.rstrip("/")


def _dashboard_url(dashboard_id: int) -> str:
    return f"{_metabase_base_url()}/dashboard/{dashboard_id}"


def _public_dashboard_url(public_uuid: str) -> str:
    return f"{_metabase_base_url()}/public/dashboard/{parse.quote(public_uuid)}"


def _card_url(card_id: int) -> str:
    return f"{_metabase_base_url()}/question/{card_id}"


def _json_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | list[Any]:
    url = f"{_metabase_base_url()}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Metabase-Session"] = session_id

    req = request.Request(url, data=body, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if path == "/api/session" and exc.code in {401, 403}:
            raise MetabaseDashboardError(
                "Metabase rechazo las credenciales configuradas. "
                "Verifica METABASE_USERNAME y METABASE_PASSWORD en el archivo .env del backend, "
                "confirma que ese usuario puede iniciar sesion en Metabase y reinicia el backend."
            ) from exc
        raise MetabaseDashboardError(
            f"Metabase respondio {exc.code} en {method} {path}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise MetabaseDashboardError(
            f"No se pudo conectar con Metabase en {_metabase_base_url()}: {exc.reason}"
        ) from exc

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetabaseDashboardError(
            f"Metabase devolvio una respuesta no JSON en {method} {path}"
        ) from exc


def _login() -> str:
    settings = _settings()
    if not settings.metabase_username or not settings.metabase_password:
        raise MetabaseDashboardError(
            "Configura METABASE_USERNAME y METABASE_PASSWORD en el backend."
        )

    response = _json_request(
        "POST",
        "/api/session",
        payload={
            "username": settings.metabase_username,
            "password": settings.metabase_password,
        },
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise MetabaseDashboardError("Metabase no devolvio una sesion valida.")
    return str(response["id"])


def _database_payload() -> dict[str, Any]:
    settings = _settings()
    return {
        "engine": "postgres",
        "name": settings.metabase_database_name,
        "details": {
            "host": settings.metabase_pg_host,
            "port": settings.metabase_pg_port,
            "dbname": settings.metabase_pg_dbname,
            "user": settings.metabase_pg_user,
            "password": settings.metabase_pg_password,
            "ssl": False,
            "tunnel-enabled": False,
        },
        "is_full_sync": True,
        "is_on_demand": False,
        "auto_run_queries": True,
    }


def _list_databases(session_id: str) -> list[dict[str, Any]]:
    response = _json_request("GET", "/api/database", session_id=session_id)
    databases = response.get("data", response) if isinstance(response, dict) else response
    if not isinstance(databases, list):
        raise MetabaseDashboardError("No se pudo leer la lista de bases de Metabase.")
    return [db for db in databases if isinstance(db, dict)]


def _find_database_id(session_id: str) -> int | None:
    expected_name = _settings().metabase_database_name.strip().lower()
    databases = _list_databases(session_id)
    selected = next(
        (
            db
            for db in databases
            if str(db.get("name", "")).strip().lower() == expected_name
            and not db.get("is_archived")
            and db.get("id")
        ),
        None,
    )
    return None if not selected else int(selected["id"])


def _ensure_database_id(session_id: str) -> int:
    database_id = _find_database_id(session_id)
    if database_id:
        return database_id

    response = _json_request(
        "POST",
        "/api/database",
        payload=_database_payload(),
        session_id=session_id,
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise MetabaseDashboardError(
            "Metabase no devolvio el ID de la conexion PostgreSQL creada."
        )
    database_id = int(response["id"])
    try:
        _json_request(
            "POST",
            f"/api/database/{database_id}/sync_schema",
            session_id=session_id,
        )
    except MetabaseDashboardError:
        pass
    return database_id


def _native_card_payload(spec: DashboardCardSpec, database_id: int) -> dict[str, Any]:
    visualization_settings: dict[str, Any] = {}
    if spec.dimensions:
        visualization_settings["graph.dimensions"] = list(spec.dimensions)
    if spec.metrics:
        visualization_settings["graph.metrics"] = list(spec.metrics)

    return {
        "name": spec.name,
        "description": spec.description,
        "display": spec.display,
        "dataset_query": {
            "type": "native",
            "database": database_id,
            "native": {"query": spec.query},
        },
        "visualization_settings": visualization_settings,
    }


def _create_card(
    session_id: str,
    database_id: int,
    spec: DashboardCardSpec,
) -> int:
    response = _json_request(
        "POST",
        "/api/card",
        payload=_native_card_payload(spec, database_id),
        session_id=session_id,
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise MetabaseDashboardError(f"No se pudo crear la tarjeta '{spec.name}'.")
    return int(response["id"])


def _find_existing_dashboard_id(session_id: str) -> int | None:
    dashboard_name = _settings().metabase_dashboard_name.strip()
    path = f"/api/search?q={parse.quote(dashboard_name)}&models=dashboard"
    response = _json_request("GET", path, session_id=session_id)
    items = response.get("data", response) if isinstance(response, dict) else response
    if not isinstance(items, list):
        return None

    matches = [
        item
        for item in items
        if str(item.get("name", "")).strip() == dashboard_name
        and not item.get("archived")
        and item.get("id")
    ]
    if not matches:
        return None

    selected = max(
        matches,
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
    )
    return int(selected["id"])


def _create_or_reuse_dashboard(session_id: str) -> tuple[int, bool]:
    existing_id = _find_existing_dashboard_id(session_id)
    if existing_id:
        return existing_id, False

    settings = _settings()
    dashboard_response = _json_request(
        "POST",
        "/api/dashboard",
        payload={
            "name": settings.metabase_dashboard_name,
            "description": (
                "Dashboard BI generado desde la exploracion conversacional del TFM. "
                "Usa la ultima ejecucion publicada en las tablas bi_*."
            ),
        },
        session_id=session_id,
    )
    if not isinstance(dashboard_response, dict) or not dashboard_response.get("id"):
        raise MetabaseDashboardError("Metabase no devolvio el ID del dashboard creado.")
    return int(dashboard_response["id"]), True


def _dashboard_links(
    session_id: str,
    dashboard_id: int,
    *,
    ensure_public_link: bool,
) -> dict[str, Any]:
    detail = _json_request(
        "GET",
        f"/api/dashboard/{dashboard_id}",
        session_id=session_id,
    )
    public_uuid = detail.get("public_uuid") if isinstance(detail, dict) else None
    if ensure_public_link and not public_uuid:
        public_link = _json_request(
            "POST",
            f"/api/dashboard/{dashboard_id}/public_link",
            session_id=session_id,
        )
        if isinstance(public_link, dict):
            public_uuid = public_link.get("uuid") or public_link.get("public_uuid")
        if not public_uuid:
            raise MetabaseDashboardError(
                "Metabase no devolvio un enlace publico para integrar el dashboard. "
                "Verifica que Public sharing este habilitado en Administracion."
            )

    return {
        "dashboard_id": dashboard_id,
        "dashboard_url": _dashboard_url(dashboard_id),
        "embed_url": _public_dashboard_url(str(public_uuid)) if public_uuid else None,
    }


def get_conversation_dashboard_links() -> dict[str, Any]:
    session_id = _login()
    dashboard_id = _find_existing_dashboard_id(session_id)
    if dashboard_id is None:
        return {}
    return _dashboard_links(
        session_id,
        dashboard_id,
        ensure_public_link=False,
    )


def _replace_dashboard_cards(
    session_id: str,
    dashboard_id: int,
    cards: list[tuple[int, DashboardCardSpec]],
) -> None:
    payload = {
        "cards": [
            {
                "id": -index,
                "card_id": card_id,
                "row": spec.row,
                "col": spec.col,
                "size_x": spec.size_x,
                "size_y": spec.size_y,
                "parameter_mappings": [],
                "series": [],
                "visualization_settings": {},
            }
            for index, (card_id, spec) in enumerate(cards, start=1)
        ],
        "tabs": [],
    }
    _json_request(
        "PUT",
        f"/api/dashboard/{dashboard_id}/cards",
        payload=payload,
        session_id=session_id,
    )


def _ensure_reporting_state_table() -> None:
    """Keep the active BI run explicit for the generated Metabase SQL."""
    engine = create_engine(_settings().bi_database_url, pool_pre_ping=True)
    with engine.begin() as con:
        con.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_active_run (
                    active_key TEXT PRIMARY KEY,
                    run_id TEXT,
                    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )


LATEST_RUN_CTE = """
WITH active_run AS (
    SELECT ar.run_id
    FROM bi_active_run ar
    JOIN bi_runs br ON br.run_id = ar.run_id
    WHERE ar.active_key = 'metabase_report'
    LIMIT 1
),
latest_run AS (
    SELECT COALESCE(
        (SELECT run_id FROM active_run),
        (
            SELECT run_id
            FROM bi_runs
            ORDER BY created_at DESC NULLS LAST, run_id DESC
            LIMIT 1
        )
    ) AS run_id
)
"""


CATEGORY_DIMENSION_SQL = """
COALESCE(
    NULLIF(COALESCE(e.categoria, e.category, e.sector), ''),
    NULLIF(TRIM(substring(e.preview from 'Cat[^=]*=([^|]+)')), ''),
    CASE
        WHEN e.cluster_label = -1 THEN 'Casos atipicos'
        ELSE 'Cluster ' || e.cluster_label::TEXT
    END
)
""".strip()


SERVICE_DIMENSION_SQL = """
COALESCE(
    NULLIF(COALESCE(e.servicio_afectado, e.affected_service, e.service_line), ''),
    NULLIF(TRIM(substring(e.preview from 'Empresa=([^|]+)')), ''),
    CASE
        WHEN e.cluster_label = -1 THEN 'Casos atipicos'
        ELSE 'Cluster ' || e.cluster_label::TEXT
    END
)
""".strip()


INCIDENT_DIMENSION_SQL = """
COALESCE(
    NULLIF(e.incident_id, ''),
    NULLIF(TRIM(substring(e.preview from 'N[^=]*mero=([^|]+)')), ''),
    e.evidence_id
)
""".strip()


CLUSTER_DIMENSION_SQL = """
CASE
    WHEN c.cluster_label = -1 THEN 'Casos atipicos'
    ELSE 'Cluster ' || c.cluster_label::TEXT
END
""".strip()


DASHBOARD_CARDS = [
    DashboardCardSpec(
        name="Evidencias analizadas",
        description="Total de evidencias publicadas para la ejecucion activa.",
        display="scalar",
        query=f"""
{LATEST_RUN_CTE}
SELECT COUNT(*) AS evidencias
FROM bi_evidences e
JOIN latest_run r ON e.run_id = r.run_id
""".strip(),
        col=0,
        row=0,
        size_x=6,
        size_y=4,
    ),
    DashboardCardSpec(
        name="Volumen por categoria",
        description="Categorias detectadas con mayor cantidad de evidencias publicadas.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    categoria,
    evidencias
FROM (
    SELECT
        {CATEGORY_DIMENSION_SQL} AS categoria,
        COUNT(*) AS evidencias
    FROM bi_evidences e
    JOIN latest_run r ON e.run_id = r.run_id
    GROUP BY 1
) ranked
ORDER BY evidencias DESC
LIMIT 12
""".strip(),
        col=6,
        row=0,
        size_x=9,
        size_y=7,
        dimensions=("categoria",),
        metrics=("evidencias",),
    ),
    DashboardCardSpec(
        name="Volumen por empresa o servicio",
        description="Empresas o servicios con mayor cantidad de evidencias publicadas.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    servicio,
    evidencias
FROM (
    SELECT
        {SERVICE_DIMENSION_SQL} AS servicio,
        COUNT(*) AS evidencias
    FROM bi_evidences e
    JOIN latest_run r ON e.run_id = r.run_id
    GROUP BY 1
) ranked
ORDER BY evidencias DESC
LIMIT 12
""".strip(),
        col=15,
        row=0,
        size_x=9,
        size_y=7,
        dimensions=("servicio",),
        metrics=("evidencias",),
    ),
    DashboardCardSpec(
        name="Top clusters por evidencias",
        description="Grupos tecnicos con mayor volumen de evidencias publicadas.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    {CLUSTER_DIMENSION_SQL} AS grupo,
    c.evidence_count AS evidencias
FROM bi_cluster_summary c
JOIN latest_run r ON c.run_id = r.run_id
ORDER BY c.evidence_count DESC NULLS LAST
LIMIT 12
""".strip(),
        col=0,
        row=7,
        size_x=8,
        size_y=7,
        dimensions=("grupo",),
        metrics=("evidencias",),
    ),
    DashboardCardSpec(
        name="Incidencias con mas evidencias asociadas",
        description="Incidencias que aparecen repetidas en la evidencia publicada.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    incidencia,
    evidencias
FROM (
    SELECT
        {INCIDENT_DIMENSION_SQL} AS incidencia,
        COUNT(*) AS evidencias
    FROM bi_evidences e
    JOIN latest_run r ON e.run_id = r.run_id
    GROUP BY 1
) ranked
WHERE incidencia IS NOT NULL
ORDER BY evidencias DESC
LIMIT 12
""".strip(),
        col=8,
        row=7,
        size_x=8,
        size_y=7,
        dimensions=("incidencia",),
        metrics=("evidencias",),
    ),
    DashboardCardSpec(
        name="Clusters prioritarios",
        description="Clusters con mayor volumen de evidencias y metricas auxiliares si existen.",
        display="table",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    {CLUSTER_DIMENSION_SQL} AS grupo,
    c.evidence_count AS evidencias,
    ROUND(
        CAST(COALESCE(c.avg_risk, 100.0 * c.avg_sla_breach_rate) AS numeric),
        2
    ) AS riesgo_promedio,
    ROUND(CAST(c.avg_sla_breach_rate AS numeric), 4) AS tasa_sla,
    ROUND(CAST(c.avg_resolution_hours AS numeric), 2) AS resolucion_horas
FROM bi_cluster_summary c
JOIN latest_run r ON c.run_id = r.run_id
ORDER BY c.evidence_count DESC NULLS LAST, riesgo_promedio DESC NULLS LAST
LIMIT 50
""".strip(),
        col=16,
        row=7,
        size_x=8,
        size_y=7,
    ),
    DashboardCardSpec(
        name="Insights seleccionados por el usuario",
        description="Hallazgos guardados desde la exploracion conversacional.",
        display="table",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    title,
    metric_label,
    metric_value,
    dimension,
    filter_kind,
    filter_value,
    selected_at
FROM bi_selected_insights i
JOIN latest_run r ON i.run_id = r.run_id
ORDER BY selected_at DESC
""".strip(),
        col=0,
        row=14,
        size_x=24,
        size_y=8,
    ),
]


def create_conversation_dashboard() -> dict[str, Any]:
    _ensure_reporting_state_table()
    session_id = _login()
    database_id = _ensure_database_id(session_id)
    dashboard_id, created = _create_or_reuse_dashboard(session_id)
    cards: list[dict[str, Any]] = []
    card_specs: list[tuple[int, DashboardCardSpec]] = []
    for spec in DASHBOARD_CARDS:
        card_id = _create_card(session_id, database_id, spec)
        card_specs.append((card_id, spec))
        cards.append(
            {
                "id": card_id,
                "name": spec.name,
                "url": _card_url(card_id),
            }
        )
    _replace_dashboard_cards(session_id, dashboard_id, card_specs)

    links = _dashboard_links(
        session_id,
        dashboard_id,
        ensure_public_link=True,
    )
    return {
        "status": "ok",
        "message": (
            "Dashboard creado en Metabase."
            if created
            else "Dashboard existente actualizado en Metabase."
        ),
        **links,
        "database_id": database_id,
        "cards": cards,
    }
