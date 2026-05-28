from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

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


def _find_database_id(session_id: str) -> int:
    expected_name = _settings().metabase_database_name.strip().lower()
    response = _json_request("GET", "/api/database", session_id=session_id)
    databases = response.get("data", response) if isinstance(response, dict) else response
    if not isinstance(databases, list):
        raise MetabaseDashboardError("No se pudo leer la lista de bases de Metabase.")

    exact_match = next(
        (
            db
            for db in databases
            if str(db.get("name", "")).strip().lower() == expected_name
        ),
        None,
    )
    postgres_match = next(
        (db for db in databases if str(db.get("engine", "")).lower() == "postgres"),
        None,
    )
    selected = exact_match or postgres_match
    if not selected or not selected.get("id"):
        raise MetabaseDashboardError(
            "No encontre una base PostgreSQL en Metabase. Agrega la conexion "
            f"'{_settings().metabase_database_name}' antes de crear el dashboard."
        )
    return int(selected["id"])


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


LATEST_RUN_CTE = """
WITH latest_run AS (
    SELECT run_id
    FROM bi_runs
    ORDER BY created_at DESC
    LIMIT 1
)
"""


DASHBOARD_CARDS = [
    DashboardCardSpec(
        name="Evidencias analizadas",
        description="Total de evidencias publicadas para la ultima ejecucion.",
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
        name="SLA incumplido por categoria",
        description="Categorias con mayor porcentaje de incumplimiento SLA.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    category,
    ROUND(CAST(100.0 * sla_breached_count / NULLIF(evidence_count, 0) AS numeric), 2)
        AS sla_incumplido_pct,
    evidence_count AS evidencias
FROM bi_sla_by_category s
JOIN latest_run r ON s.run_id = r.run_id
ORDER BY sla_incumplido_pct DESC NULLS LAST
""".strip(),
        col=6,
        row=0,
        size_x=9,
        size_y=7,
        dimensions=("category",),
        metrics=("sla_incumplido_pct",),
    ),
    DashboardCardSpec(
        name="Servicios con mayor riesgo",
        description="Servicios afectados ordenados por riesgo operacional promedio.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    affected_service,
    ROUND(CAST(avg_risk AS numeric), 2) AS riesgo_promedio,
    evidence_count AS evidencias
FROM bi_service_risk s
JOIN latest_run r ON s.run_id = r.run_id
ORDER BY riesgo_promedio DESC NULLS LAST
LIMIT 12
""".strip(),
        col=15,
        row=0,
        size_x=9,
        size_y=7,
        dimensions=("affected_service",),
        metrics=("riesgo_promedio",),
    ),
    DashboardCardSpec(
        name="Volumen por severidad",
        description="Distribucion de evidencias por severidad.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    COALESCE(severity, 'Sin severidad') AS severity,
    COUNT(*) AS evidencias
FROM bi_evidences e
JOIN latest_run r ON e.run_id = r.run_id
GROUP BY COALESCE(severity, 'Sin severidad')
ORDER BY evidencias DESC
""".strip(),
        col=0,
        row=7,
        size_x=8,
        size_y=7,
        dimensions=("severity",),
        metrics=("evidencias",),
    ),
    DashboardCardSpec(
        name="Tiempo de resolucion por categoria",
        description="Categorias con mayor tiempo promedio de resolucion.",
        display="bar",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    category,
    ROUND(CAST(avg_resolution_hours AS numeric), 2) AS resolucion_horas
FROM bi_sla_by_category s
JOIN latest_run r ON s.run_id = r.run_id
ORDER BY resolucion_horas DESC NULLS LAST
""".strip(),
        col=8,
        row=7,
        size_x=8,
        size_y=7,
        dimensions=("category",),
        metrics=("resolucion_horas",),
    ),
    DashboardCardSpec(
        name="Clusters prioritarios",
        description="Clusters con mayor riesgo promedio y volumen de evidencias.",
        display="table",
        query=f"""
{LATEST_RUN_CTE}
SELECT
    cluster_label,
    evidence_count AS evidencias,
    ROUND(CAST(avg_risk AS numeric), 2) AS riesgo_promedio,
    ROUND(CAST(avg_sla_breach_rate AS numeric), 4) AS tasa_sla,
    ROUND(CAST(avg_resolution_hours AS numeric), 2) AS resolucion_horas
FROM bi_cluster_summary c
JOIN latest_run r ON c.run_id = r.run_id
ORDER BY riesgo_promedio DESC NULLS LAST, evidencias DESC
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
    session_id = _login()
    database_id = _find_database_id(session_id)
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

    dashboard_url = _dashboard_url(dashboard_id)
    return {
        "status": "ok",
        "message": (
            "Dashboard creado en Metabase."
            if created
            else "Dashboard existente actualizado en Metabase."
        ),
        "dashboard_id": dashboard_id,
        "dashboard_url": dashboard_url,
        "database_id": database_id,
        "cards": cards,
    }
