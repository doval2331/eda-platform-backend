from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import pandas as pd

from app.schemas import ConversationChartDataResponse
from app.services.conversation.dashboard_spec import (
    _column_key,
    _column_summary,
    _humanize_column,
    _semantic_variables,
)
from app.services.runs.duckdb_store import load_run_evidences

PREVIEW_COLUMN_MAP = {
    "Número": "incident_id",
    "Numero": "incident_id",
    "Nro": "incident_id",
    "CI_Cat": "categoria",
    "CI_Subcat": "subcategoria",
    "Category": "category",
    "Catálogo": "category",
    "Catalogo": "category",
    "Catálogo.1": "category",
    "Catalogo.1": "category",
    "Catálogo.2": "category",
    "Catalogo.2": "category",
    "Nombre": "ci_cat",
    "Subcategoría": "ci_subcat",
    "Subcategoria": "ci_subcat",
    "Elemento de configuración": "affected_service",
    "Elemento de configuracion": "affected_service",
    "Priority": "prioridad",
    "Prioridad": "prioridad",
    "Status": "status",
    "Estado de la incidencia": "status",
    "Valor": "status",
    "Impact": "impact",
    "Impacto": "impact",
    "Urgency": "urgency",
    "Urgencia": "urgency",
    "Affected_Service": "affected_service",
    "Service": "servicio_afectado",
    "Business_Service": "business_service",
    "Assignment_Group": "assignment_group",
    "Grupo de asignación": "assignment_group",
    "Grupo de asignacion": "assignment_group",
    "Empresa": "company",
    "Empresa.1": "company",
    "Tipo de contacto": "channel",
    "No_Of_Reassignments": "no_of_reassignments",
    "No_of_Reassignments": "no_of_reassignments",
    "Volver a abrir recuento": "reopen_count",
    "No_of_Related_Interactions": "related_interactions_count",
    "No_of_Related_Incidents": "related_incidents_count",
    "No_of_Related_Changes": "related_changes_count",
    "Closure_Code": "close_code",
    "Resolver_Group": "resolver_group",
    "Assigned_To": "assigned_to",
    "Close_Code": "close_code",
    "Código de resolución": "close_code",
    "Codigo de resolucion": "close_code",
    "Tipo de Solución": "close_code",
    "Tipo de Solucion": "close_code",
    "Diagnóstico": "root_cause",
    "Diagnostico": "root_cause",
    "Causada por Cambio": "root_cause",
    "Environment": "environment",
    "Location": "location",
    "Service_Offering": "service_offering",
    "Opened_By": "opened_by",
    "Abierto por": "opened_by",
    "Solicitante": "opened_by",
    "Open_Time": "opened_at",
    "Inicio": "opened_at",
    "Resolved_Time": "closed_at",
    "Close_Time": "closed_at",
    "Fin": "closed_at",
    "Resuelto": "closed_at",
    "Duración": "avg_resolution_hours",
    "Duracion": "avg_resolution_hours",
    "Tiempo de trabajo": "avg_resolution_hours",
    "task.business_service": "business_service",
    "task.assignment_group": "assignment_group",
    "task.company": "company",
    "task.contact_type": "channel",
    "task.impact": "impact",
    "task.priority": "prioridad",
    "task.state": "status",
    "task.u_task_category": "category",
    "task.u_technical_subservice": "subcategoria",
    "task.reassignment_count": "no_of_reassignments",
}

FREE_TEXT_DIMENSION_COLUMNS = {
    "preview",
    "descripcion_corta",
    "short_description",
    "description",
    "summary",
    "comentario",
    "comment",
}
IDENTIFIER_DIMENSION_COLUMNS = {
    "run_id",
    "evidence_id",
    "incident_id",
    "ticket_id",
    "number",
    "numero",
    "nro",
    "id",
    "uuid",
}
SERVICE_DIMENSIONS = ("affected_service", "servicio_afectado", "service", "service_line", "business_service")
CATEGORY_DIMENSIONS = ("categoria", "category", "ci_cat", "sector", "subcategoria", "ci_subcat", "close_code")
PRIORITY_DIMENSIONS = ("prioridad", "priority", "severity")
GROUP_DIMENSIONS = ("cluster_label", "assignment_group")
STATUS_DIMENSIONS = ("status", "estado")
CHANNEL_DIMENSIONS = ("channel", "contact_type")

SAMPLE_PREVIEW_TARGETS = {
    "incident_id",
    "descripcion_corta",
    "prioridad",
    "servicio_afectado",
    "affected_service",
    "categoria",
    "category",
    "status",
    "no_of_reassignments",
}

NUMERIC_PREVIEW_TARGETS = {
    "avg_resolution_hours",
    "business_impact_score",
    "cost_impact_usd",
    "escalation_count",
    "no_of_reassignments",
    "number_cnt",
    "reopen_count",
    "related_changes_count",
    "related_incidents_count",
    "related_interactions_count",
    "resolution_minutes",
    "sla_breach_rate",
    "urgency_score",
}


def build_conversation_chart_data(
    *,
    run_id: str,
    visualization: dict[str, Any],
    limit: int = 12,
    evidence_limit: int = 12,
    project_id: str | None = None,
) -> ConversationChartDataResponse:
    safe_limit = _bounded_int(limit, default=12, maximum=20)
    safe_evidence_limit = _bounded_int(evidence_limit, default=12, maximum=60)

    df = _enrich_preview_columns(
        load_run_evidences(run_id),
        targets=_preview_targets_for_visualization(visualization),
    )
    if df.empty:
        return _empty_response(
            run_id=run_id,
            visualization=visualization,
            warning="No hay evidencias materializadas en DuckDB para esta ejecucion.",
        )

    columns = _column_summary(df, project_id=project_id)
    semantic_items = _semantic_variables(columns, project_id=project_id)
    semantic_by_name = {item["name"]: item for item in semantic_items}
    warnings: list[str] = []
    missing: list[str] = []

    filtered_df = _apply_filters(df, visualization.get("filters") or [])
    if filtered_df.empty:
        filtered_df = df
        warnings.append("Los filtros de la visualizacion no devolvieron registros; se uso el conjunto completo.")

    x_col = _resolve_column(
        filtered_df,
        visualization.get("x") or visualization.get("group_by"),
        role="dimension",
        semantic_by_name=semantic_by_name,
    )
    if not x_col:
        x_col = _best_dimension(filtered_df, columns, visualization)
        if x_col:
            warnings.append(f"El eje sugerido no existe o no es interpretable; se uso {_humanize_column(x_col)}.")
    if not x_col:
        missing.append("dimension")

    metric_col = _resolve_column(
        filtered_df,
        visualization.get("metric") or visualization.get("y"),
        role="metric",
        semantic_by_name=semantic_by_name,
    )
    aggregation = str(visualization.get("aggregation") or "count").lower()
    if not metric_col or metric_col == "count":
        metric_col = "count"
        aggregation = "count"
    elif not pd.api.types.is_numeric_dtype(filtered_df[metric_col]):
        warnings.append(f"La metrica {_humanize_column(metric_col)} no es numerica; se uso conteo.")
        metric_col = "count"
        aggregation = "count"

    if not x_col:
        return _empty_response(
            run_id=run_id,
            visualization=visualization,
            warning="No se encontro una dimension valida para construir el grafico.",
            missing=missing,
            semantic_items=semantic_items,
        )

    series_df = _aggregate(filtered_df, x_col=x_col, metric_col=metric_col, aggregation=aggregation)
    if series_df.empty:
        return _empty_response(
            run_id=run_id,
            visualization=visualization,
            warning="No se encontraron datos suficientes para agregar la visualizacion.",
            missing=missing,
            semantic_items=semantic_items,
        )

    series_df = series_df.sort_values(["value", "count"], ascending=False).head(safe_limit)
    samples_by_key: dict[str, list[dict[str, Any]]] = {}
    series: list[dict[str, Any]] = []
    dimension_values = filtered_df[x_col].fillna("Sin dato").astype(str).replace("", "Sin dato")
    for row in series_df.to_dict(orient="records"):
        key = str(row["key"])
        segment = filtered_df[dimension_values == key]
        samples = _evidence_samples(segment, metric_col=metric_col, limit=safe_evidence_limit)
        samples_by_key[key] = samples
        value = _safe_float(row.get("value")) or 0
        series.append(
            {
                "key": key,
                "label": key,
                "value": value,
                "count": int(row["count"] or 0),
                "metric": metric_col,
                "filter": {"column": x_col, "operator": "eq", "value": key},
            }
        )

    selected_key = series[0]["key"] if series else ""
    evidence_samples = samples_by_key.get(selected_key, [])
    evidence_truncated = bool(series and len(evidence_samples) < int(series[0].get("count") or 0))
    validation = _validation_payload(
        visualization=visualization,
        x_col=x_col,
        metric_col=metric_col,
        semantic_by_name=semantic_by_name,
        warnings=warnings,
        missing=missing,
        chart_is_buildable=bool(series),
        evidence_returned=len(evidence_samples),
    )
    return ConversationChartDataResponse.model_validate(
        {
            "run_id": run_id,
            "visualization_id": str(visualization.get("id") or ""),
            "title": str(visualization.get("title") or _humanize_column(x_col)),
            "chart_type": str(visualization.get("chart_type") or "bar"),
            "x": x_col,
            "metric": metric_col,
            "aggregation": aggregation,
            "total_records": int(len(filtered_df)),
            "evidence_returned": len(evidence_samples),
            "evidence_truncated": evidence_truncated,
            "series": series,
            "evidence_samples": evidence_samples,
            "samples_by_key": samples_by_key,
            "semantic_dictionary": semantic_items,
            "validation": validation,
        }
    )


def build_conversation_chart_error_response(
    *,
    run_id: str,
    visualization: dict[str, Any],
    warning: str,
) -> ConversationChartDataResponse:
    return _empty_response(
        run_id=run_id,
        visualization=visualization,
        warning=warning,
        missing=["backend_error"],
    )


def _bounded_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(1, min(number, maximum))


def _aggregate(df: pd.DataFrame, *, x_col: str, metric_col: str, aggregation: str) -> pd.DataFrame:
    work = df.copy()
    work[x_col] = work[x_col].fillna("Sin dato").astype(str).replace("", "Sin dato")
    grouped = work.groupby(x_col, dropna=False)
    if metric_col == "count" or aggregation == "count":
        result = grouped.size().reset_index(name="value")
    elif aggregation in {"sum", "total"}:
        result = grouped[metric_col].sum(numeric_only=True).reset_index(name="value")
    elif aggregation in {"max", "maximum"}:
        result = grouped[metric_col].max(numeric_only=True).reset_index(name="value")
    elif aggregation in {"min", "minimum"}:
        result = grouped[metric_col].min(numeric_only=True).reset_index(name="value")
    else:
        result = grouped[metric_col].mean(numeric_only=True).reset_index(name="value")
    counts = grouped.size().reset_index(name="count")
    result = result.merge(counts, on=x_col, how="left")
    result = result.rename(columns={x_col: "key"})
    result["value"] = _finite_numeric_series(result["value"], default=0)
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0).astype(int)
    return result


def _finite_numeric_series(series: pd.Series, *, default: float = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([math.inf, -math.inf], pd.NA)
    return numeric.fillna(default)


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = df
    for item in filters:
        column = _find_column(result, item.get("column") or item.get("field") or item.get("x"))
        if not column:
            continue
        operator = str(item.get("operator") or item.get("op") or "eq").lower()
        value = item.get("value")
        if operator in {"eq", "=", "=="}:
            result = result[result[column].astype(str) == str(value)]
        elif operator in {"neq", "!=", "not_eq"}:
            result = result[result[column].astype(str) != str(value)]
        elif operator in {"in", "contains_any"} and isinstance(value, list):
            allowed = {str(entry) for entry in value}
            result = result[result[column].astype(str).isin(allowed)]
    return result


def _resolve_column(
    df: pd.DataFrame,
    candidate: Any,
    *,
    role: str,
    semantic_by_name: dict[str, dict[str, Any]],
) -> str:
    text = str(candidate or "").strip()
    if not text or text == "count":
        return "count" if role == "metric" else ""
    column = _find_column(df, text)
    if not column:
        return ""
    semantic = semantic_by_name.get(column) or {}
    if role == "dimension" and semantic.get("can_chart") is False:
        return ""
    if role == "dimension" and not _is_usable_dimension(df, column):
        return ""
    if role == "dimension" and not _has_values(df, column):
        return ""
    if role == "metric" and column != "count" and not _has_values(df, column):
        return ""
    if role == "metric" and semantic.get("avoid_as_metric"):
        return ""
    return column


def _find_column(df: pd.DataFrame, candidate: Any) -> str:
    text = str(candidate or "").strip()
    if not text:
        return ""
    if text in df.columns:
        return text
    normalized = _normalize_name(text)
    for column in df.columns:
        if _normalize_name(column) == normalized:
            return str(column)
    candidate_key = _column_key(text)
    for column in df.columns:
        if _column_key(column) == candidate_key:
            return str(column)
    return ""


def _best_dimension(df: pd.DataFrame, columns: dict[str, Any], visualization: dict[str, Any]) -> str:
    for column in _preferred_dimensions_for_visualization(visualization):
        real_column = _find_column(df, column)
        if real_column and _is_usable_dimension(df, real_column):
            return real_column
    for column in columns.get("business") or []:
        if _is_usable_dimension(df, str(column)):
            return str(column)
    for column in columns.get("available") or []:
        if (
            column not in set(columns.get("technical") or [])
            and column not in set(columns.get("numeric") or [])
            and _is_usable_dimension(df, str(column))
        ):
            return str(column)
    if "cluster_label" in df.columns and _is_usable_dimension(df, "cluster_label", allow_technical=True):
        return "cluster_label"
    return ""


def _preferred_dimensions_for_visualization(visualization: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(visualization.get(key) or "")
        for key in ("id", "title", "reason", "question_answered", "evidence_used", "what_to_analyze")
    ).lower()
    preferred: list[str] = []
    if any(token in text for token in ("cluster", "clusters", "grupo", "grupos")):
        preferred.extend(GROUP_DIMENSIONS)
    if any(token in text for token in ("servicio", "service", "aplicacion", "application")):
        preferred.extend(SERVICE_DIMENSIONS)
    if any(token in text for token in ("categoria", "category", "ci_cat", "familia")):
        preferred.extend(CATEGORY_DIMENSIONS)
    if any(token in text for token in ("prioridad", "priority", "urgencia", "severity")):
        preferred.extend(PRIORITY_DIMENSIONS)
    if any(token in text for token in ("estado", "status", "abierto", "cerrado")):
        preferred.extend(STATUS_DIMENSIONS)
    if any(token in text for token in ("canal", "channel", "contacto", "contact")):
        preferred.extend(CHANNEL_DIMENSIONS)
    if any(token in text for token in ("cierre", "closure", "resolucion", "resolution")):
        preferred.extend(("close_code", "category", "subcategoria"))
    preferred.extend((
        *SERVICE_DIMENSIONS,
        *CATEGORY_DIMENSIONS,
        *PRIORITY_DIMENSIONS,
        *STATUS_DIMENSIONS,
        *CHANNEL_DIMENSIONS,
        "assignment_group",
        "company",
        "cluster_label",
    ))
    return _unique_texts(preferred)


def _preview_targets_for_visualization(visualization: dict[str, Any]) -> set[str]:
    targets = set(SAMPLE_PREVIEW_TARGETS)
    targets.update(_preferred_dimensions_for_visualization(visualization))
    for value in (
        visualization.get("x"),
        visualization.get("y"),
        visualization.get("metric"),
        visualization.get("group_by"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        if text in PREVIEW_COLUMN_MAP.values():
            targets.add(text)
        mapped = PREVIEW_COLUMN_MAP.get(text)
        if mapped:
            targets.add(mapped)
        normalized = _normalize_name(text)
        for key, target in PREVIEW_COLUMN_MAP.items():
            if _normalize_name(key) == normalized:
                targets.add(target)
    return targets


def _enrich_preview_columns(
    df: pd.DataFrame,
    *,
    targets: set[str] | None = None,
) -> pd.DataFrame:
    if df.empty or "preview" not in df.columns:
        return df
    result = df.copy()
    target_filter = set(targets or PREVIEW_COLUMN_MAP.values())
    lookup: dict[str, str] = {}
    for key, target in PREVIEW_COLUMN_MAP.items():
        if target not in target_filter:
            continue
        lookup[_normalize_name(key)] = target

    required_targets: list[str] = []
    for target in sorted(set(lookup.values())):
        if target in result.columns and _has_values(result, target):
            continue
        if target not in result.columns:
            result[target] = None
        required_targets.append(target)

    if required_targets:
        preview = result["preview"].fillna("").astype(str)
        parsed = preview.map(lambda text: _extract_preview_values(text, lookup))
    else:
        parsed = None

    for target in required_targets:
        extracted = parsed.map(lambda values, field=target: values.get(field)) if parsed is not None else None
        current = result[target]
        current_text = current.fillna("").astype(str).str.strip().str.lower()
        should_fill = current.isna() | current_text.isin({"", "nan", "none", "null"})
        result[target] = current.where(~should_fill, extracted)
    for target in NUMERIC_PREVIEW_TARGETS:
        if target not in result.columns:
            continue
        result[target] = _coerce_numeric_series(result[target])
    return result


def _has_values(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    values = df[column].dropna().astype(str).str.strip().str.lower()
    values = values[~values.isin({"", "nan", "none", "null"})]
    return not values.empty


def _is_usable_dimension(df: pd.DataFrame, column: str, *, allow_technical: bool = False) -> bool:
    if column not in df.columns or not _has_values(df, column):
        return False
    key = _normalize_name(column)
    blocked = {_normalize_name(item) for item in FREE_TEXT_DIMENSION_COLUMNS | IDENTIFIER_DIMENSION_COLUMNS}
    if key in blocked:
        return False
    if not allow_technical and key in {"x", "y", "umapx", "umapy", "pcax", "pcay"}:
        return False
    values = df[column].dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin({"", "nan", "none", "null"})]
    if values.empty:
        return False
    unique_count = int(values.nunique(dropna=True))
    total = int(len(values))
    if unique_count <= 0:
        return False
    if key == "clusterlabel":
        return True
    if unique_count > 200 and unique_count / max(total, 1) > 0.3:
        return False
    average_length = values.head(200).map(len).mean()
    if average_length and average_length > 80:
        return False
    return True


def _evidence_samples(df: pd.DataFrame, *, metric_col: str, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    if metric_col != "count" and metric_col in work:
        work["_metric_value"] = pd.to_numeric(work[metric_col], errors="coerce")
        work = work.sort_values("_metric_value", ascending=False, na_position="last")
    elif "evidence_index" in work:
        work = work.sort_values("evidence_index")
    samples: list[dict[str, Any]] = []
    for row in work.head(limit).to_dict(orient="records"):
        samples.append(
            {
                "evidence_id": _as_text(row.get("evidence_id")),
                "incident_id": _as_text(_first_present(row, "incident_id", "ticket_id")),
                "title": _as_text(_first_present(row, "descripcion_corta", "short_description", "preview")),
                "preview": _as_text(_first_present(row, "preview", "descripcion_corta", "short_description")),
                "source": _as_text(row.get("source")),
                "group": _as_text(row.get("cluster_label")),
                "priority": _as_text(_first_present(row, "prioridad", "priority", "severity")),
                "service": _as_text(_first_present(row, "servicio_afectado", "affected_service", "service_line")),
                "category": _as_text(_first_present(row, "categoria", "category", "sector")),
                "metric_value": _safe_float(row.get(metric_col)) if metric_col != "count" else None,
                "fields": {
                    "ticket": _as_text(_first_present(row, "incident_id", "ticket_id", "evidence_id")),
                    "grupo": _as_text(row.get("cluster_label")),
                    "prioridad": _as_text(_first_present(row, "prioridad", "priority", "severity")),
                    "servicio": _as_text(_first_present(row, "servicio_afectado", "affected_service", "service_line")),
                    "categoria": _as_text(_first_present(row, "categoria", "category", "sector")),
                    "estado": _as_text(_first_present(row, "status", "estado")),
                    "reasignaciones": _as_text(_first_present(row, "no_of_reassignments", "number_of_reassignments", "reassignment_count")),
                    "descripcion": _as_text(
                        _first_present(row, "descripcion_corta", "short_description", "preview")
                    ),
                },
            }
        )
    return samples


def _validation_payload(
    *,
    visualization: dict[str, Any],
    x_col: str,
    metric_col: str,
    semantic_by_name: dict[str, dict[str, Any]],
    warnings: list[str],
    missing: list[str],
    chart_is_buildable: bool,
    evidence_returned: int = 0,
) -> dict[str, Any]:
    x_semantic = semantic_by_name.get(x_col) or {}
    metric_semantic = semantic_by_name.get(metric_col) or {}
    x_role = x_semantic.get("role") or "unknown"
    metric_role = metric_semantic.get("role") or "metric"
    uses_technical = x_role in {"technical", "identifier"} or metric_role in {"technical", "identifier"}
    interpretable = x_role in {"business", "unknown"} and not x_semantic.get("avoid_as_metric")
    if metric_col != "count":
        interpretable = interpretable and metric_role == "metric"
    if not interpretable:
        warnings.append("La visualizacion usa al menos una variable tecnica; revisar si aporta lectura funcional.")
    possibly_invented = bool(missing) or any(
        "no existe" in str(warning).lower() or "no es interpretable" in str(warning).lower()
        for warning in warnings
    )
    quality_score = 100
    if not chart_is_buildable:
        quality_score -= 45
    if missing:
        quality_score -= 25
    if uses_technical:
        quality_score -= 15
    if not interpretable:
        quality_score -= 15
    if possibly_invented:
        quality_score -= 20
    if evidence_returned <= 0:
        quality_score -= 20
    quality_score = max(0, min(100, quality_score))
    operation_ready = bool(
        chart_is_buildable
        and not missing
        and evidence_returned > 0
        and not possibly_invented
    )
    if not chart_is_buildable:
        recommended_action = "Pide al agente una vista con dimension, metrica y tipo de grafico construible."
        validation_summary = "No esta listo para operacion porque el grafico no se pudo construir con datos reales."
    elif evidence_returned <= 0:
        recommended_action = "Guarda o selecciona evidencias antes de usar esta vista para decidir."
        validation_summary = "El grafico existe, pero no recupero tickets/evidencias para drill-down operativo."
    elif uses_technical:
        recommended_action = (
            "Usa esta vista solo como trazabilidad tecnica y pide una variable de negocio para decision funcional."
        )
        validation_summary = "La vista es calculable, pero usa variables tecnicas que pueden confundir al usuario funcional."
    elif possibly_invented:
        recommended_action = "Revisa la sugerencia del agente porque fue ajustada a columnas reales disponibles."
        validation_summary = "La vista fue corregida por el backend para evitar usar campos inexistentes."
    else:
        recommended_action = "Haz drill-down sobre una barra, revisa tickets y envia la seleccion al agente para accion."
        validation_summary = "Lista para operacion: usa datos reales, es graficable y tiene evidencias para revisar."
    return {
        "status": "ok" if chart_is_buildable and not missing else "warning",
        "quality_score": quality_score,
        "operation_ready": operation_ready,
        "llm_used_available_data": not missing,
        "chose_interpretable_variables": bool(interpretable),
        "chart_is_buildable": chart_is_buildable,
        "uses_real_data": chart_is_buildable,
        "requires_data": bool(missing),
        "uses_technical_variable": bool(uses_technical),
        "possibly_invented": bool(possibly_invented),
        "evidence_returned": int(evidence_returned),
        "validation_summary": validation_summary,
        "recommended_action": recommended_action,
        "warnings": warnings,
        "missing": missing,
        "source": "duckdb",
    }


def _empty_response(
    *,
    run_id: str,
    visualization: dict[str, Any],
    warning: str,
    missing: list[str] | None = None,
    semantic_items: list[dict[str, Any]] | None = None,
) -> ConversationChartDataResponse:
    return ConversationChartDataResponse.model_validate(
        {
            "run_id": run_id,
            "visualization_id": str(visualization.get("id") or ""),
            "title": str(visualization.get("title") or "Visualizacion"),
            "chart_type": str(visualization.get("chart_type") or "bar"),
            "x": str(visualization.get("x") or visualization.get("group_by") or ""),
            "metric": str(visualization.get("metric") or visualization.get("y") or "count"),
            "aggregation": str(visualization.get("aggregation") or "count"),
            "total_records": 0,
            "evidence_returned": 0,
            "evidence_truncated": False,
            "series": [],
            "evidence_samples": [],
            "samples_by_key": {},
            "semantic_dictionary": semantic_items or [],
            "validation": {
                "status": "warning",
                "quality_score": 0,
                "operation_ready": False,
                "llm_used_available_data": False,
                "chose_interpretable_variables": False,
                "chart_is_buildable": False,
                "uses_real_data": False,
                "requires_data": True,
                "uses_technical_variable": False,
                "possibly_invented": True,
                "evidence_returned": 0,
                "validation_summary": "No esta listo para operacion porque no hay datos suficientes para graficar.",
                "recommended_action": "Revisa que existan evidencias guardadas y que la vista tenga dimension y metrica validas.",
                "warnings": [warning],
                "missing": missing or [],
                "source": "duckdb",
            },
        }
    )


def _normalize_name(value: Any) -> str:
    text = _repair_mojibake(str(value or "").lower())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(ch for ch in text if ch.isalnum())
    for damaged, repaired in {
        "nmero": "numero",
        "catlogo": "catalogo",
        "subcategora": "subcategoria",
        "configuracin": "configuracion",
        "asignacin": "asignacion",
        "duracin": "duracion",
        "resolucin": "resolucion",
        "diagnstico": "diagnostico",
    }.items():
        text = text.replace(damaged, repaired)
    return text


def _repair_mojibake(value: str) -> str:
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def _extract_preview_value(preview: Any, field_name: str) -> str | None:
    target_key = _normalize_name(field_name)
    for chunk in str(preview or "").split("|"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if _normalize_name(key) == target_key:
            text = value.strip()
            return text or None
    return None


def _extract_preview_values(preview: Any, lookup: dict[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    if not lookup:
        return values
    for chunk in str(preview or "").split("|"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        target = lookup.get(_normalize_name(key))
        if not target or target in values:
            continue
        text = value.strip()
        if text:
            values[target] = text
    return values


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.replace(r"[^\d,.\-]", "", regex=True)
    has_comma_decimal = text.str.contains(",", regex=False)
    normalized = text.where(
        ~has_comma_decimal,
        text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(normalized, errors="coerce")


def _unique_texts(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text[:500]


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if _as_text(value):
            return value
    return None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
