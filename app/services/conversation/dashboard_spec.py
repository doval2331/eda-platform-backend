from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.db import AnalysisRun
from app.schemas import ConversationDashboardSpec
from app.services.agents.llm_agent import design_dashboard_with_llm
from app.services.conversation.semantic_dictionary import (
    get_semantic_dictionary,
    load_configured_semantic_variables,
    semantic_dictionary_path,
)
from app.services.datasets.dataset_store import get_dataset_meta
from app.services.runs.duckdb_store import (
    list_agent_cluster_insights,
    list_agent_decisions,
    list_agent_recommendations,
    list_chat_messages,
    load_run_evidences,
)

BUSINESS_COLUMNS = (
    "affected_service",
    "business_service",
    "service",
    "servicio_afectado",
    "service_line",
    "priority",
    "prioridad",
    "severity",
    "category",
    "categoria",
    "sector",
    "assignment_group",
    "Assignment_Group",
    "support_channel",
    "canal_entrada",
    "channel",
    "contact_type",
    "Tipo de contacto",
    "status",
    "Status",
    "Estado de la incidencia",
    "Valor",
    "causa_raiz_simulada",
    "Diagnóstico",
    "Causada por Cambio",
    "ci_cat",
    "CI_Cat",
    "Nombre",
    "ci_subcat",
    "CI_Subcat",
    "Subcategoría",
    "Elemento de configuración",
    "Closure_Code",
    "close_code",
    "Código de resolución",
    "Tipo de Solución",
    "category",
    "Category",
    "Catálogo",
    "Catálogo.1",
    "Catálogo.2",
    "company",
    "Empresa",
    "Empresa.1",
    "Impacto",
    "Urgencia",
    "Prioridad",
    "Grupo de asignación",
    "task.business_service",
    "task.assignment_group",
    "task.company",
    "task.contact_type",
    "task.impact",
    "task.priority",
    "task.state",
    "task.u_task_category",
    "task.u_technical_subservice",
    "short_description",
)

NUMERIC_METRIC_COLUMNS = (
    "sla_breach_rate",
    "avg_resolution_hours",
    "tiempo_resolucion_horas",
    "resolution_hours",
    "operational_risk_score",
    "business_impact_score",
    "impact_score",
    "urgency_score",
    "monthly_tickets",
    "critical_incidents",
    "downtime_hours",
    "customer_satisfaction",
    "no_of_reassignments",
    "No_of_Reassignments",
    "number_of_reassignments",
    "reassignment_count",
    "task.reassignment_count",
    "No_of_Related_Interactions",
    "No_of_Related_Incidents",
    "No_of_Related_Changes",
    "related_interactions_count",
    "related_incidents_count",
    "related_changes_count",
    "number_cnt",
    "task.business_duration",
    "task.calendar_duration",
    "task.time_worked",
    "business_duration",
    "calendar_duration",
    "duration",
    "Duración",
    "Tiempo de trabajo",
    "Volver a abrir recuento",
)

TECHNICAL_COLUMNS = {"run_id", "evidence_index", "evidence_id", "x", "y", "cluster_label", "incident_id", "ticket_id"}
VALID_CHART_TYPES = {"bar", "line", "scatter", "priority_matrix", "distribution", "ranking", "heatmap", "boxplot"}
VALID_PRIORITIES = {"alta", "media", "baja"}
VALID_AUDIENCES = {"funcional", "experto", "ambos"}
VALID_ACTION_TYPES = {"chart", "chat", "conclusion"}
VALID_VARIABLE_ROLES = {"business", "metric", "technical", "identifier", "unknown"}
VALID_CONFIDENCE = {"alta", "media", "baja"}
VALID_EVIDENCE_SOURCES = {"dataset", "pipeline", "cluster", "insight", "llm", "user"}
DASHBOARD_SPEC_SCHEMA_VERSION = "conversation-dashboard/v1"

SEMANTIC_VARIABLES: dict[str, dict[str, Any]] = {
    "affected_service": {
        "label": "Servicio afectado",
        "role": "business",
        "description": "Servicio o aplicacion impactada por la incidencia.",
        "recommended_use": "Usala como eje o segmento para encontrar focos operativos.",
    },
    "business_service": {
        "label": "Servicio de negocio",
        "role": "business",
        "description": "Servicio de negocio asociado a la incidencia.",
        "recommended_use": "Usala para explicar impacto y priorizacion en lenguaje funcional.",
    },
    "service": {
        "label": "Servicio",
        "role": "business",
        "description": "Servicio o aplicacion asociada a la incidencia.",
        "recommended_use": "Usala como eje para detectar concentracion de casos.",
    },
    "servicio_afectado": {
        "label": "Servicio afectado",
        "role": "business",
        "description": "Servicio o aplicacion impactada por la incidencia.",
        "recommended_use": "Usala como eje o segmento para encontrar focos operativos.",
    },
    "avg_resolution_hours": {
        "label": "Tiempo promedio de resolucion",
        "role": "metric",
        "description": "Promedio de horas usadas para resolver las incidencias.",
        "recommended_use": "Usala como metrica para detectar servicios o grupos lentos.",
    },
    "resolution_hours": {
        "label": "Horas de resolucion",
        "role": "metric",
        "description": "Horas necesarias para resolver la incidencia.",
        "recommended_use": "Usala para detectar demoras, extremos o segmentos con mayor tiempo de atencion.",
    },
    "tiempo_resolucion_horas": {
        "label": "Tiempo promedio de resolucion",
        "role": "metric",
        "description": "Promedio de horas usadas para resolver las incidencias.",
        "recommended_use": "Usala como metrica para detectar servicios o grupos lentos.",
    },
    "sla_breach_rate": {
        "label": "Tasa de incumplimiento SLA",
        "role": "metric",
        "description": "Proporcion de casos con incumplimiento de SLA.",
        "recommended_use": "Usala como metrica de riesgo operativo.",
    },
    "business_impact_score": {
        "label": "Impacto de negocio",
        "role": "metric",
        "description": "Puntaje estimado de impacto funcional u operativo.",
        "recommended_use": "Usala como metrica para priorizar acciones y comparar segmentos.",
    },
    "operational_risk_score": {
        "label": "Riesgo operativo",
        "role": "metric",
        "description": "Puntaje de riesgo asociado al comportamiento operativo de las incidencias.",
        "recommended_use": "Usala para ordenar grupos, servicios o categorias que requieren atencion.",
    },
    "impact_score": {
        "label": "Impacto",
        "role": "metric",
        "description": "Puntaje de impacto estimado para la incidencia o grupo.",
        "recommended_use": "Usala para priorizar evidencias de mayor efecto operativo.",
    },
    "urgency_score": {
        "label": "Urgencia",
        "role": "metric",
        "description": "Puntaje de urgencia estimado para la incidencia o grupo.",
        "recommended_use": "Usala para ordenar evidencias que requieren revision temprana.",
    },
    "no_of_reassignments": {
        "label": "Cantidad de reasignaciones",
        "role": "metric",
        "description": "Numero de veces que una incidencia fue reasignada entre equipos o responsables.",
        "recommended_use": "Usala para detectar complejidad operativa, friccion de soporte o mala derivacion inicial.",
    },
    "number_of_reassignments": {
        "label": "Cantidad de reasignaciones",
        "role": "metric",
        "description": "Numero de veces que una incidencia fue reasignada entre equipos o responsables.",
        "recommended_use": "Usala para detectar complejidad operativa, friccion de soporte o mala derivacion inicial.",
    },
    "reassignment_count": {
        "label": "Cantidad de reasignaciones",
        "role": "metric",
        "description": "Numero de reasignaciones acumuladas para una incidencia o grupo.",
        "recommended_use": "Usala como metrica operativa para identificar casos con mucha transferencia.",
    },
    "priority": {
        "label": "Prioridad",
        "role": "business",
        "description": "Nivel de prioridad asignado a la incidencia.",
        "recommended_use": "Usala como eje para priorizar volumen, impacto o SLA.",
    },
    "prioridad": {
        "label": "Prioridad",
        "role": "business",
        "description": "Nivel de prioridad asignado a la incidencia.",
        "recommended_use": "Usala como eje para priorizar volumen, impacto o SLA.",
    },
    "category": {
        "label": "Categoria",
        "role": "business",
        "description": "Categoria funcional o tecnica de la incidencia.",
        "recommended_use": "Usala para explicar patrones en lenguaje de negocio.",
    },
    "categoria": {
        "label": "Categoria",
        "role": "business",
        "description": "Categoria funcional o tecnica de la incidencia.",
        "recommended_use": "Usala para explicar patrones en lenguaje de negocio.",
    },
    "status": {
        "label": "Estado",
        "role": "business",
        "description": "Estado operativo de la incidencia.",
        "recommended_use": "Usala para separar casos abiertos, cerrados o pendientes.",
    },
    "assignment_group": {
        "label": "Grupo asignado",
        "role": "business",
        "description": "Equipo o grupo responsable asociado a la incidencia.",
        "recommended_use": "Usala para detectar concentracion de carga o reasignaciones.",
    },
    "ci_cat": {
        "label": "Categoria CI",
        "role": "business",
        "description": "Categoria del elemento de configuracion o aplicacion relacionada.",
        "recommended_use": "Usala para traducir patrones tecnicos a familias funcionales.",
    },
    "short_description": {
        "label": "Descripcion corta",
        "role": "business",
        "description": "Texto breve de la incidencia; puede aportar contexto para interpretar patrones.",
        "recommended_use": "Usala para resumenes, no como eje principal de una grafica agregada.",
    },
    "cluster_label": {
        "label": "Grupo tecnico",
        "role": "technical",
        "description": "Identificador del cluster creado por el algoritmo.",
        "recommended_use": "Usalo para trazabilidad tecnica y drill-down, no como metrica funcional.",
        "avoid_as_metric": True,
    },
    "x": {
        "label": "Coordenada X",
        "role": "technical",
        "description": "Coordenada de proyeccion del mapa.",
        "recommended_use": "Usala solo para mapas tecnicos de dispersion.",
        "avoid_as_metric": True,
    },
    "y": {
        "label": "Coordenada Y",
        "role": "technical",
        "description": "Coordenada de proyeccion del mapa.",
        "recommended_use": "Usala solo para mapas tecnicos de dispersion.",
        "avoid_as_metric": True,
    },
    "incident_id": {
        "label": "ID de incidencia",
        "role": "identifier",
        "description": "Identificador unico del ticket o incidencia.",
        "recommended_use": "Usalo solo para drill-down y trazabilidad de evidencias.",
        "avoid_as_metric": True,
    },
    "ticket_id": {
        "label": "ID de ticket",
        "role": "identifier",
        "description": "Identificador unico del ticket.",
        "recommended_use": "Usalo solo para drill-down y trazabilidad de evidencias.",
        "avoid_as_metric": True,
    },
}


def build_dashboard_spec(
    *,
    db: Session,
    user_id: str,
    run_id: str | None,
    insights: list[dict[str, Any]],
) -> ConversationDashboardSpec:
    context = build_dashboard_context(db=db, user_id=user_id, run_id=run_id, insights=insights)
    fallback = _fallback_spec(context, insights)
    llm_result, raw_spec = design_dashboard_with_llm(user_payload=context)
    if llm_result.used and isinstance(raw_spec, dict):
        spec = _sanitize_llm_spec(raw_spec, fallback)
        spec.llm_used = True
        spec.llm_mode = llm_result.mode
        spec.llm_detail = llm_result.detail
        return _refresh_dashboard_contract(spec)
    fallback.llm_used = False
    fallback.llm_mode = llm_result.mode
    fallback.llm_detail = llm_result.detail
    return _refresh_dashboard_contract(fallback)


def build_dashboard_context(
    *,
    db: Session,
    user_id: str,
    run_id: str | None,
    insights: list[dict[str, Any]],
) -> dict[str, Any]:
    run_ids = _run_ids_from_context(run_id, insights)
    run_rows = _load_run_rows(db, run_ids)
    semantic_project_id = _semantic_project_id(run_rows)
    evidence_by_run = _load_evidences(run_ids)
    all_evidences = [df for df in evidence_by_run.values() if not df.empty]
    merged_evidences = pd.concat(all_evidences, ignore_index=True) if all_evidences else pd.DataFrame()
    dataset_summaries = _dataset_summaries(run_rows, user_id=user_id)
    columns = _column_summary(merged_evidences, project_id=semantic_project_id)
    semantic_variables = _semantic_variables(columns, project_id=semantic_project_id)
    metrics = _metrics_summary(run_rows, merged_evidences, insights)
    clusters = _cluster_summary(merged_evidences)
    evidence_summary = _evidence_summary(merged_evidences)
    agent_recommendations = _safe_agent_payload(run_ids, list_agent_recommendations)
    agent_cluster_insights = _safe_agent_payload(run_ids, list_agent_cluster_insights)
    agent_decisions = _safe_agent_payload(run_ids, lambda rid: list_agent_decisions(rid, limit=6))
    recommendation_feedback = _feedback_summary(run_ids=run_ids, user_id=user_id)
    dashboard_usage_summary = _usage_summary(run_ids=run_ids, user_id=user_id)
    operational_readiness = _operational_readiness(
        run_ids=run_ids,
        evidence_by_run=evidence_by_run,
        evidence_summary=evidence_summary,
        semantic_variables=semantic_variables,
        insights=insights,
        project_id=semantic_project_id,
    )

    return {
        "instruction": (
            "Disena una propuesta de dashboard para este analisis. Devuelve JSON estructurado y valido. "
            "Sugiere graficos, conclusiones, preguntas y acciones. Diferencia usuario funcional y experto. "
            "No inventes datos; usa solo el contexto recibido."
        ),
        "run_id": run_id,
        "run_ids": run_ids,
        "dataset_summary": dataset_summaries,
        "columns": columns,
        "semantic_variables": semantic_variables,
        "semantic_project_id": semantic_project_id,
        "metrics": metrics,
        "clusters": clusters,
        "insights": [_compact_insight(item) for item in insights[:30]],
        "parameters": [_run_parameters(row) for row in run_rows],
        "execution_history": _execution_history(run_rows, agent_decisions),
        "existing_recommendations": agent_recommendations[:20],
        "recommendation_feedback": recommendation_feedback,
        "dashboard_usage_summary": dashboard_usage_summary,
        "cluster_interpretations": agent_cluster_insights[:20],
        "evidence_summary": evidence_summary,
        "operational_readiness": operational_readiness,
    }


def _semantic_project_id(run_rows: list[AnalysisRun]) -> str:
    project_ids = _unique(
        [str(getattr(row, "project_id", "") or "") for row in run_rows if getattr(row, "project_id", None)]
    )
    return project_ids[0] if len(project_ids) == 1 else ""


def _feedback_summary(*, run_ids: list[str], user_id: str | None) -> dict[str, Any]:
    """Summarize persisted dashboard feedback without exposing full chat history to the LLM."""
    rows: list[dict[str, Any]] = []
    for rid in run_ids[:8]:
        try:
            messages = list_chat_messages(run_id=rid, user_id=user_id, limit=500)
        except Exception:
            continue
        for message in messages:
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if metadata.get("kind") != "conversation_dashboard_feedback":
                continue
            helpful = metadata.get("helpful")
            if isinstance(helpful, str):
                helpful = helpful.strip().lower() in {"true", "1", "yes", "si", "util", "useful"}
            status = "useful" if helpful is True else "not_useful" if helpful is False else "unknown"
            recommendation_id = str(
                metadata.get("recommendation_id")
                or metadata.get("target_id")
                or metadata.get("id")
                or ""
            ).strip()
            title = str(
                metadata.get("recommendation_title")
                or metadata.get("target_title")
                or metadata.get("title")
                or recommendation_id
                or "Recomendacion sin titulo"
            ).strip()
            rows.append(
                {
                    "run_id": rid,
                    "recommendation_id": recommendation_id,
                    "title": title,
                    "status": status,
                    "chart_validated": bool(metadata.get("chart_validated")),
                    "has_warning": bool(metadata.get("has_warning")),
                    "visualization_id": str(metadata.get("visualization_id") or ""),
                    "evidence_materialized": bool(metadata.get("evidence_materialized")),
                    "evidence_records": int(_safe_float(metadata.get("evidence_records")) or 0),
                    "created_at": str(message.get("created_at") or ""),
                }
            )

    useful = [row for row in rows if row["status"] == "useful"]
    not_useful = [row for row in rows if row["status"] == "not_useful"]
    useful_ids = _unique([row["recommendation_id"] for row in useful if row["recommendation_id"]])
    not_useful_ids = _unique([row["recommendation_id"] for row in not_useful if row["recommendation_id"]])
    useful_titles = _unique([row["title"] for row in useful if row["title"]])
    not_useful_titles = _unique([row["title"] for row in not_useful if row["title"]])
    return {
        "total": len(rows),
        "useful": len(useful),
        "not_useful": len(not_useful),
        "useful_recommendation_ids": useful_ids[:20],
        "not_useful_recommendation_ids": not_useful_ids[:20],
        "useful_titles": useful_titles[:12],
        "not_useful_titles": not_useful_titles[:12],
        "recent": rows[-12:],
        "guidance": (
            "Prioriza recomendaciones parecidas a las marcadas como utiles y revisa o reformula "
            "las marcadas como no utiles. No ocultes una recomendacion si la evidencia real la respalda, "
            "pero explica por que vuelve a aparecer."
        )
        if rows
        else "Sin feedback persistido del usuario para este dashboard.",
    }


def _usage_summary(*, run_ids: list[str], user_id: str | None) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for rid in run_ids[:8]:
        try:
            messages = list_chat_messages(run_id=rid, user_id=user_id, limit=500)
        except Exception:
            continue
        for message in messages:
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if metadata.get("kind") != "conversation_dashboard_event":
                continue
            event_type = str(metadata.get("event_type") or "unknown").strip() or "unknown"
            events.append(
                {
                    "run_id": rid,
                    "event_type": event_type,
                    "target_id": str(metadata.get("target_id") or metadata.get("visualization_id") or ""),
                    "target_title": str(metadata.get("target_title") or metadata.get("visualization_title") or ""),
                    "ticket_count": int(_safe_float(metadata.get("ticket_count")) or 0),
                    "created_at": str(message.get("created_at") or ""),
                }
            )
    counts: dict[str, int] = {}
    for event in events:
        event_type = event["event_type"]
        counts[event_type] = counts.get(event_type, 0) + 1
    return {
        "total": len(events),
        "events_by_type": counts,
        "charts_opened": counts.get("recommendation_graph_opened", 0) + counts.get("visualization_selected", 0),
        "tickets_sent_to_agent": counts.get("tickets_sent_to_agent", 0),
        "exports": counts.get("tickets_exported", 0),
        "reports_prepared": counts.get("operational_selection_saved", 0),
        "recent": events[-12:],
        "guidance": (
            "Usa estas senales para priorizar recomendaciones que el usuario realmente abre, manda al agente "
            "o convierte en evidencia operativa."
        )
        if events
        else "Sin eventos de uso persistidos para este dashboard.",
    }


def _run_ids_from_context(run_id: str | None, insights: list[dict[str, Any]]) -> list[str]:
    if run_id:
        return [run_id]
    seen: list[str] = []
    for item in insights:
        candidate = str(item.get("run_id") or "").strip()
        if candidate and candidate not in seen:
            seen.append(candidate)
        if len(seen) >= 8:
            break
    return seen


def _load_run_rows(db: Session, run_ids: list[str]) -> list[AnalysisRun]:
    if not run_ids:
        return []
    rows = db.query(AnalysisRun).filter(AnalysisRun.id.in_(run_ids)).all()
    by_id = {row.id: row for row in rows}
    return [by_id[rid] for rid in run_ids if rid in by_id]


def _load_evidences(run_ids: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for rid in run_ids[:8]:
        try:
            result[rid] = load_run_evidences(rid)
        except Exception:
            result[rid] = pd.DataFrame()
    return result


def _dataset_summaries(rows: list[AnalysisRun], *, user_id: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = row.dataset_id or row.source_id
        meta: dict[str, Any] = {}
        if dataset_id:
            try:
                meta = get_dataset_meta(str(dataset_id), user_id=user_id)
            except Exception:
                meta = {}
        summaries.append(
            {
                "run_id": row.id,
                "dataset_id": dataset_id,
                "dataset_name": meta.get("filename") or row.source_name or row.project_name or row.id,
                "project_name": row.project_name,
                "source_name": row.source_name,
                "source_type": row.source_type,
                "rows": _safe_int(meta.get("n_rows")) or row.n_samples,
                "columns": _safe_int(meta.get("n_cols")),
                "numeric_columns": list(meta.get("numeric_columns") or [])[:20],
                "categorical_columns": list(meta.get("categorical_columns") or [])[:20],
                "all_columns": list(meta.get("all_columns") or [])[:40],
            }
        )
    return summaries


def _column_summary(df: pd.DataFrame, *, project_id: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {"available": [], "business": [], "numeric": [], "technical": []}
    available = list(map(str, df.columns.tolist()))
    business_keys = {_column_key(col) for col in BUSINESS_COLUMNS}
    metric_keys = {_column_key(col) for col in NUMERIC_METRIC_COLUMNS}
    technical_keys = {_column_key(col) for col in TECHNICAL_COLUMNS}
    business: list[str] = []
    numeric: list[str] = []
    technical: list[str] = []
    for col in available:
        key = _column_key(col)
        base = _semantic_base(col, project_id=project_id)
        role = str(base.get("role") or "")
        if key in technical_keys or role in {"technical", "identifier"} or _looks_identifier(col):
            technical.append(col)
        if key in business_keys or role == "business":
            business.append(col)
        if (pd.api.types.is_numeric_dtype(df[col]) or key in metric_keys or role == "metric") and col not in technical:
            numeric.append(col)
    return {
        "available": available[:80],
        "business": _unique(business)[:30],
        "numeric": _unique(numeric)[:30],
        "technical": _unique(technical),
        "column_count": len(available),
    }


def _semantic_variables(columns: dict[str, Any], *, project_id: str | None = None) -> list[dict[str, Any]]:
    available = list(columns.get("available") or [])
    business = set(columns.get("business") or [])
    numeric = set(columns.get("numeric") or [])
    technical = set(columns.get("technical") or [])
    items: list[dict[str, Any]] = []

    for name in available[:80]:
        text = str(name)
        base = dict(_semantic_base(text, project_id=project_id))
        role = base.get("role")
        if not role:
            if text in technical or _looks_identifier(text):
                role = "technical" if text in technical else "identifier"
            elif text in business:
                role = "business"
            elif text in numeric:
                role = "metric"
            else:
                role = "unknown"
        label = base.get("label") or _humanize_column(text)
        avoid = bool(base.get("avoid_as_metric")) or role in {"technical", "identifier"}
        items.append(
            {
                "name": text,
                "label": label,
                "role": role,
                "description": base.get("description") or _semantic_description(text, role),
                "recommended_use": base.get("recommended_use") or _semantic_use(label, role),
                "avoid_as_metric": avoid,
                "can_chart": bool(base.get("can_chart", True)) and role not in {"identifier"},
                "semantic_type": str(base.get("semantic_type") or ""),
            }
        )
    return items


def _metrics_summary(
    rows: list[AnalysisRun],
    df: pd.DataFrame,
    insights: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "records_count": int(len(df)) if not df.empty else sum(row.n_samples for row in rows),
        "runs_count": len(rows),
        "selected_insights_count": len(insights),
        "key_metrics": _unique([str(item.get("metric_label") or "") for item in insights if item.get("metric_label")])[:12],
    }
    column_lookup = {_column_key(col): col for col in df.columns} if not df.empty else {}
    for col in NUMERIC_METRIC_COLUMNS:
        actual_col = col if col in df.columns else column_lookup.get(_column_key(col))
        if not actual_col:
            continue
        values = pd.to_numeric(df[actual_col], errors="coerce").dropna()
        if values.empty:
            continue
        metrics[actual_col] = {
            "avg": _round(values.mean()),
            "max": _round(values.max()),
            "min": _round(values.min()),
            "count": int(values.shape[0]),
        }
    return metrics


def _cluster_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    cluster_columns = _candidate_columns(df, ("cluster_label",))
    if df.empty or not cluster_columns:
        return []
    cluster_col = cluster_columns[0]
    cluster_df = df.copy()
    cluster_df[cluster_col] = pd.to_numeric(cluster_df[cluster_col], errors="coerce")
    cluster_df = cluster_df[cluster_df[cluster_col].notna()]
    if cluster_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for label, group in cluster_df.groupby(cluster_col):
        if len(rows) >= 15:
            break
        rows.append(
            {
                "cluster_label": int(label),
                "count": int(len(group)),
                "top_service": _top_value(group, ("affected_service", "servicio_afectado", "service_line")),
                "top_priority": _top_value(group, ("priority", "prioridad", "severity")),
                "top_category": _top_value(group, ("category", "categoria", "sector")),
                "avg_risk": _mean_first(group, ("operational_risk_score", "business_impact_score")),
                "avg_resolution_hours": _mean_first(group, ("avg_resolution_hours", "tiempo_resolucion_horas")),
                "avg_sla": _mean_first(group, ("sla_breach_rate",)),
            }
        )
    rows.sort(key=lambda item: (item.get("avg_risk") or 0, item["count"]), reverse=True)
    return rows


def _safe_agent_payload(run_ids: list[str], loader) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for rid in run_ids[:4]:
        try:
            items = loader(rid)
        except Exception:
            items = []
        for item in items:
            if isinstance(item, dict):
                payload.append(_jsonable(item))
    return payload


def _run_parameters(row: AnalysisRun) -> dict[str, Any]:
    metrics = {}
    try:
        payload = json.loads(row.result_json or "{}")
        metrics = dict(payload.get("metrics") or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        metrics = {}
    return {
        "run_id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "reduction_method": row.reduction_method,
        "seed": row.seed,
        "n_samples": row.n_samples,
        "outliers_count": row.outliers_count,
        "silhouette": _safe_float(row.silhouette) if row.silhouette is not None else metrics.get("silhouette"),
        "davies_bouldin": _safe_float(row.davies_bouldin) if row.davies_bouldin is not None else metrics.get("davies_bouldin"),
        "project_name": row.project_name,
        "source_name": row.source_name,
        "source_type": row.source_type,
    }


def _execution_history(rows: list[AnalysisRun], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = [
        {
            "source": "pipeline",
            "title": f"Ejecucion {row.id[:8]}",
            "description": f"{row.reduction_method}, seed {row.seed}, {row.n_samples} registros.",
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }
        for row in rows[:8]
    ]
    for decision in decisions[:8]:
        history.append(
            {
                "source": "llm" if "llm" in str(decision.get("model_name") or "").lower() else "pipeline",
                "title": str(decision.get("decision_type") or decision.get("agent_name") or "Decision de agente"),
                "description": str(decision.get("response") or "")[:500],
                "created_at": str(decision.get("created_at") or ""),
            }
        )
    return history[:16]


def _evidence_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"records_count": 0, "sample_previews": []}
    previews = []
    if "preview" in df.columns:
        previews = [str(value)[:180] for value in df["preview"].dropna().astype(str).head(6).tolist()]
    return {
        "records_count": int(len(df)),
        "columns_count": int(len(df.columns)),
        "sample_previews": previews,
    }


def _operational_readiness(
    *,
    run_ids: list[str],
    evidence_by_run: dict[str, pd.DataFrame],
    evidence_summary: dict[str, Any],
    semantic_variables: list[dict[str, Any]],
    insights: list[dict[str, Any]],
    project_id: str | None = None,
) -> dict[str, Any]:
    evidence_records = int(evidence_summary.get("records_count") or 0)
    evidence_runs = sum(1 for rid in run_ids if not evidence_by_run.get(rid, pd.DataFrame()).empty)
    selected_insights = len(insights)
    configured_count = len(load_configured_semantic_variables(project_id=project_id))
    business_variables = [
        item for item in semantic_variables if str(item.get("role") or "").lower() == "business"
    ]
    run_scope = "empty" if not run_ids else "single_run" if len(run_ids) == 1 else "multi_run"
    warnings: list[str] = []

    if run_scope == "multi_run":
        warnings.append(
            "La vista combina varias ejecuciones; selecciona una ejecucion concreta para decisiones operativas."
        )
    if evidence_records <= 0:
        warnings.append("No hay tickets o evidencias materializadas para drill-down operativo.")
    if selected_insights <= 0:
        warnings.append("No hay hallazgos guardados; el dashboard queda sin foco de analisis.")
    if not business_variables:
        warnings.append("No se detectaron variables de negocio suficientes para graficos funcionales.")
    if configured_count <= 0:
        warnings.append("El diccionario semantico usa la base por defecto; conviene gobernarlo por configuracion.")

    if (
        run_scope == "single_run"
        and evidence_records > 0
        and selected_insights > 0
        and business_variables
    ):
        status = "operational"
        summary = "Listo para analisis operativo: hay evidencia real, hallazgos guardados y variables interpretables."
        decision_level = "operational"
        functional_message = "Hay casos reales y hallazgos guardados para tomar decisiones con respaldo."
        expert_message = (
            "Vista operativa: existen evidencias materializadas, hallazgos seleccionados y variables "
            "de negocio para graficos y drill-down."
        )
        recommended_next_step = "Abre un grafico, filtra los casos relacionados y envia la seleccion al agente."
    elif evidence_records > 0 or selected_insights > 0:
        status = "interpretive"
        summary = (
            "Lectura interpretativa con soporte parcial; revisa advertencias antes de tomar decisiones."
        )
        decision_level = "assisted_review"
        functional_message = (
            "La lectura sirve para orientar la revision, pero necesita completar casos o hallazgos "
            "antes de decidir."
        )
        expert_message = (
            "Soporte parcial: hay evidencia o hallazgos, pero falta completar la cadena de decision "
            "con datos, variables de negocio o seleccion de evidencias."
        )
        recommended_next_step = "Completa la base de evidencia o guarda hallazgos antes de usarlo como decision."
    else:
        status = "limited"
        summary = "Contexto limitado: ejecuta el pipeline y guarda hallazgos para habilitar analisis operativo."
        decision_level = "interpretive"
        functional_message = "Todavia falta contexto real para convertir esta lectura en acciones."
        expert_message = "Contexto limitado: no hay evidencia materializada ni hallazgos suficientes para operar."
        recommended_next_step = "Ejecuta el analisis, guarda hallazgos y vuelve a generar el dashboard."

    return {
        "status": status,
        "run_scope": run_scope,
        "active_run_id": run_ids[0] if len(run_ids) == 1 else "",
        "run_ids": run_ids[:8],
        "decision_level": decision_level,
        "evidence_materialized": evidence_records > 0,
        "evidence_records": evidence_records,
        "evidence_runs": evidence_runs,
        "selected_insights": selected_insights,
        "semantic_dictionary_configured": configured_count > 0,
        "semantic_dictionary_source": str(semantic_dictionary_path(project_id)),
        "semantic_dictionary_total": len(semantic_variables),
        "semantic_dictionary_configured_count": configured_count,
        "llm_validated": False,
        "summary": summary,
        "functional_message": functional_message,
        "expert_message": expert_message,
        "recommended_next_step": recommended_next_step,
        "warnings": warnings[:6],
    }


def _fallback_spec(context: dict[str, Any], insights: list[dict[str, Any]]) -> ConversationDashboardSpec:
    dataset = (context.get("dataset_summary") or [{}])[0] if context.get("dataset_summary") else {}
    metrics = context.get("metrics") or {}
    columns = context.get("columns") or {}
    semantic_variables = context.get("semantic_variables") or []
    clusters = context.get("clusters") or []
    findings = _fallback_findings(insights)
    visualizations = _fallback_visualizations(context, findings)
    active_id = visualizations[0]["id"] if visualizations else ""
    conclusions = _fallback_conclusions(findings, visualizations)
    payload = {
            "schema_version": DASHBOARD_SPEC_SCHEMA_VERSION,
            "executive_summary": {
                "title": "Resumen ejecutivo",
                "dataset_name": str(dataset.get("dataset_name") or dataset.get("source_name") or "Analisis actual"),
                "analysis_objective": "Priorizar hallazgos guardados y explicar que acciones conviene investigar.",
                "records_count": int(metrics.get("records_count") or dataset.get("rows") or 0),
                "columns_count": int(columns.get("column_count") or dataset.get("columns") or 0),
                "main_variables": (columns.get("business") or dataset.get("categorical_columns") or [])[:8],
                "key_metrics": metrics.get("key_metrics") or _unique([item.get("metric_label") for item in insights if item.get("metric_label")])[:8],
                "summary": _summary_text(metrics, insights, clusters),
            },
            "semantic_variables": semantic_variables,
            "priority_findings": findings,
            "agent_recommendations": _fallback_recommendations(context, findings),
            "suggested_visualizations": visualizations,
            "active_chart_default": {
                "visualization_id": active_id,
                "explanation": "Vista inicial sugerida por reglas locales con datos disponibles.",
            },
            "conclusions": conclusions,
            "evidence_line": _fallback_evidence_line(context, findings),
            "suggested_questions": {
                "functional_user": [
                    "Que hallazgo deberia revisar primero y por que?",
                    "Que conclusion puedo presentar a un usuario funcional?",
                    "Que accion concreta recomiendan estos datos?",
                ],
                "expert_user": [
                    "Que variables explican mejor los hallazgos seleccionados?",
                    "Que parametros o clusters conviene validar tecnicamente?",
                    "Que evidencia de DuckDB soporta la conclusion principal?",
                ],
            },
            "operational_readiness": context.get("operational_readiness") or {},
            "recommendation_feedback": context.get("recommendation_feedback") or {},
            "dashboard_usage_summary": context.get("dashboard_usage_summary") or {},
            "llm_used": False,
            "llm_mode": "rules",
            "llm_detail": "Especificacion generada por reglas locales.",
        }
    payload = _normalize_dashboard_payload(payload, payload)
    payload = _apply_dashboard_contract_metadata(payload)
    spec = ConversationDashboardSpec.model_validate(payload)
    return spec


def _fallback_findings(insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        insights,
        key=lambda item: _safe_float(item.get("metric_value")) or 0,
        reverse=True,
    )[:6]
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(ranked, start=1):
        priority = _priority_from_value(item.get("metric_value"), ranked)
        title = str(item.get("title") or f"Hallazgo {index}")
        findings.append(
            {
                "id": str(item.get("id") or f"finding-{index}"),
                "title": title,
                "priority": priority,
                "impact": _metric_sentence(item),
                "urgency": "Revisar primero" if priority == "alta" else "Revisar despues de los hallazgos de prioridad alta",
                "evidence": str(item.get("description") or "Hallazgo guardado por el usuario."),
                "suggested_action": "Analizar con el agente y validar la evidencia antes de reportar.",
                "related_variables": _related_variables(item),
                "suggested_question": f"Explica el hallazgo '{title}' y que accion recomiendas.",
            }
        )
    return findings


def _fallback_recommendations(context: dict[str, Any], findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = context.get("columns") or {}
    clusters = context.get("clusters") or []
    recs = [
        {
            "id": "rec-priority-review",
            "title": "Empezar por los hallazgos de mayor prioridad",
            "why_it_matters": "Ayuda a convertir el dashboard en una lista accionable y no solo descriptiva.",
            "what_to_analyze": "Impacto, urgencia, evidencia y metrica asociada de los hallazgos principales.",
            "recommended_next_step": "Aplicar la vista de ranking o matriz de prioridad.",
            "audience": "funcional",
            "action_type": "chart",
            "linked_visualization_id": "viz-priority-matrix",
            "evidence_needed": "Hallazgos guardados con prioridad y metricas asociadas.",
        },
        {
            "id": "rec-business-variables",
            "title": "Usar variables de negocio antes que identificadores tecnicos",
            "why_it_matters": "Servicios, prioridad, categoria o estado son mas interpretables que ids tecnicos.",
            "what_to_analyze": ", ".join((columns.get("business") or [])[:6]) or "Variables de negocio disponibles.",
            "recommended_next_step": "Crear una vista por dimension de negocio si hay datos suficientes.",
            "audience": "ambos",
            "action_type": "chart" if columns.get("business") else "chat",
            "linked_visualization_id": "viz-business-dimension" if columns.get("business") else "",
            "evidence_needed": "Columnas de negocio detectadas y conteos por dimension.",
        },
    ]
    if clusters:
        recs.append(
            {
                "id": "rec-cluster-validation",
                "title": "Validar clusters con evidencia tecnica",
                "why_it_matters": "Los clusters explican patrones, pero requieren contrastar muestras y metricas.",
                "what_to_analyze": "Tamanio de cluster, riesgo, servicio dominante y ejemplos representativos.",
                "recommended_next_step": "Usar scatter o ranking por cluster y preguntar al agente por diferencias.",
                "audience": "experto",
                "action_type": "chart",
                "linked_visualization_id": "viz-cluster-scatter",
                "evidence_needed": "Coordenadas de proyeccion, etiqueta de cluster y muestras relacionadas.",
            }
        )
    if findings:
        recs.append(
            {
                "id": "rec-functional-conclusion",
                "title": "Preparar una conclusion presentable",
                "why_it_matters": "El usuario funcional necesita resultado, evidencia y accion sugerida.",
                "what_to_analyze": findings[0]["title"],
                "recommended_next_step": "Preguntar al agente por una conclusion ejecutiva con evidencia.",
                "audience": "funcional",
                "action_type": "conclusion",
                "linked_visualization_id": "",
                "evidence_needed": "Hallazgos priorizados, evidencia resumida y grafico activo.",
            }
        )
    return recs[:5]


def _fallback_visualizations(context: dict[str, Any], findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = context.get("columns") or {}
    clusters = context.get("clusters") or []
    business = columns.get("business") or []
    numeric = columns.get("numeric") or []
    visuals: list[dict[str, Any]] = [
        {
            "id": "viz-priority-matrix",
            "title": "Matriz de prioridad de hallazgos",
            "chart_type": "priority_matrix",
            "x": "impact",
            "y": "urgency",
            "metric": "metric_value",
            "group_by": "priority",
            "aggregation": "score",
            "filters": [],
            "reason": "Permite identificar rapidamente que hallazgos requieren atencion.",
            "evidence_used": "Hallazgos seleccionados y metricas asociadas.",
            "question_answered": "Que hallazgo conviene investigar primero?",
            "audience": "funcional",
            "what_i_am_seeing": "Distribucion de hallazgos por prioridad estimada.",
            "why_it_matters": "Permite empezar por los grupos que requieren mayor atencion operativa.",
            "suggested_action": "Abrir el drill-down de prioridad alta y revisar evidencias relacionadas.",
            "drilldown": "priority",
        },
        {
            "id": "viz-finding-ranking",
            "title": "Ranking de hallazgos priorizados",
            "chart_type": "ranking",
            "x": "title",
            "y": "metric_value",
            "metric": "metric_value",
            "group_by": "metric_label",
            "aggregation": "max",
            "filters": [],
            "reason": "Ordena los hallazgos para revisar primero los de mayor intensidad.",
            "evidence_used": "Insights guardados en el dashboard conversacional.",
            "question_answered": "Cuales son los hallazgos mas relevantes?",
            "audience": "ambos",
            "what_i_am_seeing": "Ranking de hallazgos ordenados por intensidad de la metrica.",
            "why_it_matters": "Ayuda a decidir que revisar primero cuando hay muchas evidencias.",
            "suggested_action": "Seleccionar un hallazgo y pedir al agente una lectura causal con ejemplos.",
            "drilldown": "insight",
        },
        {
            "id": "viz-type-distribution",
            "title": "Distribucion por tipo de hallazgo",
            "chart_type": "distribution",
            "x": "metric_label",
            "y": "count",
            "metric": "count",
            "group_by": "metric_kind",
            "aggregation": "count",
            "filters": [],
            "reason": "Muestra si el dashboard esta concentrado en impacto, urgencia, SLA, clusters u otro foco.",
            "evidence_used": "Tipos de metricas en los hallazgos seleccionados.",
            "question_answered": "Sobre que tipo de problema esta concentrado el dashboard?",
            "audience": "funcional",
            "what_i_am_seeing": "Cantidad de hallazgos agrupados por tipo de metrica.",
            "why_it_matters": "Muestra si el analisis esta sesgado hacia volumen, prioridad, SLA u otro foco.",
            "suggested_action": "Cambiar el filtro de tipo para comparar la lectura del dashboard.",
            "drilldown": "metric_kind",
        },
    ]
    if clusters:
        visuals.append(
            {
                "id": "viz-cluster-scatter",
                "title": "Mapa de clusters y casos atipicos",
                "chart_type": "scatter",
                "x": "x",
                "y": "y",
                "metric": "cluster_label",
                "group_by": "cluster_label",
                "aggregation": "none",
                "filters": [],
                "reason": "Ayuda a validar separacion de grupos y casos atipicos.",
                "evidence_used": "Coordenadas 2D y etiquetas de cluster persistidas en DuckDB.",
                "question_answered": "Los grupos tienen separacion visual suficiente?",
                "audience": "experto",
                "what_i_am_seeing": "Mapa tecnico de agrupamientos y posibles casos atipicos.",
                "why_it_matters": "Permite validar si la reduccion dimensional separa patrones utiles.",
                "suggested_action": "Inspeccionar clusters grandes o aislados con muestras representativas.",
                "drilldown": "cluster",
            }
        )
    if business:
        visuals.append(
            {
                "id": "viz-business-dimension",
                "title": f"Distribucion por {business[0]}",
                "chart_type": "bar",
                "x": business[0],
                "y": "count",
                "metric": "count",
                "group_by": business[0],
                "aggregation": "count",
                "filters": [],
                "reason": "Traduce el analisis a una dimension de negocio interpretable.",
                "evidence_used": f"Columna {business[0]} detectada en evidencias.",
                "question_answered": f"Que valores de {business[0]} concentran mas registros?",
                "audience": "ambos",
                "what_i_am_seeing": f"Conteo de evidencias por {_humanize_column(str(business[0]))}.",
                "why_it_matters": "Traduce clusters tecnicos a una dimension de negocio accionable.",
                "suggested_action": "Abrir los segmentos con mayor volumen para revisar tickets relacionados.",
                "drilldown": "business_dimension",
            }
        )
    if numeric:
        visuals.append(
            {
                "id": "viz-numeric-distribution",
                "title": f"Distribucion de {numeric[0]}",
                "chart_type": "boxplot",
                "x": business[0] if business else "",
                "y": numeric[0],
                "metric": numeric[0],
                "group_by": business[0] if business else "",
                "aggregation": "distribution",
                "filters": [],
                "reason": "Permite revisar dispersion, extremos y diferencias por grupo.",
                "evidence_used": f"Columna numerica {numeric[0]} detectada en evidencias.",
                "question_answered": f"Hay valores extremos en {numeric[0]}?",
                "audience": "experto",
                "what_i_am_seeing": f"Dispersion y valores extremos de {_humanize_column(str(numeric[0]))}.",
                "why_it_matters": "Ayuda a detectar outliers o grupos con comportamiento anomalo.",
                "suggested_action": "Filtrar los extremos y revisar si corresponden a casos operativos reales.",
                "drilldown": "metric_distribution",
            }
        )
    return visuals[:6]


def _fallback_conclusions(
    findings: list[dict[str, Any]],
    visualizations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conclusions: list[dict[str, Any]] = []
    for index, finding in enumerate(findings[:4], start=1):
        related_chart = visualizations[min(index - 1, len(visualizations) - 1)]["id"] if visualizations else ""
        conclusions.append(
            {
                "id": f"conclusion-{index}",
                "conclusion": f"{finding['title']} debe revisarse con prioridad {finding['priority']}.",
                "evidence": finding.get("evidence") or "",
                "related_chart": related_chart,
                "related_metric": ", ".join(finding.get("related_variables") or []),
                "related_items": _unique([finding.get("id"), finding.get("title"), *finding.get("related_variables", [])])[:8],
                "confidence": finding["priority"] if finding["priority"] in VALID_CONFIDENCE else "media",
                "recommended_action": finding.get("suggested_action") or "Validar evidencia y accion.",
                "source": "rules",
                "evidence_quality": "Basada en hallazgos guardados y metricas disponibles; requiere validacion funcional.",
            }
        )
    return conclusions


def _fallback_evidence_line(
    context: dict[str, Any],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset = (context.get("dataset_summary") or [{}])[0] if context.get("dataset_summary") else {}
    metrics = context.get("metrics") or {}
    clusters = context.get("clusters") or []
    steps = [
        {
            "step": 1,
            "title": "Datos cargados",
            "description": f"{dataset.get('dataset_name') or 'Dataset'} con {metrics.get('records_count') or 0} registros evaluados.",
            "source": "dataset",
            "related_items": [str(dataset.get("dataset_id") or "")],
        },
        {
            "step": 2,
            "title": "Pipeline ejecutado",
            "description": "Se usaron parametros de reduccion, clustering y metricas persistidas para la ejecucion.",
            "source": "pipeline",
            "related_items": context.get("run_ids") or [],
        },
        {
            "step": 3,
            "title": "Clusters y evidencias",
            "description": f"{len(clusters)} grupos resumidos desde DuckDB.",
            "source": "cluster",
            "related_items": [str(item.get("cluster_label")) for item in clusters[:6]],
        },
        {
            "step": 4,
            "title": "Hallazgos seleccionados",
            "description": f"{len(findings)} hallazgos priorizados para el dashboard.",
            "source": "insight",
            "related_items": [item["id"] for item in findings[:8]],
        },
    ]
    return steps


def _sanitize_llm_spec(raw: dict[str, Any], fallback: ConversationDashboardSpec) -> ConversationDashboardSpec:
    fallback_payload = fallback.model_dump()
    payload = {
        "schema_version": DASHBOARD_SPEC_SCHEMA_VERSION,
        "executive_summary": _merge_dict(
            fallback_payload["executive_summary"],
            _dict_value(raw.get("executive_summary")),
        ),
        "semantic_variables": _sanitize_list(raw.get("semantic_variables"), fallback_payload["semantic_variables"], _sanitize_semantic_variable),
        "priority_findings": _sanitize_list(raw.get("priority_findings"), fallback_payload["priority_findings"], _sanitize_finding),
        "agent_recommendations": _sanitize_list(raw.get("agent_recommendations"), fallback_payload["agent_recommendations"], _sanitize_recommendation),
        "suggested_visualizations": _sanitize_list(raw.get("suggested_visualizations"), fallback_payload["suggested_visualizations"], _sanitize_visualization),
        "active_chart_default": _merge_dict(
            fallback_payload["active_chart_default"],
            _dict_value(raw.get("active_chart_default")),
        ),
        "conclusions": _sanitize_list(raw.get("conclusions"), fallback_payload["conclusions"], _sanitize_conclusion),
        "evidence_line": _sanitize_list(raw.get("evidence_line"), fallback_payload["evidence_line"], _sanitize_evidence_step),
        "suggested_questions": _merge_dict(
            fallback_payload["suggested_questions"],
            _dict_value(raw.get("suggested_questions")),
        ),
        "operational_readiness": fallback_payload.get("operational_readiness") or {},
        "recommendation_feedback": fallback_payload.get("recommendation_feedback") or {},
        "dashboard_usage_summary": fallback_payload.get("dashboard_usage_summary") or {},
        "llm_used": False,
        "llm_mode": "rules",
        "llm_detail": None,
    }
    if not payload["suggested_visualizations"]:
        payload["suggested_visualizations"] = fallback_payload["suggested_visualizations"]
    if not payload["active_chart_default"].get("visualization_id") and payload["suggested_visualizations"]:
        payload["active_chart_default"]["visualization_id"] = payload["suggested_visualizations"][0]["id"]
    payload = _normalize_dashboard_payload(payload, fallback_payload)
    payload = _apply_dashboard_contract_metadata(payload)
    return ConversationDashboardSpec.model_validate(payload)


def _refresh_dashboard_contract(spec: ConversationDashboardSpec) -> ConversationDashboardSpec:
    payload = spec.model_dump()
    payload = _apply_dashboard_contract_metadata(payload)
    return ConversationDashboardSpec.model_validate(payload)


def _apply_dashboard_contract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    warnings, risk_flags = _dashboard_contract_findings(payload)
    payload["schema_version"] = DASHBOARD_SPEC_SCHEMA_VERSION
    payload["contract_warnings"] = warnings
    payload["llm_risk_flags"] = risk_flags
    blocking_flags = {"missing_variable", "invalid_chart_type", "unlinked_chart", "requires_data"}
    payload["contract_status"] = "warning" if blocking_flags.intersection(risk_flags) else "valid"
    readiness = dict(payload.get("operational_readiness") or {})
    readiness["llm_validated"] = bool(payload.get("llm_used")) and not blocking_flags.intersection(risk_flags)
    if payload.get("llm_used") and blocking_flags.intersection(risk_flags):
        readiness.setdefault("warnings", [])
        readiness["warnings"] = _unique(
            [
                *readiness.get("warnings", []),
                "El LLM propuso elementos que el backend ajusto o marco con advertencias.",
            ]
        )[:8]
    payload["operational_readiness"] = readiness
    return payload


def _dashboard_contract_findings(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    semantic_items = payload.get("semantic_variables") or []
    semantic_map = {str(item.get("name")): item for item in semantic_items if item.get("name")}
    visualizations = payload.get("suggested_visualizations") or []
    visual_ids = {str(item.get("id")) for item in visualizations if item.get("id")}
    warnings: list[str] = []
    risk_flags: list[str] = []

    if payload.get("llm_used"):
        risk_flags.append("llm_generated")

    for visualization in visualizations:
        viz_id = str(visualization.get("id") or visualization.get("title") or "visualizacion")
        chart_type = str(visualization.get("chart_type") or "").lower()
        if chart_type not in VALID_CHART_TYPES:
            warnings.append(f"{viz_id}: tipo de grafico no soportado '{chart_type or 'vacio'}'.")
            risk_flags.append("invalid_chart_type")

        dimension = str(visualization.get("x") or visualization.get("group_by") or "").strip()
        metric = str(visualization.get("metric") or visualization.get("y") or "count").strip()
        if not dimension:
            warnings.append(f"{viz_id}: falta una dimension para graficar.")
            risk_flags.append("requires_data")
        else:
            _append_variable_contract_findings(
                warnings,
                risk_flags,
                semantic_map,
                variable=dimension,
                owner=viz_id,
                usage="dimension",
            )

        if metric and _column_key(metric) not in {"count", "conteo", "cantidad", "tickets", "incidencias", "registros"}:
            _append_variable_contract_findings(
                warnings,
                risk_flags,
                semantic_map,
                variable=metric,
                owner=viz_id,
                usage="metrica",
            )

    for recommendation in payload.get("agent_recommendations") or []:
        if str(recommendation.get("action_type") or "").lower() != "chart":
            continue
        linked_id = str(recommendation.get("linked_visualization_id") or "")
        if linked_id not in visual_ids:
            rec_id = str(recommendation.get("id") or recommendation.get("title") or "recomendacion")
            warnings.append(f"{rec_id}: recomendacion marcada como grafico sin visualizacion valida.")
            risk_flags.append("unlinked_chart")

    for conclusion in payload.get("conclusions") or []:
        related_chart = str(conclusion.get("related_chart") or "")
        if related_chart and related_chart not in visual_ids:
            conclusion_id = str(conclusion.get("id") or "conclusion")
            warnings.append(f"{conclusion_id}: referencia un grafico que no existe en la especificacion.")
            risk_flags.append("unlinked_chart")
        if not str(conclusion.get("evidence") or "").strip():
            risk_flags.append("low_evidence")

    return _unique(warnings)[:12], _unique(risk_flags)[:12]


def _append_variable_contract_findings(
    warnings: list[str],
    risk_flags: list[str],
    semantic_map: dict[str, dict[str, Any]],
    *,
    variable: str,
    owner: str,
    usage: str,
) -> None:
    if not variable or _column_key(variable) in {"count", "conteo", "cantidad", "tickets", "incidencias", "registros"}:
        return
    semantic = semantic_map.get(variable)
    if not semantic:
        warnings.append(f"{owner}: la {usage} '{variable}' no existe en el diccionario semantico del run.")
        risk_flags.append("missing_variable")
        return
    role = str(semantic.get("role") or "").lower()
    if role in {"technical", "identifier"} or (usage == "metrica" and semantic.get("avoid_as_metric")):
        warnings.append(f"{owner}: usa '{variable}' como {usage}, pero es una variable tecnica o identificador.")
        risk_flags.append("technical_variable")


def _normalize_dashboard_payload(payload: dict[str, Any], fallback_payload: dict[str, Any]) -> dict[str, Any]:
    semantic_items = payload.get("semantic_variables") or fallback_payload.get("semantic_variables") or []
    semantic_map = {str(item.get("name")): item for item in semantic_items if item.get("name")}
    visualizations = payload.get("suggested_visualizations") or []
    metric_candidates = [
        str(item.get("name"))
        for item in semantic_items
        if item.get("role") == "metric" and not item.get("avoid_as_metric")
    ]
    business_candidates = [
        str(item.get("name"))
        for item in semantic_items
        if item.get("role") == "business"
    ]

    for visualization in visualizations:
        _normalize_visualization_semantics(visualization, semantic_map, metric_candidates, business_candidates)

    visual_ids = {str(item.get("id")) for item in visualizations if item.get("id")}
    for recommendation in payload.get("agent_recommendations") or []:
        action_type = str(recommendation.get("action_type") or "chat").lower()
        linked_id = str(recommendation.get("linked_visualization_id") or "")
        if action_type != "chart":
            continue
        if linked_id not in visual_ids:
            matched_id = _best_visualization_id_for_text(_recommendation_text(recommendation), visualizations)
            if matched_id:
                recommendation["linked_visualization_id"] = matched_id
                continue
            recommendation["action_type"] = "chat"
            recommendation["linked_visualization_id"] = ""
            recommendation["evidence_needed"] = (
                recommendation.get("evidence_needed")
                or "El agente no vinculo una visualizacion construible para esta accion."
            )

    for conclusion in payload.get("conclusions") or []:
        related_chart = str(conclusion.get("related_chart") or "")
        if related_chart and related_chart not in visual_ids:
            matched_id = _best_visualization_id_for_text(_conclusion_text(conclusion), visualizations)
            conclusion["related_chart"] = matched_id or ""
        if not conclusion.get("related_items"):
            conclusion["related_items"] = _unique(
                [
                    conclusion.get("related_metric"),
                    conclusion.get("related_chart"),
                    *_extract_short_tokens(conclusion.get("evidence")),
                ]
            )[:8]

    active_id = str(payload.get("active_chart_default", {}).get("visualization_id") or "")
    if active_id and active_id not in visual_ids and visualizations:
        payload["active_chart_default"]["visualization_id"] = visualizations[0]["id"]
    return payload


def _normalize_visualization_semantics(
    visualization: dict[str, Any],
    semantic_map: dict[str, dict[str, Any]],
    metric_candidates: list[str],
    business_candidates: list[str],
) -> None:
    x_value = str(visualization.get("x") or visualization.get("group_by") or "")
    metric_value = str(visualization.get("metric") or visualization.get("y") or "")
    x_value = _resolve_payload_column_name(x_value, semantic_map, business_candidates)
    metric_value = _resolve_payload_metric_name(metric_value, semantic_map, metric_candidates)
    if x_value:
        visualization["x"] = x_value
        visualization["group_by"] = x_value
    if metric_value:
        visualization["metric"] = metric_value
        if visualization.get("y"):
            visualization["y"] = metric_value
    x_semantic = semantic_map.get(x_value) or _semantic_base(x_value)
    metric_semantic = semantic_map.get(metric_value) or _semantic_base(metric_value)

    if x_semantic.get("avoid_as_metric") and business_candidates:
        visualization["x"] = business_candidates[0]
        visualization["group_by"] = business_candidates[0]
    if metric_semantic.get("avoid_as_metric") and metric_candidates:
        visualization["metric"] = metric_candidates[0]
        if visualization.get("y") == metric_value:
            visualization["y"] = metric_candidates[0]
    if not visualization.get("metric"):
        visualization["metric"] = "count"
    if not visualization.get("x") and business_candidates:
        visualization["x"] = business_candidates[0]
    if not visualization.get("group_by") and visualization.get("x"):
        visualization["group_by"] = visualization["x"]

    x_label = _humanize_column(str(visualization.get("x") or visualization.get("group_by") or "hallazgos"))
    metric_label = _humanize_column(str(visualization.get("metric") or visualization.get("y") or "conteo"))
    if not visualization.get("what_i_am_seeing"):
        visualization["what_i_am_seeing"] = f"Esta grafica muestra {metric_label} por {x_label}."
    if not visualization.get("why_it_matters"):
        visualization["why_it_matters"] = "Importa porque ayuda a priorizar soporte, validar foco operativo y decidir que revisar primero."
    if not visualization.get("evidence_used"):
        visualization["evidence_used"] = "Hallazgos guardados, evidencias persistidas y metricas calculadas para la ejecucion actual."
    if not visualization.get("suggested_action"):
        visualization["suggested_action"] = "Abrir el detalle de evidencias relacionadas antes de reportar una conclusion."


def _resolve_payload_column_name(
    value: str,
    semantic_map: dict[str, dict[str, Any]],
    business_candidates: list[str],
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in semantic_map:
        return text
    lookup = {_column_key(name): name for name in semantic_map}
    actual = lookup.get(_column_key(text))
    if actual:
        return actual
    actual = _resolve_semantic_alias_to_actual(text, semantic_map)
    if actual:
        return actual
    base = _semantic_base(text)
    if base.get("role") == "business" and business_candidates:
        actual = _best_actual_for_role(base, semantic_map, preferred=business_candidates)
        return actual or business_candidates[0]
    return text


def _resolve_payload_metric_name(
    value: str,
    semantic_map: dict[str, dict[str, Any]],
    metric_candidates: list[str],
) -> str:
    text = str(value or "").strip()
    if not text or _column_key(text) in {"count", "conteo", "cantidad", "tickets", "incidencias", "registros"}:
        return "count"
    if text in semantic_map:
        return text
    lookup = {_column_key(name): name for name in semantic_map}
    actual = lookup.get(_column_key(text))
    if actual:
        return actual
    actual = _resolve_semantic_alias_to_actual(text, semantic_map)
    if actual:
        return actual
    base = _semantic_base(text)
    if base.get("role") == "metric" and metric_candidates:
        actual = _best_actual_for_role(base, semantic_map, preferred=metric_candidates)
        return actual or metric_candidates[0]
    return text


def _resolve_semantic_alias_to_actual(
    value: str,
    semantic_map: dict[str, dict[str, Any]],
) -> str:
    value_key = _column_key(value)
    if not value_key:
        return ""
    for actual_name, semantic in semantic_map.items():
        candidates = [
            actual_name,
            semantic.get("label"),
            *(semantic.get("aliases") or []),
        ]
        if value_key in {_column_key(candidate) for candidate in candidates if candidate}:
            return actual_name
    return ""


def _best_actual_for_role(
    target_semantic: dict[str, Any],
    semantic_map: dict[str, dict[str, Any]],
    *,
    preferred: list[str],
) -> str:
    target_label = _column_key(target_semantic.get("label"))
    target_aliases = {_column_key(alias) for alias in target_semantic.get("aliases") or []}
    for actual_name in preferred:
        semantic = semantic_map.get(actual_name) or {}
        actual_keys = {
            _column_key(actual_name),
            _column_key(semantic.get("label")),
            *(_column_key(alias) for alias in semantic.get("aliases") or []),
        }
        if target_label and target_label in actual_keys:
            return actual_name
        if target_aliases.intersection(actual_keys):
            return actual_name
    return ""


def _recommendation_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("title", "why_it_matters", "what_to_analyze", "recommended_next_step", "evidence_needed")
    )


def _conclusion_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("conclusion", "evidence", "related_metric", "recommended_action")
    )


def _best_visualization_id_for_text(text: str, visualizations: list[dict[str, Any]]) -> str:
    text_tokens = _extract_short_tokens(text)
    if not text_tokens:
        return ""
    scored: list[tuple[int, str]] = []
    for visualization in visualizations:
        visual_text = " ".join(str(visualization.get(key) or "") for key in (
            "id",
            "title",
            "chart_type",
            "x",
            "y",
            "metric",
            "group_by",
            "reason",
            "question_answered",
        )).lower()
        score = sum(1 for token in text_tokens if token in visual_text)
        if score:
            scored.append((score, str(visualization.get("id") or "")))
    scored.sort(reverse=True)
    return scored[0][1] if scored else ""


def _extract_short_tokens(value: Any) -> list[str]:
    text = str(value or "").lower()
    tokens = re.findall(r"[a-z0-9_]{4,}", text)
    ignored = {"esta", "este", "para", "como", "datos", "grafico", "hallazgo", "hallazgos", "evidencia"}
    return _unique([token for token in tokens if token not in ignored])[:12]


def _sanitize_finding(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _slug(item.get("id") or item.get("title") or f"finding-{index}"),
        "title": _text(item.get("title"), f"Hallazgo {index}"),
        "priority": _choice(item.get("priority"), VALID_PRIORITIES, "media"),
        "impact": _text(item.get("impact")),
        "urgency": _text(item.get("urgency")),
        "evidence": _text(item.get("evidence")),
        "suggested_action": _text(item.get("suggested_action")),
        "related_variables": _string_list(item.get("related_variables"), limit=8),
        "suggested_question": _text(item.get("suggested_question")),
    }


def _sanitize_semantic_variable(item: dict[str, Any], index: int) -> dict[str, Any]:
    name = _text(item.get("name"), f"variable_{index}")
    role = _choice(item.get("role"), VALID_VARIABLE_ROLES, "unknown")
    return {
        "name": name,
        "label": _text(item.get("label"), _humanize_column(name)),
        "role": role,
        "description": _text(item.get("description"), _semantic_description(name, role)),
        "recommended_use": _text(item.get("recommended_use"), _semantic_use(_humanize_column(name), role)),
        "avoid_as_metric": bool(item.get("avoid_as_metric")) or role in {"technical", "identifier"},
    }


def _sanitize_recommendation(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _slug(item.get("id") or item.get("title") or f"recommendation-{index}"),
        "title": _text(item.get("title"), f"Recomendacion {index}"),
        "why_it_matters": _text(item.get("why_it_matters")),
        "what_to_analyze": _text(item.get("what_to_analyze")),
        "recommended_next_step": _text(item.get("recommended_next_step")),
        "audience": _choice(item.get("audience"), VALID_AUDIENCES, "ambos"),
        "action_type": _choice(item.get("action_type"), VALID_ACTION_TYPES, "chat"),
        "linked_visualization_id": _text(item.get("linked_visualization_id")),
        "evidence_needed": _text(item.get("evidence_needed")),
    }


def _sanitize_visualization(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _slug(item.get("id") or item.get("title") or f"visualization-{index}"),
        "title": _text(item.get("title"), f"Visualizacion {index}"),
        "chart_type": _choice(item.get("chart_type"), VALID_CHART_TYPES, "bar"),
        "x": _text(item.get("x")),
        "y": _text(item.get("y")),
        "metric": _text(item.get("metric")),
        "group_by": _text(item.get("group_by")),
        "aggregation": _text(item.get("aggregation")),
        "filters": item.get("filters") if isinstance(item.get("filters"), list) else [],
        "reason": _text(item.get("reason")),
        "evidence_used": _text(item.get("evidence_used")),
        "question_answered": _text(item.get("question_answered")),
        "audience": _choice(item.get("audience"), VALID_AUDIENCES, "ambos"),
        "what_i_am_seeing": _text(item.get("what_i_am_seeing")),
        "why_it_matters": _text(item.get("why_it_matters")),
        "suggested_action": _text(item.get("suggested_action")),
        "drilldown": _text(item.get("drilldown")),
    }


def _sanitize_conclusion(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _slug(item.get("id") or f"conclusion-{index}"),
        "conclusion": _text(item.get("conclusion"), f"Conclusion {index}"),
        "evidence": _text(item.get("evidence")),
        "related_chart": _text(item.get("related_chart")),
        "related_metric": _text(item.get("related_metric")),
        "related_items": _string_list(item.get("related_items"), limit=10),
        "confidence": _choice(item.get("confidence"), VALID_CONFIDENCE, "media"),
        "recommended_action": _text(item.get("recommended_action")),
        "source": _text(item.get("source")),
        "evidence_quality": _text(item.get("evidence_quality")),
    }


def _sanitize_evidence_step(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "step": _safe_int(item.get("step")) or index,
        "title": _text(item.get("title"), f"Paso {index}"),
        "description": _text(item.get("description")),
        "source": _choice(item.get("source"), VALID_EVIDENCE_SOURCES, "dataset"),
        "related_items": _string_list(item.get("related_items"), limit=12),
    }


def _sanitize_list(raw: Any, fallback: list[dict[str, Any]], sanitizer) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return fallback
    items = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            items.append(sanitizer(item, index))
    return items or fallback


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def _compact_insight(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "description": item.get("description"),
        "metric_label": item.get("metric_label"),
        "metric_value": item.get("metric_value"),
        "dimension": item.get("dimension"),
        "filter_kind": item.get("filter_kind"),
        "filter_value": item.get("filter_value"),
        "run_id": item.get("run_id"),
    }


def _summary_text(metrics: dict[str, Any], insights: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> str:
    return (
        f"Se analizaron {metrics.get('records_count') or 0} registros con "
        f"{len(insights)} hallazgos seleccionados y {len(clusters)} clusters resumidos. "
        "El dashboard propone priorizar evidencia, visualizaciones y acciones verificables."
    )


def _priority_from_value(value: Any, ranked: list[dict[str, Any]]) -> str:
    number = _safe_float(value)
    values = [_safe_float(item.get("metric_value")) or 0 for item in ranked]
    maximum = max(values) if values else 0
    if maximum <= 0 or number is None:
        return "media"
    ratio = number / maximum
    if ratio >= 0.75:
        return "alta"
    if ratio >= 0.35:
        return "media"
    return "baja"


def _metric_sentence(item: dict[str, Any]) -> str:
    label = item.get("metric_label")
    value = item.get("metric_value")
    if label and value is not None:
        return f"{label}: {_round(value)}"
    return "Impacto basado en hallazgo guardado."


def _related_variables(item: dict[str, Any]) -> list[str]:
    values = [item.get("dimension"), item.get("filter_kind"), item.get("metric_label")]
    return _unique([str(value) for value in values if value])


def _column_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "Ã" in text or "Â" in text:
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "no_of_reassignments": "no_of_reassignments",
        "number_of_reassignments": "number_of_reassignments",
        "num_reassignments": "number_of_reassignments",
        "reassignments": "reassignment_count",
        "incidencias": "count",
        "incidencia": "count",
        "tickets": "count",
        "ticket": "count",
        "registros": "count",
        "registro": "count",
        "n_mero": "incident_id",
        "affected_service": "affected_service",
        "service_affected": "affected_service",
        "affected_service_name": "affected_service",
        "business_service": "affected_service",
        "task_business_service": "affected_service",
        "task_service_offering": "affected_service",
        "elemento_de_configuracion": "affected_service",
        "elemento_de_configuraci_n": "affected_service",
        "servicio": "affected_service",
        "servicio_afectado": "affected_service",
        "ci_category": "ci_cat",
        "ci_cat": "ci_cat",
        "ci_cat_name": "ci_cat",
        "configuration_item_category": "ci_cat",
        "nombre": "ci_cat",
        "ci_subcat": "ci_subcat",
        "ci_subcategory": "ci_subcat",
        "ci_sub_cat": "ci_subcat",
        "configuration_item_subcategory": "ci_subcat",
        "subcategor_a": "ci_subcat",
        "subcategoria": "ci_subcat",
        "subcategoria_ci": "ci_subcat",
        "modulo_de_subcategoria": "ci_subcat",
        "m_dulo_de_subcategor_a": "ci_subcat",
        "closure_code": "close_code",
        "close_code": "close_code",
        "codigo_cierre": "close_code",
        "codigo_de_resolucion": "close_code",
        "c_digo_de_resoluci_n": "close_code",
        "tipo_de_solucion": "close_code",
        "tipo_de_soluci_n": "close_code",
        "task_close_code": "close_code",
        "task_assignment_group": "assignment_group",
        "grupo_de_asignacion": "assignment_group",
        "grupo_de_asignaci_n": "assignment_group",
        "task_company": "company",
        "empresa": "company",
        "task_contact_type": "channel",
        "contact_type": "channel",
        "tipo_de_contacto": "channel",
        "diagn_stico": "root_cause",
        "diagnostico": "root_cause",
        "causada_por_cambio": "root_cause",
        "task_state": "status",
        "estado_de_la_incidencia": "status",
        "valor": "status",
        "estado": "status",
        "task_priority": "priority",
        "prioridad": "priority",
        "task_impact": "impact",
        "impacto": "impact",
        "task_urgency": "urgency",
        "urgencia": "urgency",
        "task_u_task_category": "category",
        "catalogo": "category",
        "cat_logo": "category",
        "avg_resolution_hours": "avg_resolution_hours",
        "average_resolution_hours": "avg_resolution_hours",
        "resolution_hours": "avg_resolution_hours",
        "handle_time_hrs": "avg_resolution_hours",
        "duracion": "avg_resolution_hours",
        "duraci_n": "avg_resolution_hours",
        "tiempo_de_trabajo": "avg_resolution_hours",
        "task_business_duration": "avg_resolution_hours",
        "task_calendar_duration": "avg_resolution_hours",
        "task_time_worked": "avg_resolution_hours",
        "calendar_duration": "avg_resolution_hours",
        "time_worked": "avg_resolution_hours",
        "business_impact": "business_impact_score",
        "business_impact_score": "business_impact_score",
        "impact_score": "business_impact_score",
        "urgency_score": "urgency_score",
        "opened_at": "opened_at",
        "opened": "opened_at",
        "created_at": "opened_at",
        "open_time": "opened_at",
        "task_opened_at": "opened_at",
        "task_sys_created_on": "opened_at",
        "closed_at": "closed_at",
        "closed": "closed_at",
        "resolved_at": "closed_at",
        "resolved_time": "closed_at",
        "close_time": "closed_at",
        "task_closed_at": "closed_at",
        "reopen_count": "reopen_count",
        "reopened_count": "reopen_count",
        "reaperturas": "reopen_count",
        "volver_a_abrir_recuento": "reopen_count",
        "escalation_count": "escalation_count",
        "escalations": "escalation_count",
        "escalados": "escalation_count",
        "channel": "channel",
        "no_of_related_interactions": "related_interactions_count",
        "no_of_related_incidents": "related_incidents_count",
        "no_of_related_changes": "related_changes_count",
    }
    return aliases.get(text, text)


def _semantic_base(value: Any, *, project_id: str | None = None) -> dict[str, Any]:
    text = str(value or "")
    dictionary = get_semantic_dictionary(SEMANTIC_VARIABLES, project_id=project_id)
    return dictionary.get(text) or dictionary.get(_column_key(text)) or {}


def _looks_identifier(value: str) -> bool:
    text = _column_key(value)
    return (
        text in {"id", "uuid", "incident_id", "case_id", "ticket_id"}
        or text.endswith("_id")
        or text.endswith("id")
        or text.startswith("cluster_")
        or text.startswith("metric_")
        or text.startswith("embedding")
    )


def _humanize_column(value: str) -> str:
    semantic = _semantic_base(value)
    if semantic.get("label"):
        return str(semantic["label"])
    text = value.replace("_", " ").replace("-", " ").strip()
    replacements = {
        "avg": "promedio",
        "sla": "SLA",
        "cnt": "cantidad",
        "num": "numero",
        "no of": "numero de",
    }
    lowered = text.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered[:1].upper() + lowered[1:]


def _semantic_description(name: str, role: str) -> str:
    label = _humanize_column(name)
    if role == "business":
        return f"Dimension de negocio para segmentar o explicar incidencias: {label}."
    if role == "metric":
        return f"Metrica cuantitativa util para comparar intensidad, volumen o desempeno: {label}."
    if role == "technical":
        return f"Variable tecnica generada por el pipeline para trazabilidad: {label}."
    if role == "identifier":
        return f"Identificador tecnico; sirve para buscar registros, no para resumir negocio: {label}."
    return f"Variable detectada en el dataset: {label}."


def _semantic_use(label: str, role: str) -> str:
    if role == "business":
        return f"Usar {label} como eje, filtro o segmento del dashboard."
    if role == "metric":
        return f"Usar {label} como metrica para comparar grupos o prioridades."
    if role in {"technical", "identifier"}:
        return f"Usar {label} solo para trazabilidad o drill-down tecnico."
    return f"Validar si {label} aporta lectura de negocio antes de graficar."


def _top_value(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in _candidate_columns(df, candidates):
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return str(values.value_counts().index[0])
    return ""


def _mean_first(df: pd.DataFrame, candidates: tuple[str, ...]) -> float | None:
    for col in _candidate_columns(df, candidates):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.empty:
            return _round(values.mean())
    return None


def _candidate_columns(df: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    lookup = {_column_key(col): col for col in df.columns}
    result: list[str] = []
    for candidate in candidates:
        actual = candidate if candidate in df.columns else lookup.get(_column_key(candidate))
        if actual and actual not in result:
            result.append(actual)
    return result


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _round(value: Any, digits: int = 3) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _unique(values: list[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:160] for item in value if str(item or "").strip()][:limit]


def _choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:700] if text else default


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:80] or "item"


def _jsonable(item: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in item.items():
        if hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result
