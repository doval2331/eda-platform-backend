from __future__ import annotations

import math
import unicodedata

import pandas as pd

from app.schemas import ChatResponse, InsightCandidate
from app.services.duckdb_store import load_run_evidences
from app.services.llm_agent import explain_with_llm


SUGGESTED_QUESTIONS = [
    "Que puedo analizar con estas incidencias?",
    "Que servicios incumplen mas SLA?",
    "Que prioridades tienen mas demoras?",
    "Que causas raiz se repiten?",
    "Que clusters son mas criticos?",
    "Que incidencias parecen anomalas?",
    "Que alternativas de decision conviene priorizar?",
    "Que acciones recomendadas puedo evaluar?",
]

FALLBACK_SUGGESTED_QUESTIONS = SUGGESTED_QUESTIONS

SERVICE_COLS = ("servicio_afectado", "affected_service", "service_line")
PRIORITY_COLS = ("prioridad", "severity")
CATEGORY_COLS = ("categoria", "category", "sector")
CHANNEL_COLS = ("canal_entrada", "support_channel")
RESOLUTION_COLS = ("tiempo_resolucion_horas", "avg_resolution_hours")
SLA_COLS = ("sla_breach_rate", "sla_incumplido", "sla_breached")
REOPEN_COLS = ("reaperturas", "reopenings", "reopen_rate")
ESCALATION_COLS = ("escalados", "escalations", "escalation_rate")
SATISFACTION_COLS = ("satisfaccion_usuario", "customer_satisfaction")
COST_COLS = ("coste_estimado", "estimated_cost")
ROOT_CAUSE_COLS = ("causa_raiz_simulada", "root_cause")
RISK_COLS = ("operational_risk_score", "business_impact_score")


def _has_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> bool:
    return any(name in df.columns for name in aliases)


def build_suggested_questions_for_run(run_id: str) -> list[str]:
    df = load_run_evidences(run_id)
    if df.empty:
        return FALLBACK_SUGGESTED_QUESTIONS
    return _suggested_questions_for_df(df)


def _suggested_questions_for_df(df: pd.DataFrame) -> list[str]:
    questions: list[str] = ["Que puedo analizar con estas fuentes?"]

    has_service = _has_col(df, SERVICE_COLS)
    has_priority = _has_col(df, PRIORITY_COLS)
    has_category = _has_col(df, CATEGORY_COLS)
    has_sla = _has_col(df, SLA_COLS)
    has_resolution = _has_col(df, RESOLUTION_COLS)
    has_root_cause = _has_col(df, ROOT_CAUSE_COLS)
    has_risk = _has_col(df, RISK_COLS)
    has_reopen = _has_col(df, REOPEN_COLS)
    has_escalation = _has_col(df, ESCALATION_COLS)
    has_cluster = "cluster_label" in df.columns

    if has_sla and has_service:
        questions.append("Que servicios concentran mas incumplimiento de SLA?")
    elif has_sla:
        questions.append("Como esta el incumplimiento de SLA en esta ejecucion?")

    if has_resolution and has_service:
        questions.append("Que servicios tardan mas en resolverse?")
    elif has_resolution:
        questions.append("Donde se concentran los mayores tiempos de resolucion?")

    if has_priority and (has_service or has_category):
        questions.append("Que grupos tienen mayor urgencia o prioridad?")
    elif has_priority:
        questions.append("Como se distribuyen las incidencias por prioridad?")

    if has_risk:
        questions.append("Que grupos tienen mayor impacto o riesgo operativo?")

    if has_root_cause:
        questions.append("Que causas raiz se repiten con mayor frecuencia?")

    if has_reopen:
        questions.append("Que patrones aparecen en los casos reabiertos?")

    if has_escalation:
        questions.append("Que grupos tienen mas escalaciones?")

    if has_cluster:
        questions.append("Que grupos detectados deberia revisar primero?")
        questions.append("Hay casos atipicos que convenga revisar por separado?")

    questions.append("Que hallazgos conviene llevar al dashboard?")

    deduped: list[str] = []
    for question in questions:
        if question not in deduped:
            deduped.append(question)
    return deduped[:8]


def _normalize(text: str) -> str:
    clean = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in clean if unicodedata.category(ch) != "Mn")


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt_pct(value) -> str:
    number = _finite(value)
    if number is None:
        return "sin dato"
    pct = number if abs(number) > 1 else number * 100
    return f"{pct:.1f}%"


def _fmt_hours(value) -> str:
    number = _finite(value)
    if number is None:
        return "sin dato"
    return f"{number:.1f} h"


def _fmt_number(value) -> str:
    number = _finite(value)
    if number is None:
        return "sin dato"
    if abs(number) >= 100:
        return f"{number:,.0f}".replace(",", ".")
    return f"{number:.1f}"


def _first_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in df.columns), None)


def _first_numeric_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name not in df.columns:
            continue
        if _numeric_series(df, name).notna().any():
            return name
    return None


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_object_dtype(series):
        normalized = series.astype(str).map(_normalize)
        mapped = normalized.map(
            {
                "true": 1,
                "si": 1,
                "yes": 1,
                "1": 1,
                "false": 0,
                "no": 0,
                "0": 0,
            }
        )
        return pd.to_numeric(mapped.fillna(series), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _mean(df: pd.DataFrame, aliases: tuple[str, ...]) -> float | None:
    col = _first_numeric_col(df, aliases)
    if not col:
        return None
    return _finite(_numeric_series(df, col).mean())


def _group_mean(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    *,
    top_n: int = 3,
    ascending: bool = False,
) -> pd.DataFrame:
    if group_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()
    filtered = pd.DataFrame(
        {
            group_col: df[group_col],
            metric_col: _numeric_series(df, metric_col),
        }
    ).dropna()
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.groupby(group_col, dropna=True)
        .agg(value=(metric_col, "mean"), count=(metric_col, "size"))
        .reset_index()
        .sort_values(["value", "count"], ascending=[ascending, False])
        .head(top_n)
    )


def _group_count(df: pd.DataFrame, group_col: str, *, top_n: int = 3) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    filtered = df[[group_col]].dropna()
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.groupby(group_col, dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def _top_value(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    col = _first_col(df, aliases)
    if not col:
        return None
    top = _group_count(df, col, top_n=1)
    if top.empty:
        return None
    return str(top.iloc[0][col])


def _safe_slug(value: object) -> str:
    text = _normalize(str(value)).strip()
    return "-".join(part for part in text.replace("/", " ").split() if part)


def _add_unique(insights: list[InsightCandidate], item: InsightCandidate) -> None:
    if not any(existing.id == item.id for existing in insights):
        insights.append(item)


def _tool_summary(
    tool_name: str,
    answer: str,
    insights: list[InsightCandidate],
) -> dict:
    return {
        "herramienta": tool_name,
        "respuesta_base": answer,
        "hallazgos": [
            {
                "titulo": item.title,
                "descripcion": item.description,
                "metrica": item.metric_label,
                "valor": item.metric_value,
                "dimension": item.dimension,
                "filtro": item.filter_value,
            }
            for item in insights[:5]
        ],
    }


def _overview(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    n_rows = len(df)
    clusters = (
        sorted(int(c) for c in df["cluster_label"].dropna().unique().tolist() if int(c) >= 0)
        if "cluster_label" in df
        else []
    )
    outliers = int((df["cluster_label"] == -1).sum()) if "cluster_label" in df else 0
    avg_sla = _mean(df, SLA_COLS)
    avg_resolution = _mean(df, RESOLUTION_COLS)
    avg_risk = _mean(df, RISK_COLS)
    service = _top_value(df, SERVICE_COLS)
    priority = _top_value(df, PRIORITY_COLS)
    answer = (
        f"Con {n_rows} incidencias puedo explorar volumen, SLA, tiempos de "
        f"resolucion, servicios afectados, prioridad, causas raiz, clusters y "
        f"anomalias. Esta corrida tiene {len(clusters)} clusters y {outliers} "
        f"incidencias anomalas. Promedios: SLA incumplido {_fmt_pct(avg_sla)}, "
        f"resolucion {_fmt_hours(avg_resolution)} y riesgo {_fmt_number(avg_risk)}. "
        f"Como primer foco, revisaria servicio {service or 'sin dato'} y prioridad "
        f"{priority or 'sin dato'}."
    )
    insights = [
        InsightCandidate(
            id="overview-sla",
            title="SLA global de incidencias",
            description=f"Incumplimiento promedio de SLA: {_fmt_pct(avg_sla)}.",
            metric_label="sla_breach_rate",
            metric_value=_finite(avg_sla),
            dimension="run",
            filter_kind="run_id",
            filter_value="current",
        )
    ]
    return answer, insights


def _sla_answer(df: pd.DataFrame, question: str) -> tuple[str, list[InsightCandidate]]:
    sla_col = _first_numeric_col(df, SLA_COLS)
    if not sla_col:
        return ("No encontre un campo de SLA para esta corrida de incidencias.", [])

    if "prioridad" in question or "severidad" in question or "critic" in question:
        group_col = _first_col(df, PRIORITY_COLS)
        label = "prioridades"
    elif "cluster" in question:
        group_col = "cluster_label" if "cluster_label" in df.columns else None
        label = "clusters"
    else:
        group_col = _first_col(df, SERVICE_COLS)
        label = "servicios"

    global_sla = _mean(df, SLA_COLS)
    if not group_col:
        return (
            f"El incumplimiento SLA global es {_fmt_pct(global_sla)}, pero no hay una dimension confiable para desagregarlo.",
            [],
        )

    top = _group_mean(df, group_col, sla_col)
    if top.empty:
        return (
            f"El incumplimiento SLA global es {_fmt_pct(global_sla)}, pero no encontre datos suficientes por {label}.",
            [],
        )

    pieces = [
        f"{row[group_col]}: {_fmt_pct(row['value'])} ({int(row['count'])} incidencias)"
        for _, row in top.iterrows()
    ]
    answer = (
        f"El incumplimiento SLA global es {_fmt_pct(global_sla)}. "
        f"Los {label} con mayor incumplimiento son " + "; ".join(pieces) + "."
    )
    insights = [
        InsightCandidate(
            id=f"sla-{group_col}-{_safe_slug(row[group_col])}",
            title=f"SLA alto en {row[group_col]}",
            description=f"{row[group_col]} tiene {_fmt_pct(row['value'])} de incumplimiento promedio de SLA.",
            metric_label="sla_breach_rate",
            metric_value=_finite(row["value"]),
            dimension=group_col,
            filter_kind=group_col,
            filter_value=str(row[group_col]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _resolution_answer(df: pd.DataFrame, question: str) -> tuple[str, list[InsightCandidate]]:
    resolution_col = _first_numeric_col(df, RESOLUTION_COLS)
    if not resolution_col:
        return ("No encontre un campo de tiempo de resolucion para esta corrida.", [])

    if "prioridad" in question or "severidad" in question or "critic" in question:
        group_col = _first_col(df, PRIORITY_COLS)
        label = "prioridades"
    elif "causa" in question:
        group_col = _first_col(df, ROOT_CAUSE_COLS)
        label = "causas raiz"
    else:
        group_col = _first_col(df, SERVICE_COLS)
        label = "servicios"

    avg_resolution = _mean(df, RESOLUTION_COLS)
    if not group_col:
        return (
            f"El tiempo promedio de resolucion es {_fmt_hours(avg_resolution)}, sin desglose disponible.",
            [],
        )

    top = _group_mean(df, group_col, resolution_col)
    if top.empty:
        return (
            f"El tiempo promedio de resolucion es {_fmt_hours(avg_resolution)}, pero no hay datos suficientes por {label}.",
            [],
        )
    pieces = [f"{row[group_col]}: {_fmt_hours(row['value'])}" for _, row in top.iterrows()]
    answer = (
        f"El tiempo promedio de resolucion es {_fmt_hours(avg_resolution)}. "
        f"Las {label} con mayor demora son " + "; ".join(pieces) + "."
    )
    insights = [
        InsightCandidate(
            id=f"resolution-{group_col}-{_safe_slug(row[group_col])}",
            title=f"Demora alta en {row[group_col]}",
            description=f"{row[group_col]} promedia {_fmt_hours(row['value'])} de resolucion.",
            metric_label="resolution_hours",
            metric_value=_finite(row["value"]),
            dimension=group_col,
            filter_kind=group_col,
            filter_value=str(row[group_col]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _services_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    service_col = _first_col(df, SERVICE_COLS)
    if not service_col:
        return ("No encontre una columna de servicio afectado para esta corrida.", [])
    top = _group_count(df, service_col)
    if top.empty:
        return ("No hay datos suficientes para medir volumen por servicio.", [])
    pieces = [
        f"{row[service_col]}: {int(row['count'])} incidencias"
        for _, row in top.iterrows()
    ]
    answer = "Los servicios con mas incidencias son " + "; ".join(pieces) + "."
    insights = [
        InsightCandidate(
            id=f"volume-{_safe_slug(row[service_col])}",
            title=f"Volumen alto en {row[service_col]}",
            description=f"{row[service_col]} concentra {int(row['count'])} incidencias.",
            metric_label="incident_count",
            metric_value=_finite(row["count"]),
            dimension=service_col,
            filter_kind=service_col,
            filter_value=str(row[service_col]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _priority_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    priority_col = _first_col(df, PRIORITY_COLS)
    if not priority_col:
        risk = _mean(df, RISK_COLS)
        return (
            f"No encontre prioridad explicita; como referencia, el riesgo promedio es {_fmt_number(risk)}.",
            [],
        )
    top = _group_count(df, priority_col, top_n=4)
    if top.empty:
        return ("No hay datos suficientes para resumir prioridades.", [])

    pieces = [f"{row[priority_col]}: {int(row['count'])}" for _, row in top.iterrows()]
    answer = "La distribucion de prioridad de incidencias es " + "; ".join(pieces) + "."
    insights = [
        InsightCandidate(
            id=f"priority-{_safe_slug(row[priority_col])}",
            title=f"Prioridad {row[priority_col]}",
            description=f"Hay {int(row['count'])} incidencias con prioridad {row[priority_col]}.",
            metric_label="priority_count",
            metric_value=_finite(row["count"]),
            dimension=priority_col,
            filter_kind=priority_col,
            filter_value=str(row[priority_col]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _root_cause_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    cause_col = _first_col(df, ROOT_CAUSE_COLS)
    if not cause_col:
        return (
            "Todavia no encontre causa_raiz_simulada en esta corrida. Cuando el dataset de incidencias la incluya, podre mostrar causas repetidas y agregarlas al dashboard.",
            [],
        )
    top = _group_count(df, cause_col, top_n=5)
    if top.empty:
        return ("No hay datos suficientes para resumir causas raiz.", [])
    pieces = [
        f"{row[cause_col]}: {int(row['count'])} incidencias" for _, row in top.iterrows()
    ]
    answer = (
        "Las causas raiz mas repetidas son "
        + "; ".join(pieces)
        + ". Esto sugiere patrones para investigar, no una causa definitiva."
    )
    insights = [
        InsightCandidate(
            id=f"root-cause-{_safe_slug(row[cause_col])}",
            title=f"Causa raiz frecuente: {row[cause_col]}",
            description=f"{row[cause_col]} aparece en {int(row['count'])} incidencias.",
            metric_label="root_cause_count",
            metric_value=_finite(row["count"]),
            dimension=cause_col,
            filter_kind=cause_col,
            filter_value=str(row[cause_col]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _anomalies_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    if "cluster_label" not in df.columns:
        return ("No encontre etiquetas de cluster para revisar anomalias.", [])
    outliers = df[df["cluster_label"] == -1]
    count = len(outliers)
    if count == 0:
        return (
            "No hay incidencias marcadas como anomalas en esta corrida. Conviene revisar igual los clusters con peor SLA o mayor demora.",
            [],
        )
    service = _top_value(outliers, SERVICE_COLS)
    priority = _top_value(outliers, PRIORITY_COLS)
    avg_sla = _mean(outliers, SLA_COLS)
    avg_resolution = _mean(outliers, RESOLUTION_COLS)
    answer = (
        f"Hay {count} incidencias anomalas. No encajan bien con los grupos principales, "
        f"por eso conviene revisarlas una por una. Predominan servicio "
        f"{service or 'sin dato'} y prioridad {priority or 'sin dato'}. "
        f"Su SLA promedio es {_fmt_pct(avg_sla)} y su resolucion promedio "
        f"{_fmt_hours(avg_resolution)}."
    )
    insights = [
        InsightCandidate(
            id="anomaly-outliers",
            title="Incidencias anomalas",
            description=f"{count} incidencias quedaron como casos atipicos para revision individual.",
            metric_label="anomaly_count",
            metric_value=_finite(count),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value="-1",
        )
    ]
    return answer, insights


def _clusters_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    if "cluster_label" not in df.columns:
        return ("Esta corrida no tiene etiquetas de cluster disponibles.", [])
    filtered = df[df["cluster_label"] >= 0]
    if filtered.empty:
        return ("Todas las incidencias quedaron como anomalias; conviene revisar datos y parametros.", [])

    rows = []
    for cluster_label, items in filtered.groupby("cluster_label"):
        avg_sla = _mean(items, SLA_COLS)
        avg_resolution = _mean(items, RESOLUTION_COLS)
        avg_risk = _mean(items, RISK_COLS)
        service = _top_value(items, SERVICE_COLS)
        priority = _top_value(items, PRIORITY_COLS)
        cause = _top_value(items, ROOT_CAUSE_COLS)
        priority_score = (
            (_finite(avg_sla) or 0) * 100
            + (_finite(avg_risk) or 0)
            + (_finite(avg_resolution) or 0)
            + (len(items) ** 0.5)
        )
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "count": len(items),
                "sla": avg_sla,
                "risk": avg_risk,
                "resolution": avg_resolution,
                "service": service,
                "priority": priority,
                "cause": cause,
                "score": priority_score,
            }
        )

    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:3]
    pieces = [
        (
            f"cluster {row['cluster_label']}: {row['count']} incidencias, "
            f"SLA {_fmt_pct(row['sla'])}, resolucion {_fmt_hours(row['resolution'])}, "
            f"servicio {row['service'] or 'sin dato'}"
        )
        for row in top
    ]
    answer = (
        "Los clusters mas criticos son "
        + "; ".join(pieces)
        + ". Interpretalos como grupos de incidencias parecidas que merecen revision operativa, no como clasificacion definitiva."
    )
    insights = [
        InsightCandidate(
            id=f"cluster-{row['cluster_label']}",
            title=f"Cluster {row['cluster_label']} critico",
            description=(
                f"Cluster {row['cluster_label']}: {row['count']} incidencias similares, "
                f"SLA {_fmt_pct(row['sla'])}, resolucion {_fmt_hours(row['resolution'])}, "
                f"prioridad dominante {row['priority'] or 'sin dato'} y causa "
                f"{row['cause'] or 'sin dato'}."
            ),
            metric_label="cluster_critical_score",
            metric_value=_finite(row["score"]),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value=str(row["cluster_label"]),
        )
        for row in top
    ]
    return answer, insights


def _decision_score(
    *,
    sla: float | None = None,
    resolution: float | None = None,
    risk: float | None = None,
    count: int | float | None = None,
) -> float:
    sla_pct = 0.0
    finite_sla = _finite(sla)
    if finite_sla is not None:
        sla_pct = finite_sla if finite_sla > 1 else finite_sla * 100

    resolution_score = min((_finite(resolution) or 0) * 2.5, 35)
    risk_score = min((_finite(risk) or 0), 45)
    volume_score = min(math.sqrt(_finite(count) or 0) * 2.5, 25)
    return min(100, sla_pct * 0.45 + resolution_score + risk_score * 0.35 + volume_score)


def _decision_alternatives(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    alternatives: list[InsightCandidate] = []

    service_col = _first_col(df, SERVICE_COLS)
    priority_col = _first_col(df, PRIORITY_COLS)
    cause_col = _first_col(df, ROOT_CAUSE_COLS)
    sla_col = _first_numeric_col(df, SLA_COLS)
    resolution_col = _first_numeric_col(df, RESOLUTION_COLS)

    if service_col and sla_col:
        top_sla = _group_mean(df, service_col, sla_col, top_n=1)
        if not top_sla.empty:
            row = top_sla.iloc[0]
            service = str(row[service_col])
            service_df = df[df[service_col] == row[service_col]]
            avg_resolution = _mean(service_df, RESOLUTION_COLS)
            avg_risk = _mean(service_df, RISK_COLS)
            score = _decision_score(
                sla=_finite(row["value"]),
                resolution=avg_resolution,
                risk=avg_risk,
                count=int(row["count"]),
            )
            alternatives.append(
                InsightCandidate(
                    id=f"decision-sla-{_safe_slug(service)}",
                    title=f"Priorizar SLA en {service}",
                    description=(
                        f"Alternativa: revisar primero {service}, porque concentra "
                        f"{int(row['count'])} incidencias y muestra {_fmt_pct(row['value'])} "
                        f"de incumplimiento SLA. Proximo paso: analizar responsables, "
                        f"causas repetidas y ventanas de mayor demora."
                    ),
                    metric_label="decision_alternative",
                    metric_value=score,
                    dimension=service_col,
                    filter_kind=service_col,
                    filter_value=service,
                )
            )

    if priority_col and resolution_col:
        top_resolution = _group_mean(df, priority_col, resolution_col, top_n=1)
        if not top_resolution.empty:
            row = top_resolution.iloc[0]
            priority = str(row[priority_col])
            priority_df = df[df[priority_col] == row[priority_col]]
            avg_sla = _mean(priority_df, SLA_COLS)
            avg_risk = _mean(priority_df, RISK_COLS)
            score = _decision_score(
                sla=avg_sla,
                resolution=_finite(row["value"]),
                risk=avg_risk,
                count=int(row["count"]),
            )
            alternatives.append(
                InsightCandidate(
                    id=f"decision-resolution-{_safe_slug(priority)}",
                    title=f"Reducir demoras en prioridad {priority}",
                    description=(
                        f"Alternativa: revisar las incidencias de prioridad {priority}, "
                        f"porque promedian {_fmt_hours(row['value'])} de resolucion. "
                        f"Proximo paso: comparar colas, escalados y reaperturas antes "
                        f"de cambiar el proceso."
                    ),
                    metric_label="decision_alternative",
                    metric_value=score,
                    dimension=priority_col,
                    filter_kind=priority_col,
                    filter_value=priority,
                )
            )

    if cause_col:
        top_cause = _group_count(df, cause_col, top_n=1)
        if not top_cause.empty:
            row = top_cause.iloc[0]
            cause = str(row[cause_col])
            cause_df = df[df[cause_col] == row[cause_col]]
            score = _decision_score(
                sla=_mean(cause_df, SLA_COLS),
                resolution=_mean(cause_df, RESOLUTION_COLS),
                risk=_mean(cause_df, RISK_COLS),
                count=int(row["count"]),
            )
            alternatives.append(
                InsightCandidate(
                    id=f"decision-root-cause-{_safe_slug(cause)}",
                    title=f"Investigar causa raiz {cause}",
                    description=(
                        f"Alternativa: tomar {cause} como linea de investigacion, "
                        f"porque aparece en {int(row['count'])} incidencias. "
                        f"Proximo paso: validar con el equipo operativo si el patron "
                        f"es real o efecto del dataset sintetico."
                    ),
                    metric_label="decision_alternative",
                    metric_value=score,
                    dimension=cause_col,
                    filter_kind=cause_col,
                    filter_value=cause,
                )
            )

    if "cluster_label" in df.columns:
        outliers = df[df["cluster_label"] == -1]
        if not outliers.empty:
            score = _decision_score(
                sla=_mean(outliers, SLA_COLS),
                resolution=_mean(outliers, RESOLUTION_COLS),
                risk=_mean(outliers, RISK_COLS),
                count=len(outliers),
            )
            alternatives.append(
                InsightCandidate(
                    id="decision-anomalies",
                    title="Revisar incidencias anomalas",
                    description=(
                        f"Alternativa: revisar {len(outliers)} incidencias anomalas "
                        f"que no encajan bien con los grupos principales. Proximo paso: "
                        f"validar si son errores de registro, casos especiales o riesgos "
                        f"operativos no cubiertos por los procesos actuales."
                    ),
                    metric_label="decision_alternative",
                    metric_value=score,
                    dimension="cluster_label",
                    filter_kind="cluster_label",
                    filter_value="-1",
                )
            )

    if not alternatives:
        return (
            "Todavia no hay suficientes metricas agregadas para proponer alternativas de decision. Conviene revisar primero SLA, tiempos, servicios y clusters.",
            [],
        )

    alternatives = sorted(
        alternatives,
        key=lambda item: item.metric_value if item.metric_value is not None else 0,
        reverse=True,
    )[:3]
    pieces = [
        f"{index + 1}) {item.title}: {item.description}"
        for index, item in enumerate(alternatives)
    ]
    answer = (
        "Estas alternativas no son decisiones automaticas, sino opciones de "
        "priorizacion para discutir con el equipo: "
        + " ".join(pieces)
    )
    return answer, alternatives


def build_chat_response(run_id: str, question: str) -> ChatResponse:
    df = load_run_evidences(run_id)
    if df.empty:
        return ChatResponse(
            answer="No encontre incidencias materializadas en DuckDB para esta ejecucion. Ejecuta el pipeline nuevamente.",
            suggested_questions=FALLBACK_SUGGESTED_QUESTIONS,
        )

    normalized = _normalize(question)
    suggested_questions = _suggested_questions_for_df(df)
    answer_parts: list[str] = []
    insights: list[InsightCandidate] = []
    tool_summaries: list[dict] = []

    def collect(tool_name: str, result: tuple[str, list[InsightCandidate]]) -> None:
        answer, items = result
        answer_parts.append(answer)
        tool_summaries.append(_tool_summary(tool_name, answer, items))
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["que puedo", "analizar", "explorar"]):
        collect("resumen_general", _overview(df))

    if any(term in normalized for term in ["sla", "incumpl"]):
        collect("analizar_sla", _sla_answer(df, normalized))

    if any(term in normalized for term in ["tiempo", "resolucion", "resolver", "demora"]):
        collect("analizar_tiempos_resolucion", _resolution_answer(df, normalized))

    if any(term in normalized for term in ["servicio", "afectado", "volumen"]):
        collect("analizar_servicios", _services_answer(df))

    if any(term in normalized for term in ["prioridad", "severidad", "critic", "riesgo", "impacto"]):
        collect("analizar_prioridades", _priority_answer(df))

    if any(term in normalized for term in ["causa", "raiz", "root"]):
        collect("analizar_causas_raiz", _root_cause_answer(df))

    if any(term in normalized for term in ["anomalia", "anomalo", "atipic", "outlier"]):
        collect("analizar_anomalias", _anomalies_answer(df))

    if any(term in normalized for term in ["cluster", "grupo similar"]):
        collect("analizar_clusters", _clusters_answer(df))

    if any(
        term in normalized
        for term in [
            "decision",
            "decidir",
            "alternativa",
            "alternativas",
            "recomend",
            "accion",
            "acciones",
            "priorizar",
            "que hago",
            "proximo paso",
        ]
    ):
        collect("alternativas_decision", _decision_alternatives(df))

    if not answer_parts:
        collect("resumen_general", _overview(df))

    fallback_answer = " ".join(answer_parts)
    llm_result = explain_with_llm(
        question=question,
        tool_summaries=tool_summaries,
        fallback_answer=fallback_answer,
    )

    return ChatResponse(
        answer=llm_result.answer,
        suggested_questions=suggested_questions,
        insights=insights[:8],
        llm_used=llm_result.used,
        llm_mode=llm_result.mode,
        llm_detail=llm_result.detail,
    )
