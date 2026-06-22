from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd

from app.schemas import ChatResponse, InsightCandidate
from app.services.runs.duckdb_store import load_run_evidences
from app.services.agents.llm_agent import explain_with_llm


SUGGESTED_QUESTIONS = [
    "Que puedo analizar con estas fuentes?",
    "Que columnas o variables estan disponibles para analizar?",
    "Que grupos detectados deberia revisar primero?",
    "Que hallazgos conviene llevar al dashboard?",
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
SOURCE_COLS = ("source_name", "source", "dataset", "archivo", "filename")

SOURCE_TYPE_LABELS = {
    "incidents": "incidencias",
    "change_mgmt": "gestion del cambio",
    "software": "problemas software",
    "hardware": "problemas hardware",
    "dictionary": "diccionario de datos",
    "notes": "notas o documentacion",
    "other": "otra fuente",
    "tabular": "fuente tabular",
}


def _has_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> bool:
    return any(name in df.columns for name in aliases)


def _first_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _clean_text_series(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[
        ~values.str.lower().isin(
            {"", "nan", "none", "null", "sin dato", "no aplica", "unknown", "n/a"}
        )
    ]


def _has_value_signal(df: pd.DataFrame, aliases: tuple[str, ...]) -> bool:
    column = _first_alias(df, aliases)
    if not column:
        return False
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return bool(series.dropna().shape[0])
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return bool(numeric.shape[0])
    return bool(_clean_text_series(series).shape[0])


def _has_repeated_text_signal(df: pd.DataFrame, aliases: tuple[str, ...]) -> bool:
    column = _first_alias(df, aliases)
    if not column:
        return False
    counts = _clean_text_series(df[column]).value_counts()
    return bool((counts >= 2).any())


def _has_cluster_signal(df: pd.DataFrame) -> bool:
    if "cluster_label" not in df.columns:
        return False
    labels = pd.to_numeric(df["cluster_label"], errors="coerce").dropna()
    return bool((labels >= 0).any())


def _has_outlier_signal(df: pd.DataFrame) -> bool:
    if "cluster_label" not in df.columns:
        return False
    labels = pd.to_numeric(df["cluster_label"], errors="coerce").dropna()
    return bool((labels == -1).any())


def _scenario_sources(run_context: dict | None) -> list[dict]:
    return [
        item
        for item in ((run_context or {}).get("sources") or [])
        if isinstance(item, dict)
    ]


def _cluster_count(df: pd.DataFrame) -> int:
    if "cluster_label" not in df.columns:
        return 0
    labels = pd.to_numeric(df["cluster_label"], errors="coerce").dropna()
    return int(labels[labels >= 0].nunique())


def _has_any_assignment_col(df: pd.DataFrame) -> bool:
    return bool(
        _first_col(df, SERVICE_COLS)
        or _first_col(df, PRIORITY_COLS)
        or _first_col(df, SLA_COLS)
    )


def _tool_catalog(df: pd.DataFrame, run_context: dict | None = None) -> list[dict[str, object]]:
    sources = _scenario_sources(run_context)
    has_sources = bool(sources)
    has_multi_sources = len(sources) > 1
    has_service = _has_value_signal(df, SERVICE_COLS)
    has_priority = _has_value_signal(df, PRIORITY_COLS)
    has_category = _has_value_signal(df, CATEGORY_COLS)
    has_sla = _has_value_signal(df, SLA_COLS)
    has_resolution = _has_value_signal(df, RESOLUTION_COLS)
    has_root_cause = _has_repeated_text_signal(df, ROOT_CAUSE_COLS)
    has_risk = _has_value_signal(df, RISK_COLS)
    has_reopen = _has_value_signal(df, REOPEN_COLS)
    has_escalation = _has_value_signal(df, ESCALATION_COLS)
    has_cluster = _has_cluster_signal(df)
    has_outliers = _has_outlier_signal(df)
    cluster_count = _cluster_count(df)
    has_columns = bool(df.columns.tolist())

    return [
        {
            "id": "scenario_overview",
            "name": "Resumen general del escenario",
            "applicable": True,
            "question": "Que puedo analizar con estas fuentes?",
        },
        {
            "id": "source_inventory",
            "name": "Fuentes del escenario",
            "applicable": has_sources,
            "question": "Que fuentes tiene este escenario y cual conviene revisar primero?",
        },
        {
            "id": "source_comparison",
            "name": "Comparacion entre fuentes",
            "applicable": has_multi_sources,
            "question": "Que diferencias hay entre las fuentes cargadas?",
        },
        {
            "id": "data_quality",
            "name": "Calidad de datos",
            "applicable": has_columns,
            "question": "Que columnas tienen datos faltantes o problemas de calidad?",
        },
        {
            "id": "columns_meaning",
            "name": "Columnas disponibles y significado probable",
            "applicable": has_columns,
            "question": "Que columnas detecto la app y para que sirven?",
        },
        {
            "id": "missing_assignments",
            "name": "Registros sin servicio/prioridad/SLA",
            "applicable": _has_any_assignment_col(df),
            "question": "Hay registros sin servicio, prioridad o SLA asignado?",
        },
        {
            "id": "critical_services",
            "name": "Ranking de servicios criticos",
            "applicable": has_service and (has_sla or has_resolution or has_risk or has_priority),
            "question": "Que servicios deberia revisar primero?",
        },
        {
            "id": "critical_priorities",
            "name": "Ranking de prioridades criticas",
            "applicable": has_priority and (has_sla or has_resolution or has_risk or has_service or has_category),
            "question": "Que prioridades o urgencias concentran mas riesgo?",
        },
        {
            "id": "sla_analysis",
            "name": "Analisis de SLA",
            "applicable": has_sla,
            "question": "Como esta el incumplimiento de SLA en esta ejecucion?",
        },
        {
            "id": "time_analysis",
            "name": "Analisis de tiempos",
            "applicable": has_resolution,
            "question": "Donde se concentran los mayores tiempos de resolucion?",
        },
        {
            "id": "outlier_analysis",
            "name": "Analisis de outliers",
            "applicable": has_outliers,
            "question": "Hay casos atipicos que convenga revisar por separado?",
        },
        {
            "id": "critical_clusters",
            "name": "Clusters criticos",
            "applicable": has_cluster,
            "question": "Que grupos detectados deberia revisar primero?",
        },
        {
            "id": "cluster_samples",
            "name": "Muestras representativas por grupo",
            "applicable": has_cluster,
            "question": "Me muestras ejemplos representativos de los grupos?",
        },
        {
            "id": "dashboard_findings",
            "name": "Hallazgos para dashboard",
            "applicable": True,
            "question": "Que hallazgos conviene llevar al dashboard?",
        },
        {
            "id": "business_recommendations",
            "name": "Recomendaciones de negocio",
            "applicable": has_cluster or has_service or has_sla or has_resolution or has_risk,
            "question": "Que acciones recomendadas puedo evaluar?",
        },
        {
            "id": "dynamic_questions",
            "name": "Preguntas sugeridas dinamicas",
            "applicable": True,
            "question": "Que preguntas puedo hacer con los datos cargados?",
        },
        {
            "id": "cluster_explanation",
            "name": "Explicacion de un cluster concreto",
            "applicable": has_cluster,
            "question": "Puedes explicar un grupo concreto?",
        },
        {
            "id": "cluster_comparison",
            "name": "Comparacion entre clusters",
            "applicable": cluster_count >= 2,
            "question": "Que diferencias hay entre dos grupos detectados?",
        },
        {
            "id": "equivalent_columns",
            "name": "Deteccion de columnas equivalentes",
            "applicable": has_columns,
            "question": "Que columnas parecen equivalentes a servicio, prioridad, SLA o tiempo?",
        },
        {
            "id": "next_steps",
            "name": "Guia de proximos pasos para usuario inexperto",
            "applicable": True,
            "question": "Por donde empiezo y que debo revisar paso a paso?",
        },
    ]


def _applicable_tools(df: pd.DataFrame, run_context: dict | None = None) -> list[dict[str, object]]:
    return [tool for tool in _tool_catalog(df, run_context) if bool(tool["applicable"])]


def build_suggested_questions_for_run(run_id: str, run_context: dict | None = None) -> list[str]:
    df = load_run_evidences(run_id)
    if df.empty:
        return FALLBACK_SUGGESTED_QUESTIONS
    return _suggested_questions_for_df(df, run_context=run_context)


def _suggested_questions_for_df(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> list[str]:
    questions: list[str] = [
        str(tool["question"])
        for tool in _applicable_tools(df, run_context)
        if tool.get("question")
    ]

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
    filtered = filtered[~_missing_mask(filtered[group_col])]
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
    filtered = filtered[~_missing_mask(filtered[group_col])]
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


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.isna()
    cleaned = series.astype(str).str.strip().str.lower()
    return series.isna() | cleaned.isin(
        {
            "",
            "nan",
            "none",
            "null",
            "sin dato",
            "sin asignar",
            "no asignado",
            "no asignada",
            "no aplica",
            "n/a",
            "unknown",
            "-",
        }
    )


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


def _source_type_label(source_type: str | None) -> str:
    if not source_type:
        return "fuente analizada"
    return SOURCE_TYPE_LABELS.get(str(source_type), str(source_type).replace("_", " "))


def _source_review_answer(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> tuple[str, list[InsightCandidate]]:
    run_context = run_context or {}
    source_name = str(run_context.get("source_name") or "").strip()
    source_type = _source_type_label(run_context.get("source_type"))
    project_name = str(run_context.get("project_name") or "").strip()
    project_strategy = str(run_context.get("project_strategy") or "").strip()
    active_source_id = str(run_context.get("source_id") or "").strip()
    scenario_sources = [
        item
        for item in (run_context.get("sources") or [])
        if isinstance(item, dict)
    ]

    source_col = _first_col(df, SOURCE_COLS)
    source_counts = _group_count(df, source_col, top_n=5) if source_col else pd.DataFrame()
    if not source_name and not source_counts.empty:
        first = source_counts.iloc[0]
        raw_name = str(first[source_col]).strip()
        if raw_name and raw_name.lower() not in {"tabular", "texto", "imagen", "multimodal"}:
            source_name = raw_name

    avg_sla = _mean(df, SLA_COLS)
    avg_resolution = _mean(df, RESOLUTION_COLS)
    avg_risk = _mean(df, RISK_COLS)
    service = _top_value(df, SERVICE_COLS)
    priority = _top_value(df, PRIORITY_COLS)
    category = _top_value(df, CATEGORY_COLS)
    outliers = int((df["cluster_label"] == -1).sum()) if "cluster_label" in df else 0
    clusters = (
        int(df[df["cluster_label"] >= 0]["cluster_label"].nunique())
        if "cluster_label" in df
        else 0
    )

    active_source = None
    if scenario_sources:
        for item in scenario_sources:
            if active_source_id and str(item.get("id") or "") == active_source_id:
                active_source = item
                break
        if active_source is None and source_name:
            for item in scenario_sources:
                if str(item.get("source_name") or "").strip() == source_name:
                    active_source = item
                    break

    if active_source and not source_name:
        source_name = str(active_source.get("source_name") or active_source.get("filename") or "").strip()

    if scenario_sources:
        source_intro = (
            f"El escenario tiene {len(scenario_sources)} fuentes cargadas"
        )
        if project_name:
            source_intro += f" en el proyecto \"{project_name}\""
        if source_name:
            source_intro += f". La ejecucion abierta ahora corresponde a \"{source_name}\""
        else:
            source_intro += ". La ejecucion abierta corresponde a una fuente tabular"
    elif source_name:
        source_intro = f"La fuente a revisar para esta ejecucion es \"{source_name}\""
    else:
        source_intro = "No tengo el nombre original del archivo en DuckDB; esta ejecucion aparece como fuente tabular"
    if project_name:
        source_intro += "" if scenario_sources else f" del proyecto \"{project_name}\""

    focus_parts = [
        f"{len(df)} registros analizados",
        f"{clusters} grupos detectados" if clusters else "grupos sin dato",
        f"{outliers} casos atipicos",
        f"SLA {_fmt_pct(avg_sla)}",
        f"tiempo medio {_fmt_hours(avg_resolution)}",
        f"riesgo {_fmt_number(avg_risk)}",
    ]
    business_focus = []
    if service:
        business_focus.append(f"servicio {service}")
    if category:
        business_focus.append(f"categoria {category}")
    if priority:
        business_focus.append(f"prioridad {priority}")

    answer = (
        f"{source_intro}. Para esta ejecucion, la fuente activa es de tipo {source_type}. "
        f"Conviene revisar esta ejecucion mirando primero: {', '.join(focus_parts)}. "
    )
    if scenario_sources:
        source_lines = []
        for index, item in enumerate(scenario_sources[:8], start=1):
            name = str(item.get("source_name") or item.get("filename") or f"fuente {index}").strip()
            kind = _source_type_label(item.get("source_type"))
            rows = item.get("n_rows")
            fmt = str(item.get("original_format") or "").strip().upper()
            status = str(item.get("processing_status") or "").strip()
            details = [kind]
            if fmt:
                details.append(fmt)
            if rows is not None:
                details.append(f"{rows} filas")
            if status:
                details.append(status)
            marker = "actual" if active_source_id and str(item.get("id") or "") == active_source_id else None
            suffix = f" ({marker})" if marker else ""
            source_lines.append(f"{index}) {name}{suffix}: " + ", ".join(details))
        answer += " Fuentes del escenario: " + "; ".join(source_lines) + ". "
        if len(scenario_sources) > 8:
            answer += f"Hay {len(scenario_sources) - 8} fuentes adicionales no listadas en esta respuesta. "
        if project_strategy == "per_source":
            answer += (
                "Como el escenario esta en modo analisis por fuente, cada archivo tabular genera una ejecucion "
                "con clusters independientes. Para comparar todas, revisa el historial del proyecto o usa "
                "unificado multifuente si quieres un solo clustering compartido. "
            )
        elif project_strategy == "merged":
            answer += (
                "Como el escenario esta en modo unificado multifuente, todas las fuentes tabulares se combinaron "
                "en un unico clustering compartido. Los clusters pueden mezclar filas de distintas fuentes; "
                "revisa _fuente_tipo y _fuente_nombre para interpretarlos. "
            )
        elif project_strategy == "unified":
            answer += (
                "Como el escenario esta en modo fuente principal, la lectura debe priorizar la fuente "
                "principal de incidencias; las demas fuentes tabulares no entraron al clustering. "
            )
    if business_focus:
        answer += "Dentro de esa fuente, el primer foco de negocio seria " + ", ".join(business_focus) + ". "
    answer += (
        "No tomes esta recomendacion como una decision automatica: es una guia para revisar las fuentes y los grupos detectados."
    )

    insights = []
    if scenario_sources:
        for index, item in enumerate(scenario_sources[:5], start=1):
            name = str(item.get("source_name") or item.get("filename") or f"fuente {index}").strip()
            rows = item.get("n_rows")
            insights.append(
                InsightCandidate(
                    id=f"source-review-{_safe_slug(name)}",
                    title=f"Fuente del escenario: {name}",
                    description=(
                        f"{name} contiene {rows or 'sin dato'} filas. "
                        "Revisar estado, columnas detectadas y ejecuciones asociadas."
                    ),
                    metric_label="records",
                    metric_value=_finite(rows),
                    dimension="source",
                    filter_kind="source",
                    filter_value=name,
                )
            )
    if not insights:
        insights.append(
            InsightCandidate(
                id=f"source-review-{_safe_slug(source_name or source_type)}",
                title=f"Fuente a revisar: {source_name or source_type}",
                description=(
                    f"{source_name or source_type} contiene {len(df)} registros analizados; "
                    f"prioriza SLA, tiempos, casos atipicos y los grupos con mayor riesgo."
                ),
                metric_label="records",
                metric_value=_finite(len(df)),
                dimension="source",
                filter_kind="source",
                filter_value=source_name or source_type,
            )
        )
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


def _missing_assignment_answer(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> tuple[str, list[InsightCandidate]]:
    service_col = _first_col(df, SERVICE_COLS)
    priority_col = _first_col(df, PRIORITY_COLS)
    source_name = str((run_context or {}).get("source_name") or "").strip()
    sources = [
        item
        for item in ((run_context or {}).get("sources") or [])
        if isinstance(item, dict)
    ]

    if not service_col and not priority_col:
        source_hint = f" En la ejecucion abierta figura la fuente {source_name}." if source_name else ""
        if sources:
            source_names = [
                str(item.get("source_name") or item.get("filename") or "").strip()
                for item in sources[:5]
            ]
            source_names = [name for name in source_names if name]
            if source_names:
                source_hint += " Fuentes del escenario: " + "; ".join(source_names) + "."
        return (
            "No encontre columnas equivalentes a servicio afectado ni prioridad en esta ejecucion. "
            "Por eso no puedo contar registros sin asignacion; primero hay que revisar el perfilado de columnas "
            "y confirmar que campos del archivo representan servicio y prioridad." + source_hint,
            [],
        )

    service_missing = (
        _missing_mask(df[service_col])
        if service_col
        else pd.Series([True] * len(df), index=df.index)
    )
    priority_missing = (
        _missing_mask(df[priority_col])
        if priority_col
        else pd.Series([True] * len(df), index=df.index)
    )
    both_missing = service_missing & priority_missing
    either_missing = service_missing | priority_missing
    missing_df = df[both_missing]
    either_df = df[either_missing]

    pieces = [
        f"registros analizados: {len(df)}",
        f"sin servicio: {int(service_missing.sum())}" if service_col else "servicio: columna no detectada",
        f"sin prioridad: {int(priority_missing.sum())}" if priority_col else "prioridad: columna no detectada",
        f"sin servicio y sin prioridad: {int(both_missing.sum())}",
        f"con algun faltante: {int(either_missing.sum())}",
    ]

    where_to_review = []
    if source_name:
        where_to_review.append(f"fuente actual: {source_name}")
    if service_col:
        where_to_review.append(f"columna de servicio: {service_col}")
    if priority_col:
        where_to_review.append(f"columna de prioridad: {priority_col}")
    id_col = _first_col(df, ("evidence_id", "incident_id", "id"))
    if id_col and not missing_df.empty:
        examples = _clean_text_series(missing_df[id_col]).head(5).tolist()
        if examples:
            where_to_review.append("ejemplos: " + ", ".join(examples))

    cluster_detail = ""
    if "cluster_label" in df.columns and not missing_df.empty:
        top_clusters = _group_count(missing_df, "cluster_label", top_n=3)
        if not top_clusters.empty:
            cluster_detail = " Se concentran principalmente en " + "; ".join(
                f"grupo {row['cluster_label']} ({int(row['count'])})"
                for _, row in top_clusters.iterrows()
            ) + "."

    if missing_df.empty:
        answer = (
            "No encontre registros que esten sin servicio y sin prioridad al mismo tiempo. "
            "Resumen de calidad de asignacion: " + "; ".join(pieces) + ". "
        )
    else:
        answer = (
            "Si hay registros sin servicio y sin prioridad. "
            "Resumen de calidad de asignacion: " + "; ".join(pieces) + ". "
        )
    if where_to_review:
        answer += "Revisalo en " + "; ".join(where_to_review) + ". "
    answer += (
        cluster_detail
        + " Siguiente paso: abrir una muestra de esos registros, validar si el faltante es real o viene de un mapeo de columnas, "
        "y luego decidir si conviene corregir la fuente o crear una regla de normalizacion."
    )

    insights = [
        InsightCandidate(
            id="quality-missing-service-priority",
            title="Registros sin servicio y prioridad",
            description=(
                f"{int(both_missing.sum())} registros no tienen servicio ni prioridad asignados. "
                "Revisar mapeo de columnas y muestra de tickets."
            ),
            metric_label="missing_service_priority",
            metric_value=_finite(both_missing.sum()),
            dimension="data_quality",
            filter_kind="data_quality",
            filter_value="missing_service_priority",
        )
    ]
    if int(either_missing.sum()) != int(both_missing.sum()):
        insights.append(
            InsightCandidate(
                id="quality-any-missing-assignment",
                title="Registros con algun dato de asignacion faltante",
                description=f"{int(either_missing.sum())} registros tienen servicio o prioridad sin informar.",
                metric_label="missing_assignment_any",
                metric_value=_finite(either_missing.sum()),
                dimension="data_quality",
                filter_kind="data_quality",
                filter_value="missing_assignment_any",
            )
        )
    return answer, insights


def _dashboard_findings_answer(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> tuple[str, list[InsightCandidate]]:
    candidates: list[InsightCandidate] = []

    def add_from(result: tuple[str, list[InsightCandidate]]) -> None:
        for item in result[1]:
            _add_unique(candidates, item)

    add_from(_decision_alternatives(df))
    if _has_cluster_signal(df):
        add_from(_clusters_answer(df))
    if _has_outlier_signal(df):
        add_from(_anomalies_answer(df))
    if _has_value_signal(df, SLA_COLS):
        add_from(_sla_answer(df, "sla"))
    if _has_value_signal(df, RESOLUTION_COLS):
        add_from(_resolution_answer(df, "tiempo"))
    if _has_value_signal(df, SERVICE_COLS):
        add_from(_services_answer(df))
    if _has_repeated_text_signal(df, ROOT_CAUSE_COLS):
        add_from(_root_cause_answer(df))
    if run_context and run_context.get("sources"):
        add_from(_source_review_answer(df, run_context))

    if not candidates:
        return (
            "Todavia no hay hallazgos suficientemente claros para llevar al dashboard. "
            "Primero revisaria calidad de datos, columnas detectadas y grupos generados.",
            [],
        )

    ranked = sorted(
        candidates,
        key=lambda item: item.metric_value if item.metric_value is not None else 0,
        reverse=True,
    )[:5]
    pieces = [
        f"{index + 1}) {item.title}: {item.description}"
        for index, item in enumerate(ranked)
    ]
    answer = (
        "Para el dashboard llevaria estos hallazgos porque son los mas accionables con los datos disponibles: "
        + " ".join(pieces)
        + " Siguiente paso: agrega solo los hallazgos que quieras explicar en el informe y usa el chat para pedir detalle de cada uno."
    )
    return answer, ranked


def _dynamic_questions_answer(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> tuple[str, list[InsightCandidate]]:
    tools = _applicable_tools(df, run_context)
    questions = [str(tool["question"]) for tool in tools[:10] if tool.get("question")]
    tool_names = [str(tool["name"]) for tool in tools[:10]]
    answer = (
        "Con las fuentes y columnas disponibles, estas son preguntas que la app puede responder con datos reales: "
        + "; ".join(questions)
        + ". Si preguntas por algo que no aparece, el asistente intentara resolverlo y te dira que dato falta si no es posible."
    )
    insights = [
        InsightCandidate(
            id="available-analytic-tools",
            title="Herramientas analiticas disponibles",
            description="Herramientas aplicables: " + "; ".join(tool_names),
            metric_label="available_tools",
            metric_value=_finite(len(tools)),
            dimension="assistant",
            filter_kind="tool_catalog",
            filter_value="available",
        )
    ]
    return answer, insights


def _columns_meaning_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    groups = [
        ("servicio", SERVICE_COLS, "permite ubicar el area, aplicacion o servicio afectado"),
        ("prioridad/severidad", PRIORITY_COLS, "sirve para ordenar urgencia o criticidad"),
        ("categoria", CATEGORY_COLS, "ayuda a describir el tipo de incidencia o dominio"),
        ("SLA", SLA_COLS, "permite medir incumplimiento o riesgo de servicio"),
        ("tiempo", RESOLUTION_COLS, "permite medir demora o esfuerzo de resolucion"),
        ("causa raiz", ROOT_CAUSE_COLS, "sirve para detectar motivos repetidos"),
        ("riesgo/impacto", RISK_COLS, "sirve para priorizar impacto de negocio"),
        ("canal/asignacion", CHANNEL_COLS, "ayuda a entender entrada o equipo responsable"),
    ]
    detected = []
    for label, aliases, meaning in groups:
        cols = [col for col in aliases if col in df.columns]
        if cols:
            detected.append(f"{label}: {', '.join(cols)} ({meaning})")

    other_cols = [
        col
        for col in df.columns
        if col not in {"run_id", "evidence_index", "x", "y", "cluster_label"}
    ][:12]
    if not detected:
        answer = (
            "Detecte columnas, pero no pude asociarlas claramente a servicio, prioridad, SLA, tiempos o causa raiz. "
            "Conviene revisar el diccionario de datos o mapear columnas equivalentes."
        )
    else:
        answer = (
            "Estas columnas parecen utiles para el analisis: "
            + "; ".join(detected)
            + ". Otras columnas disponibles para revisar: "
            + ", ".join(other_cols)
            + "."
        )
    insights = [
        InsightCandidate(
            id="columns-detected",
            title="Columnas detectadas",
            description=f"La ejecucion tiene {len(df.columns)} columnas; {len(detected)} grupos semanticos reconocidos.",
            metric_label="column_count",
            metric_value=_finite(len(df.columns)),
            dimension="schema",
            filter_kind="columns",
            filter_value="detected",
        )
    ]
    return answer, insights


def _data_quality_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    rows = []
    for col in df.columns:
        if col in {"run_id", "evidence_index", "x", "y"}:
            continue
        missing = int(_missing_mask(df[col]).sum())
        missing_rate = missing / len(df) if len(df) else 0
        if missing > 0:
            rows.append((col, missing, missing_rate))
    rows = sorted(rows, key=lambda item: (item[2], item[1]), reverse=True)[:8]
    if not rows:
        return (
            "No detecte faltantes relevantes en las columnas principales de esta ejecucion. Igual conviene revisar tipos de datos y valores atipicos.",
            [],
        )
    pieces = [
        f"{col}: {missing} registros ({rate * 100:.1f}%)"
        for col, missing, rate in rows
    ]
    answer = (
        "Las columnas con mas datos faltantes o no informados son "
        + "; ".join(pieces)
        + ". Siguiente paso: abrir el detalle de esas columnas en la fuente original y decidir si se corrigen, se imputan o se excluyen del analisis."
    )
    insights = [
        InsightCandidate(
            id=f"quality-missing-{_safe_slug(col)}",
            title=f"Faltantes en {col}",
            description=f"{missing} registros sin valor util en {col}.",
            metric_label="missing_rate",
            metric_value=_finite(rate),
            dimension="data_quality",
            filter_kind="column",
            filter_value=col,
        )
        for col, missing, rate in rows[:5]
    ]
    return answer, insights


def _source_comparison_answer(
    df: pd.DataFrame,
    run_context: dict | None = None,
) -> tuple[str, list[InsightCandidate]]:
    sources = _scenario_sources(run_context)
    if len(sources) < 2:
        return (
            "Esta ejecucion no tiene suficientes fuentes de escenario para comparar. Carga dos o mas fuentes en el proyecto para activar esta lectura.",
            [],
        )
    rows = []
    for item in sources:
        name = str(item.get("source_name") or item.get("filename") or "fuente").strip()
        kind = _source_type_label(item.get("source_type"))
        n_rows = item.get("n_rows")
        n_cols = item.get("n_cols")
        fmt = str(item.get("original_format") or "").upper()
        rows.append((name, kind, n_rows, n_cols, fmt))
    pieces = [
        f"{name}: {kind}, {n_rows or 'sin dato'} filas, {n_cols or 'sin dato'} columnas, {fmt or 'formato sin dato'}"
        for name, kind, n_rows, n_cols, fmt in rows[:8]
    ]
    answer = (
        "Comparacion de fuentes del escenario: "
        + "; ".join(pieces)
        + ". Para una comparacion analitica completa, revisa las ejecuciones generadas por cada fuente y compara clusters, faltantes y metricas."
    )
    insights = [
        InsightCandidate(
            id=f"source-compare-{_safe_slug(name)}",
            title=f"Fuente: {name}",
            description=f"{kind}, {n_rows or 'sin dato'} filas y {n_cols or 'sin dato'} columnas.",
            metric_label="source_rows",
            metric_value=_finite(n_rows),
            dimension="source",
            filter_kind="source",
            filter_value=name,
        )
        for name, kind, n_rows, n_cols, _fmt in rows[:5]
    ]
    return answer, insights


def _critical_services_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    service_col = _first_col(df, SERVICE_COLS)
    if not service_col:
        return ("No encontre una columna de servicio para construir un ranking critico.", [])
    rows = []
    for service, group in df.groupby(service_col, dropna=True):
        if _missing_mask(pd.Series([service])).iloc[0]:
            continue
        count = len(group)
        score = _decision_score(
            sla=_mean(group, SLA_COLS),
            resolution=_mean(group, RESOLUTION_COLS),
            risk=_mean(group, RISK_COLS),
            count=count,
        )
        rows.append((str(service), count, score, _mean(group, SLA_COLS), _mean(group, RESOLUTION_COLS)))
    rows = sorted(rows, key=lambda item: item[2], reverse=True)[:5]
    if not rows:
        return ("No hay datos suficientes para ordenar servicios criticos.", [])
    answer = "Servicios a revisar primero: " + "; ".join(
        f"{name}: {count} registros, score {_fmt_number(score)}, SLA {_fmt_pct(sla)}, tiempo {_fmt_hours(time)}"
        for name, count, score, sla, time in rows
    ) + "."
    insights = [
        InsightCandidate(
            id=f"critical-service-{_safe_slug(name)}",
            title=f"Servicio critico: {name}",
            description=f"{name} combina volumen, SLA, tiempo o riesgo con score {_fmt_number(score)}.",
            metric_label="critical_service_score",
            metric_value=_finite(score),
            dimension=service_col,
            filter_kind=service_col,
            filter_value=name,
        )
        for name, _count, score, _sla, _time in rows
    ]
    return answer, insights


def _critical_priorities_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    priority_col = _first_col(df, PRIORITY_COLS)
    if not priority_col:
        return ("No encontre una columna de prioridad o severidad para construir ranking.", [])
    rows = []
    for priority, group in df.groupby(priority_col, dropna=True):
        if _missing_mask(pd.Series([priority])).iloc[0]:
            continue
        score = _decision_score(
            sla=_mean(group, SLA_COLS),
            resolution=_mean(group, RESOLUTION_COLS),
            risk=_mean(group, RISK_COLS),
            count=len(group),
        )
        rows.append((str(priority), len(group), score, _mean(group, SLA_COLS), _mean(group, RESOLUTION_COLS)))
    rows = sorted(rows, key=lambda item: item[2], reverse=True)[:5]
    if not rows:
        return ("No hay datos suficientes para ordenar prioridades criticas.", [])
    answer = "Prioridades o urgencias a revisar primero: " + "; ".join(
        f"{name}: {count} registros, score {_fmt_number(score)}, SLA {_fmt_pct(sla)}, tiempo {_fmt_hours(time)}"
        for name, count, score, sla, time in rows
    ) + "."
    insights = [
        InsightCandidate(
            id=f"critical-priority-{_safe_slug(name)}",
            title=f"Prioridad critica: {name}",
            description=f"{name} tiene {count} registros y score {_fmt_number(score)}.",
            metric_label="critical_priority_score",
            metric_value=_finite(score),
            dimension=priority_col,
            filter_kind=priority_col,
            filter_value=name,
        )
        for name, count, score, _sla, _time in rows
    ]
    return answer, insights


def _cluster_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    if "cluster_label" not in df.columns:
        return []
    rows = []
    for cluster_label, group in df[df["cluster_label"] >= 0].groupby("cluster_label"):
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "count": len(group),
                "service": _top_value(group, SERVICE_COLS),
                "priority": _top_value(group, PRIORITY_COLS),
                "category": _top_value(group, CATEGORY_COLS),
                "sla": _mean(group, SLA_COLS),
                "resolution": _mean(group, RESOLUTION_COLS),
                "risk": _mean(group, RISK_COLS),
                "df": group,
            }
        )
    return sorted(
        rows,
        key=lambda row: _decision_score(
            sla=row["sla"], resolution=row["resolution"], risk=row["risk"], count=row["count"]
        ),
        reverse=True,
    )


def _extract_cluster_ids(question: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?:cluster|grupo)\s*#?\s*(-?\d+)", question)]


def _cluster_sample_answer(df: pd.DataFrame, question: str) -> tuple[str, list[InsightCandidate]]:
    rows = _cluster_rows(df)
    if not rows:
        return ("No encontre grupos con etiquetas de cluster para mostrar muestras.", [])
    requested = _extract_cluster_ids(question)
    row = next((item for item in rows if item["cluster_label"] in requested), rows[0])
    group = row["df"]
    id_col = _first_col(group, ("evidence_id", "incident_id", "id"))
    preview_col = _first_col(group, ("preview", "descripcion_corta", "description"))
    samples = []
    for _, item in group.head(5).iterrows():
        sample_id = str(item.get(id_col, "")) if id_col else ""
        preview = str(item.get(preview_col, ""))[:120] if preview_col else ""
        samples.append(f"{sample_id or 'registro'}: {preview or 'sin vista previa'}")
    answer = (
        f"Muestra del grupo {row['cluster_label']} ({row['count']} registros): "
        + "; ".join(samples)
        + ". Usa estos ejemplos para validar si el patron del grupo tiene sentido para negocio."
    )
    return answer, [
        InsightCandidate(
            id=f"cluster-sample-{row['cluster_label']}",
            title=f"Muestra del grupo {row['cluster_label']}",
            description=f"{row['count']} registros; servicio {row['service'] or 'sin dato'}; prioridad {row['priority'] or 'sin dato'}.",
            metric_label="cluster_records",
            metric_value=_finite(row["count"]),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value=str(row["cluster_label"]),
        )
    ]


def _cluster_explanation_answer(df: pd.DataFrame, question: str) -> tuple[str, list[InsightCandidate]]:
    rows = _cluster_rows(df)
    if not rows:
        return ("No encontre grupos con etiquetas de cluster para explicar.", [])
    requested = _extract_cluster_ids(question)
    row = next((item for item in rows if item["cluster_label"] in requested), rows[0])
    answer = (
        f"El grupo {row['cluster_label']} contiene {row['count']} registros. "
        f"Patron dominante: servicio {row['service'] or 'sin dato'}, prioridad {row['priority'] or 'sin dato'}, "
        f"categoria {row['category'] or 'sin dato'}, SLA {_fmt_pct(row['sla'])}, tiempo {_fmt_hours(row['resolution'])} "
        f"y riesgo {_fmt_number(row['risk'])}. Recomendacion: revisar una muestra y validar si el nombre de negocio del grupo es correcto."
    )
    return answer, [
        InsightCandidate(
            id=f"cluster-explain-{row['cluster_label']}",
            title=f"Explicacion del grupo {row['cluster_label']}",
            description=f"Servicio {row['service'] or 'sin dato'}, prioridad {row['priority'] or 'sin dato'}, {row['count']} registros.",
            metric_label="cluster_records",
            metric_value=_finite(row["count"]),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value=str(row["cluster_label"]),
        )
    ]


def _cluster_comparison_answer(df: pd.DataFrame, question: str) -> tuple[str, list[InsightCandidate]]:
    rows = _cluster_rows(df)
    if len(rows) < 2:
        return ("Necesito al menos dos grupos detectados para comparar clusters.", [])
    requested = _extract_cluster_ids(question)
    selected = [row for row in rows if row["cluster_label"] in requested][:2]
    if len(selected) < 2:
        selected = rows[:2]
    first, second = selected
    answer = (
        f"Comparacion de grupos: grupo {first['cluster_label']} tiene {first['count']} registros, "
        f"servicio {first['service'] or 'sin dato'}, prioridad {first['priority'] or 'sin dato'}, "
        f"SLA {_fmt_pct(first['sla'])} y tiempo {_fmt_hours(first['resolution'])}. "
        f"Grupo {second['cluster_label']} tiene {second['count']} registros, servicio {second['service'] or 'sin dato'}, "
        f"prioridad {second['priority'] or 'sin dato'}, SLA {_fmt_pct(second['sla'])} y tiempo {_fmt_hours(second['resolution'])}. "
        "Usa esta comparacion para decidir cual revisar primero o si ambos representan perfiles distintos."
    )
    insights = [
        InsightCandidate(
            id=f"cluster-compare-{row['cluster_label']}",
            title=f"Grupo {row['cluster_label']}",
            description=f"{row['count']} registros; servicio {row['service'] or 'sin dato'}; prioridad {row['priority'] or 'sin dato'}.",
            metric_label="cluster_records",
            metric_value=_finite(row["count"]),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value=str(row["cluster_label"]),
        )
        for row in selected
    ]
    return answer, insights


def _equivalent_columns_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    groups = [
        ("servicio", SERVICE_COLS),
        ("prioridad/severidad", PRIORITY_COLS),
        ("SLA", SLA_COLS),
        ("tiempo de resolucion", RESOLUTION_COLS),
        ("causa raiz", ROOT_CAUSE_COLS),
        ("riesgo/impacto", RISK_COLS),
        ("categoria", CATEGORY_COLS),
    ]
    pieces = []
    for label, aliases in groups:
        matches = [col for col in df.columns if col in aliases]
        if matches:
            pieces.append(f"{label}: {', '.join(matches)}")
    unmatched = [
        col
        for col in df.columns
        if col not in {"run_id", "evidence_index", "x", "y", "cluster_label"}
        and not any(col in aliases for _, aliases in groups)
    ][:10]
    answer = (
        "Mapeo automatico de columnas equivalentes: "
        + ("; ".join(pieces) if pieces else "no encontre equivalencias fuertes")
        + ". Columnas que podrian requerir revision manual: "
        + (", ".join(unmatched) if unmatched else "ninguna destacada")
        + "."
    )
    return answer, [
        InsightCandidate(
            id="equivalent-columns",
            title="Columnas equivalentes detectadas",
            description=f"{len(pieces)} grupos de equivalencias detectados.",
            metric_label="equivalence_groups",
            metric_value=_finite(len(pieces)),
            dimension="schema",
            filter_kind="columns",
            filter_value="equivalent",
        )
    ]


def _next_steps_answer(df: pd.DataFrame, run_context: dict | None = None) -> tuple[str, list[InsightCandidate]]:
    steps = ["1) Revisa que fuentes y columnas detecto la app."]
    if _has_any_assignment_col(df):
        steps.append("2) Revisa calidad de datos: servicio, prioridad y SLA faltantes.")
    else:
        steps.append("2) Mapea columnas equivalentes para servicio, prioridad y SLA si existen en la fuente.")
    if _has_cluster_signal(df):
        steps.append("3) Abre los grupos mas criticos y valida una muestra de tickets.")
    if _has_outlier_signal(df):
        steps.append("4) Revisa los casos atipicos por separado.")
    steps.append("5) Lleva al dashboard solo hallazgos que puedas explicar con datos reales.")
    answer = "Guia sugerida para empezar: " + " ".join(steps)
    return answer, [
        InsightCandidate(
            id="guided-next-steps",
            title="Guia de proximos pasos",
            description="Ruta sugerida para revisar el escenario sin conocimientos tecnicos previos.",
            metric_label="steps",
            metric_value=_finite(len(steps)),
            dimension="assistant",
            filter_kind="guide",
            filter_value="next_steps",
        )
    ]


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


def build_chat_response(
    run_id: str,
    question: str,
    *,
    run_context: dict | None = None,
    history: list[dict[str, str]] | None = None,
) -> ChatResponse:
    df = load_run_evidences(run_id)
    if df.empty:
        return ChatResponse(
            answer="No encontre incidencias materializadas en DuckDB para esta ejecucion. Ejecuta el pipeline nuevamente.",
            suggested_questions=FALLBACK_SUGGESTED_QUESTIONS,
        )

    normalized = _normalize(question)
    suggested_questions = _suggested_questions_for_df(df, run_context=run_context)
    answer_parts: list[str] = []
    insights: list[InsightCandidate] = []
    tool_summaries: list[dict] = []

    def collect(tool_name: str, result: tuple[str, list[InsightCandidate]]) -> None:
        answer, items = result
        answer_parts.append(answer)
        tool_summaries.append(_tool_summary(tool_name, answer, items))
        for item in items:
            _add_unique(insights, item)

    def has_any(*terms: str) -> bool:
        return any(term in normalized for term in terms)

    asks_dashboard = "dashboard" in normalized or (
        "hallazgo" in normalized
        and any(term in normalized for term in ["llevar", "agregar", "mostrar", "conviene"])
    )
    asks_missing_assignment = (
        any(
            term in normalized
            for term in [
                "sin servicio",
                "sin prioridad",
                "sin asignar",
                "no asignado",
                "no asignada",
                "no tienen asignado",
                "no tiene asignado",
                "faltante",
                "faltan",
                "no informado",
                "no informada",
            ]
        )
        and any(term in normalized for term in ["servicio", "prioridad", "asignacion"])
    )

    asks_dynamic_questions = (
        has_any("que preguntas", "preguntas puedo", "puedo preguntar", "preguntas sugeridas")
        or ("pregunta" in normalized and has_any("hacer", "suger"))
    )
    asks_overview = has_any(
        "que puedo analizar",
        "que puedo explorar",
        "resumen general",
        "vision general",
    )
    asks_columns = has_any("columna", "columnas", "variable", "variables", "campo", "campos")
    asks_equivalent_columns = asks_columns and has_any("equivalent", "mapear", "mapeo", "parecen", "significado")
    asks_data_quality = has_any(
        "calidad",
        "faltante",
        "faltan",
        "vacio",
        "vacios",
        "nulo",
        "nulos",
        "incompleto",
        "sin dato",
        "sin datos",
    )
    asks_source_comparison = has_any("fuente", "fuentes", "archivo", "archivos", "dataset", "datasets") and has_any(
        "compar", "diferencia", "relacion", "relaciones", "cruzar"
    )
    asks_source_inventory = has_any("fuente", "fuentes", "archivo", "archivos", "excel", "dataset", "origen")
    asks_critical_services = has_any("servicio", "servicios", "afectado") and has_any(
        "critic", "priorizar", "primero", "ranking", "revisar", "peor"
    )
    asks_critical_priorities = has_any("prioridad", "prioridades", "severidad", "urgencia") and has_any(
        "critic", "riesgo", "ranking", "concentran", "primero", "peor"
    )
    asks_cluster_compare = has_any("compar") and has_any("cluster", "clusters", "grupo", "grupos")
    asks_cluster_sample = has_any("muestra", "muestras", "ejemplo", "ejemplos", "representativo")
    asks_cluster_explain = has_any("explica", "explicar", "detalle", "significa", "entender") and has_any(
        "cluster", "clusters", "grupo", "grupos"
    )
    asks_clusters = has_any("cluster", "clusters", "grupo", "grupos", "agrupamiento")
    asks_next_steps = has_any(
        "por donde empiezo",
        "paso a paso",
        "proximo paso",
        "siguiente paso",
        "guiame",
        "guia",
        "que debo revisar",
        "donde revisar",
    )
    asks_recommendations = has_any(
        "decision",
        "decidir",
        "alternativa",
        "alternativas",
        "recomend",
        "accion",
        "acciones",
        "priorizar",
        "que hago",
    )

    if asks_dynamic_questions:
        collect("preguntas_dinamicas", _dynamic_questions_answer(df, run_context))

    if asks_overview:
        collect("resumen_general", _overview(df))

    if asks_source_comparison:
        collect("comparar_fuentes", _source_comparison_answer(df, run_context))
    elif asks_source_inventory:
        collect("analizar_fuentes", _source_review_answer(df, run_context))

    if asks_missing_assignment:
        collect("calidad_asignacion", _missing_assignment_answer(df, run_context))
    elif asks_data_quality:
        collect("calidad_datos", _data_quality_answer(df))

    if asks_equivalent_columns:
        collect("columnas_equivalentes", _equivalent_columns_answer(df))
    elif asks_columns:
        collect("columnas_detectadas", _columns_meaning_answer(df))

    if asks_dashboard:
        collect("hallazgos_dashboard", _dashboard_findings_answer(df, run_context))

    if asks_cluster_compare:
        collect("comparar_clusters", _cluster_comparison_answer(df, normalized))
    elif asks_cluster_sample:
        collect("muestras_clusters", _cluster_sample_answer(df, normalized))
    elif asks_cluster_explain:
        collect("explicar_cluster", _cluster_explanation_answer(df, normalized))
    elif asks_clusters:
        collect("analizar_clusters", _clusters_answer(df))

    if not asks_missing_assignment and has_any("sla", "incumpl"):
        collect("analizar_sla", _sla_answer(df, normalized))

    if not asks_missing_assignment and has_any("tiempo", "resolucion", "resolver", "demora"):
        collect("analizar_tiempos_resolucion", _resolution_answer(df, normalized))

    if not asks_missing_assignment and asks_critical_services:
        collect("ranking_servicios_criticos", _critical_services_answer(df))
    elif not asks_missing_assignment and has_any("servicio", "servicios", "afectado", "volumen"):
        collect("analizar_servicios", _services_answer(df))

    if not asks_missing_assignment and asks_critical_priorities:
        collect("ranking_prioridades_criticas", _critical_priorities_answer(df))
    elif not asks_missing_assignment and has_any("prioridad", "prioridades", "severidad", "urgencia", "riesgo", "impacto"):
        collect("analizar_prioridades", _priority_answer(df))

    if has_any("causa", "raiz", "root"):
        collect("analizar_causas_raiz", _root_cause_answer(df))

    if has_any("anomalia", "anomalo", "atipic", "outlier"):
        collect("analizar_anomalias", _anomalies_answer(df))

    if asks_recommendations and not asks_dashboard:
        collect("alternativas_decision", _decision_alternatives(df))

    if asks_next_steps:
        collect("guia_proximos_pasos", _next_steps_answer(df, run_context))

    if not answer_parts and has_any("que puedo", "analizar", "explorar", "resumen"):
        collect("resumen_general", _overview(df))

    if not answer_parts:
        collect("resumen_general", _overview(df))

    fallback_answer = " ".join(answer_parts)
    llm_result = explain_with_llm(
        question=question,
        tool_summaries=tool_summaries,
        fallback_answer=fallback_answer,
        conversation_history=history,
    )

    return ChatResponse(
        answer=llm_result.answer,
        suggested_questions=suggested_questions,
        insights=insights[:8],
        llm_used=llm_result.used,
        llm_mode=llm_result.mode,
        llm_detail=llm_result.detail,
    )
