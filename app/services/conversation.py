from __future__ import annotations

import math
import unicodedata

import pandas as pd

from app.schemas import ChatResponse, InsightCandidate
from app.services.duckdb_store import load_run_evidences


SUGGESTED_QUESTIONS = [
    "Que puedo analizar con esta data?",
    "Que grupos incumplen SLA?",
    "Que servicios concentran mas volumen?",
    "Que clusters tienen mayor riesgo?",
    "Que afecta los tiempos de resolucion?",
]


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
    return f"{number * 100:.1f}%"


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
    filtered = df[[group_col, metric_col]].dropna()
    if filtered.empty:
        return pd.DataFrame()
    grouped = (
        filtered.groupby(group_col, dropna=True)
        .agg(value=(metric_col, "mean"), count=(metric_col, "size"))
        .reset_index()
        .sort_values(["value", "count"], ascending=[ascending, False])
        .head(top_n)
    )
    return grouped


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


def _add_unique(insights: list[InsightCandidate], item: InsightCandidate) -> None:
    if not any(existing.id == item.id for existing in insights):
        insights.append(item)


def _overview(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    n_rows = len(df)
    clusters = sorted(
        int(c) for c in df["cluster_label"].dropna().unique().tolist() if int(c) >= 0
    )
    outliers = int((df["cluster_label"] == -1).sum()) if "cluster_label" in df else 0
    avg_sla = df["sla_breach_rate"].mean() if "sla_breach_rate" in df else None
    avg_resolution = (
        df["avg_resolution_hours"].mean() if "avg_resolution_hours" in df else None
    )
    avg_risk = (
        df["operational_risk_score"].mean()
        if "operational_risk_score" in df
        else None
    )
    answer = (
        f"Con {n_rows} evidencias puedo explorar volumen, SLA, tiempos de "
        f"resolucion, servicios afectados, severidad/riesgo, clusters y outliers. "
        f"Esta corrida tiene {len(clusters)} clusters y {outliers} outliers. "
        f"Promedios: SLA incumplido {_fmt_pct(avg_sla)}, resolucion "
        f"{_fmt_hours(avg_resolution)} y riesgo {_fmt_number(avg_risk)}."
    )
    insights = [
        InsightCandidate(
            id="overview-sla",
            title="SLA global",
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
    group_col = "assignment_group" if "grupo" in question or "equipo" in question else "affected_service"
    label = "grupos de asignacion" if group_col == "assignment_group" else "servicios"
    top = _group_mean(df, group_col, "sla_breach_rate")
    global_sla = df["sla_breach_rate"].mean() if "sla_breach_rate" in df else None
    if top.empty:
        return (
            f"El incumplimiento SLA global es {_fmt_pct(global_sla)}, pero no encontre una dimension confiable para desagregarlo.",
            [],
        )

    pieces = [
        f"{row[group_col]}: {_fmt_pct(row['value'])} ({int(row['count'])} evidencias)"
        for _, row in top.iterrows()
    ]
    answer = (
        f"El incumplimiento SLA global es {_fmt_pct(global_sla)}. "
        f"Los {label} con mayor incumplimiento son " + "; ".join(pieces) + "."
    )
    insights = [
        InsightCandidate(
            id=f"sla-{group_col}-{str(row[group_col]).lower().replace(' ', '-')}",
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


def _resolution_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    top = _group_mean(df, "affected_service", "avg_resolution_hours")
    avg_resolution = (
        df["avg_resolution_hours"].mean() if "avg_resolution_hours" in df else None
    )
    if top.empty:
        return (
            f"El tiempo promedio de resolucion es {_fmt_hours(avg_resolution)}, sin desglose por servicio disponible.",
            [],
        )
    pieces = [
        f"{row['affected_service']}: {_fmt_hours(row['value'])}"
        for _, row in top.iterrows()
    ]
    answer = (
        f"El tiempo promedio de resolucion es {_fmt_hours(avg_resolution)}. "
        f"Los servicios mas lentos son " + "; ".join(pieces) + "."
    )
    insights = [
        InsightCandidate(
            id=f"resolution-{str(row['affected_service']).lower().replace(' ', '-')}",
            title=f"Resolucion lenta en {row['affected_service']}",
            description=f"{row['affected_service']} promedia {_fmt_hours(row['value'])} de resolucion.",
            metric_label="avg_resolution_hours",
            metric_value=_finite(row["value"]),
            dimension="affected_service",
            filter_kind="affected_service",
            filter_value=str(row["affected_service"]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _services_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    top = _group_count(df, "affected_service")
    if top.empty:
        return ("No encontre una columna de servicio afectado para esta corrida.", [])
    pieces = [
        f"{row['affected_service']}: {int(row['count'])} evidencias"
        for _, row in top.iterrows()
    ]
    answer = "Los servicios con mas volumen son " + "; ".join(pieces) + "."
    insights = [
        InsightCandidate(
            id=f"volume-{str(row['affected_service']).lower().replace(' ', '-')}",
            title=f"Volumen concentrado en {row['affected_service']}",
            description=f"{row['affected_service']} concentra {int(row['count'])} evidencias.",
            metric_label="evidence_count",
            metric_value=_finite(row["count"]),
            dimension="affected_service",
            filter_kind="affected_service",
            filter_value=str(row["affected_service"]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _severity_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    top = _group_count(df, "severity", top_n=4)
    risk = (
        df["operational_risk_score"].mean()
        if "operational_risk_score" in df
        else None
    )
    if top.empty:
        return (
            f"El riesgo operativo promedio es {_fmt_number(risk)}, pero no hay severidad derivada para esta corrida.",
            [],
        )
    pieces = [f"{row['severity']}: {int(row['count'])}" for _, row in top.iterrows()]
    answer = (
        f"El riesgo operativo promedio es {_fmt_number(risk)}. "
        f"La distribucion de severidad es " + "; ".join(pieces) + "."
    )
    insights = [
        InsightCandidate(
            id=f"severity-{str(row['severity']).lower()}",
            title=f"Severidad {row['severity']}",
            description=f"Hay {int(row['count'])} evidencias con severidad {row['severity']}.",
            metric_label="evidence_count",
            metric_value=_finite(row["count"]),
            dimension="severity",
            filter_kind="severity",
            filter_value=str(row["severity"]),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def _clusters_answer(df: pd.DataFrame) -> tuple[str, list[InsightCandidate]]:
    if "cluster_label" not in df.columns:
        return ("Esta corrida no tiene etiquetas de cluster disponibles.", [])
    filtered = df[df["cluster_label"] >= 0]
    if filtered.empty:
        return ("HDBSCAN marco todas las evidencias como ruido/outliers.", [])
    grouped = (
        filtered.groupby("cluster_label")
        .agg(
            count=("cluster_label", "size"),
            sla=("sla_breach_rate", "mean"),
            risk=("operational_risk_score", "mean"),
            resolution=("avg_resolution_hours", "mean"),
        )
        .reset_index()
    )
    grouped["priority"] = (
        grouped["sla"].fillna(0) * 100
        + grouped["risk"].fillna(0)
        + grouped["resolution"].fillna(0)
    )
    top = grouped.sort_values("priority", ascending=False).head(3)
    pieces = [
        (
            f"cluster {int(row['cluster_label'])}: {int(row['count'])} evidencias, "
            f"SLA {_fmt_pct(row['sla'])}, riesgo {_fmt_number(row['risk'])}"
        )
        for _, row in top.iterrows()
    ]
    answer = "Los clusters mas prioritarios son " + "; ".join(pieces) + "."
    insights = [
        InsightCandidate(
            id=f"cluster-{int(row['cluster_label'])}",
            title=f"Cluster {int(row['cluster_label'])} prioritario",
            description=(
                f"Cluster {int(row['cluster_label'])}: SLA {_fmt_pct(row['sla'])}, "
                f"riesgo {_fmt_number(row['risk'])}, resolucion {_fmt_hours(row['resolution'])}."
            ),
            metric_label="priority_score",
            metric_value=_finite(row["priority"]),
            dimension="cluster_label",
            filter_kind="cluster_label",
            filter_value=str(int(row["cluster_label"])),
        )
        for _, row in top.iterrows()
    ]
    return answer, insights


def build_chat_response(run_id: str, question: str) -> ChatResponse:
    df = load_run_evidences(run_id)
    if df.empty:
        return ChatResponse(
            answer="No encontre evidencias materializadas en DuckDB para esta ejecucion. Ejecuta el pipeline nuevamente.",
            suggested_questions=SUGGESTED_QUESTIONS,
        )

    normalized = _normalize(question)
    answer_parts: list[str] = []
    insights: list[InsightCandidate] = []

    if any(term in normalized for term in ["que puedo", "analizar", "explorar"]):
        answer, items = _overview(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["sla", "incumpl"]):
        answer, items = _sla_answer(df, normalized)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["tiempo", "resolucion", "resolver"]):
        answer, items = _resolution_answer(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["servicio", "afectado", "volumen"]):
        answer, items = _services_answer(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["severidad", "critic", "riesgo", "impacto"]):
        answer, items = _severity_answer(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if any(term in normalized for term in ["cluster", "grupo similar", "outlier"]):
        answer, items = _clusters_answer(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    if not answer_parts:
        answer, items = _overview(df)
        answer_parts.append(answer)
        for item in items:
            _add_unique(insights, item)

    return ChatResponse(
        answer=" ".join(answer_parts),
        suggested_questions=SUGGESTED_QUESTIONS,
        insights=insights[:6],
    )

